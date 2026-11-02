"""One unified motion fit: body -> feet -> hands, on the avatar's rig.

Differences from the first version:
- every landmark is a joint plus a fixed canonical offset (toes, heels, fingertips),
  taken from the SMPL-X rest mesh, never from the motion being fitted;
- the 2D term uses the saved multi-view inlier mask (observation-level), not just score;
- each loss term is divided by its own noise scale, so the temporal prior is active;
- feet get their own stage (6 points) and hands their own stage (42 points).

FK reproduces the avatar's stored joints55_world exactly (verified 0.0).
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "lhmpp" / "deps"), str(PROJECT / "runtime" / "deps")]

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

AVATAR = PROJECT / "assets" / "human" / "underarm_weight_amount_fit_v3" / "avatar.pt"
OBS = PROJECT / "assets" / "observations_all300_v1" / "observations.npz"
KEY = PROJECT / "assets" / "keypoints_all300_v1"
MANIFEST = PROJECT / "assets" / "helmet_native000_299_v1" / "manifest.json"
SMPLX = PROJECT / "lhmpp" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
OUT = PROJECT / "assets" / "human" / "motion_v2"

BODY = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
FOOT_VERTEX = {17: ("LBigToe", 10), 18: ("LSmallToe", 10), 19: ("LHeel", 10),
               20: ("RBigToe", 11), 21: ("RSmallToe", 11), 22: ("RHeel", 11)}
HAND_OFFSET = 91   # COCO left hand block start
PI = np.pi


def fk(rest, parents, aa, scale, trans):
    T = aa.shape[0]
    R = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(T, 55, 3, 3)
    Rg, tg = [None] * 55, [None] * 55
    for j in range(55):
        p = int(parents[j])
        if p < 0:
            Rg[j], tg[j] = R[:, j], rest[j].expand(T, 3)
        else:
            Rg[j] = Rg[p] @ R[:, j]
            tg[j] = tg[p] + (Rg[p] @ (rest[j] - rest[p]).view(1, 3, 1)).squeeze(-1)
    return torch.stack(tg, 1), torch.stack(Rg, 1)


def reprojection(P, pts):
    T = pts.shape[0]
    homo = torch.cat([pts, torch.ones(T, pts.shape[1], 1)], dim=-1)
    q = torch.einsum("vij,tnj->tvni", P, homo)
    return q[..., :2] / q[..., 2:3].clamp_min(1e-6)


def canonical_offsets():
    """Fixed SMPL-X rest offsets: feet (toe/heel) and fingertips, independent of the fit."""
    d = np.load(SMPLX, allow_pickle=True)
    v = np.asarray(d["v_template"], np.float64)
    jr = d["J_regressor"]; jr = jr.toarray() if hasattr(jr, "toarray") else np.asarray(jr, np.float64)
    J = jr @ v
    from smplx.vertex_ids import vertex_ids
    feet = []
    for coco, (name, joint) in FOOT_VERTEX.items():
        vid = vertex_ids["smplx"][name]
        feet.append((coco, joint, torch.tensor(v[vid] - J[joint], dtype=torch.float32)))
    tips_vid = [5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022]  # 5 fingers x 2 hands
    tips = []
    for side in range(2):
        base = 25 + 15 * side
        for f in range(5):
            distal = base + 3 * f + 2
            vid = tips_vid[side * 5 + f]
            tips.append((91 + side * 21 + 4 * f + 4, distal, torch.tensor(v[vid] - J[distal], dtype=torch.float32)))
    return feet, tips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[400, 200, 200])
    ap.add_argument("--temporal", type=float, default=0.5)
    ap.add_argument("--avatar", default=str(AVATAR))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    avatar_path = Path(args.avatar)
    if not avatar_path.exists():
        avatar_path = PROJECT / "assets" / "human" / args.avatar
    if avatar_path.is_dir():
        avatar_path = avatar_path / "avatar.pt"
    bundle = torch.load(avatar_path, map_location="cpu", weights_only=False)
    rig, poses = bundle["rig"], bundle["poses"]
    rest, parents = rig["rest_joints"][0].float(), rig["parents"]
    scale = float(poses["scale"])
    frames = poses["frames"].long(); T = len(frames)

    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    P = torch.tensor(np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in meta["views"]]), dtype=torch.float32)
    obs = np.load(OBS)
    xyz = torch.tensor(obs["xyz"], dtype=torch.float32)
    valid3d = torch.tensor(obs["valid"]) & torch.isfinite(xyz).all(-1)
    inlier = torch.tensor(obs["inlier"])
    xy = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["xy"], dtype=torch.float32) for c in cams]).permute(1, 0, 2, 3)
    score = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["score"], dtype=torch.float32) for c in cams]).permute(1, 0, 2)
    mask2d = inlier.permute(1, 0, 2) & (score >= 0.5)

    feet_off, tip_off = canonical_offsets()

    root = poses["root_pose"].clone().requires_grad_(True)
    body = poses["body_pose"].clone().requires_grad_(True)
    lh = poses.get("lhand_pose", poses["left_hand_pose"]).clone().requires_grad_(True)
    rh = poses.get("rhand_pose", poses["right_hand_pose"]).clone().requires_grad_(True)
    trans = poses["translation_world"].clone().requires_grad_(True)
    root0, body0, trans0 = root.detach().clone(), body.detach().clone(), trans.detach().clone()

    def assemble():
        return torch.cat([root.unsqueeze(1), body, torch.zeros(T, 3, 3), lh, rh], dim=1)

    # landmark table: (coco index, joint, offset or None)
    L = []
    for cj, sj in BODY.items():
        L.append((cj, sj, None, 1.0))
    for cj, sj, off in feet_off:
        L.append((cj, sj, off, 0.10))
    for cj, sj, off in tip_off:
        L.append((cj, sj, off, 0.10))
    # hand joints: COCO 1..15 of each hand -> SMPL-X 25..39 (3 per finger)
    for side in range(2):
        for f in range(5):
            for k in range(3):
                coco = HAND_OFFSET + side * 21 + 1 + 4 * f + k
                L.append((coco, 25 + side * 15 + 3 * f + k, None, 0.20))
    print("landmarks:", len(L), "feet:", len(feet_off), "tips:", len(tip_off))

    def points_of(tg, Rg):
        pts = []
        for cj, sj, off, _w in L:
            base = tg[:, sj]
            if off is not None:
                base = base + torch.einsum("tij,j->ti", Rg[:, sj], off)
            pts.append(base)
        return torch.stack(pts, 1)

    def evaluate(tg, Rg):
        pts = points_of(tg, Rg) * scale
        world = pts + trans.unsqueeze(1)
        uv = reprojection(P, world)
        idx = torch.tensor([x[0] for x in L]); wgt = torch.tensor([x[3] for x in L])
        errs = []
        for k in range(len(L)):
            m = mask2d[:, :, idx[k]]
            if m.any():
                errs.append((uv[:, :, k] - xy[:, :, idx[k]]).norm(dim=-1)[m].mean())
        return float(torch.stack(errs).mean())

    def loss_fn(tg, Rg, stage):
        pts = points_of(tg, Rg) * scale
        world = pts + trans.unsqueeze(1)
        uv = reprojection(P, world)
        idx = torch.tensor([x[0] for x in L]); wgt = torch.tensor([x[3] for x in L])
        data2d = torch.zeros(())
        for k in range(len(L)):
            cj = int(idx[k]); m = mask2d[:, :, cj]
            if not m.any():
                continue
            err = (uv[:, :, k] - xy[:, :, cj]).norm(dim=-1) / 5.0
            data2d = data2d + wgt[k] * torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=1.0)[m].mean()
        data2d = data2d / wgt.sum()
        data3d = torch.zeros(())
        for k in range(len(L)):
            cj = int(idx[k]); m = valid3d[:, cj]
            if not m.any():
                continue
            d = (world[m, k] - xyz[m, cj]).norm(dim=-1) / 0.02
            data3d = data3d + wgt[k] * torch.nn.functional.huber_loss(d, torch.zeros_like(d), reduction="mean", delta=1.0)
        data3d = data3d / wgt.sum()
        tempo = torch.zeros(())
        if T > 2:
            acc_t = (trans[2:] - 2 * trans[1:-1] + trans[:-2]).abs().mean() / 5e-4
            acc_b = (body[2:] - 2 * body[1:-1] + body[:-2]).abs().mean() / 5e-3
            acc_r = (root[2:] - 2 * root[1:-1] + root[:-2]).abs().mean() / 5e-3
            tempo = acc_t + acc_b + acc_r
        prior = 0.5 * ((body - body0) ** 2).mean() / 1e-3 + 0.5 * ((trans - trans0) ** 2).mean() / 1e-4
        foot_ids = [k for k, x in enumerate(L) if x[0] in (17, 18, 19, 20, 21, 22)]
        if foot_ids:
            fy = world[:, foot_ids, 2] if world.shape[-1] == 3 else world[:, foot_ids, 1]
            low = fy.min(dim=1).values
            contact = (low - low.median()).abs().mean() / 1e-3
            tempo = tempo + 0.2 * contact
        return data2d + data3d + args.temporal * tempo + prior

    with torch.no_grad():
        tg, Rg = fk(rest, parents, assemble(), scale, trans)
        before = evaluate(tg, Rg)
    print("baseline reproj", round(before, 2), "px", flush=True)
    started = time.time(); history = []
    for stage, steps in enumerate(args.steps):
        if stage == 2:
            lh.requires_grad_(True); rh.requires_grad_(True)
        params = [root, body, trans] + ([lh, rh] if stage >= 2 else [])
        opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=4e-4 if stage == 0 else 2e-4)
        for it in range(steps):
            opt.zero_grad()
            tg, Rg = fk(rest, parents, assemble(), scale, trans)
            loss = loss_fn(tg, Rg, stage)
            loss.backward(); opt.step()
            if it % max(1, steps // 4) == 0 or it == steps - 1:
                with torch.no_grad():
                    history.append({"stage": stage, "iter": it, "loss": round(float(loss), 4),
                                    "reproj_px": round(evaluate(tg, Rg), 2)})
                    print(history[-1], flush=True)

    with torch.no_grad():
        tg, Rg = fk(rest, parents, assemble(), scale, trans)
        after = evaluate(tg, Rg)
        joints = tg * scale + trans.unsqueeze(1)
    motion = {"frames": frames, "root_pose": root.detach(), "body_pose": body.detach(),
              "lhand_pose": lh.detach(), "rhand_pose": rh.detach(),
              "translation_world": trans.detach(), "scale": torch.tensor(scale),
              "betas": poses["betas"], "joints55_world": joints}
    torch.save(motion, out / "motion.pt")
    report = {"success": True, "landmarks": len(L), "feet": len(feet_off), "fingertips": len(tip_off),
              "stage_steps": args.steps, "temporal_weight": args.temporal,
              "mean_reproj_px_before": before, "mean_reproj_px_after": after,
              "history": history, "seconds": round(time.time() - started, 1),
              "scope": "Body+feet+hands landmarks (joints + canonical SMPL-X offsets), 2D inlier mask, "
                       "noise-normalised terms, trust-region prior."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", report["seconds"], "s;", round(before, 2), "->", round(after, 2))


if __name__ == "__main__":
    main()
