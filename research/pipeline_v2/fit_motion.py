"""One unified motion fit: body -> feet -> hands (+ face stage), on the avatar's rig.

The forward kinematics reproduces the avatar's stored joints55_world exactly, so the
optimiser works in the parameterisation the renderer consumes. A trust-region prior
keeps the result close to the accepted initialisation where that was already good.

Outputs assets/human/motion_unified_v1/{motion.pt, report.json}
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
OUT = PROJECT / "assets" / "human" / "motion_unified_v1"

COCO2SX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
FEET_SX = [7, 8, 10, 11]
SCORE_MIN = 1.0


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
    return torch.stack(tg, dim=1) * scale + trans.unsqueeze(1)


def reprojection(P, joints):
    T = joints.shape[0]
    homo = torch.cat([joints, torch.ones(T, joints.shape[1], 1)], dim=-1)
    q = torch.einsum("vij,tnj->tvni", P, homo)
    return q[..., :2] / q[..., 2:3].clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[400, 200, 200])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(AVATAR, map_location="cpu", weights_only=False)
    rig, poses = bundle["rig"], bundle["poses"]
    rest, parents = rig["rest_joints"][0].float(), rig["parents"]
    scale = float(poses["scale"])
    frames = poses["frames"].long()
    T = len(frames)

    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    P = torch.tensor(np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in meta["views"]]), dtype=torch.float32)

    obs = np.load(OBS)
    xyz = torch.tensor(obs["xyz"], dtype=torch.float32)
    valid3d = torch.tensor(obs["valid"]) & torch.isfinite(xyz).all(-1)
    xy = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["xy"], dtype=torch.float32) for c in cams]).permute(1, 0, 2, 3)
    score = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["score"], dtype=torch.float32) for c in cams]).permute(1, 0, 2)

    root = poses["root_pose"].clone().requires_grad_(True)
    body = poses["body_pose"].clone().requires_grad_(True)
    lh = poses["lhand_pose"].clone().requires_grad_(True)
    rh = poses["rhand_pose"].clone().requires_grad_(True)
    trans = poses["translation_world"].clone().requires_grad_(True)
    root0, body0, trans0 = root.detach().clone(), body.detach().clone(), trans.detach().clone()

    def assemble():
        zero = torch.zeros(T, 3, 3, dtype=root.dtype)
        return torch.cat([root.unsqueeze(1), body, zero, lh, rh], dim=1)

    def losses(joints, stage):
        loss = torch.zeros((), dtype=torch.float32)
        for cj, sj in COCO2SX.items():
            m = valid3d[:, cj]
            if m.any():
                d = (joints[m, sj] - xyz[m, cj]).norm(dim=-1)
                w = torch.ones_like(d) * (2.0 if (stage >= 1 and sj in FEET_SX) else 1.0)
                loss = loss + torch.nn.functional.huber_loss(d, torch.zeros_like(d), reduction="mean", delta=0.02) * w.mean()
        uv = reprojection(P, joints)
        for cj, sj in COCO2SX.items():
            w = (score[:, :, cj] >= SCORE_MIN).float()
            if w.sum() == 0:
                continue
            err = (uv[:, :, sj] - xy[:, :, cj]).norm(dim=-1)
            per = torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=20.0)
            loss = loss + (per * w).sum() / w.sum() * (1.2 if stage >= 1 else 1.0)
        if T > 2:
            acc = (root[2:] - 2 * root[1:-1] + root[:-2]).abs().mean()
            acc_t = (trans[2:] - 2 * trans[1:-1] + trans[:-2]).abs().mean()
            acc_b = (body[2:] - 2 * body[1:-1] + body[:-2]).abs().mean()
            loss = loss + 0.5 * (acc + acc_t) + 0.05 * acc_b
        loss = loss + 0.15 * ((root - root0) ** 2).mean() + 0.10 * ((trans - trans0) ** 2).mean() + 0.05 * ((body - body0) ** 2).mean()
        return loss

    def evaluate(joints):
        uv = reprojection(P, joints)
        errs = []
        for cj, sj in COCO2SX.items():
            m = score[:, :, cj] >= SCORE_MIN
            if m.any():
                errs.append((uv[:, :, sj] - xy[:, :, cj]).norm(dim=-1)[m].mean())
        return float(torch.stack(errs).mean())

    with torch.no_grad():
        before = evaluate(fk(rest, parents, assemble(), scale, trans))
    print("baseline reproj", round(before, 2), "px", flush=True)
    started = time.time()
    history = []
    for stage, steps in enumerate(args.steps):
        if stage == 2:
            lh.requires_grad_(True); rh.requires_grad_(True)
        params = [root, body, trans] + ([lh, rh] if stage >= 2 else [])
        opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=0.004 if stage == 0 else 0.002)
        for it in range(steps):
            opt.zero_grad()
            joints = fk(rest, parents, assemble(), scale, trans)
            loss = losses(joints, stage)
            loss.backward()
            opt.step()
            if it % max(1, steps // 5) == 0 or it == steps - 1:
                with torch.no_grad():
                    history.append({"stage": stage, "iter": it, "loss": round(float(loss), 4),
                                    "reproj_px": round(evaluate(joints), 2)})
                    print(history[-1], flush=True)

    with torch.no_grad():
        joints = fk(rest, parents, assemble(), scale, trans)
        after = evaluate(joints)
    motion = {"frames": frames, "root_pose": root.detach(), "body_pose": body.detach(),
              "lhand_pose": lh.detach(), "rhand_pose": rh.detach(),
              "translation_world": trans.detach(), "scale": torch.tensor(scale),
              "betas": poses["betas"], "joints55_world": joints}
    torch.save(motion, out / "motion.pt")
    report = {"success": True, "source_avatar": str(AVATAR), "stage_steps": args.steps,
              "frames": int(T), "seconds": round(time.time() - started, 1),
              "mean_reproj_px_before": before, "mean_reproj_px_after": after,
              "history": history,
              "scope": "Body+feet+hands on the avatar's own rig; beta/scale frozen; trust-region prior "
                       "to the accepted initialisation. Face stage lives in fit_face_stage.py."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", report["seconds"], "s;", round(before, 2), "->", round(after, 2))


if __name__ == "__main__":
    main()
