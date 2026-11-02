"""Face stage: refine neck/head rotation against the 5 face keypoints.

The rig has no face landmarks, so head-local offsets for nose/eyes/ears are learned
from the current motion (median of triangulated points in the head frame), then the
face stage optimises only neck (body_pose[11]) and head (body_pose[14]) so those
points reproject onto the detected 2D keypoints in all eight views.

Input : assets/human/motion_unified_v1/{motion.pt,avatar.pt}, assets/observations_all300_v1,
        assets/keypoints_all300_v1, helmet manifest
Output: assets/human/motion_face_v1/{motion.pt,report.json}
"""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "lhmpp" / "deps"), str(PROJECT / "runtime" / "deps")]

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

SRC = PROJECT / "assets" / "human" / "motion_unified_v1"
OUT = PROJECT / "assets" / "human" / "motion_face_v1"
OBS = PROJECT / "assets" / "observations_all300_v1" / "observations.npz"
KEY = PROJECT / "assets" / "keypoints_all300_v1"
MANIFEST = PROJECT / "assets" / "helmet_native000_299_v1" / "manifest.json"
NECK, HEAD = 11, 14          # body_pose indices for joints 12 and 15
FACE_KP = [0, 1, 2, 3, 4]    # nose, eyes, ears


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = torch.load(SRC / "motion.pt", map_location="cpu", weights_only=False)
    av = torch.load(SRC / "avatar.pt", map_location="cpu", weights_only=False)
    rest, parents = av["rig"]["rest_joints"][0].float(), av["rig"]["parents"]
    scale = float(m["scale"])
    root = m["root_pose"].clone(); body = m["body_pose"].clone()
    lh = m["lhand_pose"].clone(); rh = m["rhand_pose"].clone(); trans = m["translation_world"].clone()
    T = len(root)

    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    P = torch.tensor(np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in meta["views"]]), dtype=torch.float32)
    xy = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["xy"][:, FACE_KP], dtype=torch.float32) for c in cams]).permute(1, 0, 2, 3)
    score = torch.stack([torch.tensor(np.load(KEY / f"{c}.npz")["score"][:, FACE_KP], dtype=torch.float32) for c in cams]).permute(1, 0, 2)
    obs = np.load(OBS)
    xyz = torch.tensor(obs["xyz"][:, FACE_KP], dtype=torch.float32)
    valid3d = torch.tensor(obs["valid"][:, FACE_KP]) & torch.isfinite(xyz).all(-1)

    def assemble(body_):
        zero = torch.zeros(T, 3, 3)
        return torch.cat([root.unsqueeze(1), body_, zero, lh, rh], dim=1)

    with torch.no_grad():
        tg, Rg = fk(rest, parents, assemble(body), scale, trans)
        head_w = tg[:, 15] * scale + trans
        local = torch.einsum("tji,tkj->tki", Rg[:, 15], (xyz - head_w.unsqueeze(1)) / scale)
        # local shape (T,5,3); median over valid frames
        L = []
        for k in range(len(FACE_KP)):
            v = local[valid3d[:, k], k]
            L.append(v.median(0).values if len(v) else torch.zeros(3))
        L = torch.stack(L)                                                   # (5,3)
        print("head-local landmarks:", np.round(L.numpy(), 4).tolist())

    d_neck = torch.zeros(T, 3, requires_grad=True)
    d_head = torch.zeros(T, 3, requires_grad=True)
    opt = torch.optim.Adam([d_neck, d_head], lr=0.01)
    history = []

    def face_loss():
        body_ = body.clone()
        body_[:, NECK] = body[:, NECK] + d_neck
        body_[:, HEAD] = body[:, HEAD] + d_head
        tg2, Rg2 = fk(rest, parents, assemble(body_), scale, trans)
        head2 = tg2[:, 15] * scale + trans
        world = head2.unsqueeze(1) + scale * torch.einsum("tij,kj->tki", Rg2[:, 15], L)
        homo = torch.cat([world, torch.ones(T, len(FACE_KP), 1)], -1)
        q = torch.einsum("vij,tkj->tvki", P, homo)
        uv = q[..., :2] / q[..., 2:3].clamp_min(1e-6)
        w = (score >= 0.5).float()
        err = (uv - xy).norm(dim=-1)
        data = (torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=20.0) * w).sum() / w.sum().clamp_min(1.0)
        reg = 0.002 * (d_neck ** 2).sum() / T + 0.002 * (d_head ** 2).sum() / T
        acc = ((d_neck[2:] - 2 * d_neck[1:-1] + d_neck[:-2]) ** 2).mean() + ((d_head[2:] - 2 * d_head[1:-1] + d_head[:-2]) ** 2).mean()
        return data + reg + 5.0 * acc, err, w

    started = time.time()
    with torch.no_grad():
        _, err0, w0 = face_loss()
        before = float(((err0 * w0).sum() / w0.sum()).item())
    for it in range(200):
        opt.zero_grad()
        loss, _, _ = face_loss()
        loss.backward()
        opt.step()
        if it % 40 == 0 or it == 199:
            with torch.no_grad():
                _, err, w = face_loss()
                history.append({"iter": it, "loss": round(float(loss), 4),
                                "face_reproj_px": round(float((err * w).sum() / w.sum()), 3)})
                print(history[-1], flush=True)

    with torch.no_grad():
        body_f = body.clone(); body_f[:, NECK] += d_neck; body_f[:, HEAD] += d_head
        tg2, _ = fk(rest, parents, assemble(body_f), scale, trans)
        _, err, w = face_loss()
        after = float((err * w).sum() / w.sum())
        out = {k: m[k] for k in m}
        out["body_pose"] = body_f
        out["joints55_world"] = (tg2 * scale + trans.unsqueeze(1))
    torch.save(out, OUT / "motion.pt")
    report = {"success": True, "source": str(SRC), "face_keypoints": FACE_KP,
              "face_reproj_px_before": before, "face_reproj_px_after": after,
              "neck_head_delta_mean": float(d_neck.abs().mean() + d_head.abs().mean()),
              "history": history, "seconds": round(time.time() - started, 1),
              "scope": "Neck/head only; head-local nose/eye/ear offsets learned from the current motion. "
                       "Weak 5-point constraint, not a dense face fit."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", report["seconds"], "s; face reproj", round(before, 2), "->", round(after, 2))


if __name__ == "__main__":
    main()
