"""Batch MAP with a singularity-free (6D) rotation parameterisation.

Root cause of the twitch was axis-angle: the fitted root sat on the 180 degree
singular surface and the optimiser bounced across it. Here every joint is optimised
as a 6D rotation (Gram-Schmidt -> matrix); the data term and the manifold smoothness
prior are the same as the previous batch MAP, with no anchor to another motion.
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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion

COCO2SX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


def sixd_to_matrix(v):
    a, b = v[..., :3], v[..., 3:]
    a = torch.nn.functional.normalize(a, dim=-1)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = torch.nn.functional.normalize(b, dim=-1)
    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], dim=-1)


def fk_mat(rest, parents, R, scale, trans):
    T = R.shape[0]
    Rg, tg = [None] * 55, [None] * 55
    for j in range(55):
        p = int(parents[j])
        if p < 0:
            Rg[j], tg[j] = R[:, j], rest[j].expand(T, 3)
        else:
            Rg[j] = Rg[p] @ R[:, j]
            tg[j] = tg[p] + (Rg[p] @ (rest[j] - rest[p]).view(1, 3, 1)).squeeze(-1)
    return torch.stack(tg, 1) * scale + trans.unsqueeze(1), torch.stack(Rg, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="assets/human/avatar_shape_v1")
    ap.add_argument("--motion", default="assets/human/motion_shape_fit_v2")
    ap.add_argument("--out", default="assets/human/motion_shape_6d_v1")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--w-pos", type=float, default=2.0)
    args = ap.parse_args()

    av = torch.load(PROJECT / args.avatar / "avatar.pt", map_location="cpu", weights_only=False)
    mo = torch.load(PROJECT / args.motion / "motion.pt", map_location="cpu", weights_only=False)
    rig, ps = av["rig"], av["poses"]
    rest = rig["rest_joints"][0].double(); parents = rig["parents"]; scale = float(mo["scale"])
    meta = json.loads((PROJECT / "assets/helmet_native000_299_v1/manifest.json").read_text(encoding="utf-8"))
    P = torch.tensor(np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in meta["views"]]), dtype=torch.float64)
    xy = torch.stack([torch.tensor(np.load(PROJECT / "assets/keypoints_all300_v1" / f"{v['serial']}.npz")["xy"], dtype=torch.float64) for v in meta["views"]]).permute(1, 0, 2, 3)
    sc = torch.stack([torch.tensor(np.load(PROJECT / "assets/keypoints_all300_v1" / f"{v['serial']}.npz")["score"], dtype=torch.float64) for v in meta["views"]]).permute(1, 0, 2)

    T = len(mo["root_pose"])
    aa0 = torch.cat([mo["root_pose"][:, None], mo["body_pose"],
                     torch.zeros(T, 3, 3, dtype=torch.float32), mo["lhand_pose"], mo["rhand_pose"]], 1)
    R0 = axis_angle_to_matrix(aa0.reshape(-1, 3).double()).reshape(T, 55, 3, 3)
    # initial 6D = first two columns flattened
    q0 = torch.cat([R0[..., 0], R0[..., 1]], dim=-1).clone()
    tr = mo["translation_world"].double().clone()

    q = q0.clone().requires_grad_(True)
    t = tr.clone().requires_grad_(True)
    with torch.no_grad():
        R_ = sixd_to_matrix(q0)
        dR_ = R_[1:] @ R_[:-1].transpose(-1, -2)
        w_ = matrix_to_axis_angle(dR_.reshape(-1, 3, 3)).reshape(T - 1, 55, 3)
        _, world_ = fk_mat(rest, parents, R_, scale, tr)
        rot0 = (w_[1:] - w_[:-1]).abs().mean().clamp_min(1e-9)
        pos0 = (world_[2:] - 2 * world_[1:-1] + world_[:-2]).abs().mean().clamp_min(1e-9)
        tr0 = (tr[2:] - 2 * tr[1:-1] + tr[:-2]).abs().mean().clamp_min(1e-9)
    print("normalisers", float(rot0), float(pos0), float(tr0), flush=True)
    cj = list(COCO2SX.keys()); sj = list(COCO2SX.values())
    mask = sc[:, :, cj] >= 1.0

    def reproj_and_positions():
        R = sixd_to_matrix(q)
        world, _ = fk_mat(rest, parents, R, scale, t)
        body = world[:, sj]
        homo = torch.cat([body, torch.ones(T, len(sj), 1, dtype=torch.float64)], -1)
        p = torch.einsum("vij,tnj->tvni", P, homo)
        uv = p[..., :2] / p[..., 2:3].clamp_min(1e-9)
        return uv, world

    def loss_fn():
        uv, world = reproj_and_positions()
        err = (uv - xy[:, :, cj]).norm(dim=-1)
        data = err[mask].mean()
        R = sixd_to_matrix(q)
        dR = R[1:] @ R[:-1].transpose(-1, -2)
        w = matrix_to_axis_angle(dR.reshape(-1, 3, 3)).reshape(T - 1, 55, 3)
        acc_rot = (w[1:] - w[:-1]).abs().mean()
        acc_pos = (world[2:] - 2 * world[1:-1] + world[:-2]).abs().mean()
        acc_tr = (t[2:] - 2 * t[1:-1] + t[:-2]).abs().mean()
        prior = acc_rot / rot0 + args.w_pos * acc_pos / pos0 + acc_tr / tr0
        return data + args.lam * prior, data

    opt = torch.optim.Adam([q, t], lr=3e-3)
    started = time.time()

    with torch.no_grad():
        _, d0 = loss_fn()
    print("initial data", float(d0), flush=True)
    for it in range(args.iters):
        opt.zero_grad()
        loss, dcur = loss_fn()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            print("iter", it, "data", round(float(dcur), 3), "loss", round(float(loss), 3), flush=True)
    with torch.no_grad():
        loss, d1 = loss_fn()
        R = sixd_to_matrix(q)
        world, _ = fk_mat(rest, parents, R, scale, t)
        dR = R[1:] @ R[:-1].transpose(-1, -2)
        ang = torch.rad2deg(matrix_to_axis_angle(dR.reshape(-1, 3, 3)).reshape(T - 1, 55, 3).norm(dim=-1))
        step = ang[:, [0, 3, 6, 9, 12, 15]]
    print("final data", float(d1), "root step p95", float(torch.quantile(step[:, 0], 0.95)), "max", float(step[:, 0].max()), flush=True)

    aa = matrix_to_axis_angle(R.reshape(-1, 3, 3)).reshape(T, 55, 3)
    out = {"frames": mo["frames"], "root_pose": aa[:, 0].float().detach(), "body_pose": aa[:, 1:22].float().detach(),
           "lhand_pose": aa[:, 25:40].float().detach(), "rhand_pose": aa[:, 40:55].float().detach(),
           "translation_world": t.float().detach(), "scale": torch.tensor(scale),
           "betas": mo["betas"], "joints55_world": world.float().detach()}
    d = PROJECT / args.out; d.mkdir(parents=True, exist_ok=False)
    torch.save(out, d / "motion.pt")
    (d / "report.json").write_text(json.dumps({
        "success": True, "parameterisation": "6D rotation for all 55 joints, Gram-Schmidt to matrix",
        "anchor": None, "lam": args.lam, "w_pos": args.w_pos, "iters": args.iters,
        "data_initial_px": float(d0), "data_final_px": float(d1),
        "root_step_deg": {"p95": float(torch.quantile(step[:, 0], 0.95)), "max": float(step[:, 0].max())},
        "seconds": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")
    print("saved", d)


if __name__ == "__main__":
    main()
