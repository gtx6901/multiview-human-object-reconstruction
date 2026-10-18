"""Refit the worn helmet attachment on a new head surface.

The joint-frame transport preserves the helmet's transform relative to the head joint,
which is only exact when the skull geometry is unchanged. Here we instead put the
helmet vertices and the avatar's head Gaussians in the *head-local* frame (where the
head geometry is static) and solve one constant rigid correction that seats the helmet
on the new surface, keeping the held phase untouched.

Input : avatar (rig+poses), helmet sequence, phase report
Output: assets/helmet/<out>/{sequence.pt, report.json}
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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def resolve(name):
        for base in (PROJECT, PROJECT / "assets", PROJECT / "assets" / "helmet", PROJECT / "assets" / "human"):
            q = base / name
            if q.exists():
                return q
        return PROJECT / "assets" / name
    av = torch.load(resolve(args.avatar) / "avatar.pt", map_location="cpu", weights_only=False)
    m = torch.load(resolve(args.motion) / "sequence.pt", map_location="cpu", weights_only=False)
    phases = json.loads((resolve(args.phases) / "report.json").read_text(encoding="utf-8"))["phases"]
    rig, poses = av["rig"], av["poses"]
    rest, parents = rig["rest_joints"][0].float(), rig["parents"]
    scale = float(poses["scale"])
    T = len(poses["frames"])
    zero = torch.cat([poses["root_pose"][:, None], poses["body_pose"], torch.zeros(T, 3, 3),
                      poses.get("lhand_pose", poses["left_hand_pose"]),
                      poses.get("rhand_pose", poses["right_hand_pose"])], 1)
    tg, Rg = fk(rest, parents, zero, scale, poses["translation_world"])
    head_pos = tg[:, 15] * scale + poses["translation_world"]
    head_R = Rg[:, 15]

    # head Gaussians in the head-local frame (static)
    w = rig["weights"].float()
    coords = rig["zero_coords"].float()
    rel = coords - rest[15]
    near = (rel.norm(dim=-1) < 0.16) & (rel[:, 1] > 0.02)
    mask = (w[:, 15] > 0.6) & near
    idx = torch.nonzero(mask).squeeze(-1)
    if len(idx) > 4000:
        idx = idx[torch.linspace(0, len(idx) - 1, 4000).long()]
    g_local = coords[idx] - rest[15]
    print("head gaussians", len(idx), flush=True)

    verts = m["vertices_local"].float() * float(torch.exp(m["logscale"]))
    if len(verts) > 2500:
        vidx = torch.linspace(0, len(verts) - 1, 2500).long()
        verts = verts[vidx]

    attached = torch.zeros(T, dtype=torch.bool)
    for ph in phases:
        attached[ph["start"]:ph["stop_exclusive"]] = True
    worn = torch.nonzero(attached).squeeze(-1)
    sample = worn[::max(1, len(worn) // 24)]
    print("worn frames", len(worn), "samples", len(sample), flush=True)

    # helmet vertices in the head frame, per sample frame (independent of the correction)
    def to_head(frames):
        out = []
        for t in frames:
            Rw, tw = m["R"][t], m["translation_world"][t]
            world = verts @ Rw.T + tw
            out.append((world - head_pos[t]) @ head_R[t] / scale)
        return torch.stack(out)

    vh = to_head(sample)                       # (S,V,3)
    target = g_local.unsqueeze(0).expand(len(sample), -1, -1)

    def chamfer(src):
        d = torch.cdist(src, target)            # (S,V,G)
        return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()

    dR = torch.zeros(3, requires_grad=True)
    dt = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([dR, dt], lr=0.01)
    started = time.time()
    best = None
    for it in range(300):
        opt.zero_grad()
        R = axis_angle_to_matrix(dR)
        src = torch.einsum("ij,svj->svi", R, vh) + dt
        loss = chamfer(src) + 20.0 * (dR ** 2).sum() + 200.0 * (dt ** 2).sum()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == 299:
            with torch.no_grad():
                R = axis_angle_to_matrix(dR)
                src = torch.einsum("ij,svj->svi", R, vh) + dt
                print({"iter": it, "chamfer": round(float(chamfer(src)), 5),
                       "dR_deg": round(float(torch.rad2deg(dR.norm())), 2),
                       "dt": [round(float(x), 4) for x in dt]}, flush=True)
                best = (dR.detach().clone(), dt.detach().clone())

    dR, dt = best
    Rc = axis_angle_to_matrix(dR)
    Rn, tn = m["R"].clone(), m["translation_world"].clone()
    for t in worn.tolist():
        local_R = head_R[t].T @ m["R"][t]
        local_t = head_R[t].T @ (m["translation_world"][t] - head_pos[t]) / scale
        new_local_R = Rc @ local_R
        new_local_t = Rc @ local_t + dt
        Rn[t] = head_R[t] @ new_local_R
        tn[t] = head_pos[t] + scale * (head_R[t] @ new_local_t)
    out = resolve(args.out) if (PROJECT / args.out).exists() is False else PROJECT / args.out
    out = PROJECT / "assets" / args.out if not str(args.out).startswith("assets/") else PROJECT / args.out
    out.mkdir(parents=True, exist_ok=False)
    torch.save({**m, "R": Rn, "translation_world": tn}, out / "sequence.pt")
    report = {"success": True, "avatar": args.avatar, "motion": args.motion,
              "worn_frames": int(len(worn)), "sample_frames": int(len(sample)),
              "correction_rotation_deg": float(torch.rad2deg(dR.norm())),
              "correction_translation_model": [float(x) for x in dt],
              "seconds": round(time.time() - started, 1),
              "scope": "Constant head-local rigid correction fitted to the new head Gaussian surface; "
                       "held phase untouched; helmet geometry unchanged."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
