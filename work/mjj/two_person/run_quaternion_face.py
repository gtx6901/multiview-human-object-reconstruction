#!/usr/bin/env python
"""Direct quaternion neck/head face stage, original 5-point control.

This is the q-native counterpart of reference/fit_face_stage.py / run_original_face.py:
* the same current-person 5 COCO nose/eye/ear landmarks and actual source scores;
* the same head-local L median from current-person triangulated 3D in the head frame;
* the same 200 Adam steps at lr 0.01;
* only neck joint 12 and head joint 15 local quaternions are optimized, with all
  other 53 rotations, hands, translation, scale/beta fixed;
* FK and reprojection are matrix-direct from q; no AA round-trip in the objective;
* mask before any residual arithmetic; no NaN backward.

The original AA correction terms are explicitly changed to SO(3): correction
q_delta=q_current*inv(q_initial), its relative-log squared regularizer, and its
relative angular acceleration.  This is not claimed to be exactly the old AA cost.

No front/inlier/rank gate, FaceBoxes, canonical nose direction, expression model,
or SMPL nose proxy is introduced.  L is from real current-person stage data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
LHM = PROJECT_ROOT / "lhmpp"
WORK_ROOT = PROJECT_ROOT / "work" / "mjj"
REF_FACE = THIS_DIR / "reference" / "fit_face_stage.py"
FACE_KP = [0, 1, 2, 3, 4]  # nose, left eye, right eye, left ear, right ear
NECK_JOINT = 12
HEAD_JOINT = 15


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def configure_runtime_paths() -> None:
    for p in (LHM, LHM / "vendor", LHM / "deps", PROJECT_ROOT / "runtime" / "deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from runtime_imports import configure

        configure()
    except Exception as exc:
        raise RuntimeError(f"runtime_imports.configure failed: {exc}") from exc


def fk_q(q: torch.Tensor, rest: torch.Tensor, parents: torch.Tensor):
    """Unscaled joint positions and global matrices, matrix-direct from q."""
    from pytorch3d.transforms import quaternion_to_matrix

    q = torch.as_tensor(q)
    R = quaternion_to_matrix(q)
    T = q.shape[0]
    Rg, tg = [None] * 55, [None] * 55
    for j in range(55):
        p = int(parents[j])
        if p < 0:
            Rg[j], tg[j] = R[:, j], rest[j].expand(T, 3)
        else:
            Rg[j] = Rg[p] @ R[:, j]
            tg[j] = tg[p] + (Rg[p] @ (rest[j] - rest[p]).view(1, 3, 1)).squeeze(-1)
    return torch.stack(tg, 1), torch.stack(Rg, 1)


def q_with_face(q_base: torch.Tensor, q_neck: torch.Tensor, q_head: torch.Tensor) -> torch.Tensor:
    q = q_base.clone()
    q[:, NECK_JOINT : NECK_JOINT + 1] = q_neck
    q[:, HEAD_JOINT : HEAD_JOINT + 1] = q_head
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    configure_runtime_paths()
    from rotation_contract import (
        load_motion_quaternions,
        normalize_quaternions,
        q_to_matrix,
        relative_quaternion,
        relative_log,
        temporal_angular_accel,
        export_motion_with_quaternions,
    )
    from pytorch3d.transforms import quaternion_to_matrix

    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    motion_path = Path(args.motion).resolve()
    if motion_path.is_dir():
        motion_path = motion_path / "motion.pt"
    out = Path(args.out).resolve()
    if not out.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("output must stay below work/mjj")
    out.mkdir(parents=True, exist_ok=True)

    m = torch.load(motion_path, map_location="cpu", weights_only=False)
    q, q_info = load_motion_quaternions(m, verify_legacy=True)
    q = normalize_quaternions(q)
    T = q.shape[0]
    frames = np.asarray(m.get("frames", np.arange(T)), np.int64)
    if len(frames) != T:
        raise ValueError("frames/q T mismatch")
    av = torch.load(avatar, map_location="cpu", weights_only=False)
    rest = av["rig"]["rest_joints"][0].float()
    parents = av["rig"]["parents"].long()
    scale = float(m["scale"].reshape(-1)[0]) if torch.is_tensor(m["scale"]) else float(m["scale"])
    trans = torch.as_tensor(m["translation_world"], dtype=torch.float32).reshape(T, 3)

    meta = json.loads(Path(info["manifest"]).read_text(encoding="utf-8"))
    cams = [str(v["serial"]) for v in meta["views"]]
    P = torch.tensor(
        np.stack(
            [
                np.asarray(v["K_undistorted_native"], np.float64)
                @ np.asarray(v["W"], np.float64)[:3]
                for v in meta["views"]
            ]
        ),
        dtype=torch.float32,
    )
    key_dir = Path(info["keypoints"])
    xy_np, score_np = [], []
    for c in cams:
        with np.load(key_dir / f"{c}.npz", allow_pickle=False) as z:
            xy_np.append(np.asarray(z["xy"], np.float64))
            score_np.append(np.asarray(z["score"], np.float64))
    xy_all = torch.tensor(np.stack(xy_np), dtype=torch.float32).permute(1, 0, 2, 3)  # T,V,133,2
    score_all = torch.tensor(np.stack(score_np), dtype=torch.float32).permute(1, 0, 2)  # T,V,133
    xy = xy_all[:, :, FACE_KP]
    score = score_all[:, :, FACE_KP]
    obs = np.load(Path(info["observations"]), allow_pickle=False)
    xyz = torch.tensor(np.asarray(obs["xyz"], np.float64)[:, FACE_KP], dtype=torch.float32)
    valid3 = torch.tensor(np.asarray(obs["valid"], bool)[:, FACE_KP]) & torch.isfinite(xyz).all(-1)
    inlier = torch.tensor(np.asarray(obs["inlier"], bool)[:, :, FACE_KP])  # V,T,5

    # Original selection: score>=0.5 and finite source xy; no front/inlier gate.
    valid = (score >= 0.5) & torch.isfinite(xy).all(-1)  # T,V,5
    valid_face = valid.permute(1, 0, 2)  # V,T,5

    q_base = q.clone()
    q_neck0 = q_base[:, NECK_JOINT : NECK_JOINT + 1].clone()
    q_head0 = q_base[:, HEAD_JOINT : HEAD_JOINT + 1].clone()

    # Original head-local L from current-person screened 3D and current head pose.
    with torch.no_grad():
        tg, Rg = fk_q(q_base, rest, parents)
        head_w = tg[:, HEAD_JOINT] * scale + trans
        local = torch.einsum("tji,tkj->tki", Rg[:, HEAD_JOINT], (xyz - head_w[:, None]) / scale)
        L = []
        for k in range(len(FACE_KP)):
            good = valid3[:, k] & torch.isfinite(xyz[:, k]).all(-1)
            if not bool(good.any()):
                raise RuntimeError(f"no real triangulated 3D for face point {k}; refusing zero/SMPL proxy L")
            L.append(local[good, k].median(0).values)
        L = torch.stack(L, 0).float()  # 5,3

    q_neck = q_neck0.clone().requires_grad_(True)
    q_head = q_head0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q_neck, q_head], lr=0.01)

    def face_loss():
        q_cur = q_with_face(q_base, q_neck, q_head)
        tg2, Rg2 = fk_q(q_cur, rest, parents)
        head2 = tg2[:, HEAD_JOINT] * scale + trans
        world = head2[:, None] + scale * torch.einsum("tij,kj->tki", Rg2[:, HEAD_JOINT], L)
        homo = torch.cat([world, torch.ones(T, len(FACE_KP), 1, dtype=world.dtype)], -1)
        p = torch.einsum("vij,tkj->tvki", P, homo)
        uv = p[..., :2] / p[..., 2:3].clamp_min(1e-6)
        m = valid & torch.isfinite(uv).all(-1)
        # Mask BEFORE residual: unselected/missing xy may be NaN.
        err = torch.zeros((T, len(cams), len(FACE_KP)), dtype=uv.dtype)
        if bool(m.any()):
            err[m] = (uv[m] - xy[m]).norm(dim=-1)
        huber = torch.nn.functional.huber_loss(
            err, torch.zeros_like(err), reduction="none", delta=20.0
        )
        data = (huber * m.float()).sum() / m.float().sum().clamp_min(1.0)

        delta_neck = relative_quaternion(q_neck, q_neck0)  # T,1,4
        delta_head = relative_quaternion(q_head, q_head0)
        log_neck = relative_log(q_neck, q_neck0)  # T,1,3
        log_head = relative_log(q_head, q_head0)
        reg = 0.002 * (log_neck.square().sum() / T) + 0.002 * (log_head.square().sum() / T)
        acc_n = temporal_angular_accel(delta_neck)  # T-2,1,3
        acc_h = temporal_angular_accel(delta_head)
        acc = torch.zeros((), dtype=uv.dtype)
        if acc_n.numel():
            acc = acc_n.square().mean() + acc_h.square().mean()
        return data + reg + 5.0 * acc, err, m, {
            "q_delta_neck_rms": float(log_neck.square().mean().sqrt()),
            "q_delta_head_rms": float(log_head.square().mean().sqrt()),
        }

    before_loss, err0, m0, _ = face_loss()
    with torch.no_grad():
        before_px = float((err0 * m0).sum() / m0.sum().clamp_min(1.0))
    started = time.time()
    history = []
    for it in range(200):
        opt.zero_grad()
        loss, err, mask, stats = face_loss()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q_neck.copy_(normalize_quaternions(q_neck))
            q_head.copy_(normalize_quaternions(q_head))
        if it % 40 == 0 or it == 199:
            with torch.no_grad():
                _loss, err2, mask2, stats2 = face_loss()
                cur = float((err2 * mask2).sum() / mask2.sum().clamp_min(1.0))
                history.append(
                    {
                        "iter": it,
                        "loss": round(float(loss), 4),
                        "face_reproj_px": round(cur, 3),
                        **{k: round(v, 6) for k, v in stats2.items()},
                    }
                )
                print(history[-1], flush=True)
    with torch.no_grad():
        _loss, err_f, mask_f, stats_f = face_loss()
        after_px = float((err_f * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        q_final = normalize_quaternions(q_with_face(q_base, q_neck, q_head))
        tg_f, Rg_f = fk_q(q_final, rest, parents)
        joints55 = tg_f * scale + trans[:, None]
        # Explicit fixed-field verification.
        before_mats = q_to_matrix(q_base)
        after_mats = q_to_matrix(q_final)
        other = [i for i in range(55) if i not in (NECK_JOINT, HEAD_JOINT)]
        other_max = float((before_mats[:, other] - after_mats[:, other]).abs().max())
        neck_delta = relative_quaternion(q_final[:, NECK_JOINT : NECK_JOINT + 1], q_neck0)
        head_delta = relative_quaternion(q_final[:, HEAD_JOINT : HEAD_JOINT + 1], q_head0)
        fixed_checks = {
            "other_53_rot_matrix_max_abs": other_max,
            "translation_max_abs": float((torch.as_tensor(m["translation_world"]) - trans).abs().max()),
            "scale_abs": abs(float(torch.as_tensor(m["scale"]).reshape(-1)[0]) - scale),
            "neck_delta_angle_rad_mean": float(2.0 * torch.atan2(neck_delta[..., 1:].norm(dim=-1), neck_delta[..., 0].abs()).mean()),
            "head_delta_angle_rad_mean": float(2.0 * torch.atan2(head_delta[..., 1:].norm(dim=-1), head_delta[..., 0].abs()).mean()),
        }

    # Export q plus all legacy aliases; no copied stale q.
    exported = export_motion_with_quaternions(m, q_final, hemisphere=True)
    exported["joints55_world"] = joints55.float()
    exported["frames"] = torch.as_tensor(frames, dtype=torch.long)
    fixed_checks["translation_preserved"] = bool(
        torch.allclose(torch.as_tensor(exported["translation_world"]), torch.as_tensor(m["translation_world"]))
    )
    fixed_checks["scale_preserved"] = bool(
        torch.allclose(torch.as_tensor(exported["scale"]), torch.as_tensor(m["scale"]))
    )
    fixed_checks["betas_preserved"] = bool(
        torch.allclose(torch.as_tensor(exported["betas"]), torch.as_tensor(m["betas"]))
    )
    fixed_checks["frames_preserved"] = bool(torch.equal(exported["frames"], torch.as_tensor(m["frames"])))
    q_exported, q_export_info = load_motion_quaternions(exported, verify_legacy=True)
    torch.save(exported, out / "motion.pt")

    face_constraints_path = out / "face_constraints.npz"
    np.savez_compressed(
        face_constraints_path,        local=L.detach().cpu().numpy().astype(np.float32),
        coco=np.asarray(FACE_KP, np.int32),
        face_kp=np.asarray(FACE_KP, np.int32),
        valid=valid_face.numpy(),
        xy=xy.detach().cpu().numpy().astype(np.float32),
        score=score.detach().cpu().numpy().astype(np.float32),
        frames=frames.astype(np.int32),
        cameras=np.asarray(cams),
        inlier=inlier.numpy(),
        selection=np.asarray("score>=0.5 & finite(xy); original 5 COCO points; no front/inlier gate"),
        xy_layout=np.asarray("T,V,5,2"),
        valid_layout=np.asarray("V,T,5"),
        inlier_layout=np.asarray("V,T,5; recorded, NOT used for selection"),
    )

    report = {
        "schema": "two_person_quaternion_face_stage/v1",
        "source_face_reference": str(REF_FACE),
        "source_face_sha256": sha256_file(REF_FACE) if REF_FACE.is_file() else None,
        "source_original_face_driver": str(THIS_DIR / "run_original_face.py"),
        "source_original_face_driver_sha256": (
            sha256_file(THIS_DIR / "run_original_face.py")
            if (THIS_DIR / "run_original_face.py").is_file()
            else None
        ),
        "person": str(person),
        "person_sha256": sha256_file(person),
        "avatar": str(avatar),
        "avatar_size_bytes": avatar.stat().st_size,
        "motion_input": str(motion_path),
        "motion_input_sha256": sha256_file(motion_path),
        "observations": str(Path(info["observations"]).resolve()),
        "observations_sha256": sha256_file(Path(info["observations"])),
        "keypoints": str(key_dir),
        "keypoints_sha256": {
            c: sha256_file(key_dir / f"{c}.npz") for c in cams
        },
        "manifest": str(Path(info["manifest"]).resolve()),
        "manifest_sha256": sha256_file(Path(info["manifest"])),
        "q_input_validation": q_info,
        "identity": info.get("identity"),
        "frames": frames.tolist(),
        "face_kp": FACE_KP,
        "face_point_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "camera_order": cams,
        "L": L.tolist(),
        "L_source": "current-person screened observations xyz in current q head frame, median over valid frames",
        "valid_selection": "score>=0.5 and finite source xy; no front/inlier/rank gate",
        "valid_count": int(valid.sum()),
        "inlier_recorded_not_used": True,
        "huber_px": 20.0,
        "steps": 200,
        "lr": 0.01,
        "optimizer": "Adam",
        "optimized_joints": {"neck_joint": NECK_JOINT, "head_joint": HEAD_JOINT},
        "other_53_fixed": True,
        "fix_checks": fixed_checks,
        "q_export_consistency": q_export_info,
        "q_export_shape": list(q_exported.shape),
        "objective": {
            "data": "masked 5-point Huber 20px on matrix-direct q FK projection",
            "correction_regularizer": "0.002 * sum(relative_log(qcur*inv(qinit))^2)/T for neck and head",
            "time_prior": "5.0 * mean-square relative angular acceleration of q_delta for neck+head",
            "changed_from_original_aa": True,
            "not_claimed_same_cost": True,
        },
        "face_reproj_px_before": before_px,
        "face_reproj_px_after": after_px,
        "history": history,
        "seconds": round(time.time() - started, 1),
        "outputs": {
            "motion": str(out / "motion.pt"),
            "face_constraints": str(face_constraints_path),
            "q_field": "local_quaternions_wxyz",
            "legacy_aliases": [
                "root_pose",
                "body_pose",
                "lhand_pose",
                "rhand_pose",
                "left_hand_pose",
                "right_hand_pose",
            ],
            "joints55_world": "actual q FK",
        },
        "no_quality_threshold": True,
        "no_faceboxes": True,
        "no_front_or_inlier_gate": True,
        "no_expression_or_head_model": True,
        "cpu_threads": 4,
        "scope": "Direct q-native original 5-point neck/head stage; L and face ghosts are not fixed or quality-accepted.",
    }
    atomic_write_json(out / "quaternion_face_provenance.json", report)
    print(f"[q-face] before={before_px:.3f}px after={after_px:.3f}px L={np.round(L.numpy(),4).tolist()}")
    print(f"[q-face] wrote {out / 'motion.pt'}")
    print(f"[q-face] wrote {face_constraints_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
