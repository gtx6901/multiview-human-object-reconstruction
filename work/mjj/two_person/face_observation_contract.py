#!/usr/bin/env python
"""Face evidence contract for the current person's staged body motion.

This is the data/visibility/support part of the original face stage, isolated
from its optimizer.  It learns nose/eye head-local offsets from the current
person's own triangulated 3D points and current motion, selects only original
inlier + front-facing points (ears excluded), and records the support basis for
times that do not have five observable head degrees of freedom.

No expression fit, no ear constraint, no MJJ/FCS006 motion, no pixel offset.
The optimizer remains in `reference/fit_face_stage.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

FACE_KP = [0, 1, 2]  # nose, left eye, right eye; ears excluded
HEAD_JOINT = 15
NECK_BODY = 11
HEAD_BODY = 14


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], np.float64)


def fk(rest: torch.Tensor, parents: torch.Tensor, aa: torch.Tensor, scale: float, trans: torch.Tensor):
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


def assemble(motion: Dict[str, Any], body: torch.Tensor, T: int) -> torch.Tensor:
    lh = motion.get("lhand_pose", motion.get("left_hand_pose"))
    rh = motion.get("rhand_pose", motion.get("right_hand_pose"))
    if lh is None or rh is None:
        raise KeyError("motion lacks hand poses")
    return torch.cat(
        [motion["root_pose"].reshape(T, 1, 3), body, torch.zeros(T, 3, 3), lh, rh], 1
    )


def build_face_contract(
    person: Path,
    avatar: Path,
    motion: Path,
    out_path: Path,
) -> Dict[str, Any]:
    person = Path(person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    av = torch.load(Path(avatar).resolve(), map_location="cpu", weights_only=False)
    m = torch.load(Path(motion).resolve(), map_location="cpu", weights_only=False)
    rest = av["rig"]["rest_joints"][0].float()
    parents = av["rig"]["parents"]
    scale = float(m["scale"])
    root = m["root_pose"].clone()
    body = m["body_pose"].clone()
    trans = m["translation_world"].clone()
    T = len(root)
    aa = assemble(m, body, T)
    with torch.no_grad():
        tg, Rg = fk(rest, parents, aa, scale, trans)
        head = tg[:, HEAD_JOINT] * scale + trans
    meta = json.loads(Path(info["manifest"]).read_text(encoding="utf-8"))
    views = meta["views"]
    cams = [str(v["serial"]) for v in views]
    P = np.stack(
        [
            np.asarray(v["K_undistorted_native"], np.float64)
            @ np.asarray(v["W"], np.float64)[:3]
            for v in views
        ]
    )
    xy = np.stack(
        [
            np.load(Path(info["keypoints"]) / f"{c}.npz", allow_pickle=False)["xy"][:, FACE_KP]
            for c in cams
        ]
    ).transpose(1, 0, 2, 3)  # T,V,3,2
    score = np.stack(
        [
            np.load(Path(info["keypoints"]) / f"{c}.npz", allow_pickle=False)["score"][:, FACE_KP]
            for c in cams
        ]
    ).transpose(1, 0, 2)  # T,V,3
    obs = np.load(Path(info["observations"]), allow_pickle=False)
    xyz = np.asarray(obs["xyz"], np.float64)[:, FACE_KP]
    valid3 = np.asarray(obs["valid"], bool)[:, FACE_KP] & np.isfinite(xyz).all(-1)
    inlier = np.asarray(obs["inlier"], bool)[:, :, FACE_KP]  # V,T,3

    local = np.zeros((len(FACE_KP), 3), np.float64)
    for k in range(len(FACE_KP)):
        good = valid3[:, k] & np.isfinite(xyz[:, k]).all(-1)
        if not good.any():
            raise RuntimeError(f"no triangulated 3D for face point {k}; refusing fake head-local L")
        v = torch.einsum(
            "tji,tj->ti", Rg[good, HEAD_JOINT], (torch.tensor(xyz[good, k], dtype=torch.float32) - head[good]) / scale
        )
        local[k] = v.median(0).values.numpy()

    L = torch.tensor(local, dtype=torch.float32)
    direction = L[0] - L[1:3].mean(0)
    direction[1] = 0.0
    direction = torch.nn.functional.normalize(direction, dim=0)
    direction = torch.einsum("tij,j->ti", Rg[:, HEAD_JOINT], direction)
    W = np.stack([np.asarray(v["W"], np.float64) for v in views])
    centers = -np.einsum("vji,vj->vi", W[:, :3, :3], W[:, :3, 3])
    to_camera = centers[:, None, :] - head.numpy()[None]  # V,T,3
    to_camera /= np.linalg.norm(to_camera, axis=-1, keepdims=True) + 1e-12
    front = np.einsum("vtj,tj->vt", to_camera, direction.numpy()) > 0.15
    valid = inlier & front[..., None]  # V,T,3; ears are already absent
    selected = int(valid.sum())
    if selected == 0:
        raise RuntimeError("no visible front-facing nose/eye evidence; face stage cannot run")

    # Same observable-rank result as shared_fcs_fit.face(visible=True, supported=True).
    K = np.stack([np.asarray(v["K_undistorted_native"], np.float64) for v in views])
    headnp = head.numpy()
    necknp = (tg[:, 12] * scale + trans).numpy()
    Rnp = Rg.numpy()
    Lnp = local
    ranks = []
    for t in range(T):
        rows = []
        for vi, ki in np.argwhere(valid[:, t]):
            point = headnp[t] + scale * (Rnp[t, HEAD_JOINT] @ Lnp[ki])
            pc = W[vi, :3, :3] @ point + W[vi, :3, 3]
            q = K[vi] @ pc
            projection = (K[vi, :2] * q[2] - q[:2, None] * K[vi, 2]) / q[2] ** 2
            dpoint = np.c_[
                -skew(point - necknp[t]) @ Rnp[t, 12],
                -skew(point - headnp[t]) @ Rnp[t, HEAD_JOINT],
            ]
            rows.append(projection @ W[vi, :3, :3] @ dpoint)
        ranks.append(int(np.linalg.matrix_rank(np.concatenate(rows))) if rows else 0)
    ranks = np.minimum(np.array(ranks), 5)
    observable = ranks == 5
    knots = np.flatnonzero(observable)
    if not len(knots):
        raise RuntimeError("no time has full 5-DOF observable head evidence; cannot build support basis")
    B = np.zeros((T, len(knots)), np.float32)
    for t in range(T):
        right = int(np.searchsorted(knots, t))
        if right == 0:
            B[t, 0] = 1
        elif right == len(knots):
            B[t, -1] = 1
        else:
            left = right - 1
            f = (t - knots[left]) / (knots[right] - knots[left])
            B[t, left] = 1 - f
            B[t, right] = f
    if not (np.allclose(B.sum(1), 1) and np.all(B >= 0) and np.allclose(B[knots], np.eye(len(knots)))):
        raise RuntimeError("invalid support basis")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        local=local.astype(np.float32),
        face_kp=np.asarray(FACE_KP, np.int32),
        head_joint=np.asarray(HEAD_JOINT, np.int32),
        inlier=inlier,
        front=front,
        valid=valid,
        basis=B,
        knots=knots,
        ranks=ranks,
        score=score.astype(np.float32),
        xy=xy.astype(np.float32),
        cameras=np.asarray(cams),
        frames=np.asarray(info["frames"], np.int32),
    )
    return {
        "path": str(out_path),
        "local": local.tolist(),
        "face_kp": FACE_KP,
        "head_joint": HEAD_JOINT,
        "selected_inlier_front_points": selected,
        "before_directional_visibility": int(inlier.sum()),
        "frames_without_visible_face": int((~valid.any((0, 2))).sum()),
        "per_frame_visible_points": valid.sum((0, 2)).tolist(),
        "supported_frames": knots.tolist(),
        "inferred_frames": np.flatnonzero(~observable).tolist(),
        "head_observation_information_rank": ranks.tolist(),
        "basis_shape": list(B.shape),
        "face_expression_fitted": False,
        "ears_excluded": True,
        "front_dot_threshold": 0.15,
        "source_person": str(person),
        "source_avatar": str(Path(avatar).resolve()),
        "source_motion": str(Path(motion).resolve()),
    }
