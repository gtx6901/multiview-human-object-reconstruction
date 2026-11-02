#!/usr/bin/env python
"""Unified wxyz quaternion rotation contract for the two-person pipeline.

Canonical internal representation: normalized local quaternions, shape
``[T, 55, 4]`` in w,x,y,z order.  Quaternion sign is not physical: q and -q are
the same rotation.  All objective terms use relative-rotation logs, never raw
axis-angle branches or raw quaternion components.

The module intentionally keeps hemisphere continuity as an *equivalent
representation* for storage/export/interpolation only.  It never smooths a
motion sequence.

Layout of the 55 local rotations:
    [0] root
    [1:22] body 21
    [22:25] jaw, left eye, right eye
    [25:40] left hand 15
    [40:55] right hand 15

Legacy AA is converted only at the SMPL-X/LHM boundary.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math

import torch
from pytorch3d.transforms import (
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)

WXYZ = 4
LEGACY_FIELDS = (
    "root_pose",
    "body_pose",
    "lhand_pose",
    "rhand_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
)
LEGACY_MATRIX_TOL = 1e-5


class QuaternionContractError(RuntimeError):
    """Raised when q and legacy rotation fields disagree or are malformed."""


def normalize_quaternions(q: Any, eps: float = 1e-12) -> torch.Tensor:
    q = torch.as_tensor(q)
    if not q.is_floating_point():
        q = q.float()
    if q.shape[-1] != WXYZ:
        raise ValueError(f"quaternion last dimension must be wxyz=4, got {tuple(q.shape)}")
    n = torch.linalg.norm(q, dim=-1, keepdim=True)
    if bool((n < eps).any()):
        raise QuaternionContractError("zero-norm quaternion is not a valid rotation")
    return q / n


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternions(q)
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    # Unit quaternions only; contract normalizes before this operation.
    return quaternion_conjugate(q)


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return normalize_quaternions(quaternion_multiply(normalize_quaternions(a), normalize_quaternions(b)))


def q_to_matrix(q: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(normalize_quaternions(q))


def matrix_to_q(rot: torch.Tensor) -> torch.Tensor:
    return normalize_quaternions(matrix_to_quaternion(rot))


def relative_quaternion(q_next: torch.Tensor, q_prev: torch.Tensor) -> torch.Tensor:
    """q_next * inverse(q_prev), in the parent/world frame."""
    return qmul(q_next, quaternion_inverse(q_prev))


def shortest_log(q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Shortest SO(3) log as axis-angle, invariant to q and -q.

    q is normalized, sign-canonicalized by w>=0 (and a deterministic vector
    tie-break when w is numerically zero), then converted with atan2.  There is
    no raw axis-angle subtraction and no artificial angle floor.
    """
    q = normalize_quaternions(q)
    w = q[..., 0]
    flip = w < 0
    q = torch.where(flip.unsqueeze(-1), -q, q)
    # At an exact pi rotation w=0, fix the remaining q/-q ambiguity by making
    # the largest-magnitude vector component positive.
    near_pi = q[..., 0].abs() <= eps
    if bool(near_pi.any()):
        v = q[..., 1:]
        idx = v.abs().argmax(dim=-1, keepdim=True)
        comp = torch.gather(v, -1, idx).squeeze(-1)
        flip_pi = (comp < 0) & near_pi
        q = torch.where(flip_pi.unsqueeze(-1), -q, q)
    v = q[..., 1:]
    n2 = v.square().sum(dim=-1)
    n = torch.sqrt(n2.clamp_min(eps * eps))
    w = q[..., 0]
    angle = 2.0 * torch.atan2(n, w)
    # Continuous identity limit: log(q) ~ 2*v. The derivative at identity
    # must also be correct, rather than replacing the entire residual by zero.
    factor = torch.where(n2 < eps * eps, 2.0 + n2 / 3.0, angle / n)
    return v * factor[..., None]


def relative_logs(q: torch.Tensor) -> torch.Tensor:
    """[T-1,J,3] relative logs q[t+1] * inverse(q[t])."""
    q = normalize_quaternions(q)
    if q.ndim != 3:
        raise ValueError(f"expected T,J,4 quaternions, got {tuple(q.shape)}")
    if q.shape[0] < 2:
        return torch.zeros((0, q.shape[1], 3), dtype=q.dtype, device=q.device)
    return shortest_log(relative_quaternion(q[1:], q[:-1]))


def relative_log(q_current: torch.Tensor, q_initial: torch.Tensor) -> torch.Tensor:
    """[...,J,3] relative log q_current * inverse(q_initial)."""
    return shortest_log(relative_quaternion(q_current, q_initial))


def temporal_angular_accel(q: torch.Tensor) -> torch.Tensor:
    """[T-2,J,3] = log(relative)[1:] - log(relative)[:-1].

    No component of AA or quaternion is used as a loss variable.
    """
    logs = relative_logs(q)
    if logs.shape[0] < 2:
        return torch.zeros((0, logs.shape[1], 3), dtype=q.dtype, device=q.device)
    return logs[1:] - logs[:-1]


def body_trust_penalty(q_current: torch.Tensor, q_initial: torch.Tensor, sigma: float = 1e-3) -> torch.Tensor:
    """SO(3) body trust, 0.5*mean(log(qcur*inv(qinit))^2)/sigma."""
    residual = relative_log(q_current, q_initial)
    return 0.5 * residual.square().mean() / sigma


def temporal_penalty(q: torch.Tensor, sigma: float = 5e-3) -> torch.Tensor:
    """SO(3) temporal acceleration penalty, mean(abs(acc))/sigma."""
    acc = temporal_angular_accel(q)
    if acc.numel() == 0:
        return torch.zeros((), dtype=q.dtype, device=q.device)
    return acc.abs().mean() / sigma


def body_motion_temporal(q_root: torch.Tensor, q_body: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Body-stage temporal scalar used by body fit and full_motion_temporal.

    mean(abs(second_diff(translation)))/5e-4
    + mean(abs(temporal_angular_accel(q_body)))/5e-3
    + mean(abs(temporal_angular_accel(q_root)))/5e-3

    q_root is [T,1,4], q_body is [T,21,4].  For T<3 this returns a scalar zero
    with translation's dtype/device.
    """
    scalar = torch.zeros((), dtype=translation.dtype, device=translation.device)
    if translation.shape[0] < 3:
        return scalar
    acc_t = (translation[2:] - 2 * translation[1:-1] + translation[:-2]).abs().mean() / 5e-4
    acc_b = temporal_angular_accel(q_body).abs().mean() / 5e-3
    acc_r = temporal_angular_accel(q_root).abs().mean() / 5e-3
    return acc_t + acc_b + acc_r


def full_motion_temporal(q: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Full-root/body21/hand30/face3 temporal scalar on the same angular scale.

    The body part is exactly body_motion_temporal(q[:,:1], q[:,1:22], t).
    Hands are q[:,25:55] (30) and jaw/eyes are q[:,22:25] (3), each with the
    same 5e-3 SO(3) angular-acceleration scale.  No filter or joint lock.
    For T<3 this returns a scalar zero with translation's dtype/device.
    """
    scalar = torch.zeros((), dtype=translation.dtype, device=translation.device)
    if translation.shape[0] < 3:
        return scalar
    body = body_motion_temporal(q[:, :1], q[:, 1:22], translation)
    hands = temporal_angular_accel(q[:, 25:55]).abs().mean() / 5e-3
    face = temporal_angular_accel(q[:, 22:25]).abs().mean() / 5e-3
    return body + hands + face


def hemisphere_continuous(q: torch.Tensor) -> torch.Tensor:
    """Equivalent q/-q sign-continuous representation; storage/interp only.

    It does not alter the rotation sequence and does not smooth values.
    """
    q = normalize_quaternions(q)
    out = [q[0]]
    for t in range(1, q.shape[0]):
        prev = out[-1]
        qt = q[t]
        dot = (prev * qt).sum(-1, keepdim=True)
        out.append(torch.where(dot < 0, -qt, qt))
    return normalize_quaternions(torch.stack(out, dim=0))


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: Any, eps: float = 1e-8) -> torch.Tensor:
    """Shortest-path SLERP with stable small-angle limit and exact endpoints."""
    q0 = normalize_quaternions(q0)
    q1 = normalize_quaternions(q1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    tt = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    if tt.ndim == 0:
        tt = tt.expand(q0.shape[:-1]).unsqueeze(-1)
    elif tt.shape != q0.shape[:-1]:
        tt = tt.expand(q0.shape[:-1]).unsqueeze(-1)
    else:
        tt = tt.unsqueeze(-1)
    small = dot > 1.0 - 1e-6
    # Do not evaluate acos at 1 even in an unused torch.where branch: its
    # infinite derivative can contaminate the identity interpolation gradient.
    theta = torch.acos(torch.where(small, torch.zeros_like(dot), dot))
    sin_theta = torch.sin(theta)
    denom = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    a = torch.sin((1.0 - tt) * theta) / denom
    b = torch.sin(tt * theta) / denom
    linear = normalize_quaternions((1.0 - tt) * q0 + tt * q1)
    curved = normalize_quaternions(torch.where(small, q0, a * q0 + b * q1))
    out = torch.where(small, linear, curved)
    out = torch.where((tt <= 0).expand_as(out), q0, out)
    out = torch.where((tt >= 1).expand_as(out), q1, out)
    return normalize_quaternions(out)


def _alias_values(motion: Dict[str, Any], names: Sequence[str], field: str) -> Tuple[torch.Tensor | None, str | None]:
    vals = [(n, torch.as_tensor(motion[n])) for n in names if n in motion and motion[n] is not None]
    if not vals:
        return None, None
    base_name, base = vals[0]
    for name, value in vals[1:]:
        if tuple(value.shape) != tuple(base.shape) or not torch.allclose(value, base, atol=1e-6, rtol=1e-5):
            raise QuaternionContractError(
                f"legacy field aliases disagree for {field}: {base_name} vs {name}"
            )
    return base, base_name


def _legacy_quaternions(motion: Dict[str, Any], T: int) -> Dict[str, torch.Tensor | None]:
    out: Dict[str, torch.Tensor | None] = {}
    root, _ = _alias_values(motion, ("root_pose",), "root")
    body, _ = _alias_values(motion, ("body_pose",), "body")
    lh, _ = _alias_values(motion, ("lhand_pose", "left_hand_pose"), "left_hand")
    rh, _ = _alias_values(motion, ("rhand_pose", "right_hand_pose"), "right_hand")
    jaw, _ = _alias_values(motion, ("jaw_pose",), "jaw")
    leye, _ = _alias_values(motion, ("leye_pose",), "left_eye")
    reye, _ = _alias_values(motion, ("reye_pose",), "right_eye")
    for name, value, shape in (
        ("root", root, (T, 3)),
        ("body", body, (T, 21, 3)),
        ("left_hand", lh, (T, 15, 3)),
        ("right_hand", rh, (T, 15, 3)),
        ("jaw", jaw, (T, 3)),
        ("left_eye", leye, (T, 3)),
        ("right_eye", reye, (T, 3)),
    ):
        if value is not None and tuple(value.shape) != shape:
            raise QuaternionContractError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
        qshape = (T, 1, 4) if name in ('root','jaw','left_eye','right_eye') else shape[:-1] + (4,)
        out[name] = None if value is None else axis_angle_to_quaternion(value.reshape(-1, 3)).reshape(qshape)
    return out


def _identity(T: int, J: int, device=None, dtype=torch.float32) -> torch.Tensor:
    q = torch.zeros((T, J, 4), dtype=dtype, device=device)
    q[..., 0] = 1.0
    return q


def load_motion_quaternions(motion: Dict[str, Any], verify_legacy: bool = True, tol: float = LEGACY_MATRIX_TOL) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return canonical [T,55,4] q, verifying existing q against legacy fields.

    If ``local_quaternions_wxyz`` and a legacy field are both present, they must
    represent the same matrices.  A mismatch raises QuaternionContractError with
    the field name; no stale field is silently preferred.
    """
    q_raw = motion.get("local_quaternions_wxyz")
    info: Dict[str, Any] = {"source": "legacy", "verified_fields": []}
    if q_raw is not None:
        q = normalize_quaternions(torch.as_tensor(q_raw))
        if q.ndim != 3 or q.shape[1:] != (55, 4):
            raise QuaternionContractError(f"local_quaternions_wxyz must be T,55,4, got {tuple(q.shape)}")
        T = q.shape[0]
        info["source"] = "local_quaternions_wxyz"
    else:
        root = motion.get("root_pose")
        if root is None:
            raise QuaternionContractError("motion has neither local_quaternions_wxyz nor root_pose")
        T = int(torch.as_tensor(root).shape[0])
        q = _identity(T, 55, dtype=torch.float32)

    legacy = _legacy_quaternions(motion, T)
    if q_raw is not None:
        if not verify_legacy:
            return q, info
        segments = [
            ("root_pose", 0, 1, legacy["root"]),
            ("body_pose", 1, 22, legacy["body"]),
            ("jaw_pose", 22, 23, legacy["jaw"]),
            ("leye_pose", 23, 24, legacy["left_eye"]),
            ("reye_pose", 24, 25, legacy["right_eye"]),
            ("left_hand_pose", 25, 40, legacy["left_hand"]),
            ("right_hand_pose", 40, 55, legacy["right_hand"]),
        ]
        for name, lo, hi, lq in segments:
            if lq is None:
                continue
            qseg = q[:, lo:hi]
            if tuple(lq.shape) != tuple(qseg.shape):
                raise QuaternionContractError(f"{name}: legacy/q shape mismatch {tuple(lq.shape)} vs {tuple(qseg.shape)}")
            err = float((q_to_matrix(qseg) - q_to_matrix(lq)).abs().max())
            if err > tol:
                raise QuaternionContractError(
                    f"{name}: local_quaternions_wxyz and legacy AA are inconsistent; "
                    f"max matrix error {err:.6g} > {tol:.6g}"
                )
            info["verified_fields"].append({"field": name, "max_matrix_error": err})
        return q, info

    # No q field: build from legacy, with jaw/eyes identity when absent.
    if legacy["root"] is None or legacy["body"] is None:
        raise QuaternionContractError("legacy motion needs root_pose and body_pose")
    if legacy["left_hand"] is None or legacy["right_hand"] is None:
        raise QuaternionContractError("legacy motion needs left and right hand poses")
    q = _identity(T, 55, dtype=torch.float32)
    q[:, 0:1] = legacy["root"]
    q[:, 1:22] = legacy["body"]
    if legacy["jaw"] is not None:
        q[:, 22:23] = legacy["jaw"]
    if legacy["left_eye"] is not None:
        q[:, 23:24] = legacy["left_eye"]
    if legacy["right_eye"] is not None:
        q[:, 24:25] = legacy["right_eye"]
    q[:, 25:40] = legacy["left_hand"]
    q[:, 40:55] = legacy["right_hand"]
    info["verified_fields"] = [{"field": "legacy_conversion", "max_matrix_error": 0.0}]
    return normalize_quaternions(q), info


def motion_to_local_quaternions(motion: Dict[str, Any], verify_legacy: bool = True) -> torch.Tensor:
    return load_motion_quaternions(motion, verify_legacy=verify_legacy)[0]


def quaternions_to_legacy(q: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Legacy AA boundary conversion from canonical local quaternions."""
    q = normalize_quaternions(q)
    if q.ndim != 3 or q.shape[1:] != (55, 4):
        raise ValueError(f"expected T,55,4 quaternions, got {tuple(q.shape)}")
    aa = matrix_to_axis_angle(q_to_matrix(q).reshape(-1, 3, 3)).reshape(q.shape[0], 55, 3)
    return {
        "root_pose": aa[:, 0],
        "body_pose": aa[:, 1:22],
        "lhand_pose": aa[:, 25:40],
        "rhand_pose": aa[:, 40:55],
        "left_hand_pose": aa[:, 25:40],
        "right_hand_pose": aa[:, 40:55],
        "jaw_pose": aa[:, 22],
        "leye_pose": aa[:, 23],
        "reye_pose": aa[:, 24],
    }


def export_motion_with_quaternions(motion: Dict[str, Any], q: torch.Tensor, hemisphere: bool = True) -> Dict[str, Any]:
    """Return motion with q + consistent legacy AA aliases; no other motion edit."""
    q = normalize_quaternions(q)
    stored = hemisphere_continuous(q) if hemisphere else q
    out = dict(motion)
    out["local_quaternions_wxyz"] = stored.detach().cpu()
    legacy = quaternions_to_legacy(stored)
    for key, value in legacy.items():
        out[key] = value.detach().cpu()
    # Preserve non-identity jaw/eyes even when the input bundle did not expose
    # those legacy fields. Dropping them here would silently change the rig.
    has_jaw_eyes = any(k in motion for k in ("jaw_pose", "leye_pose", "reye_pose"))
    if not has_jaw_eyes:
        jaw_eyes = stored[:, 22:25]
        eye = torch.eye(3, dtype=q_to_matrix(jaw_eyes).dtype, device=jaw_eyes.device)
        nonidentity = (q_to_matrix(jaw_eyes) - eye).abs().max() > 1e-6
        has_jaw_eyes = bool(nonidentity)
    if not has_jaw_eyes:
        for key in ("jaw_pose", "leye_pose", "reye_pose"):
            out.pop(key, None)
    return out


def lift_axis_angles_continuous(aa: torch.Tensor) -> torch.Tensor:
    """Equivalent continuous AA representation; diagnostic/export only.

    Uses quaternion sign continuity and nearest 2*pi angle lifting.  No smoothing
    and no forced axis at zero.
    """
    aa = torch.as_tensor(aa)
    if not aa.is_floating_point():
        aa = aa.float()
    orig_shape = aa.shape
    q = axis_angle_to_quaternion(aa.reshape(-1, 3))
    q = hemisphere_continuous(q.reshape(-1, 4)).reshape(-1, 1, 4)
    q = q.reshape(-1, 4)
    lifted = []
    prev_angle = None
    for t in range(q.shape[0]):
        qv = q[t, 1:]
        qw = float(q[t, 0])
        n = float(torch.linalg.norm(qv))
        if n < 1e-12:
            aa_t = torch.zeros(3, dtype=q.dtype, device=q.device)
            angle = 0.0
        else:
            axis = qv / n
            angle = 2.0 * math.atan2(n, qw)
            if prev_angle is not None:
                k = round((prev_angle - angle) / (2.0 * torch.pi))
                angle += 2.0 * torch.pi * k
            aa_t = axis * angle
        lifted.append(aa_t)
        prev_angle = angle
    return torch.stack(lifted, 0).reshape(orig_shape)
