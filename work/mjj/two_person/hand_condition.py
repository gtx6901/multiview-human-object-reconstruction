"""Image-hand SO(3) observation condition for q-native fit/MAP.

The class consumes the per-view observe-hands NPZ, computes an all-view
SO(3) angle-IRLS center, and stores the OAS covariance of the 45D left-tangent
view residuals.  Only observed hand30 entries are initialized or used in the
loss; no MANO root/beta/depth or camera selection is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rotation_contract import matrix_to_q, normalize_quaternions, relative_log


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _oas(x: np.ndarray, assume_centered: bool = True):
    """Official sklearn _oas formula (NumPy only, no sklearn dependency)."""
    x = np.asarray(x, np.float64)
    n, p = x.shape
    if n == 0:
        raise ValueError("OAS needs at least one sample")
    if assume_centered:
        emp = (x.T @ x) / n
    else:
        xc = x - x.mean(0)
        emp = (xc.T @ xc) / n
    alpha = float(np.mean(emp ** 2))
    mu = float(np.trace(emp) / p)
    mu2 = mu * mu
    num = alpha + mu2
    den = (n + 1) * (alpha - mu2 / p)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    cov = (1.0 - shrinkage) * emp
    cov.flat[:: p + 1] += shrinkage * mu
    return cov, float(shrinkage)


def _consensus(Rs: np.ndarray) -> Rotation:
    """Existing hand-condition all-view SO(3) angle-IRLS per joint."""
    R = Rotation.from_matrix(Rs)
    estimate = R.mean()
    for _ in range(20):
        distance = (estimate.inv() * R).magnitude()
        estimate = R.mean(weights=1.0 / np.maximum(distance, 1e-6))
    return estimate


class HandCondition:
    """Per-frame/side image hand center and tangent covariance likelihood."""

    def __init__(self, path: Path, motion_frames: Sequence[int]):
        self.path = Path(path).resolve()
        self.sha256 = _sha256(self.path)
        self.motion_frames = [int(x) for x in motion_frames]
        self.T = len(self.motion_frames)
        with np.load(self.path, allow_pickle=False) as z:
            rotations = np.asarray(z["rotations"], np.float64)
            valid = np.asarray(z["valid"], bool)
            if "completed_views" in z.files and not np.asarray(z["completed_views"], bool).all():
                raise ValueError("hand observation generation is incomplete; finish all cameras before fitting")
            obs_frames = np.asarray(z["frames"], np.int64)
            self.serials = [str(x) for x in np.asarray(z["serials"]).tolist()] if "serials" in z.files else []
        if rotations.ndim != 6 or rotations.shape[2:] != (2, 15, 3, 3):
            raise ValueError(f"rotations must be V,T,2,15,3,3, got {rotations.shape}")
        if valid.shape != rotations.shape[:3]:
            raise ValueError(f"valid shape {valid.shape} vs rotations {rotations.shape}")
        if obs_frames.ndim != 1 or len(obs_frames) != rotations.shape[1]:
            raise ValueError("observations frames/rotations T mismatch")
        obs_list = [int(x) for x in obs_frames.tolist()]
        if len(set(obs_list)) != len(obs_list):
            raise ValueError("duplicate observation frames")
        self.obs_frames = obs_list
        obs_index = {f: i for i, f in enumerate(obs_list)}
        self.missing_motion_frames = [f for f in self.motion_frames if f not in obs_index]
        self.matched_frames = [f for f in self.motion_frames if f in obs_index]

        identity_q = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
        self.means = np.tile(identity_q, (self.T, 2, 15, 1))
        self.precision = np.zeros((self.T, 2, 45, 45), np.float64)
        self.initializable = np.zeros((self.T, 2), bool)
        self.evaluable = np.zeros((self.T, 2), bool)
        self.view_counts = np.zeros((self.T, 2), np.int32)
        self.available_serials: List[List[List[str]]] = [[[] for _ in range(2)] for _ in range(self.T)]
        self.oas_shrinkage: List[float] = []
        self.oas_used_count = 0
        self.covariance_failures: List[Dict[str, Any]] = []

        for t, frame in enumerate(self.motion_frames):
            oi = obs_index.get(frame)
            if oi is None:
                continue
            for side in range(2):
                view_mask = valid[:, oi, side]
                if not bool(view_mask.any()):
                    continue
                view_indices = np.flatnonzero(view_mask)
                Rs = rotations[view_indices, oi, side]
                finite = np.isfinite(Rs).all(axis=(1, 2, 3))
                view_indices = view_indices[finite]
                Rs = Rs[finite]
                if len(Rs) == 0:
                    continue
                self.view_counts[t, side] = len(Rs)
                if len(self.serials) == rotations.shape[0]:
                    self.available_serials[t][side] = [self.serials[i] for i in view_indices.tolist()]
                mean_rot = []
                mean_q = []
                for joint in range(15):
                    rot = _consensus(Rs[:, joint])
                    mean_rot.append(rot)
                    q = normalize_quaternions(matrix_to_q(torch.from_numpy(rot.as_matrix())))
                    mean_q.append(q.numpy())
                self.means[t, side] = np.stack(mean_q, 0)
                self.initializable[t, side] = True
                if len(Rs) < 2:
                    continue
                mean_R = Rotation.from_matrix(np.stack([r.as_matrix() for r in mean_rot], 0))
                residuals = np.stack(
                    [
                        (Rotation.from_matrix(Rs[v]) * mean_R.inv()).as_rotvec().reshape(45)
                        for v in range(len(Rs))
                    ],
                    0,
                )
                cov, coefficient = _oas(residuals, assume_centered=True)
                eigvals = np.linalg.eigvalsh(cov)
                if float(eigvals.min()) <= 0.0:
                    self.covariance_failures.append(
                        {"frame": int(frame), "side": side, "min_eigenvalue": float(eigvals.min())}
                    )
                    raise ValueError(
                        f"OAS covariance is not positive definite for frame {frame} side {side}: "
                        f"min eigenvalue {float(eigvals.min()):.6g}"
                    )
                self.precision[t, side] = np.linalg.inv(cov)
                self.evaluable[t, side] = True
                self.oas_shrinkage.append(coefficient)
                self.oas_used_count += 1

        self.initialize_called = False
        self.initialized_frame_sides = 0

    def initialize(self, q55: torch.Tensor) -> torch.Tensor:
        if tuple(q55.shape) != (self.T, 55, 4):
            raise ValueError(f"q55 must be {self.T},55,4, got {tuple(q55.shape)}")
        means = torch.as_tensor(self.means, device=q55.device, dtype=q55.dtype)
        with torch.no_grad():
            for t in range(self.T):
                for side in range(2):
                    if not bool(self.initializable[t, side]):
                        continue
                    lo = 25 if side == 0 else 40
                    q55[t, lo:lo + 15] = means[t, side]
        self.initialize_called = True
        self.initialized_frame_sides = int(self.initializable.sum())
        return q55

    def loss(self, q55: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((), device=q55.device, dtype=q55.dtype)
        if not bool(self.evaluable.any()):
            return zero
        if tuple(q55.shape) != (self.T, 55, 4):
            raise ValueError(f"q55 must be {self.T},55,4, got {tuple(q55.shape)}")
        means = torch.as_tensor(self.means, device=q55.device, dtype=q55.dtype)
        left = relative_log(q55[:, 25:40], means[:, 0])
        right = relative_log(q55[:, 40:55], means[:, 1])
        residual = torch.stack([left, right], dim=1).reshape(self.T, 2, 45)
        r = residual.to(torch.float64)
        precision = torch.as_tensor(self.precision, device=q55.device, dtype=torch.float64)
        mask = torch.as_tensor(self.evaluable, device=q55.device, dtype=torch.bool)
        quad = torch.einsum("tsi,tsij,tsj->ts", r, precision, r) / 45.0
        vals = quad[mask]
        if vals.numel() == 0:
            return zero
        return (0.5 * vals.mean()).to(q55.dtype)

    def provenance(self) -> Dict[str, Any]:
        single = int((self.initializable & ~self.evaluable).sum())
        no_view = int((~self.initializable).sum())
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "serials": self.serials,
            "identity_motion_frames": self.motion_frames,
            "observation_frames": self.obs_frames,
            "matched_frames": self.matched_frames,
            "missing_motion_frames": self.missing_motion_frames,
            "available_views_per_frame_side": self.view_counts.tolist(),
            "available_view_serials_per_frame_side": self.available_serials,
            "side_order": ["left", "right"],
            "consensus": "per-joint Rotation.mean init + 20 iterations weights=1/max(angle,1e-6), all valid views",
            "oas_formula_source": "local runtime/hand/oas_source.py _oas formula, NumPy implementation; no sklearn import",
            "covariance": "OAS assume_centered=True over 45D left-tangent residuals Log(Rview Rmean^-1)",
            "covariance_not_divided_by_views": True,
            "covariance_floor": None,
            "oas_used_count": self.oas_used_count,
            "oas_shrinkage": self.oas_shrinkage,
            "single_view_initialized_without_covariance": single,
            "no_view_unobserved": no_view,
            "initialize_called": self.initialize_called,
            "initialized_frame_sides": self.initialized_frame_sides,
            "loss_evaluated_frame_sides": int(self.evaluable.sum()),
            "cross_view_dispersion_is_not_calibrated_shared_bias": True,
            "not_a_quality_acceptance": True,
        }
