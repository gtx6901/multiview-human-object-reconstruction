#!/usr/bin/env python
"""Body, foot and hand observation terms for the quaternion motion fit.

The 58 base landmarks retain the production definitions, weights, raw-image
2D validity masks, screened 3D validity and Huber scales. Face anchors are
attached separately by the quaternion fit. Its quaternion MAP objective also
defines the body/translation trust prior; this module supplies data terms.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from hoi.motion.landmarks import WORLD_HUBER_SCALE

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
LANDMARKS_SOURCE = THIS_DIR / "landmarks.py"


def base_2d_valid_mask(
    xy: torch.Tensor,
    score: torch.Tensor,
    views: Sequence[Dict[str, Any]],
    in_image: torch.Tensor | None = None,
) -> torch.Tensor:
    """T,V,J 2D validity from raw metadata; never from triangulation inliers.

    xy is T,V,J,2, score is T,V,J, views are the person manifest views in V
    order, and optional in_image is V,T,J from the per-camera keypoint NPZ.
    A point qualifies only if its score is finite and >=0.5, its xy is finite,
    it lies in that camera's real manifest width/height, and (when supplied)
    the NPZ in_image flag is true.  No 3D valid, inlier, support or minimum-view
    test is used here.
    """
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be T,V,J,2, got {tuple(xy.shape)}")
    T, V, J = xy.shape[:3]
    if tuple(score.shape) != (T, V, J):
        raise ValueError(f"score must be {T},{V},{J}, got {tuple(score.shape)}")
    if len(views) != V:
        raise ValueError(f"views length {len(views)} != V={V}")
    if in_image is not None and tuple(in_image.shape) != (V, T, J):
        raise ValueError(f"in_image must be V,T,J=({V},{T},{J}), got {tuple(in_image.shape)}")

    finite = torch.isfinite(xy).all(dim=-1) & torch.isfinite(score)
    mask = finite & (score >= 0.5)
    for vi, view in enumerate(views):
        if "width" not in view or "height" not in view:
            raise KeyError(f"manifest view {vi} lacks width/height")
        width = float(view["width"])
        height = float(view["height"])
        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError(f"manifest view {vi} has invalid width/height: {width}, {height}")
        x = xy[:, vi, :, 0]
        y = xy[:, vi, :, 1]
        bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        mask[:, vi, :] = mask[:, vi, :] & bounds
    if in_image is not None:
        mask = mask & in_image.to(dtype=torch.bool).permute(1, 0, 2)
    return mask


def _load_fit_contract(smplx_path: Path):
    """Load the shared production body, foot and hand landmark definitions."""
    from hoi.motion.landmarks import BODY, HAND_OFFSET, canonical_offsets

    feet, tips = canonical_offsets(smplx_path)
    source_sha256 = hashlib.sha256(LANDMARKS_SOURCE.read_bytes()).hexdigest()
    provenance = {
        "source": str(LANDMARKS_SOURCE),
        "source_sha256": source_sha256,
        "adapted_sha256": source_sha256,
        "replacements": [],
        "loss_hyperparameters_changed": False,
    }
    return {
        "BODY": dict(BODY),
        "feet": feet,
        "tips": tips,
        "HAND_OFFSET": int(HAND_OFFSET),
        "provenance": provenance,
    }


class FullObservations58:
    """Production observation terms with 58 base body, foot and hand landmarks."""

    def __init__(
        self,
        person: Path,
        avatar: Path,
        initialization: Path,
        smplx_path: Path,
        expected_t: int | None = None,
        face_constraints: Path | None = None,
    ):
        if face_constraints is not None:
            raise ValueError("attach face constraints with hoi.motion.fit.attach_face")
        self.person_path = Path(person).resolve()
        info = json.loads(self.person_path.read_text(encoding="utf-8"))
        self.info = info
        self.avatar_path = Path(avatar).resolve()
        self.init_path = Path(initialization).resolve()
        self.smplx_path = Path(smplx_path).resolve()

        av = torch.load(self.avatar_path, map_location="cpu", weights_only=False)
        self.scale = (
            float(av["poses"]["scale"].reshape(-1)[0])
            if torch.is_tensor(av["poses"]["scale"])
            else float(av["poses"]["scale"])
        )
        meta = json.loads(Path(info["manifest"]).read_text(encoding="utf-8"))
        self.views = meta["views"]
        self.P = torch.tensor(
            np.stack(
                [
                    np.asarray(v["K_undistorted_native"], np.float64)
                    @ np.asarray(v["W"], np.float64)[:3]
                    for v in self.views
                ]
            ),
            dtype=torch.float64,
        )
        obs = np.load(Path(info["observations"]), allow_pickle=False)
        self.xyz = torch.tensor(np.asarray(obs["xyz"], np.float64), dtype=torch.float64)
        self.valid3d = torch.tensor(
            np.asarray(obs["valid"], bool), dtype=torch.bool
        ) & torch.isfinite(self.xyz).all(-1)
        inlier = np.asarray(obs["inlier"], bool)
        if inlier.ndim != 3:
            raise ValueError(f"inlier must be V,T,J, got {inlier.shape}")
        self.inlier = torch.tensor(inlier, dtype=torch.bool)  # V,T,J
        T = int(self.xyz.shape[0])
        V = len(self.views)
        if self.inlier.shape != (V, T, 133):
            raise ValueError(
                f"inlier layout must be V,T,133=({V},{T},133), got {tuple(self.inlier.shape)}"
            )
        if expected_t is not None and T != int(expected_t):
            raise ValueError(f"observations T={T}, expected {expected_t}")
        key_dir = Path(info["keypoints"])
        xy, sc, in_image_views = [], [], []
        for v in self.views:
            with np.load(key_dir / f"{v['serial']}.npz", allow_pickle=False) as z:
                xy.append(np.asarray(z["xy"], np.float64))
                sc.append(np.asarray(z["score"], np.float64))
                if "in_image" in z.files:
                    in_image_views.append(np.asarray(z["in_image"], bool))
                else:
                    in_image_views.append(None)
        self.xy = torch.tensor(np.stack(xy), dtype=torch.float64).permute(1, 0, 2, 3)  # T,V,J,2
        self.score = torch.tensor(np.stack(sc), dtype=torch.float64).permute(1, 0, 2)  # T,V,J
        if self.xy.shape[:2] != (T, len(self.views)):
            raise ValueError(f"xy shape {tuple(self.xy.shape)} inconsistent with T/V")
        if any(item is not None for item in in_image_views):
            in_image_np = np.ones((len(self.views), T, 133), bool)
            for vi, item in enumerate(in_image_views):
                if item is not None:
                    if item.shape != (T, 133):
                        raise ValueError(f"in_image shape {item.shape} != {(T, 133)} for view {vi}")
                    in_image_np[vi] = item
            in_image = torch.tensor(in_image_np, dtype=torch.bool)
        else:
            in_image = None
        self.base_2d_mask = base_2d_valid_mask(
            self.xy, self.score, self.views, in_image=in_image
        )  # T,V,133

        contract = _load_fit_contract(self.smplx_path)
        self.contract = contract
        self.BODY = contract["BODY"]
        self.table: List[Tuple[int, int, torch.Tensor | None, float]] = []
        for coco, joint in self.BODY.items():
            self.table.append((int(coco), int(joint), None, 1.0))
        for coco, joint, offset in contract["feet"]:
            self.table.append((int(coco), int(joint), offset.double(), 0.10))
        for coco, joint, offset in contract["tips"]:
            self.table.append((int(coco), int(joint), offset.double(), 0.10))
        for side in range(2):
            for finger in range(5):
                for k in range(3):
                    coco = 91 + side * 21 + 1 + 4 * finger + k
                    joint = [37, 25, 28, 34, 31][finger] + side * 15 + k
                    self.table.append((int(coco), int(joint), None, 0.20))
        if len(self.table) != 58:
            raise RuntimeError(f"expected 58 fitted landmarks, built {len(self.table)}")
        self.base_count = 58
        self.face_count = 0
        self.face_path = None
        self.coco = [x[0] for x in self.table]
        self.joints = torch.tensor([x[1] for x in self.table], dtype=torch.long)
        offsets = []
        for _, _, off, _ in self.table:
            offsets.append(np.zeros(3, np.float64) if off is None else off.numpy())
        self.offset = torch.tensor(np.stack(offsets), dtype=torch.float64)  # L,3
        self.weights = torch.tensor([x[3] for x in self.table], dtype=torch.float64)
        self.base_w_sum = self.weights[: self.base_count].sum()
        self.w_sum = self.weights.sum()

        # Base 2D validity uses raw-image metadata, not triangulation inliers.
        masks = []
        for k in range(self.base_count):
            masks.append(self.base_2d_mask[:, :, self.coco[k]])
        self.mask = torch.stack(masks, dim=-1)  # T,V,L

    def landmarks(self, world: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
        """world is fk_mat's scaled+translated joint positions; Rg all rotations."""
        base = world[:, self.joints]
        off = torch.einsum("tnij,nj->tni", Rg[:, self.joints], self.offset)
        return base + self.scale * off

    def loss(self, world: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
        points = self.landmarks(world, Rg)
        homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        q = torch.einsum("vij,tnj->tvni", self.P, homo)
        uv = q[..., :2] / q[..., 2:3].clamp_min(1e-9)

        data2d = torch.zeros((), dtype=torch.float64)
        for k, coco in enumerate(self.coco):
            m = self.mask[:, :, k]
            if not bool(m.any()):
                continue
            # Mask BEFORE subtraction/norm: unobserved xy may be NaN.
            err = (uv[:, :, k][m] - self.xy[:, :, coco][m]).norm(dim=-1) / 5.0
            term = torch.nn.functional.huber_loss(
                err, torch.zeros_like(err), reduction="mean", delta=1.0
            )
            data2d = data2d + self.weights[k].to(term.device) * term
        data2d = data2d / self.w_sum.to(data2d.device)

        data3d = torch.zeros((), dtype=torch.float64)
        for k, coco in enumerate(self.coco):
            if k >= self.base_count:
                continue  # face anchors are 2D-only; no invented 3D face point
            m = self.valid3d[:, coco]
            if not bool(m.any()):
                continue
            # Mask BEFORE subtraction/norm; no fake xyz for missing joints.
            d = (points[m, k] - self.xyz[m, coco]).norm(dim=-1) / WORLD_HUBER_SCALE
            term = torch.nn.functional.huber_loss(
                d, torch.zeros_like(d), reduction="mean", delta=1.0
            )
            data3d = data3d + self.weights[k].to(term.device) * term
        data3d = data3d / self.base_w_sum.to(data3d.device)
        return data2d + data3d

    def provenance(self) -> Dict[str, Any]:
        return {
            "landmark_count": len(self.table),
            "base_body_feet_hands": self.base_count,
            "face_landmarks": self.face_count,
            "landmark_coco": self.coco,
            "landmark_joints": [int(x) for x in self.joints.tolist()],
            "landmark_weights": [float(x) for x in self.weights.tolist()],
            "groups": {
                "body": 12,
                "feet": 6,
                "fingertips": 10,
                "hand_joints": 30,
                "face_nose_eyes": self.face_count,
            },
            "contract_source": str(LANDMARKS_SOURCE),
            "contract_replacements": self.contract["provenance"]["replacements"],
            "base_2d_mask": (
                "raw score>=0.5, finite score/xy, manifest view width/height, "
                "and per-point in_image when present; no triangulation inlier, "
                "3D validity, support or minimum-view test"
            ),
            "triangulation_inlier": "retained only in observations for 3D/source audit; not used by base 2D loss",
            "score_threshold": 0.5,
            "huber_2d_px": 5.0,
            "huber_3d_world": WORLD_HUBER_SCALE,
            "face": (
                "face-stage exact valid mask appended to the constructor's base mask; "
                "no score/inlier second gate"
                if self.face_count
                else "unfitted; no face/head constraints supplied"
            ),
            "face_constraints": self.face_path,
            "hand_locks": False,
            "all55_free": True,
            "body_translation_prior": "defined by the quaternion MAP objective, not this data-term module",
            "initialization": str(self.init_path),
        }
