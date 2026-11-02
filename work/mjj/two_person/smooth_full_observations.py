#!/usr/bin/env python
"""Full58-observation adapter for the archived 6D smoother.

The archived ``smooth_6d.py`` uses only 12 body joints for data while updating
all 55 rotations.  This module supplies the actually fitted evidence from the
current person instead:

* body 12, feet 6, fingertips 10, hand joints 30 = 58 landmarks;
* original fit_motion landmark definitions and per-landmark weights
  (body 1.0, feet 0.1, fingertips 0.1, hand joints 0.2);
* current-person saved 2D inliers (V,T,J), scores >=.5, and 3D validity;
* 2D Huber(5 px) and 3D Huber(.02 world units);
* body/translation trust prior centered on the actual ``--initialization``;
* no face/head constraints (unfitted), no hand locks, no new weights.

The source ``smooth_6d`` remains responsible for the 6D parameterisation,
temporal model, corrected FK tuple and all-55 free optimisation; this is a data
and prior adapter only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
REF_FIT = THIS_DIR / "reference" / "fit_motion.py"


def _load_fit_contract(smplx_path: Path):
    """Reuse the actual patched fit_motion.canonical_offsets and BODY/order."""
    import pipeline_adapter as pa

    pa.configure_runtime_paths()
    mod, provenance = pa.load_reference_patched(REF_FIT, pa.FIT_MOTION_REPLACEMENTS)
    mod.SMPLX = smplx_path
    feet, tips = mod.canonical_offsets()
    return {
        "BODY": dict(mod.BODY),
        "feet": feet,
        "tips": tips,
        "HAND_OFFSET": int(mod.HAND_OFFSET),
        "provenance": provenance,
    }


class FullObservations58:
    """Data/prior terms matching fit_motion's final stage2 contract."""

    def __init__(
        self,
        person: Path,
        avatar: Path,
        initialization: Path,
        smplx_path: Path,
        expected_t: int | None = None,
        face_constraints: Path | None = None,
    ):
        self.person_path = Path(person).resolve()
        info = json.loads(self.person_path.read_text(encoding="utf-8"))
        self.info = info
        self.avatar_path = Path(avatar).resolve()
        self.init_path = Path(initialization).resolve()
        self.smplx_path = Path(smplx_path).resolve()

        av = torch.load(self.avatar_path, map_location="cpu", weights_only=False)
        self.scale = float(av["poses"]["scale"].reshape(-1)[0]) if torch.is_tensor(av["poses"]["scale"]) else float(av["poses"]["scale"])
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
        self.valid3d = torch.tensor(np.asarray(obs["valid"], bool), dtype=torch.bool) & torch.isfinite(self.xyz).all(-1)
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
        xy, sc = [], []
        for v in self.views:
            with np.load(key_dir / f"{v['serial']}.npz", allow_pickle=False) as z:
                xy.append(np.asarray(z["xy"], np.float64))
                sc.append(np.asarray(z["score"], np.float64))
        self.xy = torch.tensor(np.stack(xy), dtype=torch.float64).permute(1, 0, 2, 3)  # T,V,J,2
        self.score = torch.tensor(np.stack(sc), dtype=torch.float64).permute(1, 0, 2)   # T,V,J
        if self.xy.shape[:2] != (T, len(self.views)):
            raise ValueError(f"xy shape {tuple(self.xy.shape)} inconsistent with T/V")

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
        self.face_path = None if face_constraints is None else str(Path(face_constraints).resolve())
        if face_constraints is not None:
            fcz = np.load(Path(face_constraints), allow_pickle=False)
            face_local = np.asarray(fcz["local"], np.float64).reshape(3, 3)
            face_valid = np.asarray(fcz["valid"], bool)  # V,T,3
            if face_valid.shape != (len(self.views), T, 3):
                raise ValueError(f"face valid must be V,T,3=({len(self.views)},{T},3), got {face_valid.shape}")
            for k in range(3):
                # nose/eye head-local offsets, fitted face-stage weight 1.0
                self.table.append((int(k), 15, torch.tensor(face_local[k]), 1.0))
            self.face_count = 3
            self.face_local = torch.tensor(face_local)
            self.face_valid = torch.tensor(face_valid, dtype=torch.bool)
        self.coco = [x[0] for x in self.table]
        self.joints = torch.tensor([x[1] for x in self.table], dtype=torch.long)
        offsets = []
        for _, _, off, _ in self.table:
            offsets.append(np.zeros(3, np.float64) if off is None else off.numpy())
        self.offset = torch.tensor(np.stack(offsets), dtype=torch.float64)  # L,3
        self.weights = torch.tensor([x[3] for x in self.table], dtype=torch.float64)
        self.base_w_sum = self.weights[: self.base_count].sum()
        self.w_sum = self.weights.sum()

        # Keep screen's native V,T,J; align to T,V,L without changing data.
        masks = []
        for k, c in enumerate(self.coco):
            if k < self.base_count:
                m = self.inlier[:, :, c].permute(1, 0) & (self.score[:, :, c] >= 0.5)
            else:
                fk = k - self.base_count
                m = self.face_valid[:, :, fk].permute(1, 0) & (self.score[:, :, c] >= 0.5)
            masks.append(m)
        self.mask = torch.stack(masks, dim=-1)  # T,V,L

        init = torch.load(self.init_path, map_location="cpu", weights_only=False)
        self.init_body = torch.as_tensor(init["body_pose"], dtype=torch.float64).reshape(T, 21, 3)
        self.init_trans = torch.as_tensor(init["translation_world"], dtype=torch.float64).reshape(T, 3)
        if self.init_body.shape[0] != T:
            raise ValueError("initialization T does not match observations")

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
            d = (points[m, k] - self.xyz[m, coco]).norm(dim=-1) / 0.02
            term = torch.nn.functional.huber_loss(
                d, torch.zeros_like(d), reduction="mean", delta=1.0
            )
            data3d = data3d + self.weights[k].to(term.device) * term
        data3d = data3d / self.base_w_sum.to(data3d.device)
        return data2d + data3d

    def prior(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """fit_motion source-body/translation trust, centered on init geometry.

        Only body21 local rotations and translation; root and other joints are
        not additionally anchored. No face constraint is present.
        """
        body = matrix_to_axis_angle(R[:, 1:22])
        body_prior = 0.5 * (body - self.init_body.to(body.device)).square().mean() / 1e-3
        trans_prior = 0.5 * (t - self.init_trans.to(t.device)).square().mean() / 1e-4
        return body_prior + trans_prior

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
            "contract_source": str(REF_FIT),
            "contract_replacements": self.contract["provenance"]["replacements"],
            "inlier_layout": "V,T,J retained; converted to T,V,L mask here",
            "score_threshold": 0.5,
            "huber_2d_px": 5.0,
            "huber_3d_world": 0.02,
            "face": (
                "face-stage nose/eyes local L + selected inlier/front visibility retained; "
                "face stage is head orientation only, no expression fit"
                if self.face_count
                else "unfitted; no face/head constraints supplied"
            ),
            "face_constraints": self.face_path,
            "hand_locks": False,
            "all55_free": True,
            "body_translation_prior": "fit_motion source-body21 1e-3 + translation 1e-4 centered on --initialization",
        }
