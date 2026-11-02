#!/usr/bin/env python
"""Parameterized current-person face stage, preserving original optimizer.

Adaptation of reference/fit_face_stage.py for one person:
  * same FK and neck/head-only optimization;
  * nose/eyes only, ears excluded;
  * original inliers + fixed initial front-facing dot > 0.15;
  * 5px Huber and 1e-3 body trust-region scale (verified generic fixes);
  * support basis only for head states not observable at 5 DOF;
  * missing 2D is masked before subtraction/norm/Huber (no NaN backward);
  * no expression fit, no manual camera selection, no pixel offset, no MJJ pose.

The face-local L, selected visibility, and support basis are also written to
face_constraints.npz for optional retention in the full-58 smoother.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "mjj"
REF_FACE = THIS_DIR / "reference" / "fit_face_stage.py"

FACE_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("FACE_KP = [0, 1, 2, 3, 4]    # nose, eyes, ears", "FACE_KP = [0, 1, 2]    # nose, eyes; ears excluded"),
    (
        "        w = (score >= 0.5).float()",
        "        w = (score >= 0.5).float() * FACE_VALID",
    ),
    (
        "        err = (uv - xy).norm(dim=-1)",
        "        valid = w > 0\n"
        "        err = torch.zeros_like(uv[..., 0])\n"
        "        if valid.any():\n"
        "            err[valid] = (uv[valid] - xy[valid]).norm(dim=-1)",
    ),
    (
        '        data = (torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=20.0) * w).sum() / w.sum().clamp_min(1.0)',
        '        data = (torch.nn.functional.huber_loss(err / 5.0, torch.zeros_like(err), reduction="none", delta=1.0) * w).sum() / w.sum().clamp_min(1.0)',
    ),
    (
        "        reg = 0.002 * (d_neck ** 2).sum() / T + 0.002 * (d_head ** 2).sum() / T",
        "        reg = 0.5 * ((d_neck ** 2).sum() + (d_head ** 2).sum()) / (T * 21 * 3 * 1e-3)",
    ),
    (
        "    d_neck = torch.zeros(T, 3, requires_grad=True)",
        "    d_neck_raw = torch.zeros(FACE_BASIS.shape[1], 3, requires_grad=True)",
    ),
    (
        "    d_head = torch.zeros(T, 3, requires_grad=True)",
        "    d_head_raw = torch.zeros(FACE_BASIS.shape[1], 3, requires_grad=True)",
    ),
    (
        "    opt = torch.optim.Adam([d_neck, d_head], lr=0.01)",
        "    opt = torch.optim.Adam([d_neck_raw, d_head_raw], lr=0.01)",
    ),
    (
        "        body_ = body.clone()",
        "        d_neck = FACE_BASIS @ d_neck_raw\n"
        "        d_head = FACE_BASIS @ d_head_raw\n"
        "        body_ = body.clone()",
    ),
    (
        "        body_f = body.clone(); body_f[:, NECK] += d_neck; body_f[:, HEAD] += d_head",
        "        d_neck = FACE_BASIS @ d_neck_raw; d_head = FACE_BASIS @ d_head_raw\n"
        "        body_f = body.clone(); body_f[:, NECK] += d_neck; body_f[:, HEAD] += d_head",
    ),
)


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


def stage_inputs(person: Path, avatar: Path, motion: Path, out: Path) -> Dict[str, Path]:
    stage = out / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    av = torch.load(avatar, map_location="cpu", weights_only=False)
    thin = {"rig": av["rig"], "poses": av["poses"]}
    torch.save(thin, stage / "avatar.pt")
    m = torch.load(motion, map_location="cpu", weights_only=False)
    if "lhand_pose" not in m and "left_hand_pose" in m:
        m["lhand_pose"] = m["left_hand_pose"]
    if "rhand_pose" not in m and "right_hand_pose" in m:
        m["rhand_pose"] = m["right_hand_pose"]
    if "left_hand_pose" not in m and "lhand_pose" in m:
        m["left_hand_pose"] = m["lhand_pose"]
    if "right_hand_pose" not in m and "rhand_pose" in m:
        m["right_hand_pose"] = m["rhand_pose"]
    torch.save(m, stage / "motion.pt")
    return {"stage": stage, "avatar": stage / "avatar.pt", "motion": stage / "motion.pt"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face-constraints", default=None, help="optional output path; default out/face_constraints.npz")
    args = ap.parse_args()

    import pipeline_adapter as pa
    pa.configure_runtime_paths()
    from face_observation_contract import build_face_contract

    pa.configure_runtime_paths()
    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    motion = Path(args.motion).resolve()
    if motion.is_dir():
        motion = motion / "motion.pt"
    out = Path(args.out).resolve()
    if not out.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("output must stay below work/mjj")
    out.mkdir(parents=True, exist_ok=True)
    stage = stage_inputs(person, avatar, motion, out)

    fc_path = Path(args.face_constraints).resolve() if args.face_constraints else out / "face_constraints.npz"
    contract = build_face_contract(person, avatar, motion, fc_path)
    fcz = np.load(fc_path, allow_pickle=False)
    valid = torch.tensor(fcz["valid"], dtype=torch.bool).permute(1, 0, 2).float()  # T,V,3
    basis = torch.tensor(fcz["basis"], dtype=torch.float32)
    if valid.shape[0] != basis.shape[0]:
        raise ValueError("face visibility/basis T mismatch")

    mod, provenance = pa.load_reference_patched(REF_FACE, FACE_REPLACEMENTS)
    mod.SRC = stage["stage"]
    mod.OUT = out
    mod.OBS = Path(info["observations"])
    mod.KEY = Path(info["keypoints"])
    mod.MANIFEST = Path(info["manifest"])
    mod.FACE_VALID = valid
    mod.FACE_BASIS = basis
    print(
        f"[face] {contract['selected_inlier_front_points']} front nose/eye points; "
        f"supported={len(contract['supported_frames'])}/{len(contract['head_observation_information_rank'])}; "
        "ears excluded; no expression",
        flush=True,
    )
    mod.main()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    report.update(
        {
            "schema": "two_person_face_stage/v1",
            "source_face": str(REF_FACE),
            "source_sha256": provenance["source_sha256"],
            "adapted_sha256": provenance["adapted_sha256"],
            "source_replacements": provenance["replacements"],
            "face_keypoints": [0, 1, 2],
            "ears_excluded": True,
            "face_orientation_fitted": True,
            "face_expression_fitted": False,
            "face_weight_contract": "score>=.5 * front-visible original inlier; Huber 5px; body trust 1e-3",
            "visibility": contract,
            "face_constraints": str(fc_path),
            "all55_free_in_model": True,
            "body_hand_fixed_during_face_stage_only": True,
            "mj_pose_or_fcs006_used": False,
            "note": "Neck/head orientation stage only; no expression/face identity fit and no quality result.",
        }
    )
    atomic_write_json(out / "face_provenance.json", report)
    atomic_write_json(out / "reference_provenance.json", provenance)
    print(f"[face] wrote {out / 'motion.pt'}")
    print(f"[face] constraints {fc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
