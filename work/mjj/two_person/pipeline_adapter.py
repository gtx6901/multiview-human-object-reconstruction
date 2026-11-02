#!/usr/bin/env python
"""Parameterised adapter from two-person observations to the current q pipeline.

The current recommended production entry is ``reconstruct``: it runs the three
actual q scripts (body/hand fit, face, final MAP) in sequence for one identity.
``run-motion``/``run-smooth`` remain historical frozen-AA adapters; they are not
the current recommended path.

The adapter does not copy or fork fitting algorithms.  It builds a *per-person*
observation package (the exact arrays/keys consumed by ``reference/fit_motion.py``),
applies the one FCS006 world normalization to both people, and can invoke the
original frozen scripts when explicitly requested.

Commands
--------
reconstruct
    Run the current three-stage q chain for one person into the existing shared
    stage root: ``<out>/<identity>``, ``<out>/face_<identity>``,
    ``<out>/map_<identity>``.  The three scripts are launched as real subprocesses
    with explicit image-hand observations and audited world-up.

build
    Convert ``instance_inputs`` output for one identity into:
      person/observations.npz, person/keypoints/<serial>.npz, person/manifest.json
    All 3D coordinates and camera translations use the *same* FCS006 gauge.

run-motion
    Historical frozen ``fit_motion.py`` compatibility; not the current q route.

The FCS006 normalization is a world-unit transform
``X_unit=(X_raw-center_raw)/body_height_raw``; it is applied identically to fcs
and partner.  It is not per-person centering/scaling.  No Z-up/floor assumption
is added.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_SCENE = THIS_DIR / "data" / "scene_inventory" / "scene.json"
DEFAULT_INSTANCES = THIS_DIR / "data" / "instance_inputs" / "run"
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "capture" / "manifest.json"
DEFAULT_REF = THIS_DIR / "reference" / "fit_motion.py"
RAW_SPACE = "raw_distorted_native_2048x1536"
UND_SPACE = "undistorted_native_2048x1536"


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def configure_runtime_paths() -> None:
    """Use the project's real runtime_imports hook for vendor/smplx paths.

    The frozen scripts import ``smplx``/``pytorch3d`` from lhmpp/vendor; adding
    only LHM/deps+runtime/deps is insufficient and can silently pick an
    unrelated same-name package.
    """
    lhm = PROJECT_ROOT / "lhmpp"
    for p in (
        lhm,
        lhm / "vendor",
        lhm / "deps",
        PROJECT_ROOT / "runtime" / "deps",
    ):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from runtime_imports import configure

        configure()
    except Exception as exc:
        raise RuntimeError(
            f"project runtime_imports.configure() failed; refusing to use a wrong smplx: {exc}"
        ) from exc


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def undistort_points(cam: Dict[str, Any], pts: np.ndarray) -> np.ndarray:
    import cv2

    K = np.asarray(cam["K_native"], np.float64)
    dist = np.asarray(cam["distortion_opencv5"], np.float64)
    K_und = np.asarray(cam["K_undistorted"], np.float64)
    q = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if dist.size:
        out = cv2.undistortPoints(q, K, dist, P=K_und)
    else:
        out = cv2.undistortPoints(q, K, np.zeros(5), P=K_und)
    return out.reshape(-1, 2)


def load_gauge(manifest_path: Path, scene: Dict[str, Any]) -> Dict[str, Any]:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    wn = man.get("world_normalization")
    if not wn:
        raise RuntimeError(f"{manifest_path} has no world_normalization; shared gauge is undefined")
    center = np.asarray(wn["center_raw"], np.float64)
    height = float(wn["body_height_raw_units"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(height) or height <= 0:
        raise RuntimeError("invalid FCS006 world_normalization")
    selected = (scene.get("calibration_selection") or {}).get("selected") or {}
    scene_cal_sha = selected.get("sha256")
    manifest_cal_sha = man.get("calibration_sha256")
    if scene_cal_sha and manifest_cal_sha and scene_cal_sha != manifest_cal_sha:
        raise RuntimeError(
            "refusing to apply FCS006 normalization: scene calibration SHA "
            f"{scene_cal_sha} != manifest calibration SHA {manifest_cal_sha}"
        )
    return {
        "source": str(manifest_path),
        "sequence": man.get("sequence"),
        "center_raw": center.tolist(),
        "body_height_raw_units": height,
        "transform": "X_unit=(X_raw-center_raw)/body_height_raw_units; R unchanged; t_unit=(R@center_raw+t_raw)/body_height_raw_units",
        "z_up_or_floor_assumed": False,
        "calibration_sha256": scene_cal_sha or manifest_cal_sha,
        "calibration_sha_match": bool(scene_cal_sha and manifest_cal_sha and scene_cal_sha == manifest_cal_sha),
        "note": "ONE shared world transform for fcs and partner; not an identity definition.",
    }


def world_to_unit(W: np.ndarray, center: np.ndarray, height: float) -> np.ndarray:
    W = np.asarray(W, np.float64)
    R, t = W[:3, :3], W[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = (R @ center + t) / height
    return out


def load_instance_package(index_path: Path, scene: Dict[str, Any], identity: str) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    npz_path = Path(index.get("observations_npz") or (index_path.parent / "observations.npz"))
    with np.load(npz_path, allow_pickle=False) as z:
        obs = {k: z[k] for k in z.files}
    ids = [str(x) for x in obs["identities"].tolist()]
    if identity not in ids:
        raise ValueError(f"identity {identity!r} not in {ids}")
    k = ids.index(identity)
    coord = str(index.get("coordinate_space") or RAW_SPACE)
    cams = index["cameras"]
    frames = [int(x) for x in obs["frames"].tolist()]
    return {"index": index, "obs": obs, "identity_index": k, "ids": ids, "coord": coord, "cams": cams, "frames": frames}


def run_screen_keypoints(key_dir: Path, manifest_path: Path, out_dir: Path) -> Path:
    """Call the frozen project screen_keypoints.main unchanged.

    Its module-level input/output paths are the only adapter points.  The
    original SCORE_MIN=.5, MAX_ERROR_PX=10, MIN_VIEWS=3, 28 view-pair RANSAC
    and ray-angle gate remain in force.
    """
    ref = THIS_DIR / "reference" / "screen_keypoints.py"
    mod = load_reference(ref)
    mod.KEYPOINTS = key_dir
    mod.MANIFEST = manifest_path
    mod.OUT = out_dir
    mod.main()
    return out_dir / "observations.npz"


def build(args: argparse.Namespace) -> int:
    scene_path = Path(args.scene).resolve()
    index_path = Path(args.instances).resolve()
    if index_path.is_dir():
        index_path = index_path / "index.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    pkg = load_instance_package(index_path, scene, args.identity)
    obs, ids, k, coord, cams, frames = (
        pkg["obs"], pkg["ids"], pkg["identity_index"], pkg["coord"], pkg["cams"], pkg["frames"]
    )
    if coord not in (RAW_SPACE, UND_SPACE):
        raise ValueError(f"unsupported observation coordinate space: {coord}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gauge = load_gauge(Path(args.manifest).resolve(), scene)
    center = np.asarray(gauge["center_raw"], np.float64)
    height = float(gauge["body_height_raw_units"])
    gauge_path = out.parent / "gauge.json"
    if gauge_path.is_file():
        old = json.loads(gauge_path.read_text(encoding="utf-8"))
        if not (
            np.allclose(old.get("center_raw"), gauge["center_raw"])
            and abs(float(old.get("body_height_raw_units")) - height) < 1e-12
        ):
            raise RuntimeError(
                f"{gauge_path} already contains a different gauge; refusing per-person scaling"
            )
    atomic_write_json(gauge_path, gauge)

    T, V, J = len(frames), len(cams), 133
    views = []
    for cam in cams:
        W_unit = world_to_unit(np.asarray(cam["W"], np.float64), center, height)
        views.append(
            {
                "serial": str(cam["serial"]),
                "K_undistorted_native": np.asarray(cam["K_undistorted"]).tolist(),
                "W": W_unit.tolist(),
                "width": int(cam["width"]),
                "height": int(cam["height"]),
                "source": cam.get("path"),
            }
        )
    manifest = {
        "sequence": scene.get("session", {}).get("name"),
        "identity": args.identity,
        "views": views,
        "coordinate_space": UND_SPACE,
        "world_normalization": gauge,
        "scope": (
            "Per-person adapter package for frozen screen_keypoints + fit_motion; "
            "shared FCS006 normalized world gauge."
        ),
    }
    atomic_write_json(out / "manifest.json", manifest)

    key_dir = out / "keypoints"
    raw_dir = out / "raw2d"
    key_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for vi, cam in enumerate(cams):
        source_xy = np.asarray(obs["kp"][vi, :, k], np.float64)  # T,133,2
        sc = np.asarray(obs["kp_score"][vi, :, k], np.float64)   # T,133
        if coord == RAW_SPACE:
            xy = np.stack([undistort_points(cam, x) for x in source_xy]) if T else np.zeros((0, J, 2))
        else:
            xy = source_xy.copy()
        in_image = (
            np.asarray(obs["kp_present"][vi, :, k], bool)[:, None]
            & np.isfinite(xy).all(axis=2)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] < int(cam["width"]))
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < int(cam["height"]))
        )
        atomic_savez(
            key_dir / f"{cam['serial']}.npz",
            xy=xy.astype(np.float32),
            score=sc.astype(np.float32),
            in_image=in_image,
            frames=np.asarray(frames, dtype=np.int32),
        )
        atomic_savez(
            raw_dir / f"{cam['serial']}.npz",
            xy=source_xy.astype(np.float32),
            score=sc.astype(np.float32),
            coordinate_space=np.asarray(coord),
            frames=np.asarray(frames, dtype=np.int32),
        )

    screen_out = out / "screen"
    screen_obs_path = run_screen_keypoints(key_dir, out / "manifest.json", screen_out)
    with np.load(screen_obs_path, allow_pickle=False) as z:
        screen = {name: z[name] for name in z.files}
    xyz = np.asarray(screen["xyz"], np.float32)
    valid = np.asarray(screen["valid"], bool)
    inlier_by_view = np.asarray(screen["inlier"], bool)  # V,T,J from screen_keypoints
    if xyz.shape != (T, J, 3) or valid.shape != (T, J) or inlier_by_view.shape != (V, T, J):
        raise RuntimeError(
            "frozen screen_keypoints returned unexpected shapes: "
            f"xyz={xyz.shape} valid={valid.shape} inlier={inlier_by_view.shape}; "
            f"expected {(T,J,3)} {(T,J)} {(V,T,J)}"
        )
    # fit_motion permutes this V,T,J array to T,V,J to match stacked 2D scores.
    inlier_for_fit = np.ascontiguousarray(inlier_by_view)
    xyz_raw = xyz.astype(np.float64) * height + center[None, None, :]
    atomic_savez(
        out / "observations.npz",
        xyz=xyz,
        xyz_raw=xyz_raw.astype(np.float32),
        valid=valid,
        inlier=inlier_for_fit,
        support=np.asarray(screen.get("support"), np.int16),
        median_error_px=np.asarray(screen.get("median_error_px"), np.float32),
        frames=np.asarray(screen.get("frames", frames), dtype=np.int32),
        identity=np.asarray(args.identity),
        world_normalization_center=center.astype(np.float32),
        world_normalization_height=np.asarray(height, np.float32),
        screen_inlier_layout=np.asarray("V,T,J; fit_motion permutes to T,V,J"),
    )
    person = {
        "schema": "pipeline_person_observations/v1",
        "identity": args.identity,
        "identity_index_in_instance_run": k,
        "instance_index": str(index_path),
        "instance_coordinate_space": coord,
        "out": str(out),
        "frames": [int(x) for x in frames],
        "valid_joint_count": int(valid.sum()),
        "valid_frame_joint_fraction": float(valid.mean()) if valid.size else 0.0,
        "inlier_shape_for_fit_motion": list(inlier_for_fit.shape),
        "inlier_shape_screen_native": list(inlier_by_view.shape),
        "screen_parameters": {
            "SCORE_MIN": 0.5,
            "MAX_ERROR_PX": 10.0,
            "MIN_VIEWS": 3,
            "MIN_RAY_ANGLE_DEG": 1.5,
            "pair_ransac": "28 view pairs",
        },
        "gauge": gauge,
        "world_normalization_applied_once": True,
        "manifest": str(out / "manifest.json"),
        "observations": str(out / "observations.npz"),
        "screen_observations": str(screen_obs_path),
        "screen_report": str(screen_out / "report.json"),
        "keypoints": str(key_dir),
        "raw2d": str(raw_dir),
        "raw2d_coordinate_space": coord,
        "warnings": [],
    }
    if int(valid.sum()) == 0:
        person["warnings"].append("screen_keypoints produced no valid 3D joint")
    atomic_write_json(out / "person.json", person)
    print(f"[adapter] person package: {out}")
    print(f"[adapter] screen valid joints: {int(valid.sum())}, inlier fit shape={inlier_for_fit.shape}")
    print(f"[adapter] shared gauge: center={gauge['center_raw']} height={height}")
    return 0 if int(valid.sum()) > 0 else 1


def subset_avatar(avatar: Path, frames: Sequence[int], out_path: Path, neutral: bool = True) -> Path:
    """Make a temporary avatar bundle whose pose timeline matches observations.

    ``neutral=True`` is the default: root/body/hands are zero and only the
    avatar's own scale/betas are retained.  This prevents FCS006 per-frame
    action from being used as the 008 initialization.  ``neutral=False`` is an
    explicit opt-in for a genuinely relevant source motion.
    """
    import torch

    bundle = torch.load(avatar, map_location="cpu", weights_only=False)
    poses = bundle.get("poses")
    if not isinstance(poses, dict) or "frames" not in poses:
        raise KeyError(f"{avatar} has no poses['frames']")
    old = poses["frames"]
    old_list = [int(x) for x in (old.tolist() if hasattr(old, "tolist") else old)]
    index = {f: i for i, f in enumerate(old_list)}
    T = len(frames)
    if neutral:
        new_poses = {
            "root_pose": torch.zeros(T, 3),
            "body_pose": torch.zeros(T, 21, 3),
            "left_hand_pose": torch.zeros(T, 15, 3),
            "right_hand_pose": torch.zeros(T, 15, 3),
            "translation_world": torch.zeros(T, 3),
            "scale": poses.get("scale", torch.tensor(1.0)),
            "betas": poses.get("betas", torch.zeros(1, 10)),
            "frames": torch.tensor(list(frames), dtype=torch.long),
        }
    else:
        if len(old_list) == 1:
            idx = torch.zeros(T, dtype=torch.long)
            new_frames = list(frames)
        else:
            missing = [f for f in frames if f not in index]
            if missing:
                raise ValueError(f"avatar has no pose frame for observations {missing[:5]}")
            idx = torch.tensor([index[f] for f in frames], dtype=torch.long)
            new_frames = [old_list[i] for i in idx.tolist()]
        new_poses = {}
        for k, v in poses.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == len(old_list):
                new_poses[k] = v[idx]
            else:
                new_poses[k] = v
        new_poses["frames"] = torch.tensor(new_frames, dtype=torch.long)
    bundle["poses"] = new_poses
    torch.save(bundle, out_path)
    return out_path


def load_reference(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"reference module not found: {path}. Codex is exporting the frozen tag files; "
            "do not substitute a rewritten fitter."
        )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIT_MOTION_REPLACEMENTS = (
    # Index missing observations before residual arithmetic: otherwise NaNs
    # outside an inlier mask can still poison the norm's backward computation.
    (
        "errs.append((uv[:, :, k] - xy[:, :, idx[k]]).norm(dim=-1)[m].mean())",
        "errs.append((uv[:, :, k][m] - xy[:, :, idx[k]][m]).norm(dim=-1).mean())",
    ),
    (
        "err = (uv[:, :, k] - xy[:, :, cj]).norm(dim=-1) / 5.0",
        "err = (uv[:, :, k][m] - xy[:, :, cj][m]).norm(dim=-1) / 5.0",
    ),
    (
        'reduction="none", delta=1.0)[m].mean()',
        'reduction="none", delta=1.0).mean()',
    ),
    # Verified common source fixes from work/mjj/shared_fcs_fit.py and
    # work/mjj/ROOT_CAUSE.md #5 (COCO thumb/index/middle/ring/pinky ->
    # SMPL-X 37/25/28/34/31, +15 for right) and #4 (body->feet->hands data
    # isolation). No loss hyperparameter is changed.
    ("base = 25 + 15 * side", "base = 15 * side"),
    ("distal = base + 3 * f + 2", "distal = base + [37, 25, 28, 34, 31][f] + 2"),
    (
        "25 + side * 15 + 3 * f + k",
        "[37, 25, 28, 34, 31][f] + side * 15 + k",
    ),
)

# The context replacement above must target loss_fn only; evaluate() has the
# same wgt line and no `stage`. This wider replacement is applied first.
FIT_MOTION_CONTEXT_REPLACEMENTS = (
    (
        "    def loss_fn(tg, Rg, stage):\n"
        "        pts = points_of(tg, Rg) * scale\n"
        "        world = pts + trans.unsqueeze(1)\n"
        "        uv = reprojection(P, world)\n"
        "        idx = torch.tensor([x[0] for x in L]); wgt = torch.tensor([x[3] for x in L])",
        "    def loss_fn(tg, Rg, stage):\n"
        "        pts = points_of(tg, Rg) * scale\n"
        "        world = pts + trans.unsqueeze(1)\n"
        "        uv = reprojection(P, world)\n"
        "        idx = torch.tensor([x[0] for x in L]); wgt = torch.tensor([x[3] if (stage >= 2 or (x[0] < 91 and (stage >= 1 or x[0] < 17))) else 0.0 for x in L])",
    ),
)

# fit_motion's archived foot contact uses world Z. FCS's world is not known
# Z-up; this source patch is activated only with an explicit audited --world-up.
FIT_MOTION_CONTACT_REPLACEMENTS = (
    (
        "fy = world[:, foot_ids, 2] if world.shape[-1] == 3 else world[:, foot_ids, 1]",
        "fy = torch.einsum('tij,j->ti', world[:, foot_ids], WORLD_UP.to(world.device))",
    ),
)

SMOOTH_6D_REPLACEMENTS = (
    # Verified normalizer bug from shared_fcs_fit.smooth: fk_mat returns
    # (positions, rotations), not (rotations, positions).
    ("_, world_ = fk_mat(rest, parents, R_, scale, tr)", "world_, _ = fk_mat(rest, parents, R_, scale, tr)"),
    ("d.mkdir(parents=True, exist_ok=False)", "d.mkdir(parents=True, exist_ok=True)"),
)


def load_reference_patched(path: Path, replacements: Sequence[Tuple[str, str]]) -> Tuple[Any, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    original = path.read_text(encoding="utf-8-sig")
    text = original
    changed = []
    for before, after in replacements:
        if text.count(before) != 1:
            raise RuntimeError(
                f"verified source replacement does not match exactly once in {path}: {before!r} "
                f"count={text.count(before)}"
            )
        text = text.replace(before, after)
        changed.append({"before": before, "after": after})
    mod = types.ModuleType("fcs2_" + path.stem)
    mod.__file__ = str(path)
    exec(compile(text, str(path), "exec"), mod.__dict__)
    provenance = {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "adapted_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "replacements": changed,
        "loss_hyperparameters_changed": False,
    }
    return mod, provenance


def bundle_with_initialization(avatar: Path, initialization: Path, frames: Sequence[int], out_path: Path) -> Path:
    """Stage a real geometric motion seed into the avatar bundle."""
    import torch

    bundle = torch.load(avatar, map_location="cpu", weights_only=False)
    seed = torch.load(initialization, map_location="cpu", weights_only=False)
    if "frames" not in seed:
        raise KeyError(f"{initialization} has no frames")
    sf = [int(x) for x in (seed["frames"].tolist() if hasattr(seed["frames"], "tolist") else seed["frames"])]
    idx = {f: i for i, f in enumerate(sf)}
    missing = [f for f in frames if f not in idx]
    if missing:
        raise ValueError(f"initialization has no frame for observations {missing[:5]}")
    take = torch.tensor([idx[f] for f in frames], dtype=torch.long)
    selected = {}
    for key, value in seed.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == len(sf):
            selected[key] = value[take]
        else:
            selected[key] = value
    selected["frames"] = torch.tensor(list(frames), dtype=torch.long)
    ap = bundle.get("poses", {})
    for key in ("scale", "betas"):
        if key in selected and key in ap:
            a = torch.as_tensor(selected[key], dtype=torch.float32)
            b = torch.as_tensor(ap[key], dtype=torch.float32)
            if a.shape != b.shape or not torch.allclose(a, b, rtol=1e-4, atol=1e-5):
                raise ValueError(
                    f"initialization {key} does not match avatar {key}; refusing inconsistent rig/scale"
                )
    bundle["poses"] = selected
    torch.save(bundle, out_path)
    return out_path


def run_motion(args: argparse.Namespace) -> int:
    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    if not avatar.is_file():
        raise FileNotFoundError(avatar)
    frames = [int(x) for x in info["frames"]]
    if args.initialization:
        fit_avatar = bundle_with_initialization(
            avatar, Path(args.initialization).resolve(), frames, out / "_avatar_fit.pt"
        )
        initialization_source = str(Path(args.initialization).resolve())
    else:
        if args.init_mode != "neutral":
            raise RuntimeError(
                "run-motion requires --initialization from initialize_pose.py; "
                "neutral is only an explicit --init-mode neutral fallback"
            )
        fit_avatar = subset_avatar(avatar, frames, out / "_avatar_fit.pt", neutral=True)
        initialization_source = "explicit_neutral"
    ref = Path(args.reference).resolve()
    root = PROJECT_ROOT
    configure_runtime_paths()
    import torch

    if args.world_up is None:
        raise RuntimeError(
            "fit_motion's archived foot contact hardcodes world Z. FCS is not known Z-up. "
            "Pass --world-up X Y Z from the same-gauge world_up_audit before running."
        )
    wu = np.asarray(args.world_up, np.float64)
    wu = wu / (np.linalg.norm(wu) + 1e-12)
    replacements = list(FIT_MOTION_REPLACEMENTS) + list(FIT_MOTION_CONTEXT_REPLACEMENTS)
    replacements += list(FIT_MOTION_CONTACT_REPLACEMENTS)
    mod, provenance = load_reference_patched(ref, replacements)
    mod.WORLD_UP = torch.tensor(wu, dtype=torch.float32)
    # The frozen script computes PROJECT from its original location; in
    # reference/ that is wrong, so patch the module to the real project root.
    mod.PROJECT = root
    mod.AVATAR = fit_avatar
    mod.OBS = Path(info["observations"])
    mod.KEY = Path(info["keypoints"])
    mod.MANIFEST = Path(info["manifest"])
    mod.OUT = out
    mod.SMPLX = root / "lhmpp" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
    sys.argv = [
        str(ref),
        "--avatar",
        str(fit_avatar),
        "--out",
        str(out),
        "--steps",
        *[str(x) for x in args.steps],
        "--temporal",
        str(args.temporal),
    ]
    print(f"[adapter] calling frozen fit_motion.main for {info['identity']}")
    mod.main()
    provenance.update(
        {
            "COCO_to_SMPLX_hand_order": [37, 25, 28, 34, 31],
            "right_hand_offset": 15,
            "stage_data_isolation": "body -> +feet -> +hands",
            "world_up": wu.tolist(),
            "world_up_source": args.world_up_source,
            "world_up_note": "must be the audited foot-plane normal in the same FCS006 normalized gauge; not an assumed Z axis",
            "initialization": initialization_source,
            "applied": True,
        }
    )
    atomic_write_json(out / "reference_provenance.json", provenance)
    print(f"[adapter] motion written: {out / 'motion.pt'}")
    return 0


def run_proportions(args: argparse.Namespace) -> int:
    static = Path(args.static).resolve()
    if not (static / "manifest.json").is_file() or not (static / "keypoints.npz").is_file():
        raise FileNotFoundError(f"static package needs manifest.json + keypoints.npz: {static}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(args.reference).resolve()
    root = PROJECT_ROOT
    configure_runtime_paths()
    mod = load_reference(ref)
    mod.PROJECT = root
    mod.STATIC = static
    mod.SMPLX = root / "lhmpp" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
    mod.OUT = out
    sys.argv = [str(ref)]
    print(f"[adapter] calling frozen fit_proportions.main for {static}")
    mod.main()
    print(f"[adapter] proportions written: {out / 'proportions.pt'}")
    return 0


def build_static_package(person_dir: Path, out: Path) -> Path:
    """Build the input expected by the frozen fit_proportions.main.

    This is an input-format adapter only: all 2D/score/in-image values are the
    person's real observations; W is the same normalized-gauge camera used by
    screen_keypoints. No DLT, threshold, or measurement code is reimplemented.
    """
    info = json.loads((person_dir / "person.json").read_text(encoding="utf-8"))
    man = json.loads(Path(info["manifest"]).read_text(encoding="utf-8"))
    views = man["views"]
    key_dir = Path(info["keypoints"])
    xy, score, in_image = [], [], []
    for v in views:
        with np.load(key_dir / f"{v['serial']}.npz", allow_pickle=False) as z:
            xy.append(np.asarray(z["xy"], np.float64))
            score.append(np.asarray(z["score"], np.float64))
            in_image.append(np.asarray(z["in_image"], bool))
    xy = np.stack(xy, axis=0)              # V,T,J,2
    score = np.stack(score, axis=0)        # V,T,J
    in_image = np.stack(in_image, axis=0)  # V,T,J
    frames = np.asarray(info.get("frames", list(range(xy.shape[1]))), np.int32)
    out.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        out / "keypoints.npz",
        xy=xy.astype(np.float32),
        score=score.astype(np.float32),
        in_image=in_image,
        frames=frames,
    )
    static_views = []
    for v in views:
        static_views.append(
            {
                "serial": v["serial"],
                "K": v["K_undistorted_native"],
                "K_undistorted_native": v["K_undistorted_native"],
                "W": v["W"],
                "width": v.get("width"),
                "height": v.get("height"),
            }
        )
    atomic_write_json(
        out / "manifest.json",
        {
            "sequence": man.get("sequence"),
            "identity": info.get("identity"),
            "views": static_views,
            "coordinate_space": UND_SPACE,
            "world_normalization": man.get("world_normalization"),
            "scope": "Input-format package for frozen fit_proportions.main; real observations, shared gauge.",
        },
    )
    return out


def fit_person_proportions(args: argparse.Namespace) -> int:
    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    static = build_static_package(Path(info["out"]), out / "static")
    sub = argparse.Namespace(static=str(static), out=str(out), reference=args.reference)
    return run_proportions(sub)


def apply_person_proportions(args: argparse.Namespace) -> int:
    """Re-export that person's avatar with fitted beta/scale from fit_proportions."""
    import torch

    identity_path = Path(args.identity_json).resolve()
    info = json.loads(identity_path.read_text(encoding="utf-8"))
    proportions_path = Path(args.proportions).resolve()
    prop = torch.load(proportions_path, map_location="cpu", weights_only=True)
    beta = torch.as_tensor(prop["beta"], dtype=torch.float32).reshape(1, 10)
    scale = torch.tensor(float(prop["scale"]), dtype=torch.float32)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    configure_runtime_paths()
    # Reuse the real LHM loader and the real reconstruct.export. The cache is
    # local; no original lhmpp/runs directory is used.
    sys.path.insert(0, str(THIS_DIR))
    import build_identity as bi

    model = bi.load_model(Path(args.cache).resolve() / "model_load")
    sys.path.insert(0, str(PROJECT_ROOT / "work" / "mjj"))
    rec_spec = importlib.util.spec_from_file_location(
        "frozen_reconstruct", PROJECT_ROOT / "work" / "mjj" / "reconstruct.py"
    )
    if rec_spec is None or rec_spec.loader is None:
        raise RuntimeError("cannot import reconstruct.py")
    rec = importlib.util.module_from_spec(rec_spec)
    rec_spec.loader.exec_module(rec)
    rec.OUT = out
    poses = {
        "root_pose": torch.zeros(1, 3),
        "body_pose": torch.zeros(1, 21, 3),
        "left_hand_pose": torch.zeros(1, 15, 3),
        "right_hand_pose": torch.zeros(1, 15, 3),
        "translation_world": torch.zeros(1, 3),
        "scale": scale,
        "betas": beta,
        "frames": torch.tensor([0]),
    }
    rec.export(
        model,
        poses,
        destination=".",
        attributes_file=str(Path(info["gaussian_attributes"]).resolve()),
        latent_file=str(Path(info["latent"]).resolve()),
    )
    atomic_write_json(
        out / "proportions_applied.json",
        {
            "source_identity": str(identity_path),
            "source_proportions": str(proportions_path),
            "beta": beta.flatten().tolist(),
            "scale": float(scale),
            "avatar": str(out / "avatar.pt"),
            "note": "Re-exported with original fit_proportions beta/scale; root/body neutral initialization, no FCS pose copied.",
        },
    )
    print(f"[adapter] proportioned avatar: {out / 'avatar.pt'}")
    return 0


def run_smooth(args: argparse.Namespace) -> int:
    person_path = Path(args.person).resolve()
    if person_path.is_dir():
        person_path = person_path / "person.json"
    info = json.loads(person_path.read_text(encoding="utf-8"))
    avatar = Path(args.avatar).resolve()
    motion = Path(args.motion).resolve()
    if not avatar.is_file() or not motion.is_file():
        raise FileNotFoundError("smooth_6d needs both independent avatar.pt and motion.pt")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(args.stage).resolve() if args.stage else out.parent / f"{out.name}_smooth_stage"
    key_src = Path(info["keypoints"])
    man_src = Path(info["manifest"])
    key_dst = stage / "assets" / "keypoints_all300_v1"
    man_dst = stage / "assets" / "helmet_native000_299_v1"
    key_dst.mkdir(parents=True, exist_ok=True)
    man_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(man_src, man_dst / "manifest.json")
    for p in sorted(key_src.glob("*.npz")):
        shutil.copy2(p, key_dst / p.name)
    ref = Path(args.reference).resolve()
    root = PROJECT_ROOT
    configure_runtime_paths()
    mod, provenance = load_reference_patched(ref, SMOOTH_6D_REPLACEMENTS)
    # smooth_6d hard-codes PROJECT/assets/helmet... and keypoints_all300;
    # stage only those inputs in its own directory, leaving originals untouched.
    mod.PROJECT = stage
    # smooth_6d appends "avatar.pt"/"motion.pt"; pass directories, not files.
    avatar_arg = str(avatar.parent if avatar.name == "avatar.pt" else avatar)
    motion_arg = str(motion.parent if motion.name == "motion.pt" else motion)
    sys.argv = [
        str(ref),
        "--avatar",
        avatar_arg,
        "--motion",
        motion_arg,
        "--out",
        str(out),
        "--iters",
        str(args.iters),
        "--lam",
        str(args.lam),
        "--w-pos",
        str(args.w_pos),
    ]
    print(f"[adapter] calling frozen smooth_6d.main for {info['identity']}")
    mod.main()
    provenance["head_hands_feet_retention"] = (
        "Not added as a new objective here: this adapter applies only the verified "
        "FK normalizer/dir fixes. Full observation retention requires the separate "
        "smooth_observations path; no frozen joint is silently asserted."
    )
    atomic_write_json(out / "reference_provenance.json", provenance)
    print(f"[adapter] smoothed motion written: {out / 'motion.pt'}")
    return 0


def _resolve_convention_file(value: str, default_name: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / default_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _safe_identity(value: Any) -> str:
    identity = str(value).strip()
    if (
        not identity
        or identity in {".", ".."}
        or Path(identity).name != identity
        or any(ch in identity for ch in '/\\:*?"<>|')
        or any(ord(ch) < 32 for ch in identity)
    ):
        raise ValueError(f"unsafe identity path segment: {identity!r}")
    return identity


def _run_reconstruct_stage(label: str, cmd: List[str], stage_dir: Path, outputs: Sequence[Path]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "fit.log"
    start_ns = time.time_ns()
    print(f"[reconstruct] {label}: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[reconstruct] {label}\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            bufsize=1,
            shell=False,
        )
        try:
            if proc.stdout is None:
                raise RuntimeError(f"{label} produced no log stream")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}; see {log_path}")
    for output in outputs:
        if not output.is_file():
            raise RuntimeError(f"{label} did not produce required output: {output}")
        if output.stat().st_mtime_ns < start_ns:
            raise RuntimeError(f"{label} left stale output instead of rewriting: {output}")
    print(f"[reconstruct] {label}: ok", flush=True)


def reconstruct(args: argparse.Namespace) -> int:
    """Run body/hand -> face -> final MAP for one identity, in place."""
    work_root = THIS_DIR.parent.resolve()
    out = Path(args.out).resolve()
    if not out.is_relative_to(work_root):
        raise ValueError(f"--out must stay below {work_root}")
    if not out.is_dir():
        raise FileNotFoundError(f"--out shared stage root does not exist: {out}")

    person = _resolve_convention_file(args.person, "person.json")
    info = json.loads(person.read_text(encoding="utf-8"))
    identity = _safe_identity(info.get("identity"))
    avatar = _resolve_convention_file(args.avatar, "avatar.pt")
    initialization = _resolve_convention_file(args.initialization, "motion.pt")
    hand_observations = Path(args.hand_observations).resolve()
    if not hand_observations.is_file():
        raise FileNotFoundError(hand_observations)
    world_up = np.asarray(args.world_up, np.float64).reshape(-1)
    if world_up.shape != (3,) or not np.isfinite(world_up).all() or np.linalg.norm(world_up) <= 0:
        raise ValueError("--world-up must be three finite non-zero floats")
    if not str(args.world_up_source).strip():
        raise ValueError("--world-up-source must be non-empty")

    body_dir = out / identity
    face_dir = out / f"face_{identity}"
    map_dir = out / f"map_{identity}"
    body_motion = body_dir / "motion.pt"
    face_motion = face_dir / "motion.pt"
    face_constraints = face_dir / "face_constraints.npz"
    map_motion = map_dir / "motion.pt"

    scripts = {
        "body": THIS_DIR / "run_quaternion_fit.py",
        "face": THIS_DIR / "run_quaternion_face.py",
        "map": THIS_DIR / "run_quaternion_map.py",
    }
    for label, script in scripts.items():
        if not script.is_file():
            raise FileNotFoundError(f"{label} script missing: {script}")

    up_args = [repr(float(x)) for x in world_up.tolist()]
    body_cmd = [
        sys.executable, str(scripts["body"]),
        "--person", str(person),
        "--avatar", str(avatar),
        "--initialization", str(initialization),
        "--hand-observations", str(hand_observations),
        "--world-up", *up_args,
        "--world-up-source", str(args.world_up_source),
        "--out", str(body_dir),
    ]
    face_cmd = [
        sys.executable, str(scripts["face"]),
        "--person", str(person),
        "--avatar", str(avatar),
        "--motion", str(body_motion),
        "--out", str(face_dir),
    ]
    map_cmd = [
        sys.executable, str(scripts["map"]),
        "--person", str(person),
        "--avatar", str(avatar),
        "--motion", str(face_motion),
        "--initialization", str(initialization),
        "--face-constraints", str(face_constraints),
        "--hand-observations", str(hand_observations),
        "--out", str(map_dir),
    ]

    _run_reconstruct_stage("body/hand", body_cmd, body_dir, [body_motion])
    _run_reconstruct_stage("face", face_cmd, face_dir, [face_motion, face_constraints])
    _run_reconstruct_stage("final-map", map_cmd, map_dir, [map_motion])
    print(f"[reconstruct] {identity}: {map_motion}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    rc = sub.add_parser("reconstruct", help="run the current dedicated body/hand -> face -> final-MAP q chain")
    rc.add_argument("--person", required=True, help="person directory or person.json")
    rc.add_argument("--avatar", required=True, help="that identity's avatar.pt or directory")
    rc.add_argument("--initialization", required=True, help="that identity's original initialization motion.pt or directory")
    rc.add_argument("--hand-observations", required=True, help="observations.npz from instance_inputs.py observe-hands")
    rc.add_argument("--out", required=True, help="existing shared stage root, e.g. data/quaternion_fit")
    rc.add_argument("--world-up", type=float, nargs=3, required=True)
    rc.add_argument("--world-up-source", required=True)
    rc.set_defaults(func=reconstruct)

    b = sub.add_parser("build", help="build one person's observations/manifest in shared gauge")
    b.add_argument("--scene", default=str(DEFAULT_SCENE))
    b.add_argument("--instances", default=str(DEFAULT_INSTANCES))
    b.add_argument("--identity", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    b.set_defaults(func=build)

    r = sub.add_parser("run-motion", help="historical frozen fit_motion.py; not the current q chain")
    r.add_argument("--person", required=True, help="person directory or person.json")
    r.add_argument("--avatar", required=True, help="that person's independent avatar.pt")
    r.add_argument("--out", required=True)
    r.add_argument("--reference", default=str(DEFAULT_REF))
    r.add_argument("--initialization", default=None, help="motion.pt from initialize_pose.py")
    r.add_argument("--steps", type=int, nargs="+", default=[400, 200, 200])
    r.add_argument("--temporal", type=float, default=0.5)
    r.add_argument("--init-mode", choices=("require", "neutral"), default="require")
    r.add_argument("--world-up", type=float, nargs=3, default=None, help="audited same-gauge foot-plane up vector")
    r.add_argument("--world-up-source", default=None)
    r.set_defaults(func=run_motion)

    q = sub.add_parser("run-proportions", help="call the frozen fit_proportions.py")
    q.add_argument("--static", required=True, help="one-person static package (manifest.json + keypoints.npz)")
    q.add_argument("--out", required=True)
    q.add_argument("--reference", default=str(THIS_DIR / "reference" / "fit_proportions.py"))
    q.set_defaults(func=run_proportions)

    qp = sub.add_parser("fit-proportions", help="build static input from person and call frozen fit_proportions")
    qp.add_argument("--person", required=True)
    qp.add_argument("--out", required=True)
    qp.add_argument("--reference", default=str(THIS_DIR / "reference" / "fit_proportions.py"))
    qp.set_defaults(func=fit_person_proportions)

    qa = sub.add_parser("apply-proportions", help="re-export the person's LHM avatar with fitted beta/scale")
    qa.add_argument("--identity-json", required=True)
    qa.add_argument("--proportions", required=True)
    qa.add_argument("--out", required=True)
    qa.add_argument("--cache", default=str(THIS_DIR / "cache" / "partner_lhm"))
    qa.set_defaults(func=apply_person_proportions)

    s = sub.add_parser("run-smooth", help="historical frozen smooth_6d.py; not the current q chain")
    s.add_argument("--person", required=True)
    s.add_argument("--avatar", required=True)
    s.add_argument("--motion", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--stage", default=None)
    s.add_argument("--reference", default=str(THIS_DIR / "reference" / "smooth_6d.py"))
    s.add_argument("--iters", type=int, default=600)
    s.add_argument("--lam", type=float, default=0.02)
    s.add_argument("--w-pos", type=float, default=2.0)
    s.set_defaults(func=run_smooth)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
