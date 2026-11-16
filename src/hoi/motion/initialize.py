#!/usr/bin/env python
"""Initialize body motion from multiview NLF image predictions.

Per-camera estimates are fused in the supplied shared camera gauge. Requested
source indices are retained; unavailable evidence remains explicitly missing.
World translation uses the same body landmarks and 3D Huber scale as fitting.
Previously reconstructed motion is not an implicit initialization input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
from hoi.runtime import runtime_settings
from hoi.motion.landmarks import BODY, WORLD_HUBER_SCALE

LHM = Path(runtime_settings()["lhm_root"])
DEFAULT_MODEL = PROJECT_ROOT / "runtime" / "weights" / "nlf.torchscript"
SMPLX_DEFAULT = LHM / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"


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


def atomic_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_frames(tokens: Sequence[str]) -> List[int]:
    """Comma/list/ranges; ranges are inclusive: 0:59:30 -> 0,30,59."""
    out: List[int] = []
    for token in tokens:
        for part in str(token).replace(" ", "").split(","):
            if not part:
                continue
            if ":" in part:
                bits = [int(x) for x in part.split(":")]
                if len(bits) == 2:
                    start, end, step = bits[0], bits[1], 1
                elif len(bits) == 3:
                    start, end, step = bits
                else:
                    raise ValueError(f"bad frame range {part!r}")
                if step <= 0:
                    raise ValueError("frame step must be positive")
                out.extend(range(start, end + 1, step))
            else:
                out.append(int(part))
    return sorted(set(out))


def load_views(person: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    man = json.loads(Path(person["manifest"]).read_text(encoding="utf-8"))
    return {str(v["serial"]): v for v in man["views"]}


def load_seed_flags(segment: Dict[str, Any]) -> Dict[Tuple[str, str], bool]:
    """Read the original identity_seeds keypoints_enabled flags, if present."""
    raw = (segment.get("seeds") or {}).get("path")
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[Tuple[str, str], bool] = {}
    for view in payload.get("views") or []:
        serial = str(view.get("serial"))
        for p in view.get("persons") or []:
            actor = str(p.get("id"))
            if actor and actor != "None":
                out[(serial, actor)] = bool(p.get("keypoints_enabled", True))
    return out


def load_boxes(segment_index: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    raw = (segment_index.get("pipeline_inputs") or {}).get("boxes_undistorted")
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for view in payload.get("views") or []:
        serial = str(view.get("serial"))
        for fr in view.get("frames") or []:
            frame = int(fr["frame"])
            for det in fr.get("detections") or []:
                actor = str(det.get("actor", ""))
                box = det.get("box")
                if actor and box is not None:
                    out.setdefault(serial, {}).setdefault(frame, {})[actor] = np.asarray(
                        box, np.float64
                    )
    return out


def mask_bbox(cam: Dict[str, Any], source_frame: int, actor: str) -> Tuple[Any, str]:
    path = Path(cam.get("mask_npz", ""))
    if not path.is_file():
        return None, "mask_npz_missing"
    with np.load(path, allow_pickle=False) as z:
        actors = [str(x) for x in z["actors"].tolist()]
        if actor not in actors:
            return None, "actor_not_in_mask_file"
        ai = actors.index(actor)
        source = [int(x) for x in z["source_frames"].tolist()]
        if source_frame not in source:
            return None, "source_frame_not_in_mask_file"
        fi = source.index(source_frame)
        if not bool(z["present"][fi, ai]):
            return None, "sam2_mask_absent"
        b = z["bboxes"][fi, ai]
        if not np.isfinite(b).all():
            return None, "sam2_bbox_invalid"
        return np.asarray(b, np.float64), "sam2_mask_bbox"


def read_rgb(path: Path) -> np.ndarray:
    from hoi.image_io import read_image

    import cv2

    bgr = read_image(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_model(path: Path, device: str = "cuda") -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass
    with path.open("rb") as source:
        model = torch.jit.load(source, map_location="cpu").to(device).eval()
    model.cano_all = {k: v.to(device) for k, v in model.cano_all.items()}
    model.weights = {
        k: {name: value.to(device) for name, value in values.items()}
        for k, values in model.weights.items()
    }
    return model


@lru_cache(maxsize=1)
def _fk_model_arrays(path: str):
    """Keep only immutable model data; predictions and shape remain per call."""
    with np.load(path, allow_pickle=True) as model:
        template = np.array(model["v_template"], dtype=np.float64, copy=True)
        shapedirs = np.array(model["shapedirs"][:, :, :10], dtype=np.float64, copy=True)
        regressor = model["J_regressor"]
    for array in (template, shapedirs, regressor):
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return template, shapedirs, regressor


def fk_check(
    pose: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    joints3d: np.ndarray,
    rest_parents: torch.Tensor,
    smplx_path: Path,
) -> float:
    """Original NLF parameter FK check in the vendored SMPL-X, camera gauge."""
    from pytorch3d.transforms import axis_angle_to_matrix
    from smplx.lbs import batch_rigid_transform

    base, shapedirs, regressor = _fk_model_arrays(str(Path(smplx_path).resolve()))
    template = base + np.einsum("vcl,l->vc", shapedirs, betas.reshape(10))
    J = regressor @ template
    J = J.toarray() if hasattr(J, "toarray") else np.asarray(J, np.float64)
    pose_t = torch.tensor(pose, dtype=torch.float32).reshape(1, 55, 3)
    J_t = torch.tensor(J, dtype=torch.float32)[None]
    j, _ = batch_rigid_transform(axis_angle_to_matrix(pose_t), J_t, rest_parents.to(torch.long))
    expected = j * 1000.0 + torch.tensor(trans, dtype=torch.float32).reshape(1, 1, 3) * 1000.0
    err = float((expected - torch.tensor(joints3d, dtype=torch.float32)[None]).abs().max())
    return err


def consensus(poses: np.ndarray, uncert: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    n, j, _ = poses.shape
    rot = np.zeros((j, 3), np.float64)
    disagreement = np.zeros((j, n), np.float64)
    for joint in range(j):
        R = Rotation.from_rotvec(poses[:, joint])
        weights = np.maximum(uncert[:, joint], 1e-6) ** -1.5
        estimate = R.mean(weights=weights)
        for _ in range(20):
            distance = (estimate.inv() * R).magnitude()
            estimate = R.mean(weights=weights / np.maximum(distance, 1e-6))
        rot[joint] = estimate.as_rotvec()
        disagreement[joint] = np.rad2deg((estimate.inv() * R).magnitude())
    return rot, disagreement


def estimate_translation(xyz, valid, joints, scale):
    """Fit translation for a fixed pose and scale with the body 3D Huber term."""
    coco_ids = np.asarray(list(BODY), dtype=np.int64)
    joint_ids = np.asarray(list(BODY.values()), dtype=np.int64)
    observed = np.asarray(xyz, dtype=np.float64)[coco_ids]
    predicted = np.asarray(joints, dtype=np.float64)[joint_ids] * float(scale)
    usable = (
        np.asarray(valid, dtype=bool)[coco_ids]
        & np.isfinite(observed).all(axis=1)
        & np.isfinite(predicted).all(axis=1)
    )
    if not usable.any():
        raise ValueError("No valid BODY observations for translation initialization")
    offsets = observed[usable] - predicted[usable]
    translation = offsets.mean(axis=0)
    tolerance = np.sqrt(np.finfo(np.float64).eps)
    for _ in range(128):
        residual = np.linalg.norm(translation[None] - offsets, axis=1)
        weights = np.ones_like(residual)
        outside = residual > WORLD_HUBER_SCALE
        weights[outside] = WORLD_HUBER_SCALE / residual[outside]
        updated = (weights[:, None] * offsets).sum(axis=0) / weights.sum()
        converged = np.linalg.norm(updated - translation) <= tolerance * (
            1.0 + np.linalg.norm(translation)
        )
        translation = updated
        if converged:
            break
    return translation, coco_ids[usable]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--segment-index", required=True)
    ap.add_argument(
        "--frames", nargs="+", required=True, help="source frames/ranges; ranges inclusive"
    )
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-up", type=float, nargs=3, required=True)
    ap.add_argument("--world-up-source", required=True)
    ap.add_argument("--smplx", default=str(SMPLX_DEFAULT))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be a finite positive number")
    fps_fraction = Fraction(str(args.fps))
    fps_rational = f"{fps_fraction.numerator}/{fps_fraction.denominator}"

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    person_path = Path(args.person).resolve()
    if person_path.is_dir():
        person_path = person_path / "person.json"
    info = json.loads(person_path.read_text(encoding="utf-8"))
    identity = str(info.get("identity"))
    segment_path = Path(args.segment_index).resolve()
    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    if str(segment.get("coordinate_space")) not in {
        "undistorted_pixels",
        "undistorted_native_2048x1536",
    }:
        raise ValueError("segment index must use undistorted pixel coordinates")
    if identity not in (segment.get("actors") or {}):
        raise ValueError(f"actor {identity} missing from segment index")
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    if not avatar.is_file():
        raise FileNotFoundError(avatar)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames_requested = parse_frames(args.frames)
    if not frames_requested:
        raise ValueError("no frames requested")

    from hoi.runtime import configure_runtime

    configure_runtime()
    torch.set_num_threads(4)
    from smplx.lbs import batch_rigid_transform
    from pytorch3d.transforms import axis_angle_to_matrix
    from hoi.motion.rotations import load_motion_quaternions, export_motion_with_quaternions

    model_path = Path(args.model).resolve()
    model = None  # Load only when an uncached image really needs inference.
    model_sha = sha256_file(model_path)
    smplx_path = Path(args.smplx).resolve()
    av = torch.load(avatar, map_location="cpu", weights_only=False)
    rest = av["rig"]["rest_joints"][0].double()
    parents = av["rig"]["parents"].to(torch.long)
    rest_parents = parents
    scale = (
        float(av["poses"]["scale"].reshape(-1)[0])
        if torch.is_tensor(av["poses"]["scale"])
        else float(av["poses"]["scale"])
    )
    betas = torch.as_tensor(av["poses"]["betas"], dtype=torch.float32).reshape(1, 10)
    views = load_views(info)
    boxes = load_boxes(segment)
    seed_flags = load_seed_flags(segment)
    obs = np.load(Path(info["observations"]), allow_pickle=False)
    obs_xyz = np.asarray(obs["xyz"], np.float64)
    obs_valid = np.asarray(obs["valid"], bool) & np.isfinite(obs_xyz).all(-1)
    person_frames = [int(x) for x in info["frames"]]
    person_frame_index = {f: i for i, f in enumerate(person_frames)}
    apose = av.get("poses", {})
    if "left_hand_pose" in apose and "right_hand_pose" in apose:
        lh = torch.as_tensor(apose["left_hand_pose"], dtype=torch.float32).reshape(-1, 15, 3)[0]
        rh = torch.as_tensor(apose["right_hand_pose"], dtype=torch.float32).reshape(-1, 15, 3)[0]
    else:
        lh, rh = torch.zeros(15, 3), torch.zeros(15, 3)
    up = np.asarray(args.world_up, np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    up_t = torch.tensor(up, dtype=torch.float32, device=args.device)

    cams_by_serial = {str(c["serial"]): c for c in segment.get("cameras") or []}
    raw_records: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    computed_frames: List[int] = []
    motions: List[Dict[str, Any]] = []
    anchor_records: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for frame in frames_requested:
            preds = []
            frame_unavailable = []
            anchor = {
                "source_frame": int(frame),
                "anchor_type": None,
                "body_landmarks": [],
                "used_for_translation": False,
            }
            for serial, view in views.items():
                cam = cams_by_serial.get(serial)
                if cam is None:
                    frame_unavailable.append(
                        {"serial": serial, "reason": "camera_not_in_segment_index"}
                    )
                    continue
                cam_frame = next(
                    (f for f in cam.get("frames") or [] if int(f["source_frame"]) == frame), None
                )
                if cam_frame is None:
                    frame_unavailable.append(
                        {"serial": serial, "reason": "source_frame_not_in_segment"}
                    )
                    continue
                seed_meta = cam.get("seed_input") or {}
                kp_enabled = seed_flags.get(
                    (serial, identity),
                    (seed_meta.get("keypoints_enabled") or {}).get(identity, True),
                )
                if not bool(kp_enabled):
                    frame_unavailable.append(
                        {"serial": serial, "reason": "keypoints_enabled_false"}
                    )
                    continue
                if bool(cam.get("head_only", False)) or bool(seed_meta.get("head_only", False)):
                    frame_unavailable.append({"serial": serial, "reason": "head_only_crop"})
                    continue
                box = boxes.get(serial, {}).get(frame, {}).get(identity)
                box_reason = "segment_boxes"
                if box is None:
                    box, box_reason = mask_bbox(cam, frame, identity)
                if box is None:
                    frame_unavailable.append({"serial": serial, "reason": box_reason})
                    continue
                image_path = Path(cam_frame["jpeg"])
                expected_rotation_only = np.asarray(view["W"], np.float64).copy()
                expected_rotation_only[:3, 3] = 0.0
                raw_path = out / "raw" / f"{frame:04d}" / f"{serial}.pt"
                if raw_path.exists():
                    cached = torch.load(raw_path, map_location="cpu", weights_only=True)
                    same = (
                        cached.get("identity") == identity
                        and cached.get("source_frame") == frame
                        and cached.get("model_sha256") == model_sha
                        and cached.get("image_sha256") == sha256_file(image_path)
                        and np.allclose(cached.get("box_xyxy"), box)
                        and np.allclose(cached.get("world_up"), up)
                        and np.allclose(cached.get("intrinsic"), view["K_undistorted_native"])
                        and cached.get("extrinsic_rotation_only") is not None
                        and np.allclose(cached["extrinsic_rotation_only"], expected_rotation_only)
                    )
                    if (
                        same
                        and cached.get("fk_check_max") is not None
                        and cached["fk_check_max"] < 0.01
                    ):
                        raw_records.append(cached)
                        preds.append(cached)
                        print(f"[nlf] cached frame {frame} {serial}", flush=True)
                        continue
                if model is None:
                    model = load_model(model_path, args.device)
                rgb = read_rgb(image_path)
                image_t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device)
                xyxy = np.asarray(box, np.float64)
                wh = np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]], np.float32)
                box_t = torch.tensor(wh[None], dtype=torch.float32, device=args.device)
                K = torch.tensor(
                    np.asarray(view["K_undistorted_native"], np.float32)[None], device=args.device
                )
                W = np.asarray(view["W"], np.float64).copy()
                W[:3, 3] = 0.0  # rotation-only camera; no NLF translation gauge
                W_t = torch.tensor(W, dtype=torch.float32, device=args.device)[None]
                pred = model.estimate_smpl_batched(
                    image_t,
                    [box_t],
                    intrinsic_matrix=K,
                    extrinsic_matrix=W_t,
                    world_up_vector=up_t,
                    model_name="smplx",
                    internal_batch_size=1,
                    num_aug=1,
                )
                item = {}
                for key in (
                    "pose",
                    "betas",
                    "trans",
                    "joints3d",
                    "joints2d",
                    "joint_uncertainties",
                ):
                    item[key] = pred[key][0].detach().cpu()
                pose = item["pose"].numpy().reshape(55, 3)
                beta = item["betas"].numpy().reshape(10)
                trans = item["trans"].numpy().reshape(3)
                joints3d = item["joints3d"].numpy().reshape(55, 3)
                raw = {
                    "schema": "nlf_scene_raw_prediction/v1",
                    "identity": identity,
                    "source_frame": int(frame),
                    "serial": serial,
                    "camera_index": cam.get("camera_index"),
                    "image": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "model_sha256": model_sha,
                    "calibration_sha256": (info.get("gauge") or {}).get("calibration_sha256"),
                    "box_xyxy": xyxy.tolist(),
                    "box_source": box_reason,
                    "world_up": up.tolist(),
                    "world_up_source": args.world_up_source,
                    "extrinsic_rotation_only": W.tolist(),
                    "intrinsic": K[0].detach().cpu().numpy().tolist(),
                    "pose": item["pose"],
                    "betas": item["betas"],
                    "trans": item["trans"],
                    "joints3d": item["joints3d"],
                    "joints2d": item["joints2d"],
                    "joint_uncertainties": item["joint_uncertainties"],
                    "fk_check_max": None,
                    "num_aug": 1,
                    "internal_batch_size": 1,
                }
                raw_path = out / "raw" / f"{frame:04d}" / f"{serial}.pt"
                atomic_torch_save(raw_path, raw)
                fk_err = fk_check(pose, beta, trans, joints3d, rest_parents, smplx_path)
                raw["fk_check_max"] = fk_err
                raw["raw_path"] = str(raw_path)
                atomic_torch_save(raw_path, raw)
                raw_records.append(raw)
                preds.append(raw)
                if fk_err >= 0.01:
                    raise RuntimeError(
                        f"NLF parameter FK mismatch frame {frame} {serial}: {fk_err} >= 0.01"
                    )
            if len(preds) < 2:
                unavailable.append(
                    {
                        "source_frame": int(frame),
                        "reason": f"only {len(preds)} visible NLF predictions; need >=2",
                        "cameras": frame_unavailable,
                    }
                )
                anchor["anchor_type"] = "unavailable_insufficient_nlf"
                anchor_records.append(anchor)
                continue
            poses = np.stack([p["pose"].numpy().reshape(55, 3) for p in preds])
            uncert = np.stack([p["joint_uncertainties"].numpy().reshape(55) for p in preds])
            rot, disagreement = consensus(poses, uncert)
            if frame not in person_frame_index:
                unavailable.append(
                    {"source_frame": int(frame), "reason": "frame absent from person observations"}
                )
                continue
            pi = person_frame_index[frame]
            aa = torch.cat(
                [
                    torch.tensor(rot[0], dtype=torch.float64).reshape(1, 3),
                    torch.tensor(rot[1:22], dtype=torch.float64),
                    torch.zeros(3, 3, dtype=torch.float64),
                    lh.double(),
                    rh.double(),
                ],
                0,
            )[None]
            j, _ = batch_rigid_transform(axis_angle_to_matrix(aa), rest[None], parents)
            try:
                translation, body_landmarks = estimate_translation(
                    obs_xyz[pi], obs_valid[pi], j[0].numpy(), scale
                )
            except ValueError as error:
                unavailable.append({"source_frame": int(frame), "reason": str(error)})
                anchor["anchor_type"] = "unavailable_body_anchor"
                anchor_records.append(anchor)
                continue
            trans = torch.tensor(translation, dtype=torch.float64)
            anchor.update(
                anchor_type="screen_body_huber_translation",
                body_landmarks=body_landmarks.tolist(),
                used_for_translation=True,
            )
            root = torch.tensor(rot[0], dtype=torch.float32)
            body = torch.tensor(rot[1:22], dtype=torch.float32)
            motion = {
                "frames": torch.tensor([frame], dtype=torch.long),
                "root_pose": root[None],
                "body_pose": body[None],
                "left_hand_pose": lh[None],
                "right_hand_pose": rh[None],
                "lhand_pose": lh[None],
                "rhand_pose": rh[None],
                "translation_world": trans.float()[None],
                "scale": torch.tensor(scale, dtype=torch.float32),
                "betas": betas,
                "fps": float(args.fps),
                "fps_rational": fps_rational,
            }
            motions.append(motion)
            computed_frames.append(frame)
            anchor_records.append(anchor)
            print(f"[nlf] frame {frame}: N={len(preds)} body_translation finite", flush=True)

    if not motions:
        raise RuntimeError("no requested source frame had a usable NLF consensus")
    motion = {
        k: torch.cat([m[k] for m in motions], 0)
        for k in (
            "root_pose",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
            "lhand_pose",
            "rhand_pose",
            "translation_world",
        )
    }
    motion.update(
        {
            "frames": torch.tensor(computed_frames, dtype=torch.long),
            "source_frames": torch.tensor(computed_frames, dtype=torch.long),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "betas": betas,
            "fps": float(args.fps),
            "fps_rational": fps_rational,
        }
    )
    # NLF's native model output is a boundary AA representation. Convert once
    # to the shared internal rotation contract without altering its pose.
    init_q, _ = load_motion_quaternions(motion)
    motion = export_motion_with_quaternions(motion, init_q)
    atomic_torch_save(out / "motion.pt", motion)
    provenance = {
        "schema": "nlf_scene_initialization",
        "identity": identity,
        "person": str(person_path),
        "avatar": str(avatar),
        "segment_index": str(segment_path),
        "model": str(model_path),
        "model_sha256": model_sha,
        "smplx": str(smplx_path),
        "smplx_sha256": sha256_file(smplx_path),
        "calibration_sha256": (info.get("gauge") or {}).get("calibration_sha256"),
        "world_up": up.tolist(),
        "world_up_source": args.world_up_source,
        "inventory_cameras": len(cams_by_serial),
        "visible_predictions_per_frame": {
            str(f): sum(1 for r in raw_records if int(r["source_frame"]) == f)
            for f in frames_requested
        },
        "requested_frames": frames_requested,
        "computed_frames": computed_frames,
        "unavailable": unavailable,
        "translation_anchor": {
            "type": "screen_body_huber_translation",
            "landmark_mapping": BODY,
            "huber_3d_world": WORLD_HUBER_SCALE,
            "validity": "observations.valid and finite xyz/FK; no new screening",
            "per_frame": anchor_records,
            "no_interpolation": True,
            "no_zero_fill": True,
        },
        "anchor_types": {int(r["source_frame"]): r["anchor_type"] for r in anchor_records},
        "num_aug": 1,
        "internal_batch_size": 1,
        "extrinsics": "rotation-only shared unit-gauge W; translation zero",
        "consensus": "existing NLF uncertainty^-1.5 + inverse-angle IRLS, no additional trim",
        "hands": "avatar hand pose/neutral; image hand observations fitted in the body stage",
        "beta_scale": "avatar-provided beta/scale; NLF beta/translation not adopted",
        "translation": (
            "IRLS translation from all valid shared BODY correspondences at fixed NLF pose "
            "and avatar scale, using the fitting term's radial Huber objective. "
            "No previously reconstructed motion is read."
        ),
        "face": "not fitted; NLF body/pose only",
        "interpolation": "none; motion.pt contains only requested computed source frames",
        "scope": "Initialization control only; not a quality result or adopted motion.",
        "raw_records": [r["raw_path"] for r in raw_records],
    }
    atomic_write_json(out / "provenance.json", provenance)
    print(f"[nlf] motion {out / 'motion.pt'} frames={computed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
