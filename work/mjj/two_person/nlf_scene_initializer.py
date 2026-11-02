#!/usr/bin/env python
"""Parameterized NLF0.3.2 image-body initialization control for the two-person scene.

This is a ROOT-CAUSE comparison for the current geometric skeleton seed.  It
uses only the recorded local NLF TorchScript estimator, the current person's
real undistorted images and SAM2 boxes, the FCS006 normalized camera gauge, and
the supplied same-gauge world-up.  It does not adopt MJJ/FCS006 motion, does not
add a detector, does not use tests/crops selected by appearance, and does not
feed NLF betas/translation into the avatar gauge.

Per-camera raw NLF outputs are cached before fusion.  Missing camera/instance,
keypoints_enabled=false, absent mask/box, or insufficient visible predictions
produce explicit unavailable records; no fake view, prediction, or translation
is created.  Only requested source frames with actual NLF consensus and valid
triangulated hips are written to motion.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
LHM = PROJECT_ROOT / "lhmpp"
DEFAULT_MODEL = PROJECT_ROOT / "work" / "mjj" / "runtime" / "nlf_l_multi_0.3.2.torchscript"
SMPLX_DEFAULT = LHM / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
ACTORS = ("fcs", "partner")


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


def configure_runtime_paths() -> None:
    for p in (LHM, LHM / "vendor", LHM / "deps", PROJECT_ROOT / "runtime" / "deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from runtime_imports import configure

        configure()
    except Exception as exc:
        raise RuntimeError(f"runtime_imports.configure failed: {exc}") from exc


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
            if actor in ACTORS:
                out[(serial, actor)] = bool(p.get("keypoints_enabled", True))
    return out


def load_boxes(segment_index: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    raw = ((segment_index.get("pipeline_inputs") or {}).get("boxes_undistorted"))
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
                if actor in ACTORS and box is not None:
                    out.setdefault(serial, {}).setdefault(frame, {})[actor] = np.asarray(box, np.float64)
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
    sys.path.insert(0, str(THIS_DIR.parent / "dual_track"))
    from image_io import read_image

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
    with path.open('rb') as source:
        model = torch.jit.load(source, map_location="cpu").to(device).eval()
    model.cano_all = {k: v.to(device) for k, v in model.cano_all.items()}
    model.weights = {
        k: {name: value.to(device) for name, value in values.items()}
        for k, values in model.weights.items()
    }
    return model


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

    d = np.load(smplx_path, allow_pickle=True)
    template = np.asarray(d["v_template"], np.float64) + np.einsum(
        "vcl,l->vc", np.asarray(d["shapedirs"], np.float64)[:, :, :10], betas.reshape(10)
    )
    J = d["J_regressor"] @ template
    J = J.toarray() if hasattr(J, "toarray") else np.asarray(J, np.float64)
    pose_t = torch.tensor(pose, dtype=torch.float32).reshape(1, 55, 3)
    J_t = torch.tensor(J, dtype=torch.float32)[None]
    j, _ = batch_rigid_transform(
        axis_angle_to_matrix(pose_t), J_t, rest_parents.to(torch.long)
    )
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--segment-index", default=str(THIS_DIR / "data" / "segment_instances" / "index.json"))
    ap.add_argument("--frames", nargs="+", default=["0", "30", "59"], help="source frames/ranges; ranges inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-up", type=float, nargs=3, required=True)
    ap.add_argument("--world-up-source", required=True)
    ap.add_argument("--source-motion", default=None, help="existing hand-pose source for requested source frames")
    ap.add_argument("--smplx", default=str(SMPLX_DEFAULT))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    person_path = Path(args.person).resolve()
    if person_path.is_dir():
        person_path = person_path / "person.json"
    info = json.loads(person_path.read_text(encoding="utf-8"))
    identity = str(info.get("identity"))
    if identity not in ACTORS:
        raise ValueError(f"person identity {identity!r} not in {ACTORS}")
    segment_path = Path(args.segment_index).resolve()
    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    if str(segment.get("coordinate_space")) != "undistorted_native_2048x1536":
        raise ValueError("segment index must be undistorted_native_2048x1536")
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

    configure_runtime_paths()
    torch.set_num_threads(4)
    from smplx.lbs import batch_rigid_transform
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, quaternion_to_matrix
    from rotation_contract import load_motion_quaternions, export_motion_with_quaternions

    model_path = Path(args.model).resolve()
    model = None  # Load only when an uncached image really needs inference.
    model_sha = sha256_file(model_path)
    smplx_path = Path(args.smplx).resolve()
    smplx = np.load(smplx_path, allow_pickle=True)
    rest_parents = None
    av = torch.load(avatar, map_location="cpu", weights_only=False)
    rest = av["rig"]["rest_joints"][0].double()
    parents = av["rig"]["parents"].to(torch.long)
    rest_parents = parents
    scale = float(av["poses"]["scale"].reshape(-1)[0]) if torch.is_tensor(av["poses"]["scale"]) else float(av["poses"]["scale"])
    betas = torch.as_tensor(av["poses"]["betas"], dtype=torch.float32).reshape(1, 10)
    views = load_views(info)
    boxes = load_boxes(segment)
    seed_flags = load_seed_flags(segment)
    obs = np.load(Path(info["observations"]), allow_pickle=False)
    obs_xyz = np.asarray(obs["xyz"], np.float64)
    person_frames = [int(x) for x in info["frames"]]
    person_frame_index = {f: i for i, f in enumerate(person_frames)}
    source_motion = None
    source_motion_path = None
    source_motion_sha = None
    source_q_info = None
    source_frames: List[int] = []
    source_index: Dict[int, int] = {}
    source_hips_world = None
    source_scale = None
    source_betas = None
    source_scale_match = None
    source_beta_match = None
    source_joints55_ignored = True
    if args.source_motion:
        source_motion_path = Path(args.source_motion).resolve()
        source_motion = torch.load(source_motion_path, map_location="cpu", weights_only=False)
        source_motion_sha = sha256_file(source_motion_path)
        source_q, source_q_info = load_motion_quaternions(source_motion, verify_legacy=True)
        source_q = source_q.double()
        sframes_raw = source_motion["frames"]
        source_frames = [int(x) for x in (sframes_raw.tolist() if hasattr(sframes_raw, "tolist") else sframes_raw)]
        if source_frames and len(source_frames) != source_q.shape[0]:
            raise ValueError("source_motion frames/q T mismatch")
        source_index = {f: i for i, f in enumerate(source_frames)}
        if "scale" not in source_motion or "betas" not in source_motion:
            raise ValueError("source_motion must provide scale and betas for the explicit prior")
        source_scale = float(torch.as_tensor(source_motion["scale"]).reshape(-1)[0])
        source_betas = torch.as_tensor(source_motion["betas"], dtype=torch.float32).reshape(-1)
        source_scale_match = bool(abs(source_scale - scale) <= 1e-6)
        source_beta_match = bool(
            source_betas.shape == betas.reshape(-1).shape
            and torch.allclose(source_betas, betas.reshape(-1), rtol=1e-5, atol=1e-6)
        )
        if not source_scale_match or not source_beta_match:
            raise ValueError(
                "source_motion scale/beta are inconsistent with the current avatar: "
                f"scale {source_scale} vs {scale}; beta_match={source_beta_match}"
            )
        source_trans = torch.as_tensor(source_motion["translation_world"], dtype=torch.float64).reshape(-1, 3)
        if len(source_trans) != source_q.shape[0]:
            raise ValueError("source_motion translation/q T mismatch")
        with torch.inference_mode():
            source_joints, _ = batch_rigid_transform(
                quaternion_to_matrix(source_q),
                rest[None].expand(source_q.shape[0], -1, -1),
                parents,
            )
            source_hips_world = (
                source_joints * source_scale + source_trans[:, None]
            )[:, [1, 2]].mean(dim=1)
    up = np.asarray(args.world_up, np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    up_t = torch.tensor(up, dtype=torch.float32, device=args.device)

    cams_by_serial = {str(c["serial"]): c for c in segment.get("cameras") or []}
    raw_records: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    computed_frames: List[int] = []
    motions: List[Dict[str, Any]] = []
    anchor_records: List[Dict[str, Any]] = []
    screen_hip_missing_frames: List[int] = []
    screen_anchor_frames: List[int] = []
    source_prior_frames: List[int] = []

    with torch.inference_mode():
        for frame in frames_requested:
            preds = []
            frame_unavailable = []
            pi0 = person_frame_index.get(frame)
            screen_hips0 = obs_xyz[pi0, [11, 12]] if pi0 is not None else None
            screen_ok0 = bool(screen_hips0 is not None and np.isfinite(screen_hips0).all())
            if not screen_ok0:
                screen_hip_missing_frames.append(int(frame))
            anchor = {
                "source_frame": int(frame),
                "screen_hip_finite": screen_ok0,
                "screen_hip_status": "screen_triangulated_hip_pair" if screen_ok0 else "screen_hip_pair_missing",
                "source_motion_frame": int(frame) if source_motion is not None and int(frame) in source_index else None,
                "source_motion_hip_fk_finite": None,
                "anchor_type": None,
                "used_for_translation": False,
            }
            for serial, view in views.items():
                cam = cams_by_serial.get(serial)
                if cam is None:
                    frame_unavailable.append({"serial": serial, "reason": "camera_not_in_segment_index"})
                    continue
                cam_frame = next(
                    (f for f in cam.get("frames") or [] if int(f["source_frame"]) == frame), None
                )
                if cam_frame is None:
                    frame_unavailable.append({"serial": serial, "reason": "source_frame_not_in_segment"})
                    continue
                seed_meta = (cam.get("seed_input") or {})
                kp_enabled = seed_flags.get(
                    (serial, identity),
                    (seed_meta.get("keypoints_enabled") or {}).get(identity, True),
                )
                if not bool(kp_enabled):
                    frame_unavailable.append({"serial": serial, "reason": "keypoints_enabled_false"})
                    continue
                if bool(cam.get("head_only", False)) or bool(seed_meta.get("head_only", False)):
                    frame_unavailable.append({"serial": serial, "reason": "head_only_crop"})
                    continue
                box = (boxes.get(serial, {}).get(frame, {}).get(identity))
                box_reason = "segment_boxes"
                if box is None:
                    box, box_reason = mask_bbox(cam, frame, identity)
                if box is None:
                    frame_unavailable.append({"serial": serial, "reason": box_reason})
                    continue
                image_path = Path(cam_frame["jpeg"])
                expected_rotation_only = np.asarray(view['W'], np.float64).copy()
                expected_rotation_only[:3, 3] = 0.0
                raw_path = out / "raw" / f"{frame:04d}" / f"{serial}.pt"
                if raw_path.exists():
                    cached = torch.load(raw_path, map_location='cpu', weights_only=True)
                    same = (cached.get('identity') == identity
                            and cached.get('source_frame') == frame
                            and cached.get('model_sha256') == model_sha
                            and cached.get('image_sha256') == sha256_file(image_path)
                            and np.allclose(cached.get('box_xyxy'), box)
                            and np.allclose(cached.get('world_up'), up)
                            and np.allclose(cached.get('intrinsic'), view['K_undistorted_native'])
                            and cached.get('extrinsic_rotation_only') is not None
                            and np.allclose(cached['extrinsic_rotation_only'], expected_rotation_only))
                    if same and cached.get('fk_check_max') is not None and cached['fk_check_max'] < .01:
                        raw_records.append(cached); preds.append(cached)
                        print(f'[nlf] cached frame {frame} {serial}', flush=True)
                        continue
                if model is None:
                    model = load_model(model_path, args.device)
                rgb = read_rgb(image_path)
                image_t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device)
                xyxy = np.asarray(box, np.float64)
                wh = np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]], np.float32)
                box_t = torch.tensor(wh[None], dtype=torch.float32, device=args.device)
                K = torch.tensor(np.asarray(view["K_undistorted_native"], np.float32)[None], device=args.device)
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
                for key in ("pose", "betas", "trans", "joints3d", "joints2d", "joint_uncertainties"):
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
                unavailable.append({"source_frame": int(frame), "reason": "frame absent from person observations"})
                continue
            pi = person_frame_index[frame]
            hips = obs_xyz[pi, [11, 12]]
            screen_ok = bool(np.isfinite(hips).all())
            source_hip = None
            if source_motion is not None and int(frame) in source_index:
                si_prior = source_index[int(frame)]
                if source_hips_world is not None:
                    source_hip = source_hips_world[si_prior].detach().cpu().numpy()
                    anchor["source_motion_hip_fk_finite"] = bool(np.isfinite(source_hip).all())
            if screen_ok:
                hip_target = torch.tensor(hips.mean(axis=0), dtype=torch.float64)
                anchor["anchor_type"] = "screen_triangulated_hip_pair"
                screen_anchor_frames.append(int(frame))
            elif source_hip is not None and np.isfinite(source_hip).all():
                hip_target = torch.tensor(source_hip, dtype=torch.float64)
                anchor["anchor_type"] = "explicit_source_motion_posterior_fk"
                source_prior_frames.append(int(frame))
                print(
                    f"[nlf] frame {frame}: screen hips missing; using explicit source_motion posterior FK hip prior",
                    flush=True,
                )
            else:
                if source_motion is None:
                    reason = "screen hip 3D missing and no explicit --source-motion"
                elif int(frame) not in source_index:
                    reason = "screen hip 3D missing and explicit source_motion lacks this source frame"
                else:
                    reason = "screen hip 3D missing and explicit source_motion FK hip is non-finite"
                unavailable.append({"source_frame": int(frame), "reason": reason})
                anchor["anchor_type"] = "unavailable_hip_anchor"
                anchor_records.append(anchor)
                continue
            anchor["used_for_translation"] = True
            # Hands: existing source motion if supplied; otherwise avatar hands.
            if source_motion is not None:
                if int(frame) not in source_index:
                    raise ValueError(f"source-motion lacks requested frame {frame}")
                si = source_index[int(frame)]
                lh = source_motion.get("left_hand_pose", source_motion.get("lhand_pose"))
                rh = source_motion.get("right_hand_pose", source_motion.get("rhand_pose"))
                if lh is None or rh is None:
                    raise ValueError("source-motion lacks hand poses")
                lh = torch.as_tensor(lh[si], dtype=torch.float32).reshape(15, 3)
                rh = torch.as_tensor(rh[si], dtype=torch.float32).reshape(15, 3)
                hands_source = str(Path(args.source_motion).resolve())
            else:
                apose = av.get("poses", {})
                if "left_hand_pose" in apose:
                    lh = torch.as_tensor(apose["left_hand_pose"], dtype=torch.float32).reshape(-1, 15, 3)[0]
                    rh = torch.as_tensor(apose["right_hand_pose"], dtype=torch.float32).reshape(-1, 15, 3)[0]
                else:
                    lh = torch.zeros(15, 3)
                    rh = torch.zeros(15, 3)
                hands_source = "avatar_pose_or_neutral"
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
            j, _ = batch_rigid_transform(
                axis_angle_to_matrix(aa), rest[None], parents
            )
            trans = hip_target - j[0, [1, 2]].mean(0) * scale
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
                "fps": 30.0,
                "fps_rational": "30/1",
            }
            motions.append(motion)
            computed_frames.append(frame)
            anchor_records.append(anchor)
            print(f"[nlf] frame {frame}: N={len(preds)} hip_translation finite", flush=True)

    if not motions:
        raise RuntimeError("no requested source frame had a usable NLF consensus")
    motion = {k: torch.cat([m[k] for m in motions], 0) for k in ("root_pose", "body_pose", "left_hand_pose", "right_hand_pose", "lhand_pose", "rhand_pose", "translation_world")}
    motion.update(
        {
            "frames": torch.tensor(computed_frames, dtype=torch.long),
            "source_frames": torch.tensor(computed_frames, dtype=torch.long),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "betas": betas,
            "fps": 30.0,
            "fps_rational": "30/1",
        }
    )
    # NLF's native model output is a boundary AA representation. Convert once
    # to the shared internal rotation contract without altering its pose.
    init_q, _ = load_motion_quaternions(motion)
    motion = export_motion_with_quaternions(motion, init_q)
    atomic_torch_save(out / "motion.pt", motion)
    provenance = {
        "schema": "two_person_nlf_scene_initializer/v1",
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
        "inventory_cameras": 8,
        "visible_predictions_per_frame": {
            str(f): sum(1 for r in raw_records if int(r["source_frame"]) == f) for f in frames_requested
        },
        "requested_frames": frames_requested,
        "computed_frames": computed_frames,
        "unavailable": unavailable,
        "source_motion": {
            "path": str(source_motion_path) if source_motion_path else None,
            "sha256": source_motion_sha,
            "frames": source_frames,
            "q_validation": source_q_info,
            "scale": source_scale,
            "scale_matches_avatar": source_scale_match,
            "betas": None if source_betas is None else source_betas.tolist(),
            "betas_match_avatar": source_beta_match,
            "fk_inputs": "shared rotation_contract q -> matrix; own avatar rest/parents; source scale/translation",
            "joints55_world_stored_field_ignored": bool(source_joints55_ignored),
        },
        "translation_anchor": {
            "screen_type": "screen_triangulated_hip_pair",
            "missing_screen_type": "explicit_source_motion_posterior_fk",
            "missing_screen_is_prior_only": True,
            "missing_screen_not_new_measurement": True,
            "screen_hip_missing_frames": screen_hip_missing_frames,
            "screen_hip_missing_count": len(screen_hip_missing_frames),
            "screen_anchored_frames": screen_anchor_frames,
            "source_prior_frames": source_prior_frames,
            "source_prior_count": len(source_prior_frames),
            "per_frame": anchor_records,
            "no_interpolation": True,
            "no_zero_fill": True,
        },
        "anchor_types": {
            int(r["source_frame"]): r["anchor_type"] for r in anchor_records
        },
        "num_aug": 1,
        "internal_batch_size": 1,
        "extrinsics": "rotation-only FCS006 unit-gauge W; translation zero",
        "consensus": "existing NLF uncertainty^-1.5 + inverse-angle IRLS, no additional trim",
        "hands": "existing source-motion hands if supplied, else avatar hand pose/neutral; no NLF hands",
        "beta_scale": "avatar-provided beta/scale; NLF beta/translation not adopted",
        "translation": (
            "screen double-hip mean when both screen hips are finite; otherwise explicit "
            "same-person source-motion FK hip mean is used ONLY as initialization translation "
            "anchor/prior for the missing screen frame. It is not a new observation; the "
            "subsequent full-view fit still optimizes translation against all valid measurements."
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
