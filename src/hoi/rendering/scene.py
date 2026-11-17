"""Render any number of independently animated Gaussian identities in one scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def frame_list(poses: Dict[str, Any]) -> List[int]:
    frames = poses["frames"]
    if hasattr(frames, "tolist"):
        return [int(x) for x in frames.tolist()]
    return [int(x) for x in frames]


def compare_decoders(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    da, db = a.get("decoder") or {}, b.get("decoder") or {}
    if set(da) != set(db):
        raise RuntimeError(
            "avatar DPT decoders have different keys: "
            f"only_first={sorted(set(da) - set(db))[:8]} only_other={sorted(set(db) - set(da))[:8]}"
        )
    for key in da:
        ta, tb = da[key], db[key]
        if not torch.is_tensor(ta) or not torch.is_tensor(tb):
            if ta != tb:
                raise RuntimeError(f"decoder entry {key} differs and is not a tensor")
            continue
        if tuple(ta.shape) != tuple(tb.shape) or ta.dtype != tb.dtype:
            raise RuntimeError(f"decoder entry {key} shape/dtype differs")
        if not torch.equal(ta.cpu(), tb.cpu()):
            raise RuntimeError(
                f"decoder entry {key} differs numerically; a shared DPT decode is not valid"
            )
    if a.get("decoder_config") != b.get("decoder_config"):
        raise RuntimeError("avatar decoder_config differs; shared DPT decode is not valid")
    return {"keys": len(da), "all_tensors_equal": True, "config_equal": True}


def load_bundle(path: Path) -> Dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("rig", "poses", "features", "background_embedding", "decoder", "decoder_config"):
        if key not in bundle:
            raise KeyError(f"{path} missing bundle[{key!r}]")
    return bundle


def _normalize_hand_aliases(motion: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for short, long in (("lhand_pose", "left_hand_pose"), ("rhand_pose", "right_hand_pose")):
        has_s = motion.get(short) is not None
        has_l = motion.get(long) is not None
        if has_s and has_l:
            import torch

            if not torch.allclose(torch.as_tensor(motion[short]), torch.as_tensor(motion[long])):
                raise RuntimeError(f"motion hand alias mismatch: {short} vs {long}")
            out[short] = long
        elif has_s:
            motion[long] = motion[short]
            out[short] = long
        elif has_l:
            motion[short] = motion[long]
            out[short] = long
    return out


def load_motion(path: Optional[Path], fallback: Dict[str, Any], rig_dtype: Any) -> Dict[str, Any]:
    import torch

    if path is None:
        motion = dict(fallback)
        source_kind, source_path = "avatar_bundle_poses", None
    else:
        motion = torch.load(path, map_location="cpu", weights_only=False)
        source_kind, source_path = "motion_file", str(path)

    q_present = motion.get("local_quaternions_wxyz") is not None
    q_info: Optional[Dict[str, Any]] = None
    legacy_regenerated = False
    if q_present:
        from hoi.motion.rotations import load_motion_quaternions, quaternions_to_legacy

        q, q_info = load_motion_quaternions(motion, verify_legacy=True)
        legacy = quaternions_to_legacy(q)
        for key, value in legacy.items():
            motion[key] = value
        legacy_regenerated = True

    hand_aliases = _normalize_hand_aliases(motion)
    fields_adapted = []
    for key in (
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
        "lhand_pose",
        "rhand_pose",
        "translation_world",
        "scale",
        "joints55_world",
    ):
        if key in motion and motion[key] is not None:
            motion[key] = torch.as_tensor(motion[key]).to(dtype=rig_dtype)
            fields_adapted.append(key)
    q_shape = (
        None if not q_present else list(torch.as_tensor(motion["local_quaternions_wxyz"]).shape)
    )
    motion["_joint_render_motion_boundary"] = {
        "source_kind": source_kind,
        "source_path": source_path,
        "source_sha256": sha256_file(Path(source_path)) if source_path else None,
        "q_present": bool(q_present),
        "q_verified": bool(q_info is not None),
        "q_verified_fields": None if q_info is None else q_info.get("verified_fields"),
        "q_shape": q_shape,
        "legacy_regenerated_from_canonical_q": bool(legacy_regenerated),
        "conversion": (
            "load_motion_quaternions(verify_legacy=True) -> quaternions_to_legacy once; "
            "frozen avatar_rig receives consistent legacy AA"
            if q_present
            else "legacy passthrough with hand alias normalization only"
        ),
        "stale_aa_used": False,
        "hand_aliases": hand_aliases,
        "rig_dtype": str(rig_dtype),
        "fields_adapted_to_rig_dtype": fields_adapted,
        "q_field_preserved_for_report": bool(q_present),
        "not_converted": [
            "bundle",
            "features",
            "decoder",
            "camera",
            "K",
            "W",
            "fps",
            "frames",
            "image_size",
        ],
    }
    return motion


def to_device(obj: Any, device: str) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def world_to_unit(W: np.ndarray, center: np.ndarray, height: float) -> np.ndarray:
    W = np.asarray(W, np.float64)
    R, t = W[:3, :3], W[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = (R @ center + t) / height
    return out


def camera_from_scene(scene, serial, center, height, width, height_px, camera_space="raw"):
    import torch

    view = next((v for v in scene["views"] if str(v["serial"]) == serial), None)
    if view is None:
        raise KeyError(f"serial {serial} not in scene")
    cal = view.get("calibration", view)
    intrinsics = cal.get("K_undistorted")
    if intrinsics is None:
        intrinsics = cal["K_undistorted_native"]
    K = np.asarray(intrinsics, np.float64).copy()
    native_w = float(
        cal.get("native_width")
        or view.get("native_width")
        or view.get("decode", {}).get("width", 0)
    )
    native_h = float(
        cal.get("native_height")
        or view.get("native_height")
        or view.get("decode", {}).get("height", 0)
    )
    if native_w <= 0 or native_h <= 0:
        raise ValueError(f"camera {serial} has invalid native dimensions")
    K[0] *= width / native_w
    K[1] *= height_px / native_h
    W = np.asarray(cal["W"], np.float64)
    if camera_space == "raw":
        W = world_to_unit(W, center, height)
    elif camera_space != "normalized":
        raise ValueError("camera_space must be raw or normalized")
    return (
        torch.tensor(K, dtype=torch.float32, device="cuda"),
        torch.tensor(W, dtype=torch.float32, device="cuda"),
    )


def load_custom_cameras(path: Path) -> Tuple[Dict[str, Any], str, int, int]:
    """Adapt fixed or source-indexed camera matrices to the scene contract."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    width, height = int(payload["width"]), int(payload["height"])
    if width <= 0 or height <= 0:
        raise ValueError("custom camera dimensions must be positive")
    camera_space = payload.get("camera_space", "normalized")
    if camera_space not in {"raw", "normalized"}:
        raise ValueError("custom camera_space must be raw or normalized")
    views = []
    for source in payload.get("views") or []:
        identifier = source.get("serial") or source.get("label")
        if identifier is None:
            raise ValueError("custom camera needs a label or serial")
        serial = safe_segment(identifier)
        K = np.asarray(source["K"], np.float64)
        W = np.asarray(source["W"], np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"custom camera {serial}: K must be finite 3x3")
        view = {
            "serial": serial,
            "label": str(source.get("label", serial)),
            "calibration": {
                "K_undistorted": K,
                "W": W,
                "native_width": width,
                "native_height": height,
            },
        }
        if W.shape != (4, 4):
            if W.ndim != 3 or W.shape[1:] != (4, 4):
                raise ValueError(f"custom camera {serial}: W must be 4x4 or T,4,4")
            source_frames = source.get("frames", payload.get("frames"))
            if source_frames is None:
                raise ValueError(f"custom camera {serial}: trajectory requires source frames")
            source_frames = [int(frame) for frame in source_frames]
            if len(source_frames) != len(W) or len(source_frames) != len(set(source_frames)):
                raise ValueError(f"custom camera {serial}: frames must uniquely index every W")
            view["frames"] = source_frames
            view["frame_index"] = {frame: i for i, frame in enumerate(source_frames)}
        if not np.isfinite(W).all():
            raise ValueError(f"custom camera {serial}: W contains non-finite values")
        views.append(view)
    if not views:
        raise ValueError("custom camera file has no views")
    return {"views": views}, camera_space, width, height


def camera_at_frame(
    view: Dict[str, Any],
    frame: int,
    center: np.ndarray,
    height: float,
    width: int,
    height_px: int,
    camera_space: str,
) -> Tuple[Any, Any]:
    """Select a trajectory pose by source identity before the shared conversion."""
    index = view["frame_index"].get(frame)
    if index is None:
        raise ValueError(f"camera {view['serial']} has no pose for source frame {frame}")
    selected = dict(view)
    selected["calibration"] = dict(view["calibration"], W=view["calibration"]["W"][index])
    return camera_from_scene(
        {"views": [selected]}, str(view["serial"]), center, height, width, height_px, camera_space
    )


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def safe_segment(value: Any) -> str:
    value = str(value)
    if (
        not value
        or value in {".", ".."}
        or any(c in value for c in '/\\:*?"<>|')
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"unsafe output path segment: {value!r}")
    return value


def parse_frame_selection(requested: Sequence[str]) -> set[int]:
    selected: set[int] = set()
    for token in requested:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                bits = part.split(":")
                if len(bits) != 2:
                    raise ValueError("frame ranges use inclusive start:end")
                start, end = map(int, bits)
                if end < start:
                    raise ValueError("frame range end precedes start")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
    return selected


def select_frames(
    motions: Sequence[Dict[str, Any]], requested: Optional[Sequence[str]]
) -> Tuple[List[int], List[Dict[int, int]]]:
    indices = []
    common = None
    for motion in motions:
        frames = frame_list(motion)
        if len(frames) != len(set(frames)):
            raise ValueError("motion contains duplicate frame identifiers")
        indices.append({frame: i for i, frame in enumerate(frames)})
        common = set(frames) if common is None else common.intersection(frames)
    available = sorted(common or [])
    if requested is not None:
        selected = parse_frame_selection(requested)
        missing = selected - set(available)
        if missing:
            raise ValueError(
                f"requested frames missing from common motion frames: {sorted(missing)}"
            )
        available = [frame for frame in available if frame in selected]
    if not available:
        raise ValueError("no common motion frames selected")
    return available, indices


def select_render_frames(
    motions: Sequence[Dict[str, Any]],
    requested: Optional[Sequence[str]],
    trajectories: Sequence[Dict[str, Any]],
    pose_frame: Optional[int] = None,
) -> Tuple[List[int], List[Dict[int, int]]]:
    """Select output frames independently of an explicitly fixed source pose."""
    if pose_frame is None:
        return select_frames(motions, requested)
    if isinstance(pose_frame, bool) or not isinstance(pose_frame, (int, np.integer)):
        raise ValueError("pose_frame must be an integer source frame")
    _, indices = select_frames(motions, [str(pose_frame)])
    if trajectories:
        camera_frames = set.intersection(*(set(view["frame_index"]) for view in trajectories))
        selected = camera_frames if requested is None else parse_frame_selection(requested)
        missing = selected - camera_frames
        if missing:
            raise ValueError(
                f"requested output frames missing from camera trajectories: {sorted(missing)}"
            )
        frames = sorted(selected)
    elif requested is not None:
        frames = sorted(parse_frame_selection(requested))
    else:
        raise ValueError("a fixed pose needs explicit output frames or a camera trajectory")
    if not frames:
        raise ValueError("no output frames selected for the fixed pose")
    return frames, indices


def load_scene_config(
    config: Dict[str, Any], project_root: Path
) -> Tuple[Dict[str, Any], np.ndarray, float, Dict[str, Any]]:
    scene = config.get("scene")
    if scene is None:
        scene = json.loads(
            resolve_path(config["scene_path"], project_root).read_text(encoding="utf-8-sig")
        )
    if not isinstance(scene, dict) or not scene.get("views"):
        raise ValueError("scene must contain camera views")
    manifest_path = resolve_path(config["manifest_path"], project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    normalization = manifest.get("world_normalization")
    if not normalization:
        raise ValueError("capture manifest has no shared world_normalization")
    center = np.asarray(normalization["center_raw"], np.float64)
    height = float(normalization["body_height_raw_units"])
    if (
        center.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(height)
        or height <= 0
    ):
        raise ValueError("invalid shared world normalization")
    scene_sha = ((scene.get("calibration_selection") or {}).get("selected") or {}).get("sha256")
    manifest_sha = manifest.get("calibration_sha256")
    if scene_sha and manifest_sha and scene_sha != manifest_sha:
        raise ValueError("scene and capture manifest calibration SHA differ")
    gauge = {
        "manifest_path": str(manifest_path),
        "center_raw": center.tolist(),
        "body_height_raw_units": height,
        "scene_calibration_sha256": scene_sha,
        "manifest_calibration_sha256": manifest_sha,
        "calibration_match": bool(scene_sha and manifest_sha and scene_sha == manifest_sha),
    }
    return scene, center, height, gauge


def build_decoder(bundle: Dict[str, Any], device: str) -> Any:
    import torch
    from accelerate import init_empty_weights
    from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly

    with init_empty_weights(include_buffers=False):
        decoder = PatchDPT4DecoderOnly(**bundle["decoder_config"])
    decoder.transformer_block = torch.nn.Module()
    decoder.load_state_dict(bundle["decoder"], strict=True, assign=True)
    decoder.eval().to(device)
    return decoder


def rasterize_scene(
    rasterization: Any,
    gaussians: Sequence[Dict[str, Any]],
    features: Sequence[Any],
    K: Any,
    W: Any,
    width: int,
    height: int,
    background: Any,
    extra_features: Optional[Sequence[Any]] = None,
) -> Tuple[Any, Any]:
    """Shared splat pass; optional attributes use the same visibility as appearance."""
    import torch

    means = torch.cat([g["xyz"] for g in gaussians], 0)
    quats = torch.cat([g["rotation"] for g in gaussians], 0)
    scales = torch.cat([g["scaling"] for g in gaussians], 0)
    opacities = torch.cat([g["opacity"].flatten() for g in gaussians], 0)
    colors = torch.cat(
        [
            torch.cat([g["shs"].squeeze(1) if g["shs"].ndim == 3 else g["shs"], feature], 1)
            for g, feature in zip(gaussians, features)
        ],
        0,
    )
    raster_background = torch.cat([torch.ones(3, device=means.device), background])
    if extra_features is not None:
        extra = torch.cat(list(extra_features), 0)
        colors = torch.cat([colors, extra], -1)
        raster_background = torch.cat([raster_background, extra.new_zeros(extra.shape[-1])])
    color, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=W[None],
        Ks=K[None],
        width=width,
        height=height,
        backgrounds=raster_background[None],
        packed=False,
        render_mode="RGB+ED",
        near_plane=0.01,
        far_plane=100.0,
        eps2d=0.3,
        rasterize_mode="antialiased",
    )
    return color, alpha


def render_scene(
    decoder: Any,
    rasterization: Any,
    gaussians: Sequence[Dict[str, Any]],
    features: Sequence[Any],
    K: Any,
    W: Any,
    width: int,
    height: int,
    background: Any,
    mesh_buffer: Optional[Dict[str, Any]] = None,
    display_background: Any = None,
) -> Tuple[Any, Any]:
    import torch

    color, alpha = rasterize_scene(
        rasterization, gaussians, features, K, W, width, height, background
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb, mask = decoder.predict_rgbs(
            None, color[0, ..., 3:-1].permute(2, 0, 1)[None], None, height, width
        )
    ba = mask[0, 0].float().squeeze() * (alpha[0, ..., 0] > 0.001)
    if mesh_buffer is not None:
        from hoi.rendering.compositing import composite

        result = composite(rgb[0, 0].float() * ba[..., None], ba, color[0, ..., -1], mesh_buffer, background=display_background)
        return result["rgb"], ba
    white = torch.ones(3, device=color.device, dtype=rgb.dtype)
    out = rgb[0, 0].float() * ba[..., None] + (1.0 - ba[..., None]) * white
    return out, ba


def write_video(
    ffmpeg: Path,
    destination: Path,
    frames: Sequence[int],
    frame_dir: Path,
    width: int,
    height: int,
    fps: float,
) -> None:
    from PIL import Image

    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    cmd = [
        str(ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    # Keep error output off a pipe while streaming to avoid a blocked encoder.
    import tempfile

    with tempfile.TemporaryFile(dir=destination.parent) as errors:
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            if process.stdin is None:
                raise RuntimeError("video encoder has no input stream")
            for frame in frames:
                with Image.open(frame_dir / f"{frame:04d}.png") as image:
                    process.stdin.write(image.convert("RGB").tobytes())
            process.stdin.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError(
                    f"video encoding failed ({code}): {errors.read().decode('utf-8', errors='replace')}"
                )
        except BaseException:
            process.kill()
            process.wait()
            raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cameras", help="custom fixed cameras or source-indexed camera trajectories JSON"
    )
    parser.add_argument("--frames", nargs="+")
    parser.add_argument(
        "--pose-frame", type=int, help="hold this source pose while output camera frames advance"
    )
    parser.add_argument("--serial", default="all")
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="retain PNG inputs after successful video encoding",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--fps", type=float, help="output FPS; interpolate 3D motion on a shared clock"
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    from hoi.runtime import configure_runtime, load_rasterization, runtime_settings

    runtime = runtime_settings(config.get("runtime"))
    project_root = Path(runtime["project_root"])
    camera_space = config.get("camera_space", "raw")
    if camera_space not in {"raw", "normalized"}:
        raise ValueError("camera_space must be raw or normalized")
    fps = float(config["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    actors = config.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ValueError("actors must be a nonempty list")
    actor_ids = [safe_segment(actor["id"]) for actor in actors]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("actor ids must be unique")
    scene, center, height, gauge = load_scene_config(config, project_root)
    custom_camera_source = None
    native_width, native_height = 1008, 756
    if args.cameras:
        camera_path = resolve_path(args.cameras, project_root)
        scene, camera_space, native_width, native_height = load_custom_cameras(camera_path)
        custom_camera_source = {"path": str(camera_path), "sha256": sha256_file(camera_path)}
    args.width = native_width if args.width is None else args.width
    args.height = native_height if args.height is None else args.height
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    views = (
        scene["views"]
        if args.serial.lower() == "all"
        else [
            view
            for view in scene["views"]
            if str(view["serial"]) == args.serial or view.get("label") == args.serial
        ]
    )
    if not views:
        raise ValueError(f"no camera selected for serial {args.serial!r}")
    serials = [safe_segment(view["serial"]) for view in views]
    if len(serials) != len(set(serials)):
        raise ValueError("camera serials must be unique")
    trajectories = {str(view["serial"]): view for view in views if "frame_index" in view}

    configure_runtime(runtime)
    import torch
    from PIL import Image
    from hoi.rendering.rig import animate

    bundles = [load_bundle(resolve_path(actor["avatar"], project_root)) for actor in actors]
    decoder_checks = [compare_decoders(bundles[0], bundle) for bundle in bundles[1:]]
    motions = [
        load_motion(
            resolve_path(actor["motion"], project_root) if actor.get("motion") else None,
            bundle["poses"],
            bundle["rig"]["rest_joints"].dtype,
        )
        for actor, bundle in zip(actors, bundles)
    ]
    object_config = config.get("rigid_object")
    object_mesh = None
    object_motion = None
    object_sources = None
    if object_config is not None:
        if not isinstance(object_config, dict):
            raise ValueError("rigid_object must contain mesh and motion paths")
        mesh_path = resolve_path(object_config["mesh"], project_root)
        object_motion_path = resolve_path(object_config["motion"], project_root)
        object_mesh = torch.load(mesh_path, map_location="cpu", weights_only=False)
        object_motion = torch.load(object_motion_path, map_location="cpu", weights_only=False)
        for key in ("vertices_local", "faces", "colors"):
            if key not in object_mesh:
                raise KeyError(f"rigid object mesh missing {key}")
        for key in ("frames", "R", "logscale", "translation_world"):
            if key not in object_motion:
                raise KeyError(f"rigid object motion missing {key}")
        object_frame_count = len(frame_list(object_motion))
        if (
            len(object_motion["R"]) != object_frame_count
            or len(object_motion["translation_world"]) != object_frame_count
        ):
            raise ValueError(
                "rigid object rotation/translation length differs from its actual frames"
            )
        object_sources = {
            "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "motion": str(object_motion_path),
            "motion_sha256": sha256_file(object_motion_path),
            "compositing": "expected_depth",
        }
    frame_motions = motions if object_motion is None else [*motions, object_motion]
    pose_frame = args.pose_frame
    if pose_frame is None:
        pose_frame = (config.get("rendering") or {}).get("pose_frame")
    resampling = None
    if args.fps is not None and args.fps != fps:
        from hoi.motion.resampling import output_times, resample_motion, resample_transforms
        from hoi.motion.rotations import LEGACY_FIELDS, export_motion_with_quaternions
        from hoi.motion.rotations import load_motion_quaternions

        available, _ = select_render_frames(
            frame_motions, None, list(trajectories.values()), pose_frame
        )
        source_fps = fps
        origin_frame = available[0]
        times = (np.asarray(available) - origin_frame) / source_fps
        targets = output_times(times, source_fps, args.fps)
        resampling = {
            "source_fps": source_fps,
            "target_fps": args.fps,
            "source_origin_frame": origin_frame,
            "fixed_pose_frame": pose_frame,
            "target_times": targets.tolist(),
        }
        sampled = []
        for motion, bundle in zip(motions, bundles):
            source = dict(motion)
            source["timestamps"] = (np.asarray(frame_list(motion)) - origin_frame) / source_fps
            if pose_frame is not None:
                index = frame_list(motion).index(int(pose_frame))
                q, _ = load_motion_quaternions(motion, verify_legacy=True)
                source = export_motion_with_quaternions(source, q[index : index + 1])
                source["translation_world"] = motion["translation_world"][index : index + 1]
                source["frames"] = torch.tensor([pose_frame])
                source["timestamps"] = np.array([0.0])
            result = resample_motion(source, bundle["rig"], source_fps, args.fps, targets)
            dtype = bundle["rig"]["rest_joints"].dtype
            for key in (*LEGACY_FIELDS, "translation_world", "joints55_world"):
                if torch.is_tensor(result.get(key)):
                    result[key] = result[key].to(dtype=dtype)
            sampled.append(result)
        motions = sampled
        for view in trajectories.values():
            ids = np.asarray(view["frames"])
            camera_times = (ids - origin_frame) / source_fps
            c2w = np.linalg.inv(view["calibration"]["W"])
            rotation, position = resample_transforms(
                c2w[:, :3, :3], c2w[:, :3, 3], camera_times, targets
            )
            matrices = np.repeat(np.eye(4)[None], len(targets), axis=0)
            matrices[:, :3, :3], matrices[:, :3, 3] = rotation, position
            view["calibration"]["W"] = np.linalg.inv(matrices)
            view["frames"] = list(range(len(targets)))
            view["frame_index"] = {i: i for i in range(len(targets))}
        if object_motion is not None:
            source = dict(object_motion)
            ids = np.asarray(frame_list(source))
            object_times = (ids - origin_frame) / source_fps
            rotation = torch.as_tensor(source["R"]).cpu().numpy()
            position = torch.as_tensor(source["translation_world"]).cpu().numpy()
            if pose_frame is not None:
                index = frame_list(source).index(int(pose_frame))
                rotation, position = rotation[index : index + 1], position[index : index + 1]
                object_times = np.array([0.0])
            rotation, position = resample_transforms(rotation, position, object_times, targets)
            source.update(
                R=torch.as_tensor(rotation),
                translation_world=torch.as_tensor(position),
                frames=torch.arange(len(targets)),
            )
            object_motion = source
        fps, pose_frame = args.fps, None
        frame_motions = motions if object_motion is None else [*motions, object_motion]
    frames, frame_indices = select_render_frames(
        frame_motions, args.frames, list(trajectories.values()), pose_frame
    )
    indices = frame_indices[: len(motions)]
    object_index = None if object_motion is None else frame_indices[-1]
    for serial, view in trajectories.items():
        missing = set(frames) - set(view["frame_index"])
        if missing:
            raise ValueError(
                f"camera {serial} is missing selected source frames: {sorted(missing)}"
            )
    if args.video and any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError(
            "--video requires consecutive output frames at config fps; use PNG output for sparse frames"
        )
    background_cpu = bundles[0]["background_embedding"][2]
    backgrounds_differ = [
        actor_id
        for actor_id, bundle in zip(actor_ids[1:], bundles[1:])
        if not torch.equal(background_cpu, bundle["background_embedding"][2])
    ]
    feature_dimensions = [int(bundle["features"].shape[1]) for bundle in bundles]
    if len(set(feature_dimensions)) != 1:
        raise ValueError(f"actor feature dimensions differ: {feature_dimensions}")
    decoder = build_decoder(bundles[0], "cuda")
    rasterization = load_rasterization(runtime)
    rigs = [to_device(bundle["rig"], "cuda") for bundle in bundles]
    poses = [to_device(motion, "cuda") for motion in motions]
    features = [bundle["features"].cuda().float() for bundle in bundles]
    background = background_cpu.cuda().float()
    mesh_renderer = None
    if object_mesh is not None:
        from hoi.rendering.mesh import MeshRenderer

        object_mesh = to_device(object_mesh, "cuda")
        object_motion = to_device(object_motion, "cuda")
        mesh_renderer = MeshRenderer()
    out = resolve_path(args.out, project_root)
    cameras = []
    for serial in serials:
        if serial in trajectories:
            K, W = None, None
        else:
            K, W = camera_from_scene(
                scene, serial, center, height, args.width, args.height, camera_space
            )
        directory = out / f"camera_{serial}"
        frame_dir = directory / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        cameras.append((serial, K, W, directory, frame_dir))
    with torch.inference_mode():
        for frame in frames:
            source_frame = frame if pose_frame is None else int(pose_frame)
            gaussians = [
                animate(rig, pose, index[source_frame])
                for rig, pose, index in zip(rigs, poses, indices)
            ]
            world = None
            if object_mesh is not None:
                object_t = object_index[source_frame]
                world = (
                    object_mesh["vertices_local"].float()
                    @ object_motion["R"][object_t].float().T
                    * object_motion["logscale"].float().exp()
                    + object_motion["translation_world"][object_t].float()
                )
            for serial, K, W, directory, frame_dir in cameras:
                if serial in trajectories:
                    K, W = camera_at_frame(
                        trajectories[serial],
                        frame,
                        center,
                        height,
                        args.width,
                        args.height,
                        camera_space,
                    )
                mesh_buffer = None
                if object_mesh is not None:
                    mesh_buffer = mesh_renderer(
                        world,
                        object_mesh["faces"],
                        K,
                        W,
                        args.width,
                        args.height,
                        object_mesh["colors"].float(),
                        uv=object_mesh.get("uv"),
                        texture=object_mesh.get("texture"),
                    )
                rgb, _ = render_scene(
                    decoder,
                    rasterization,
                    gaussians,
                    features,
                    K,
                    W,
                    args.width,
                    args.height,
                    background,
                    mesh_buffer,
                )
                image = np.uint8(rgb.clamp(0, 1).cpu().numpy() * 255.0)
                Image.fromarray(image).save(frame_dir / f"{frame:04d}.png")
            print(f"[render] frame {frame}; cameras={len(cameras)}", flush=True)
    if args.video:
        from hoi.results import record_video, video_path

        for serial, K, W, directory, frame_dir in cameras:
            destination = video_path(out, f"camera_{serial}")
            write_video(
                Path(runtime["ffmpeg"]),
                destination,
                frames,
                frame_dir,
                args.width,
                args.height,
                fps,
            )
            record_video(destination, report=out / "report.json", frames=len(frames), fps=fps)
            if not args.keep_frames:
                for frame in frames:
                    (frame_dir / f"{frame:04d}.png").unlink()
                if not any(frame_dir.iterdir()):
                    frame_dir.rmdir()
    report = {
        "config": str(config_path),
        "actors": [
            {
                "id": actor_id,
                "avatar": str(resolve_path(actor["avatar"], project_root)),
                "motion": (
                    str(resolve_path(actor["motion"], project_root))
                    if actor.get("motion")
                    else None
                ),
                "motion_boundary": motion["_joint_render_motion_boundary"],
                "resampling": motion.get("resampling"),
            }
            for actor_id, actor, motion in zip(actor_ids, actors, motions)
        ],
        "frames": frames,
        "pose_frame": pose_frame,
        "source_pose_frames": (
            None
            if resampling
            else [frame if pose_frame is None else int(pose_frame) for frame in frames]
        ),
        "frame_semantics": (
            "output IDs index a shared resampled clock; original motion IDs and interpolation weights are in actor resampling"
            if resampling
            else "output IDs index camera poses; source_pose_frames index actor and rigid motions"
        ),
        "fps": fps,
        "resampling": resampling,
        "video": bool(args.video),
        "video_outputs": (
            [str(video_path(out, f"camera_{serial}")) for serial in serials]
            if args.video else []
        ),
        "image_size": [args.width, args.height],
        "camera_serials": serials,
        "gauge": gauge,
        "camera_space": camera_space,
        "decoder_checks": decoder_checks,
        "custom_cameras": custom_camera_source,
        "shared_decoder_actor": actor_ids[0],
        "shared_background_actor": actor_ids[0],
        "backgrounds_differ_from_first": backgrounds_differ,
        "feature_dimension": feature_dimensions[0],
        "rigid_object": object_sources,
        "renumbered_or_interpolated": resampling is not None,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
