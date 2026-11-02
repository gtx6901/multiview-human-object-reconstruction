#!/usr/bin/env python
"""Joint Gaussian render for two independent avatar/motion bundles.

Both avatars are animated independently, their Gaussians concatenated, and one
gsplat rasterization plus one shared DPT decode is used.  A single shared render
loop handles one real camera or all real scene cameras; only the camera/output
list differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_SCENE = THIS_DIR / "data" / "scene_inventory" / "scene.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "capture" / "manifest.json"
FFMPEG = Path("D:/Software/FormatFactory/ffmpeg.exe")
_MOTION_RUNTIME_READY = False


def configure_motion_runtime() -> None:
    global _MOTION_RUNTIME_READY
    if _MOTION_RUNTIME_READY:
        return
    import pipeline_adapter as pa

    pa.configure_runtime_paths()
    _MOTION_RUNTIME_READY = True


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def world_to_unit(W: np.ndarray, center: np.ndarray, height: float) -> np.ndarray:
    W = np.asarray(W, np.float64)
    R, t = W[:3, :3], W[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = (R @ center + t) / height
    return out


def load_scene_gauge(scene_path: Path, manifest_path: Path) -> Dict[str, Any]:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    wn = man.get("world_normalization")
    if not wn:
        raise RuntimeError("manifest has no world_normalization; shared gauge undefined")
    center = np.asarray(wn["center_raw"], np.float64)
    height = float(wn["body_height_raw_units"])
    scene_sha = ((scene.get("calibration_selection") or {}).get("selected") or {}).get("sha256")
    manifest_sha = man.get("calibration_sha256")
    if scene_sha and manifest_sha and scene_sha != manifest_sha:
        raise RuntimeError("scene calibration SHA does not match FCS006 manifest; refusing shared gauge")
    return {
        "scene": scene,
        "manifest": man,
        "center": center,
        "height": height,
        "scene_calibration_sha": scene_sha,
        "manifest_calibration_sha": manifest_sha,
        "calibration_match": bool(scene_sha and manifest_sha and scene_sha == manifest_sha),
    }


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


def select_common_frames(poses1: Dict[str, Any], poses2: Dict[str, Any], args: argparse.Namespace):
    fr1 = frame_list(poses1)
    fr2 = frame_list(poses2)
    map2 = {f: i for i, f in enumerate(fr2)}
    common_all = [f for f in fr1 if f in map2]
    available = list(common_all)
    requested = None
    if args.frames is not None:
        requested = set()
        for token in args.frames:
            for part in str(token).replace(" ", "").split(","):
                if not part:
                    continue
                if ":" in part:
                    bits = [int(x) for x in part.split(":")]
                    if len(bits) != 2:
                        raise ValueError(f"bad frame range {part!r}; use inclusive a:b")
                    start, end = bits
                    if end < start:
                        raise ValueError(f"frame range end before start: {part!r}")
                    requested.update(range(start, end + 1))
                else:
                    requested.add(int(part))
        missing = sorted(requested - set(available))
        if missing:
            raise RuntimeError(
                f"requested frame(s) not present in actual common motion frames: {missing}"
            )
        available = [f for f in available if f in requested]
        if not available:
            raise RuntimeError("no requested frame remains after common-motion filtering")
    common = available[: int(args.max_frames)]
    if not common:
        raise RuntimeError("motion frame sets do not overlap")
    return fr1, fr2, map2, common, requested, common_all


def compare_decoders(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    da, db = a.get("decoder") or {}, b.get("decoder") or {}
    if set(da) != set(db):
        raise RuntimeError(
            "avatar DPT decoders have different keys: "
            f"only_fcs={sorted(set(da) - set(db))[:8]} only_partner={sorted(set(db) - set(da))[:8]}"
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
        configure_motion_runtime()
        from rotation_contract import load_motion_quaternions, quaternions_to_legacy

        q, q_info = load_motion_quaternions(motion, verify_legacy=True)
        legacy = quaternions_to_legacy(q)
        for key, value in legacy.items():
            motion[key] = value
        legacy_regenerated = True

    hand_aliases = _normalize_hand_aliases(motion)
    fields_adapted = []
    for key in (
        "root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "left_hand_pose", "right_hand_pose", "lhand_pose", "rhand_pose",
        "translation_world", "scale", "joints55_world",
    ):
        if key in motion and motion[key] is not None:
            motion[key] = torch.as_tensor(motion[key]).to(dtype=rig_dtype)
            fields_adapted.append(key)
    q_shape = None if not q_present else list(torch.as_tensor(motion["local_quaternions_wxyz"]).shape)
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
            if q_present else "legacy passthrough with hand alias normalization only"
        ),
        "stale_aa_used": False,
        "hand_aliases": hand_aliases,
        "rig_dtype": str(rig_dtype),
        "fields_adapted_to_rig_dtype": fields_adapted,
        "q_field_preserved_for_report": bool(q_present),
        "not_converted": ["bundle", "features", "decoder", "camera", "K", "W", "fps", "frames", "image_size"],
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


def build_decoder(bundle: Dict[str, Any], device: str) -> Any:
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import render_gtcam_video as rg
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        decoder = rg.PatchDPT4DecoderOnly(**bundle["decoder_config"])
    decoder.transformer_block = torch.nn.Module()
    decoder.load_state_dict(bundle["decoder"], strict=True, assign=True)
    decoder.eval().to(device)
    return decoder, rg


def cat_gaussians(g1: Dict[str, Any], g2: Dict[str, Any], f1: Any, f2: Any, partner_opacity: float) -> Dict[str, Any]:
    import torch

    def sh_dc(g):
        s = g["shs"]
        return s.squeeze(1) if s.ndim == 3 else s

    means = torch.cat([g1["xyz"], g2["xyz"]], 0)
    quats = torch.cat([g1["rotation"], g2["rotation"]], 0)
    scales = torch.cat([g1["scaling"], g2["scaling"]], 0)
    op1 = g1["opacity"].flatten()
    op2 = g2["opacity"].flatten() * float(partner_opacity)
    opacities = torch.cat([op1, op2], 0)
    colors = torch.cat([torch.cat([sh_dc(g1), f1], 1), torch.cat([sh_dc(g2), f2], 1)], 0)
    return {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}


def render_joint(decoder, rg, g1, g2, f1, f2, K, W, width, height, bg, partner_opacity) -> Any:
    import torch

    if g2 is None:
        means, quats = g1["xyz"], g1["rotation"]
        scales, opacities = g1["scaling"], g1["opacity"].flatten()
        s = g1["shs"]
        sh = s.squeeze(1) if s.ndim == 3 else s
        colors = torch.cat([sh, f1], 1)
    else:
        packed = cat_gaussians(g1, g2, f1, f2, partner_opacity)
        means, quats, scales = packed["means"], packed["quats"], packed["scales"]
        opacities, colors = packed["opacities"], packed["colors"]
    color, alpha, _ = rg.rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=W[None], Ks=K[None], width=width, height=height,
        backgrounds=torch.cat([torch.ones(3, device=means.device), bg])[None],
        packed=False, render_mode="RGB+ED", near_plane=0.01, far_plane=100.0, eps2d=0.3,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb, mask = decoder.predict_rgbs(
            None, color[0, ..., 3:-1].permute(2, 0, 1)[None], None, height, width
        )
    ba = mask[0, 0].float().squeeze() * (alpha[0, ..., 0] > 0.001)
    white = torch.ones(3, device=means.device, dtype=rgb.dtype)
    out = rgb[0, 0].float() * ba[..., None] + (1.0 - ba[..., None]) * white
    return out, ba


def camera_from_scene(scene, serial, center, height, width, height_px):
    import torch

    view = next((v for v in scene["views"] if str(v["serial"]) == serial), None)
    if view is None:
        raise KeyError(f"serial {serial} not in scene")
    cal = view["calibration"]
    K = np.asarray(cal["K_undistorted"], np.float64).copy()
    native_w = float(cal.get("native_width") or view["decode"]["width"])
    native_h = float(cal.get("native_height") or view["decode"]["height"])
    K[0] *= width / native_w
    K[1] *= height_px / native_h
    W = world_to_unit(np.asarray(cal["W"], np.float64), center, height)
    return (
        torch.tensor(K, dtype=torch.float32, device="cuda"),
        torch.tensor(W, dtype=torch.float32, device="cuda"),
    )


def write_h264_preview(out_path: Path, frames: Sequence[int], frame_dir: Path, width: int, height: int) -> None:
    """One H.264 preview from actual PNG frames; no mp4v intermediate."""
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"local ffmpeg missing: {FFMPEG}")
    cmd = [
        str(FFMPEG), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "30",
        "-i", "-", "-an", "-vcodec", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
        "-movflags", "+faststart", str(out_path),
    ]
    from PIL import Image

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for f in frames:
            rgb = np.asarray(Image.open(frame_dir / f"{f:04d}.png").convert("RGB"), dtype=np.uint8)
            if rgb.shape != (height, width, 3):
                raise ValueError(f"frame {f} size {rgb.shape} != {(height, width, 3)}")
            proc.stdin.write(rgb.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"H.264 preview failed rc={rc}: {stderr[-2000:]}")


def build_camera_report(
    *,
    camera_serial: str,
    between_serial: Optional[str],
    all_mode: bool,
    frames: Sequence[int],
    frame_selection: Dict[str, Any],
    args: argparse.Namespace,
    center: np.ndarray,
    height: float,
    scene_gauge: Dict[str, Any],
    fcs_avatar: Path,
    partner_avatar: Path,
    poses1: Dict[str, Any],
    poses2: Dict[str, Any],
    decoder_check: Dict[str, Any],
    feature_basis: Dict[str, int],
    verify: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    report = {
        "schema": "two_person_joint_render/v1",
        "generated_at_utc": utc_now(),
        "frames": [int(x) for x in frames],
        "frame_selection": frame_selection,
        "camera_serial": camera_serial,
        "between_serial": between_serial,
        "width": args.width,
        "height": args.height,
        "shared_world_gauge": {
            "center_raw": center.tolist(),
            "body_height_raw_units": height,
            "calibration_match": scene_gauge["calibration_match"],
            "scene_calibration_sha256": scene_gauge["scene_calibration_sha"],
            "manifest_calibration_sha256": scene_gauge["manifest_calibration_sha"],
            "z_up_or_floor_assumed": False,
        },
        "avatar_fcs": str(fcs_avatar),
        "avatar_partner": str(partner_avatar),
        "motion_fcs": args.motion_fcs,
        "motion_partner": args.motion_partner,
        "motion_boundary": {
            "fcs": poses1.get("_joint_render_motion_boundary"),
            "partner": poses2.get("_joint_render_motion_boundary"),
        },
        "motion_boundary_contract": (
            "For every supplied motion, local_quaternions_wxyz (when present) is verified "
            "against all present legacy aliases by load_motion_quaternions before a single "
            "canonical-q -> legacy-AA conversion. The frozen avatar_rig receives only that "
            "converted legacy AA, cast to the actual rig dtype. No stale AA is silently used. "
            "Legacy-only motions retain alias normalization. This report makes no temporal "
            "or visual-quality claim."
        ),
        "rotation_order": "wxyz; internal q is normalized; legacy AA only at the SMPL-X/LHM rig boundary",
        "partner_opacity": args.partner_opacity,
        "decoder_check": decoder_check,
        "feature_basis": feature_basis,
        "background_ambiguity": scene_gauge.get("background_warning"),
        "joint_rasterization": True,
        "separate_rgb_layers": False,
        "verify_single": verify,
        "video": {
            "enabled": bool(args.video),
            "codec": "H.264/libx264" if args.video else None,
            "pix_fmt": "yuv420p" if args.video else None,
            "fps": 30.0 if args.video else None,
            "container": "mp4/+faststart" if args.video else None,
        },
        "note": (
            "Both avatars' Gaussians are concatenated before one rasterization and one DPT decode. "
            "Independent world translations are preserved; no RGB paste-up or FCS duplication."
        ),
    }
    if all_mode:
        report["all_mode"] = True
    return report


def render_camera_set(
    *,
    cameras: List[Dict[str, Any]],
    common: Sequence[int],
    fr1: Sequence[int],
    map2: Dict[int, int],
    args: argparse.Namespace,
    rg: Any,
    decoder: Any,
    rig1: Dict[str, Any],
    poses1: Dict[str, Any],
    rig2: Dict[str, Any],
    poses2: Dict[str, Any],
    f1: Any,
    f2: Any,
    bg1: Any,
    center: np.ndarray,
    height: float,
    scene_gauge: Dict[str, Any],
    fcs_avatar: Path,
    partner_avatar: Path,
    decoder_check: Dict[str, Any],
    frame_selection: Dict[str, Any],
    all_mode: bool,
):
    import torch
    from PIL import Image

    def animate_pair(f: int):
        return (
            rg.animate(rig1, poses1, fr1.index(f)),
            rg.animate(rig2, poses2, map2[f]),
        )

    verify_by_serial: Dict[str, Dict[str, Any]] = {}
    with torch.inference_mode():
        for n, f in enumerate(common):
            g1, g2 = animate_pair(f)
            for cam in cameras:
                rgb, _ = render_joint(
                    decoder, rg, g1, g2, f1, f2, cam["K"], cam["W"],
                    args.width, args.height, bg1, args.partner_opacity,
                )
                arr = np.uint8(rgb.clamp(0, 1).cpu().numpy() * 255.0)
                Image.fromarray(arr).save(cam["dir"] / f"{f:04d}.png")
            if args.verify_single and f == common[0]:
                for cam in cameras:
                    rgb0, _ = render_joint(
                        decoder, rg, g1, None, f1, None, cam["K"], cam["W"],
                        args.width, args.height, bg1, 0.0,
                    )
                    rgb_zero, _ = render_joint(
                        decoder, rg, g1, g2, f1, f2, cam["K"], cam["W"],
                        args.width, args.height, bg1, 0.0,
                    )
                    diff = (rgb0 - rgb_zero).abs()
                    verify = {
                        "frame": int(common[0]),
                        "single_only_vs_partner_opacity_0_max_abs": float(diff.max().item()),
                        "single_only_vs_partner_opacity_0_mean_abs": float(diff.mean().item()),
                        "exact": bool(torch.equal(rgb0, rgb_zero)),
                        "tolerance": 1e-6,
                    }
                    if all_mode:
                        verify["camera_serial"] = cam["serial"]
                    verify["passed"] = bool(
                        verify["exact"] or verify["single_only_vs_partner_opacity_0_max_abs"] <= 1e-6
                    )
                    atomic_write_json(cam["dir"] / "verify_single.json", verify)
                    verify_by_serial[cam["serial"]] = verify
            if n % 10 == 0:
                print(f"[joint_render] frame {f} cameras={len(cameras)}", flush=True)

    if args.video:
        for cam in cameras:
            write_h264_preview(
                cam["dir"] / "preview.mp4", common, cam["dir"], args.width, args.height
            )

    feature_basis = {"fcs_dim": int(f1.shape[1]), "partner_dim": int(f2.shape[1])}
    reports: Dict[str, Dict[str, Any]] = {}
    for cam in cameras:
        subreport = build_camera_report(
            camera_serial=cam["serial"],
            between_serial=None if all_mode else args.between_serial,
            all_mode=all_mode,
            frames=common,
            frame_selection=frame_selection,
            args=args,
            center=center,
            height=height,
            scene_gauge=scene_gauge,
            fcs_avatar=fcs_avatar,
            partner_avatar=partner_avatar,
            poses1=poses1,
            poses2=poses2,
            decoder_check=decoder_check,
            feature_basis=feature_basis,
            verify=verify_by_serial.get(cam["serial"]),
        )
        atomic_write_json(cam["dir"] / "report.json", subreport)
        reports[cam["serial"]] = subreport
    return verify_by_serial, reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--avatar-fcs", required=True)
    ap.add_argument("--avatar-partner", required=True)
    ap.add_argument("--motion-fcs", default=None)
    ap.add_argument("--motion-partner", default=None)
    ap.add_argument("--serial", default="21305717")
    ap.add_argument("--between-serial", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1008)
    ap.add_argument("--height", type=int, default=756)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--frames", nargs="+", default=None)
    ap.add_argument("--partner-opacity", type=float, default=1.0)
    ap.add_argument("--verify-single", action="store_true")
    args = ap.parse_args(argv)

    configure_motion_runtime()
    import torch

    is_all = str(args.serial).strip().lower() == "all"
    if is_all and args.between_serial:
        raise ValueError("--serial all cannot be combined with --between-serial")
    if args.width % 2 or args.height % 2:
        raise ValueError("width/height must be even")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fcs_avatar = Path(args.avatar_fcs).resolve()
    partner_avatar = Path(args.avatar_partner).resolve()
    if fcs_avatar == partner_avatar:
        raise RuntimeError("partner avatar path equals FCS avatar; copying the FCS identity is not allowed")
    if fcs_avatar.is_file() and partner_avatar.is_file() and sha256_file(fcs_avatar) == sha256_file(partner_avatar):
        raise RuntimeError("partner avatar is byte-identical to the FCS avatar; second identity not reconstructed")
    if not partner_avatar.is_file():
        raise FileNotFoundError(
            "partner avatar is missing; the second identity is not reconstructed. "
            f"Expected: {partner_avatar}"
        )
    scene_gauge = load_scene_gauge(Path(args.scene).resolve(), Path(args.manifest).resolve())
    scene = scene_gauge["scene"]
    center, height = scene_gauge["center"], scene_gauge["height"]

    b1 = load_bundle(fcs_avatar)
    b2 = load_bundle(partner_avatar)
    decoder_check = compare_decoders(b1, b2)
    decoder, rg = build_decoder(b1, "cuda")
    rig1 = to_device(b1["rig"], "cuda")
    rig2 = to_device(b2["rig"], "cuda")
    poses1 = to_device(
        load_motion(Path(args.motion_fcs) if args.motion_fcs else None, b1["poses"], rig1["rest_joints"].dtype),
        "cuda",
    )
    poses2 = to_device(
        load_motion(Path(args.motion_partner) if args.motion_partner else None, b2["poses"], rig2["rest_joints"].dtype),
        "cuda",
    )
    f1 = b1["features"].cuda().float()
    f2 = b2["features"].cuda().float()
    if f1.shape[1] != f2.shape[1]:
        raise RuntimeError(f"feature basis dimensions differ: {f1.shape[1]} vs {f2.shape[1]}")
    bg1 = b1["background_embedding"][2].cuda().float()
    bg2 = b2["background_embedding"][2].cuda().float()
    if not torch.equal(bg1, bg2):
        scene_gauge["background_warning"] = (
            "avatar backgrounds differ; joint render uses FCS background. "
            "This is an explicit ambiguity, not a second DPT basis."
        )

    # The only camera-list difference between single and all mode.
    if is_all:
        cameras = []
        for view in scene.get("views") or []:
            serial = str(view["serial"])
            Kc, Wc = camera_from_scene(scene, serial, center, height, args.width, args.height)
            cam_dir = out / f"cam_{serial}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            cameras.append({"serial": serial, "K": Kc, "W": Wc, "dir": cam_dir})
        if not cameras:
            raise RuntimeError("scene has no real cameras for --serial all")
    else:
        K, W = camera_from_scene(scene, args.serial, center, height, args.width, args.height)
        if args.between_serial:
            from scipy.spatial.transform import Rotation, Slerp

            _, W_other = camera_from_scene(scene, args.between_serial, center, height, args.width, args.height)
            a, b = W.cpu().numpy().astype(np.float64), W_other.cpu().numpy().astype(np.float64)
            joints = torch.cat([poses1["joints55_world"], poses2["joints55_world"]], 1)
            points = joints.reshape(-1, 3).cpu().numpy()
            target = (points.min(0) + points.max(0)) / 2
            ca, cb = -a[:3, :3].T @ a[:3, 3], -b[:3, :3].T @ b[:3, 3]
            da, db = ca - target, cb - target
            ra, rb = np.linalg.norm(da), np.linalg.norm(db)
            direction = da / ra + db / rb
            C = target + direction / np.linalg.norm(direction) * ((ra + rb) / 2)
            r_hint = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))(.5).as_matrix()
            forward = target - C
            forward /= np.linalg.norm(forward)
            right = np.cross(r_hint[1], forward)
            right /= np.linalg.norm(right)
            R = np.stack([right, np.cross(forward, right), forward])
            mid = np.eye(4)
            mid[:3, :3] = R
            mid[:3, 3] = -R @ C
            W = torch.tensor(mid, dtype=torch.float32, device="cuda")
        cameras = [{"serial": args.serial, "K": K, "W": W, "dir": out}]

    fr1, fr2, map2, common, requested, common_all = select_common_frames(poses1, poses2, args)
    frame_selection = {
        "requested": None if requested is None else sorted(int(x) for x in requested),
        "available_common": [int(x) for x in common_all],
        "selected": [int(x) for x in common],
        "max_frames": int(args.max_frames),
        "missing_requested": [],
        "renumbered_or_interpolated": False,
    }

    verify_by_serial, reports = render_camera_set(
        cameras=cameras,
        common=common,
        fr1=fr1,
        map2=map2,
        args=args,
        rg=rg,
        decoder=decoder,
        rig1=rig1,
        poses1=poses1,
        rig2=rig2,
        poses2=poses2,
        f1=f1,
        f2=f2,
        bg1=bg1,
        center=center,
        height=height,
        scene_gauge=scene_gauge,
        fcs_avatar=fcs_avatar,
        partner_avatar=partner_avatar,
        decoder_check=decoder_check,
        frame_selection=frame_selection,
        all_mode=is_all,
    )

    if is_all:
        summary = {
            "schema": "two_person_joint_render_all/v1",
            "generated_at_utc": utc_now(),
            "all_mode": True,
            "camera_count": len(cameras),
            "cameras": [
                {
                    "serial": cam["serial"],
                    "frames": [int(x) for x in common],
                    "report": str(cam["dir"] / "report.json"),
                    "directory": str(cam["dir"]),
                    "verify_single": verify_by_serial.get(cam["serial"]),
                }
                for cam in cameras
            ],
            "frames": [int(x) for x in common],
            "frame_selection": frame_selection,
            "width": args.width,
            "height": args.height,
            "avatar_fcs": str(fcs_avatar),
            "avatar_partner": str(partner_avatar),
            "motion_fcs": args.motion_fcs,
            "motion_partner": args.motion_partner,
            "motion_boundary": {
                "fcs": poses1.get("_joint_render_motion_boundary"),
                "partner": poses2.get("_joint_render_motion_boundary"),
            },
            "decoder_check": decoder_check,
            "feature_basis": {"fcs_dim": int(f1.shape[1]), "partner_dim": int(f2.shape[1])},
            "partner_opacity": args.partner_opacity,
            "shared_world_gauge": scene_gauge_payload(scene_gauge, center, height),
            "video": {
                "enabled": bool(args.video),
                "codec": "H.264/libx264" if args.video else None,
                "pix_fmt": "yuv420p" if args.video else None,
                "fps": 30.0 if args.video else None,
            },
            "notes": (
                "One process loaded both bundles/rig/decoder and verified q/decoder once. "
                "Each actual frame was animated once and the same Gaussians rendered to every real camera. "
                "Each camera owns a full two_person_joint_render report and optional verify_single.json."
            ),
            "verify_single_all_cameras": bool(args.verify_single),
        }
        atomic_write_json(out / "report.json", summary)
    if args.verify_single and any(not v["passed"] for v in verify_by_serial.values()):
        print("[joint_render] verify_single FAILED for at least one camera")
        return 2
    return 0


def scene_gauge_payload(scene_gauge: Dict[str, Any], center: np.ndarray, height: float) -> Dict[str, Any]:
    return {
        "center_raw": center.tolist(),
        "body_height_raw_units": height,
        "calibration_match": scene_gauge["calibration_match"],
        "scene_calibration_sha256": scene_gauge["scene_calibration_sha"],
        "manifest_calibration_sha256": scene_gauge["manifest_calibration_sha"],
        "z_up_or_floor_assumed": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
