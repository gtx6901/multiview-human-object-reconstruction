#!/usr/bin/env python
"""Build 99 full-frame SAM2 masks for MJJ001 from the real motion source_frames.

Inputs (read-only):
  * motion.pt                 -> `source_frames` defines the 99 real capture indices
  * assets/mjj/dynamic/<serial>/<source_frame:04d>.jpg
  * assets/mjj/keypoints/dynamic.npz -> RTMW133 topdown single-person bbox
  * runtime/vendor/sam2 + runtime/sam2/checkpoints/sam2.1_hiera_tiny.pt

Outputs (only under cloth_dynamics/data/...):
  * inputs/<serial>/00000.jpg ... 00098.jpg  hardlinks to the original frames
    (never re-encoded, never resized, no second undistortion)
  * masks/<serial>/<source_frame:04d>.png   SAM2 obj_id=1, binary 0/255
  * overlays/<serial>/<source_frame:04d>.jpg
  * index.json + source_frames.json

The 4 old diagnostic masks under
`work/mjj/reconstruction/diagnosis/surface/masks` are read only for provenance
and are never overwritten.  This script does not call a GPU itself; it exposes
`--device` (default cuda) for the Codex GPU scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "dual_track"))
from image_io import read_image, write_image  # noqa: E402

DEFAULT_MOTION = ROOT / "work/mjj/reconstruction/refinement/body/contract_fit/full_context/expression/continuous/motion.pt"
DEFAULT_IMAGES = ROOT / "assets/mjj/dynamic"
DEFAULT_KEYPOINTS = ROOT / "assets/mjj/keypoints/dynamic.npz"
DEFAULT_CALIBRATION = ROOT / "work/mjj/calibration/candidate_balanced_a.npz"
DEFAULT_OLD_MASKS = ROOT / "work/mjj/reconstruction/diagnosis/surface/masks"
DEFAULT_SAM2_ROOT = ROOT / "runtime/vendor/sam2"
DEFAULT_CKPT = ROOT / "runtime/sam2/checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_OUT = HERE / "data/mjj_sam2"
EXPECTED_FRAMES = 99
OBJ_ID = 1
BBOX_KEYS = ("boxes", "bboxes", "person_boxes", "det_boxes", "box")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_motion(path: Path) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"motion.pt is not a dict: {type(raw)}")
    sf = raw.get("source_frames", None)
    if sf is None:
        raise ValueError("motion.pt has no source_frames; refusing to infer source indices")
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.int64).reshape(-1)
    if sf.size != EXPECTED_FRAMES:
        raise ValueError(f"source_frames has {sf.size} entries, expected {EXPECTED_FRAMES}")
    if len(set(sf.tolist())) != sf.size:
        raise ValueError("source_frames contains duplicates; refusing to fabricate times")
    if (sf < 0).any():
        raise ValueError("negative source_frames")
    fps = raw.get("fps", raw.get("capture_fps", None))
    fps = float(fps) if fps is not None else None
    meta = {
        "motion": str(path), "motion_sha256": sha256_file(path),
        "T": int(sf.size), "source_frames": sf.tolist(), "fps": fps,
        "has_source_frames": True,
    }
    return sf, fps, meta


def load_serial_order(cal_path: Path, keypoint_cams: Sequence[str]) -> List[str]:
    cal = np.load(cal_path, allow_pickle=True)
    cams = [str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x) for x in np.asarray(cal["cams"]).reshape(-1)]
    if len(cams) != 8 or len(set(cams)) != 8:
        raise ValueError(f"calibration cams are not 8 unique serials: {cams}")
    kp = [str(x) for x in keypoint_cams]
    if set(kp) != set(cams):
        raise ValueError(f"keypoint cams {sorted(kp)} != calibration cams {sorted(cams)}")
    return cams


def _bbox_array(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, str]:
    for key in BBOX_KEYS:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 4 and arr.shape[2] == 1:
                arr = arr[:, :, 0, :]
            if arr.ndim == 3 and arr.shape[-1] == 4:
                return arr, key
    raise KeyError(
        "no single-person bbox array in keypoints npz; expected one of "
        f"{BBOX_KEYS} with shape (8,T,4) or (T,8,4); keys={list(z.files)}"
    )


def pick_bbox(arr: np.ndarray, key: str, cam_idx: int, kp_frame_idx: int,
              n_cams: int, n_frames: int) -> np.ndarray:
    if arr.shape[0] == n_cams and arr.shape[1] == n_frames:
        b = arr[cam_idx, kp_frame_idx]
    elif arr.shape[0] == n_frames and arr.shape[1] == n_cams:
        b = arr[kp_frame_idx, cam_idx]
    else:
        raise ValueError(f"{key} shape {arr.shape} is not (8,T,4) or (T,8,4)")
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if b.size != 4 or not np.isfinite(b).all():
        raise ValueError(f"invalid bbox {b} for cam {cam_idx}, kp frame {kp_frame_idx}")
    x0, y0, x1, y1 = [float(v) for v in b]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate bbox {b}")
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def prepare_input_links(out: Path, image_root: Path, cams: Sequence[str],
                        source_frames: np.ndarray) -> Dict[str, Any]:
    """Hardlink the 99 selected originals to contiguous SAM2 JPEG names."""
    stats: Dict[str, Any] = {"serials": {}, "hardlinks": 0, "copies": 0}
    for serial in cams:
        src_dir = image_root / serial
        dst_dir = out / "inputs" / serial
        dst_dir.mkdir(parents=True, exist_ok=True)
        for old in dst_dir.glob("*.jpg"):
            old.unlink()
        rows = []
        for t, sid in enumerate(source_frames.tolist()):
            src = src_dir / f"{int(sid):04d}.jpg"
            if not src.is_file():
                raise FileNotFoundError(f"source frame missing: {src}")
            dst = dst_dir / f"{t:05d}.jpg"
            try:
                os.link(src, dst)
                how = "hardlink"
                stats["hardlinks"] += 1
            except OSError:
                shutil.copy2(src, dst)
                how = "copy"
                stats["copies"] += 1
            rows.append({"internal_index": t, "source_frame": int(sid),
                         "image": str(dst), "link": how})
        stats["serials"][serial] = {"input_dir": str(dst_dir), "frames": rows}
    return stats


def load_seed_bboxes(keypoints: Path, cams: Sequence[str],
                     source_frames: np.ndarray) -> Dict[str, Any]:
    z = np.load(keypoints, allow_pickle=True)
    kp_cams = [str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x)
               for x in np.asarray(z["cams"]).reshape(-1)]
    if len(kp_cams) != 8 or len(set(kp_cams)) != 8:
        raise ValueError(f"keypoint cams are not 8 unique serials: {kp_cams}")
    if "frames" not in z.files:
        raise KeyError(f"keypoints npz has no frames; keys={list(z.files)}")
    kp_frames = np.asarray(z["frames"], dtype=np.int64).reshape(-1)
    seed_src = int(source_frames[0])
    hits = np.flatnonzero(kp_frames == seed_src)
    if hits.size != 1:
        raise ValueError(f"seed source frame {seed_src} appears {hits.size} times in keypoint frames")
    kp_fi = int(hits[0])
    arr, key = _bbox_array(z)
    out: Dict[str, Any] = {"key": key, "seed_source_frame": seed_src,
                           "keypoint_frame_index": kp_fi, "cameras": {}}
    for serial in cams:
        ci = kp_cams.index(serial)
        b = pick_bbox(arr, key, ci, kp_fi, len(kp_cams), len(kp_frames))
        out["cameras"][serial] = b.tolist()
    return out


def validate_seed_boxes(images: Path, cams: Sequence[str],
                        source_frames: np.ndarray, bboxes: Dict[str, list]) -> List[str]:
    """Catch normalized/wrong-coordinate boxes before opening SAM2."""
    notes = []
    for serial in cams:
        p = images / serial / f"{int(source_frames[0]):04d}.jpg"
        im = read_image(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"cannot read seed image {p}")
        h, w = im.shape[:2]
        b = np.asarray(bboxes[serial], np.float64)
        if b[0] < 0 or b[1] < 0 or b[2] > w + 2 or b[3] > h + 2:
            raise ValueError(
                f"{serial}: seed bbox {b.tolist()} outside image {w}x{h}; "
                "expected undistorted full-resolution pixels, not normalized coordinates"
            )
        if b[2] - b[0] < 8 or b[3] - b[1] < 8:
            raise ValueError(f"{serial}: seed bbox too small {b.tolist()}")
        notes.append({"serial": serial, "image_wh": [int(w), int(h)], "bbox": b.tolist()})
    return notes


def build_predictor(args: argparse.Namespace):
    sam2_root = Path(args.sam2_root).resolve()
    ckpt = Path(args.ckpt).resolve()
    if not (sam2_root / "sam2").is_dir():
        raise FileNotFoundError(f"SAM2 source not found: {sam2_root / 'sam2'}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    cwd = Path.cwd()
    os.chdir(sam2_root)
    try:
        predictor = build_sam2_video_predictor(args.sam2_config, str(ckpt), device=args.device)
    finally:
        os.chdir(cwd)
    return predictor


def overlay_frame(bgr: np.ndarray, mask: np.ndarray, bbox: np.ndarray,
                  serial: str, source_frame: int, area: int) -> np.ndarray:
    vis = bgr.copy()
    color = np.array([40, 190, 40], np.float32)  # BGR
    vis[mask] = (vis[mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    cv2.putText(vis, f"{serial} src={source_frame:04d} area={area}",
                (max(4, x0), max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def segment_camera(predictor, serial: str, input_dir: Path, out: Path,
                   source_frames: np.ndarray, seed_box: np.ndarray,
                   args: argparse.Namespace) -> Dict[str, Any]:
    frames = sorted(input_dir.glob("*.jpg"))
    if len(frames) != len(source_frames):
        raise RuntimeError(f"{serial}: {len(frames)} input jpgs != {len(source_frames)} source_frames")
    state = predictor.init_state(
        video_path=str(input_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    h, w = None, None
    records = [None] * len(source_frames)
    try:
        with torch.inference_mode():
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=OBJ_ID, box=seed_box, clear_old_points=True
            )
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=0,
                max_frame_num_to_track=len(source_frames) - 1, reverse=False,
            ):
                if not (0 <= frame_idx < len(source_frames)) or masks is None:
                    continue
                ids = list(obj_ids)
                if OBJ_ID not in ids:
                    raise RuntimeError(f"{serial} frame {frame_idx}: obj_id {OBJ_ID} missing")
                oi = ids.index(OBJ_ID)
                if masks.ndim == 4:
                    logits = masks[oi, 0]
                elif masks.ndim == 3:
                    logits = masks[oi]
                else:
                    raise RuntimeError(f"{serial} frame {frame_idx}: mask output shape {tuple(masks.shape)}")
                logits = logits.detach().float().cpu().numpy()
                if logits.ndim != 2:
                    raise RuntimeError(f"{serial} frame {frame_idx}: mask shape {logits.shape}")
                h, w = logits.shape
                mask = logits > 0.0
                area = int(mask.sum())
                ys, xs = np.where(mask)
                if area == 0 or xs.size == 0:
                    raise RuntimeError(f"{serial} frame {frame_idx}: empty SAM2 mask")
                bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
                src = int(source_frames[frame_idx])
                mask_path = out / "masks" / serial / f"{src:04d}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not write_image(str(mask_path), (mask.astype(np.uint8) * 255)):
                    raise RuntimeError(f"mask write failed: {mask_path}")
                records[frame_idx] = {
                    "internal_index": frame_idx, "source_frame": src,
                    "mask": str(mask_path), "area": area,
                    "bbox": bbox.tolist(),
                    "score": float(np.mean(1.0 / (1.0 + np.exp(-np.clip(logits[mask], -30, 30))))),
                }
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        del state
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    missing = [i for i, r in enumerate(records) if r is None]
    if missing:
        raise RuntimeError(f"{serial}: no mask for internal frames {missing[:8]}...")
    picks = sorted({0, len(records) // 2, len(records) - 1} |
                   set(range(0, len(records), max(1, args.overlay_every))))
    overlay_dir = out / "overlays" / serial
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for i in picks:
        r = records[i]
        src = int(source_frames[i])
        bgr = read_image(str(input_dir / f"{i:05d}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read input overlay frame {input_dir / f'{i:05d}.jpg'}")
        vis = overlay_frame(bgr, read_image(str(r["mask"]), cv2.IMREAD_GRAYSCALE) > 127,
                            np.asarray(r["bbox"]), serial, src, int(r["area"]))
        if vis.shape[1] > 1280:
            s = 1280.0 / vis.shape[1]
            vis = cv2.resize(vis, (1280, int(round(vis.shape[0] * s))), interpolation=cv2.INTER_AREA)
        write_image(str(overlay_dir / f"{src:04d}.jpg"), vis,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {
        "serial": serial, "frames": len(records), "image_shape": [int(w), int(h)],
        "present": len(records), "missing": missing,
        "records": records, "overlays": picks,
        "mask_dir": str(out / "masks" / serial),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default=str(DEFAULT_MOTION))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--keypoints", default=str(DEFAULT_KEYPOINTS))
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--old-masks", default=str(DEFAULT_OLD_MASKS))
    ap.add_argument("--sam2-root", default=str(DEFAULT_SAM2_ROOT))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--device", default="cuda", help="cuda for Codex execution, cpu for a tiny smoke")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--overlay-every", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if not out.is_relative_to(HERE.resolve()):
        raise ValueError(f"--out must stay under {HERE}: {out}")
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to rebuild own outputs")
    out.mkdir(parents=True, exist_ok=True)

    source_frames, fps, motion_meta = load_motion(Path(a.motion).resolve())
    z = np.load(Path(a.keypoints).resolve(), allow_pickle=True)
    kp_cams = [str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x)
               for x in np.asarray(z["cams"]).reshape(-1)]
    cams = load_serial_order(Path(a.calibration).resolve(), kp_cams)
    seeds = load_seed_bboxes(Path(a.keypoints).resolve(), cams, source_frames)
    links = prepare_input_links(out, Path(a.images).resolve(), cams, source_frames)
    seed_check = validate_seed_boxes(Path(a.images).resolve(), cams, source_frames, seeds["cameras"])

    old_dir = Path(a.old_masks).resolve()
    old_files = sorted(old_dir.glob("*/*.png")) if old_dir.is_dir() else []
    started = time.time()
    predictor = build_predictor(a)
    cameras_out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for serial in cams:
        print(f"[sam2_mjj] {serial}: {len(source_frames)} frames from {links['serials'][serial]['input_dir']}", flush=True)
        rec = segment_camera(predictor, serial, out / "inputs" / serial, out,
                             source_frames, np.asarray(seeds["cameras"][serial], np.float32), a)
        cameras_out.append(rec)
    index = {
        "schema": "mjj_sam2_masks/v1",
        "generated_at_utc": utc_now(),
        "elapsed_sec": round(time.time() - started, 2),
        "motion": motion_meta,
        "fps": fps,
        "T": int(len(source_frames)),
        "source_frames": source_frames.tolist(),
        "mapping": "internal 0..T-1 -> source_frames[t] -> assets/mjj/dynamic/<serial>/<source>:04d.jpg",
        "images": str(Path(a.images).resolve()),
        "inputs_root": str(out / "inputs"),
        "inputs": links["serials"],
        "masks_root": str(out / "masks"),
        "overlays_root": str(out / "overlays"),
        "cameras": cameras_out,
        "seed": {
            "bbox_key": seeds["key"], "seed_source_frame": seeds["seed_source_frame"],
            "keypoint_frame_index": seeds["keypoint_frame_index"],
            "bboxes": seeds["cameras"],
            "checks": seed_check,
            "note": "RTMW133 topdown single-person bbox at the first source frame; no rectification applied (images already undistorted)",
        },
        "sam2": {
            "root": str(Path(a.sam2_root).resolve()),
            "checkpoint": str(Path(a.ckpt).resolve()),
            "config": a.sam2_config,
            "device": a.device,
            "obj_id": OBJ_ID,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "async_loading_frames": False,
        },
        "old_diagnostic_masks": {
            "root": str(old_dir),
            "files_found": [str(p) for p in old_files],
            "count": len(old_files),
            "untouched": True,
            "note": "existing 4 diagnostic frames are a separate comparison; not propagated/overwritten",
        },
        "warnings": warnings + [
            "SAM2 masks are unreviewed single-person proposals, not body-boundary or clothing truth.",
            "No mask is fabricated for a missing/failed frame; the run fails instead.",
            "Inputs are hardlinks/copies of already-undistorted images; no resize, no second undistortion.",
        ],
        "scope": "input mask generation only; not a reconstruction or clothing-quality result",
    }
    atomic_json(out / "index.json", index)
    atomic_json(out / "source_frames.json", {
        "T": int(len(source_frames)), "fps": fps,
        "source_frames": source_frames.tolist(),
        "internal_to_source": {str(i): int(s) for i, s in enumerate(source_frames.tolist())},
    })
    print(f"[sam2_mjj] wrote {out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
