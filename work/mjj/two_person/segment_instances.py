#!/usr/bin/env python
"""SAM2.1 video-instance segmentation for the real FCS two-person scene.

What this script does, in one pass over cameras (one camera at a time):

1. Reads ``scene.json`` plus the raw-coordinate Codex seed boxes in
   ``../dual_track/identity_seeds.json``.
2. Sequentially decodes only the requested source frames, rectifies each frame
   with the camera's native K/distortion into the scene's K_undistorted, and
   writes the undistorted JPEG sequence used by SAM2 (``00000.jpg`` ...).
3. Rectifies every seed box through the same K/distortion -> K_undistorted
   mapping, so SAM2 boxes, masks, bboxes and later RTMW inputs all live in the
   same undistorted pixel space.
4. Adds every seeded actor as a separate SAM2 ``obj_id`` at the seed frame,
   propagates all objects together in one video state, with video/state offload
   to CPU.  A null seed means that actor has no object in that camera: no mask
   is invented.  Identity is assigned only by the seed label, never by bbox size.
5. Saves full-resolution masks (packed), per-frame visibility/bbox/area/score,
   overlays and a local contact sheet.  Masks are explicitly unreviewed SAM2
   proposals, not body-boundary truth.

Only a local GPU is used.  Nothing is uploaded and no original asset is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dual_track'))
from image_io import read_image, write_image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_SCENE = THIS_DIR / "data" / "scene_inventory" / "scene.json"
DEFAULT_SEEDS = THIS_DIR.parent / "dual_track" / "identity_seeds.json"
DEFAULT_OUT = THIS_DIR / "data" / "segment_instances"
ACTORS = ("fcs", "partner")
COLORS = {"fcs": (40, 190, 40), "partner": (255, 130, 0)}  # BGR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def load_scene(path: Path) -> Dict[str, Any]:
    scene = json.loads(path.read_text(encoding="utf-8"))
    if len(scene.get("views") or []) != 8:
        raise RuntimeError(f"scene must contain 8 calibrated views, got {len(scene.get('views') or [])}")
    return scene


def load_seeds(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coordinate_space") != "raw_distorted_native_2048x1536":
        raise RuntimeError(
            "identity_seeds.json must declare coordinate_space=raw_distorted_native_2048x1536"
        )
    ids = set((payload.get("actors") or {}).keys())
    if not set(ACTORS).issubset(ids):
        raise RuntimeError(f"identity seeds must contain actors {ACTORS}, got {sorted(ids)}")
    return payload


def timeline_pts(view: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Return per-source-frame seconds and a per-frame source tag."""
    n = int(view["decode"]["actual_frames"])
    pts = np.full(n, np.nan, dtype=np.float64)
    tags = ["unknown"] * n
    tp = view.get("timeline_npz")
    if not tp:
        return pts, tags
    path = Path(tp)
    if not path.is_file():
        return pts, tags
    with np.load(path, allow_pickle=False) as z:
        ff = np.asarray(z["ffprobe_pts_sec"], dtype=np.float64)
        cv = np.asarray(z["cv2_pos_msec"], dtype=np.float64) / 1000.0
        m = min(n, ff.size, cv.size)
        for i in range(m):
            if np.isfinite(ff[i]):
                pts[i] = ff[i]
                tags[i] = "ffprobe_pts"
            elif np.isfinite(cv[i]):
                pts[i] = cv[i]
                tags[i] = "cv2_pos_msec"
    return pts, tags


def rectify_box(
    box: Sequence[float], K: np.ndarray, dist: np.ndarray, K_und: np.ndarray, width: int, height: int
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Map a raw axis-aligned seed box to the bbox of its rectified boundary.

    A box boundary is not a 2D affine object under lens distortion, so sample a
    7x7 lattice over the full box and take the rectified min/max.  The result is
    a coarse SAM2 prompt, not a precise body box.
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    xs = np.linspace(x0, x1, 7, dtype=np.float64)
    ys = np.linspace(y0, y1, 7, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(np.float64)
    if pts.shape[0] < 4:
        return None, {"error": "too few box samples"}
    p = pts.reshape(-1, 1, 2)
    if dist.size:
        out = cv2.undistortPoints(p, K, dist, P=K_und).reshape(-1, 2)
    else:
        out = cv2.undistortPoints(p, K, np.zeros(5, np.float64), P=K_und).reshape(-1, 2)
    ux0, uy0 = np.min(out, axis=0)
    ux1, uy1 = np.max(out, axis=0)
    ux0 = float(np.clip(ux0, 0, width - 1))
    uy0 = float(np.clip(uy0, 0, height - 1))
    ux1 = float(np.clip(ux1, 0, width - 1))
    uy1 = float(np.clip(uy1, 0, height - 1))
    if ux1 - ux0 < 8.0 or uy1 - uy0 < 8.0:
        return None, {
            "error": "rectified seed box smaller than 8 px",
            "raw_box": [x0, y0, x1, y1],
            "rectified_box": [ux0, uy0, ux1, uy1],
        }
    return np.array([ux0, uy0, ux1, uy1], dtype=np.float32), {
        "raw_box": [x0, y0, x1, y1],
        "rectified_box": [float(ux0), float(uy0), float(ux1), float(uy1)],
    }


def extract_frames(
    view: Dict[str, Any],
    frames_dir: Path,
    start: int,
    end: int,
    pts: np.ndarray,
    pts_tags: Sequence[str],
) -> List[Dict[str, Any]]:
    """Sequentially decode [0,end), rectify [start,end), save JPEGs 00000.jpg..."""
    source = Path(view["path"])
    w = int(view["decode"]["width"])
    h = int(view["decode"]["height"])
    cal = view["calibration"]
    K = np.asarray(cal["K_native"], dtype=np.float64)
    dist = np.asarray(cal["distortion_opencv5"], dtype=np.float64)
    K_und = np.asarray(cal["K_undistorted"], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K_und, (w, h), cv2.CV_32FC1)

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    meta: List[Dict[str, Any]] = []
    local = 0
    try:
        for src in range(end):
            ok, raw = cap.read()
            if not ok or raw is None:
                raise RuntimeError(f"{view['serial']}: decode ended at source frame {src}, need {end}")
            if src < start:
                continue
            rect = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out_path = frames_dir / f"{local:05d}.jpg"
            if not write_image(str(out_path), rect, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"cannot write {out_path}")
            p = float(pts[src]) if src < pts.size and np.isfinite(pts[src]) else None
            tag = str(pts_tags[src]) if src < len(pts_tags) else "unknown"
            meta.append(
                {
                    "local_frame": local,
                    "source_frame": src,
                    "pts_sec": p,
                    "pts_source": tag,
                    "jpeg": str(out_path),
                }
            )
            local += 1
    finally:
        cap.release()
    if local != end - start:
        raise RuntimeError(f"{view['serial']}: extracted {local} frames, expected {end - start}")
    return meta


def build_predictor(args: argparse.Namespace) -> Any:
    sam2_root = Path(args.sam2_root).resolve()
    ckpt = Path(args.ckpt).resolve()
    if not (sam2_root / "sam2").is_dir():
        raise FileNotFoundError(f"SAM2 source not found: {sam2_root / 'sam2'}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:
        raise RuntimeError(
            "SAM2 import failed. Expected runtime/vendor/sam2 on sys.path and torch installed: "
            f"{exc}"
        ) from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SAM2 but torch.cuda.is_available() is false")
    cwd = Path.cwd()
    os.chdir(sam2_root)
    try:
        predictor = build_sam2_video_predictor(
            args.sam2_config,
            str(ckpt),
            device=args.device,
        )
    finally:
        os.chdir(cwd)
    return predictor


def empty_camera(
    view: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    actors: Sequence[str],
    reasons: Dict[str, str],
) -> Dict[str, Any]:
    f_count = len(frames)
    h = int(view["decode"]["height"])
    w = int(view["decode"]["width"])
    k = len(actors)
    return {
        "present": np.zeros((f_count, k), dtype=bool),
        "raw_area": np.zeros((f_count, k), dtype=np.int32),
        "areas": np.zeros((f_count, k), dtype=np.int32),
        "bboxes": np.full((f_count, k, 4), np.nan, dtype=np.float32),
        "scores": np.zeros((f_count, k), dtype=np.float32),
        "border_touch": np.zeros((f_count, k), dtype=bool),
        "packed": np.zeros((f_count, k, h, (w + 7) // 8), dtype=np.uint8),
        "mask_shape": (h, w),
        "reasons": reasons,
        "seed_boxes": {a: None for a in actors},
        "obj_ids": {},
    }


def propagate_camera(
    predictor: Any,
    view: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    frames_dir: Path,
    seeds_by_actor: Dict[str, Optional[np.ndarray]],
    rectified: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """One SAM2 video state for this camera; all valid actors are obj_ids."""
    import torch

    h = int(view["decode"]["height"])
    w = int(view["decode"]["width"])
    k = len(ACTORS)
    f_count = len(frames)
    present = np.zeros((f_count, k), dtype=bool)
    raw_area = np.zeros((f_count, k), dtype=np.int32)
    areas = np.zeros((f_count, k), dtype=np.int32)
    bboxes = np.full((f_count, k, 4), np.nan, dtype=np.float32)
    scores = np.zeros((f_count, k), dtype=np.float32)
    border_touch = np.zeros((f_count, k), dtype=bool)
    packed = np.zeros((f_count, k, h, (w + 7) // 8), dtype=np.uint8)
    obj_ids_by_actor: Dict[str, int] = {}
    reasons: Dict[str, str] = {}
    seed_boxes: Dict[str, Optional[np.ndarray]] = {}

    for ai, actor in enumerate(ACTORS):
        box = seeds_by_actor.get(actor)
        seed_boxes[actor] = box
        if box is None:
            reasons[actor] = rectified.get(actor, {}).get("error", "no_seed_box")
            continue
        # Stable client obj ids; never inferred from bbox.
        obj_id = ai + 1
        obj_ids_by_actor[actor] = obj_id

    if not obj_ids_by_actor:
        return {
            "present": present,
            "raw_area": raw_area,
            "areas": areas,
            "bboxes": bboxes,
            "scores": scores,
            "border_touch": border_touch,
            "packed": packed,
            "mask_shape": (h, w),
            "reasons": reasons,
            "seed_boxes": seed_boxes,
            "obj_ids": {},
        }

    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    try:
        with torch.inference_mode():
            for actor, obj_id in obj_ids_by_actor.items():
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=seed_boxes[actor],
                    clear_old_points=True,
                )
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=f_count - 1,
                reverse=False,
            ):
                if not (0 <= frame_idx < f_count):
                    continue
                if masks is None:
                    continue
                if masks.ndim == 4:
                    masks_2d = masks[:, 0]
                elif masks.ndim == 3:
                    masks_2d = masks
                else:
                    raise RuntimeError(f"unexpected SAM2 mask shape {tuple(masks.shape)}")
                ids = list(obj_ids)
                for actor, obj_id in obj_ids_by_actor.items():
                    if obj_id not in ids:
                        reasons[actor] = "obj_id_missing_from_output"
                        continue
                    ai = ACTORS.index(actor)
                    logits = masks_2d[ids.index(obj_id)].detach().float().cpu().numpy()
                    if logits.ndim != 2:
                        raise RuntimeError(f"unexpected per-object mask shape {logits.shape}")
                    mask = logits > 0.0
                    area = int(mask.sum())
                    raw_area[frame_idx, ai] = area
                    if area < int(args.min_area):
                        if area == 0:
                            reasons[actor] = "sam2_empty_mask"
                        else:
                            reasons[actor] = "sam2_area_below_min"
                        continue
                    ys, xs = np.where(mask)
                    if xs.size == 0:
                        reasons[actor] = "sam2_empty_mask"
                        continue
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    present[frame_idx, ai] = True
                    areas[frame_idx, ai] = area
                    bboxes[frame_idx, ai] = [x0, y0, x1, y1]
                    border_touch[frame_idx, ai] = bool(
                        x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h
                    )
                    fg = logits[mask]
                    scores[frame_idx, ai] = float(
                        np.mean(1.0 / (1.0 + np.exp(-np.clip(fg, -30.0, 30.0))))
                    )
                    if w % 8:
                        padded = np.pad(mask, ((0, 0), (0, 8 - (w % 8))))
                    else:
                        padded = mask
                    packed[frame_idx, ai] = np.packbits(padded, axis=1)
                    reasons[actor] = ""
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        try:
            del state
        except Exception:
            pass
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "present": present,
        "raw_area": raw_area,
        "areas": areas,
        "bboxes": bboxes,
        "scores": scores,
        "border_touch": border_touch,
        "packed": packed,
        "mask_shape": (h, w),
        "reasons": reasons,
        "seed_boxes": seed_boxes,
        "obj_ids": obj_ids_by_actor,
    }


def unpack_mask(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.unpackbits(packed, axis=1, count=w).astype(bool).reshape(h, w)


def draw_overlay(
    frame_bgr: np.ndarray,
    view: Dict[str, Any],
    frame_meta: Dict[str, Any],
    masks: Dict[str, Any],
    actors: Sequence[str],
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()
    ai_frame = int(frame_meta["local_frame"])
    for ai, actor in enumerate(actors):
        if not bool(masks["present"][ai_frame, ai]):
            continue
        mask = unpack_mask(masks["packed"][ai_frame, ai], masks["mask_shape"])
        color = np.asarray(COLORS[actor], dtype=np.float32)
        vis[mask] = (vis[mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
        b = masks["bboxes"][ai_frame, ai]
        if np.isfinite(b).all():
            p0 = (int(b[0]), int(b[1]))
            p1 = (int(b[2]), int(b[3]))
            cv2.rectangle(vis, p0, p1, COLORS[actor], 3)
            cv2.putText(
                vis,
                f"{actor} area={int(masks['areas'][ai_frame, ai])}",
                (p0[0] + 4, max(20, p0[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLORS[actor],
                2,
                cv2.LINE_AA,
            )
    missing = [a for ai, a in enumerate(actors) if not bool(masks["present"][ai_frame, ai])]
    label = (
        f"cam{view['camera_index']} {view['serial']} src={frame_meta['source_frame']:04d} "
        f"pts={frame_meta['pts_sec'] if frame_meta['pts_sec'] is not None else 'NA'}"
    )
    cv2.rectangle(vis, (0, 0), (min(w - 1, 900), 26), (0, 0, 0), -1)
    cv2.putText(vis, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if missing:
        cv2.rectangle(vis, (0, 28), (min(w - 1, 900), 54), (0, 0, 90), -1)
        cv2.putText(
            vis,
            "absent (no seed / empty SAM2): " + ", ".join(missing),
            (6, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def write_camera_outputs(
    view: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    masks: Dict[str, Any],
    out_dir: Path,
    overlay_every: int,
) -> Dict[str, Any]:
    serial = view["serial"]
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{serial}.npz"
    np.savez_compressed(
        mask_path,
        packed=masks["packed"],
        mask_shape=np.asarray(masks["mask_shape"], dtype=np.int32),
        present=masks["present"],
        raw_area=masks["raw_area"],
        areas=masks["areas"],
        bboxes=masks["bboxes"],
        scores=masks["scores"],
        border_touch=masks["border_touch"],
        actors=np.asarray(ACTORS, dtype="U16"),
        obj_ids=np.asarray([masks["obj_ids"].get(a, -1) for a in ACTORS], dtype=np.int32),
        local_frames=np.asarray([f["local_frame"] for f in frames], dtype=np.int32),
        source_frames=np.asarray([f["source_frame"] for f in frames], dtype=np.int32),
        pts_sec=np.asarray(
            [np.nan if f["pts_sec"] is None else f["pts_sec"] for f in frames],
            dtype=np.float64,
        ),
    )

    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    picks = {0, len(frames) - 1, len(frames) // 2}
    picks.update(range(0, len(frames), max(1, overlay_every)))
    picks = sorted(picks)
    for fi in picks:
        frame_path = Path(frames[fi]["jpeg"])
        bgr = read_image(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        vis = draw_overlay(bgr, view, frames[fi], masks, ACTORS)
        if vis.shape[1] > 1280:
            scale = 1280.0 / vis.shape[1]
            vis = cv2.resize(vis, (1280, int(round(vis.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        write_image(
            str(overlay_dir / f"{serial}_f{frames[fi]['source_frame']:04d}.jpg"),
            vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

    present_counts = {a: int(masks["present"][:, i].sum()) for i, a in enumerate(ACTORS)}
    return {
        "serial": serial,
        "camera_index": view["camera_index"],
        "source": str(view["path"]),
        "frames_dir": str(out_dir / "frames" / serial),
        "mask_npz": str(mask_path),
        "frame_count": len(frames),
        "present_counts": present_counts,
        "reasons": masks["reasons"],
        "obj_ids": masks["obj_ids"],
        "seed_boxes_rectified": {
            a: (None if masks["seed_boxes"].get(a) is None else [float(x) for x in masks["seed_boxes"][a]])
            for a in ACTORS
        },
        "mask_shape": list(masks["mask_shape"]),
        "frames": frames,
    }


def make_contact_sheet(
    cameras: Sequence[Dict[str, Any]],
    out_dir: Path,
    title: str,
) -> str:
    tile_w, tile_h, header, left = 320, 240, 32, 120
    rows = 3
    sheet = np.full((header + rows * (tile_h + 20), left + len(cameras) * tile_w, 3), 255, np.uint8)
    cv2.putText(sheet, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    for ci, cam in enumerate(cameras):
        serial = cam["serial"]
        frames = cam["frames"]
        picks = sorted({0, len(frames) // 2, len(frames) - 1})
        for ri, fi in enumerate(picks):
            y0 = header + ri * (tile_h + 20)
            cv2.putText(
                sheet,
                f"cam{ci} {serial}" if ri == 0 else f"f{frames[fi]['source_frame']}",
                (4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            p = out_dir / "overlays" / f"{serial}_f{frames[fi]['source_frame']:04d}.jpg"
            img = read_image(str(p), cv2.IMREAD_COLOR)
            if img is None:
                img = np.full((tile_h, tile_w, 3), 230, np.uint8)
            else:
                img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            sheet[y0 : y0 + tile_h, left + ci * tile_w : left + (ci + 1) * tile_w] = img
    path = out_dir / "contact_sheet.jpg"
    write_image(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--start", type=int, default=0, help="source frame start; seed frame must be 0")
    ap.add_argument("--end", type=int, default=60, help="source frame end, exclusive")
    ap.add_argument("--sam2-root", default=str(PROJECT_ROOT / "runtime" / "vendor" / "sam2"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "runtime" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--overlay-every", type=int, default=10)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    scene_path = Path(args.scene).resolve()
    seeds_path = Path(args.seeds).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = load_scene(scene_path)
    seed_payload = load_seeds(seeds_path)
    if args.start != 0:
        raise RuntimeError("seed frame is source frame 0; use --start 0 for SAM2 propagation")
    if args.end <= args.start:
        raise RuntimeError("--end must be greater than --start")
    seed_views = {str(v["serial"]): v for v in seed_payload.get("views") or []}

    predictors = build_predictor(args)
    cameras_out: List[Dict[str, Any]] = []
    pipeline_boxes_views: List[Dict[str, Any]] = []
    pipeline_seed_views: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for view in sorted(scene["views"], key=lambda v: v["camera_index"]):
        serial = str(view["serial"])
        print(f"[segment] cam{view['camera_index']} {serial}: extracting {args.end - args.start} frames", flush=True)
        pts, tags = timeline_pts(view)
        frames_dir = out_dir / "frames" / serial
        frames = extract_frames(view, frames_dir, args.start, args.end, pts, tags)
        cal = view["calibration"]
        K = np.asarray(cal["K_native"], dtype=np.float64)
        dist = np.asarray(cal["distortion_opencv5"], dtype=np.float64)
        K_und = np.asarray(cal["K_undistorted"], dtype=np.float64)
        w = int(view["decode"]["width"])
        h = int(view["decode"]["height"])
        sv = seed_views.get(serial)
        raw_by_actor = {a: None for a in ACTORS}
        rectified: Dict[str, Dict[str, Any]] = {}
        seeds_by_actor: Dict[str, Optional[np.ndarray]] = {}
        if sv is None:
            warnings.append(f"{serial}: no row in identity_seeds.json; both actors unknown in this camera")
        else:
            for p in sv.get("persons") or []:
                actor = str(p.get("id"))
                if actor not in ACTORS:
                    continue
                box = p.get("bbox")
                if box is None:
                    rectified[actor] = {"error": "seed_bbox_null"}
                    warnings.append(f"{serial}: {actor} seed bbox is null ({p.get('visibility')}); no mask will be created")
                    continue
                raw_by_actor[actor] = box
                ub, info = rectify_box(box, K, dist, K_und, w, h)
                rectified[actor] = dict(info, seed_visibility=p.get("visibility"), keypoints_enabled=p.get("keypoints_enabled"))
                if ub is None:
                    warnings.append(f"{serial}: {actor} seed could not be rectified: {info.get('error')}")
                else:
                    seeds_by_actor[actor] = ub
        print(f"[segment] cam{view['camera_index']} {serial}: SAM2 obj_ids={ {a: i+1 for i,a in enumerate(ACTORS) if a in seeds_by_actor} }", flush=True)
        try:
            masks = propagate_camera(predictors, view, frames, frames_dir, seeds_by_actor, rectified, args)
        except Exception as exc:
            warnings.append(f"{serial}: SAM2 failed: {type(exc).__name__}: {exc}")
            masks = empty_camera(view, frames, ACTORS, {a: "sam2_error" for a in ACTORS})
            masks["seed_boxes"] = seeds_by_actor
            masks["obj_ids"] = {}
        for actor in ACTORS:
            if actor not in seeds_by_actor:
                masks["reasons"].setdefault(actor, rectified.get(actor, {}).get("error", "no_seed_box"))
        cam = write_camera_outputs(view, frames, masks, out_dir, args.overlay_every)
        # Export the SAM2 mask bboxes as external boxes for instance_inputs.
        # These are undistorted-frame coordinates, not raw video coordinates.
        box_frames = []
        for fi, fm in enumerate(frames):
            detections = []
            for ai, actor in enumerate(ACTORS):
                if bool(masks["present"][fi, ai]):
                    b = masks["bboxes"][fi, ai]
                    if np.isfinite(b).all():
                        detections.append(
                            {
                                "box": [round(float(x), 3) for x in b.tolist()],
                                "score": round(float(masks["scores"][fi, ai]), 5),
                                "actor": actor,
                                "source": "sam2_mask_bbox",
                            }
                        )
            box_frames.append({"frame": int(fm["source_frame"]), "detections": detections})
        pipeline_boxes_views.append(
            {"serial": serial, "camera_index": view["camera_index"], "frames": box_frames}
        )
        seed_persons = []
        for actor in ACTORS:
            b = masks["seed_boxes"].get(actor)
            if b is not None:
                seed_persons.append(
                    {
                        "identity": actor,
                        "frame": 0,
                        "box": [round(float(x), 3) for x in b.tolist()],
                        "score": 1.0,
                        "source": "sam2_seed_rectified",
                        "visibility": rectified.get(actor, {}).get("seed_visibility"),
                        "keypoints_enabled": rectified.get(actor, {}).get("keypoints_enabled"),
                    }
                )
        if seed_persons:
            pipeline_seed_views.append({"serial": serial, "frame": 0, "persons": seed_persons})
        cam["calibration"] = {
            "source": cal.get("source"),
            "source_sha256": cal.get("source_sha256"),
            "K_native": K.tolist(),
            "distortion_opencv5": dist.tolist(),
            "K_undistorted": K_und.tolist(),
            "W": np.asarray(cal["W"], dtype=np.float64).tolist(),
            "native_width": w,
            "native_height": h,
        }
        cam["seed_input"] = {
            "raw_boxes": {a: (None if raw_by_actor[a] is None else [float(x) for x in raw_by_actor[a]]) for a in ACTORS},
            "rectified": rectified,
            "seed_visibility": {a: rectified.get(a, {}).get("seed_visibility") for a in ACTORS},
            "keypoints_enabled": {a: rectified.get(a, {}).get("keypoints_enabled") for a in ACTORS},
        }
        cameras_out.append(cam)
        print(f"[segment] cam{view['camera_index']} {serial}: present {cam['present_counts']} masks={cam['mask_npz']}", flush=True)

    sheet = make_contact_sheet(
        cameras_out,
        out_dir,
        f"{scene.get('session', {}).get('name')} | SAM2.1 tiny | frames {args.start}..{args.end - 1} | masks unreviewed",
    )
    boxes_path = out_dir / "boxes_undistorted.json"
    seeds_und_path = out_dir / "seeds_undistorted.json"
    atomic_write_json(
        boxes_path,
        {
            "schema": "external_person_boxes/v1",
            "coordinate_space": "undistorted_native_2048x1536",
            "source": "SAM2 mask bboxes; not precise body boundaries",
            "frame_range": [args.start, args.end],
            "views": pipeline_boxes_views,
        },
    )
    atomic_write_json(
        seeds_und_path,
        {
            "schema": "two_person_identity_seeds/v1",
            "session": scene.get("session", {}).get("name"),
            "coordinate_space": "undistorted_native_2048x1536",
            "source": "rectified Codex coarse seed boxes, not keypoint truth",
            "identities": [
                {
                    "id": "fcs",
                    "display_name": str(seed_payload.get("actors", {}).get("fcs", "fcs")),
                    "notes": "white/navy shirt, tan shorts; FCS006 appearance must still be validated",
                },
                {
                    "id": "partner",
                    "display_name": str(seed_payload.get("actors", {}).get("partner", "partner")),
                    "notes": "green shirt/black shorts; independent identity",
                },
            ],
            "views": pipeline_seed_views,
        },
    )
    index = {
        "schema": "fcs_two_person_sam2_instances/v1",
        "generated_at_utc": utc_now(),
        "elapsed_sec": round(time.time() - started, 2),
        "scene": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "seeds": {
            "path": str(seeds_path),
            "sha256": sha256_file(seeds_path),
            "coordinate_space": seed_payload.get("coordinate_space"),
            "source": seed_payload.get("source"),
        },
        "pipeline_inputs": {
            "boxes_undistorted": str(boxes_path),
            "seeds_undistorted": str(seeds_und_path),
            "coordinate_space": "undistorted_native_2048x1536",
        },
        "frame_range": [args.start, args.end],
        "frame_count": args.end - args.start,
        "coordinate_space": "undistorted_native_2048x1536",
        "rectification": (
            "cv2.initUndistortRectifyMap(K_native, dist, None, K_undistorted, (2048,1536)); "
            "seed boxes mapped through cv2.undistortPoints with P=K_undistorted"
        ),
        "sam2": {
            "root": str(Path(args.sam2_root).resolve()),
            "checkpoint": str(Path(args.ckpt).resolve()),
            "config": args.sam2_config,
            "device": args.device,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "non_overlap_masks": False,
            "min_area": args.min_area,
            "obj_ids": {a: i + 1 for i, a in enumerate(ACTORS)},
        },
        "actors": {
            "fcs": seed_payload.get("actors", {}).get("fcs"),
            "partner": seed_payload.get("actors", {}).get("partner"),
        },
        "cameras": cameras_out,
        "contact_sheet": sheet,
        "warnings": warnings,
        "notes": [
            "Masks are SAM2 video-propagation proposals in undistorted pixels, not precise body boundaries.",
            "A null seed produces no mask; a fully hidden actor is never filled by a background person.",
            "Masks may overlap at the handshake; non_overlap_masks is deliberately false.",
            "Identity comes only from Codex seed labels, not from bbox size or track order.",
            "Frames and masks are local only; no original image/video/weight left this machine.",
        ],
    }
    atomic_write_json(out_dir / "index.json", index)
    print(f"[segment] wrote {out_dir / 'index.json'}")
    print(f"[segment] contact sheet: {sheet}")
    if warnings:
        for wmsg in warnings:
            print("[segment] warning:", wmsg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
