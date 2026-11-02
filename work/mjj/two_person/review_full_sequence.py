#!/usr/bin/env python
"""Current-person full-sequence review artifacts from an existing render directory.

This is review-only.  It reads report.json and the original frame PNGs, never
rewrites a model output.  GT is aspect-fit once to the render canvas, then the
same optional crop is applied to every column/frame.  Video is H.264 via the
local FFmpeg stdin rawvideo path; no mp4v intermediate.

Required:
  --render-dir   contains report.json and original frame PNGs
  --gt-dir       original-image directory
  --out          review output directory below work/mjj

Optional:
  --compare-dir  older render directory, adds an OLD column
  --crop x0 y0 x1 y1  same crop on every fitted column

Outputs:
  comparison.mp4
  contact.jpg
  all_frames_<first>_<last>.jpg  (one per 60 actual frames)
  review_scope.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "mjj"
FFMPEG = Path("D:/Software/FormatFactory/ffmpeg.exe")
KEY_FRAMES = [0, 10, 17, 20, 23, 25, 30, 40, 59]
_FONT_CACHE: Dict[int, Any] = {}


def _font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _index_images(directory: Path) -> Dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    priority = {".png": 0, ".jpg": 1, ".jpeg": 2}
    found: Dict[int, Tuple[int, Path]] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if not stem.isdigit():
            continue
        frame = int(stem)
        pri = priority.get(p.suffix.lower(), 99)
        if frame not in found or pri < found[frame][0]:
            found[frame] = (pri, p)
    if not found:
        raise RuntimeError(f"no numeric frame images in {directory}")
    return {k: v[1] for k, v in found.items()}


def _read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit_canvas(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Aspect-preserving fit to one shared render canvas (review-only)."""
    target_w, target_h = size
    scale = min(target_w / float(img.width), target_h / float(img.height))
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def _composite(
    gt: Image.Image,
    old: Optional[Image.Image],
    new: Image.Image,
    frame: int,
    fps: float,
    render_size: Tuple[int, int],
    crop: Optional[Tuple[int, int, int, int]],
) -> Image.Image:
    target_w, target_h = render_size
    images = [_fit_canvas(gt, render_size)]
    labels = ["GT"]
    if old is not None:
        images.append(_fit_canvas(old, render_size))
        labels.append("OLD")
    images.append(_fit_canvas(new, render_size))
    labels.append("NEW")
    if crop is not None:
        x0, y0, x1, y1 = crop
        images = [im.crop((x0, y0, x1, y1)) for im in images]
        cw, ch = x1 - x0, y1 - y0
    else:
        cw, ch = target_w, target_h
    header = 34
    width = cw * len(images)
    canvas = Image.new("RGB", (width, header + ch), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"actual_frame {frame:03d} | {fps:g} fps", fill=(0, 0, 0), font=_font(18))
    for j, (im, label) in enumerate(zip(images, labels)):
        x = j * cw
        canvas.paste(im, (x, header))
        draw.text((x + 6, 8), label, fill=(0, 0, 0), font=_font(17))
    return canvas


def _sheet(
    composites: Sequence[Image.Image],
    labels: Sequence[str],
    cols: int,
    tile_w: int,
    out_path: Path,
) -> None:
    if not composites:
        return
    cols = max(1, min(int(cols), len(composites)))
    rows = int(math.ceil(len(composites) / cols))
    # Preserve each composite's aspect; same tile width for all.
    tile_h = max(1, int(round(tile_w * composites[0].height / composites[0].width)))
    label_h = 20
    cell_h = tile_h + label_h
    sheet = Image.new("RGB", (cols * tile_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(composites, labels)):
        x = (i % cols) * tile_w
        y = (i // cols) * cell_h
        thumb = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 2, y + 2), label, fill=(0, 0, 0), font=_font(13))
    sheet.save(out_path, quality=94)


def _open_ffmpeg(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"local ffmpeg missing: {FFMPEG}")
    # pad only to satisfy yuv420p; no review-column resize or distortion.
    cmd = [
        str(FFMPEG),
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:g}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
        "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-dir", default=None)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    args = ap.parse_args()

    render_dir = Path(args.render_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    compare_dir = Path(args.compare_dir).resolve() if args.compare_dir else None
    out = Path(args.out).resolve()
    if not out.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError(f"output must stay below work/mjj: {out}")
    report_path = render_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = [int(x) for x in report.get("frames", [])]
    if not frames:
        raise RuntimeError("report.json has no actual frames")
    if len(set(frames)) != len(frames):
        raise ValueError("report.json frames contain duplicates; refusing renumbering")
    fps_source = "report.fps" if report.get("fps") is not None else "default_30"
    fps = float(report.get("fps") if report.get("fps") is not None else 30.0)
    if fps <= 0:
        raise ValueError("fps must be positive")

    render_index = _index_images(render_dir)
    gt_index = _index_images(gt_dir)
    compare_index = _index_images(compare_dir) if compare_dir is not None else None
    missing_render = [f for f in frames if f not in render_index]
    missing_gt = [f for f in frames if f not in gt_index]
    missing_compare = [f for f in frames if compare_index is not None and f not in compare_index]
    if missing_render:
        raise FileNotFoundError(f"render PNG missing actual frames: {missing_render}")
    if missing_gt:
        raise FileNotFoundError(f"GT image missing actual frames: {missing_gt}")
    if missing_compare:
        raise FileNotFoundError(f"compare PNG missing actual frames: {missing_compare}")

    first = _read_rgb(render_index[frames[0]])
    render_size = (int(first.width), int(first.height))
    crop = tuple(int(x) for x in args.crop) if args.crop else None
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= render_size[0] and 0 <= y0 < y1 <= render_size[1]):
            raise ValueError(f"crop {crop} outside render canvas {render_size}")
    columns = 3 if compare_index is not None else 2
    canvas_w = (crop[2] - crop[0]) if crop is not None else render_size[0]
    canvas_h = (crop[3] - crop[1]) if crop is not None else render_size[1]
    frame_w = canvas_w * columns
    frame_h = canvas_h + 34
    out.mkdir(parents=True, exist_ok=True)

    video_path = out / "comparison.mp4"
    proc = _open_ffmpeg(video_path, frame_w, frame_h, fps)
    contact_composites: List[Image.Image] = []
    contact_labels: List[str] = []
    chunks: List[Dict[str, Any]] = []
    all_frame_names: List[str] = []
    try:
        for start in range(0, len(frames), 60):
            chunk_frames = frames[start:start + 60]
            chunk_composites: List[Image.Image] = []
            chunk_labels: List[str] = []
            for f in chunk_frames:
                gt = _read_rgb(gt_index[f])
                old = _read_rgb(compare_index[f]) if compare_index is not None else None
                new = _read_rgb(render_index[f])
                comp = _composite(gt, old, new, f, fps, render_size, crop)
                if comp.size != (frame_w, frame_h):
                    raise RuntimeError(f"unexpected composite size at frame {f}: {comp.size} != {(frame_w, frame_h)}")
                proc.stdin.write(comp.tobytes())
                chunk_composites.append(comp)
                chunk_labels.append(f"actual_frame {f:03d}")
                if f in KEY_FRAMES:
                    contact_composites.append(comp)
                    contact_labels.append(f"actual_frame {f:03d}")
            first_f = int(chunk_frames[0])
            last_f = int(chunk_frames[-1])
            sheet_name = f"all_frames_{first_f:03d}_{last_f:03d}.jpg"
            _sheet(chunk_composites, chunk_labels, 5, 360, out / sheet_name)
            all_frame_names.append(sheet_name)
            chunks.append({"first": first_f, "last": last_f, "count": len(chunk_frames), "file": sheet_name})
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"H.264 FFmpeg failed rc={rc}: {stderr[-2000:]}")
    if contact_composites:
        _sheet(contact_composites, contact_labels, 3, 620, out / "contact.jpg")
    else:
        raise RuntimeError("none of the configured key frames exists in report.frames")

    scope = {
        "schema": "current_render_review_scope/v1",
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
        "compare_dir": str(compare_dir) if compare_dir else None,
        "out": str(out),
        "frames": frames,
        "actual_frame_count": len(frames),
        "fps": fps,
        "fps_source": fps_source,
        "render_size": [render_size[0], render_size[1]],
        "review_canvas": [frame_w, frame_h],
        "gt_normalization": "aspect-preserving fit to the shared render canvas; same transform for every frame and column",
        "crop": None if crop is None else list(crop),
        "crop_scope": "same crop applied to GT, OLD (if present), and NEW after shared fit",
        "columns": ["GT", "OLD", "NEW"] if compare_dir else ["GT", "NEW"],
        "key_frames_requested": KEY_FRAMES,
        "key_frames_present": [f for f in KEY_FRAMES if f in frames],
        "all_frames_chunks": chunks,
        "comparison": "comparison.mp4 (H.264 via local FFmpeg stdin rawvideo; no mp4v intermediate)",
        "contact": "contact.jpg",
        "all_frame_sheets": all_frame_names,
        "encoder": str(FFMPEG),
        "video_note": "yuv420p padding may add at most one white row/column; no review-column scaling",
        "not_quality_acceptance": True,
        "note": "Review-only. Original images are aspect-fit/cropped for display; no model output, pose, camera, or Gaussian data is modified.",
    }
    (out / "review_scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    print(f"[review] {len(frames)} frames -> {video_path}")
    print(f"[review] contact={out / 'contact.jpg'} sheets={all_frame_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
