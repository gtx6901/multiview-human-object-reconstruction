#!/usr/bin/env python
"""Inventory the real FCS two-person session from all eight native videos.

This script is CPU-only and read-only with respect to the project inputs.  It
discovers four cameras under each of the two capture environments, decodes each
video sequentially with OpenCV (no frame-count shortcut), cross-checks the
container timestamps with ffprobe, and writes all derived artifacts below its
own ``data/`` directory.

Important facts recorded, not hidden:
  * exact duplicate decoded frames are counted but never dropped;
  * every real PTS is retained even when consecutive images are identical;
  * the current calibration/manifest files are inspected without being edited;
  * two-person presence is *not* asserted by this script; it is marked pending
    local visual review by Codex.

Example (run from the directory containing this file):

  D:/Document/3dgs/.tmp/rebuild_20261014_runtime/venv/Scripts/python.exe ^
    scene_inventory.py --session fcs_2026828am/seuvcl_202608281122_008

If Codex has already chosen a verified interaction interval, add:
  --interaction-start 20 --interaction-end 55
Otherwise the script selects the highest-motion 1-second window and labels it
``motion_candidate_unverified``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"OpenCV (cv2) is required: {exc}")

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"Pillow is required: {exc}")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_SESSION = "fcs_2026828am/seuvcl_202608281122_008"
DEFAULT_OUT = THIS_DIR / "data" / "scene_inventory"
DEFAULT_ENVS = ("e_102", "e_103")


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
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


def file_facts(path: Path) -> Dict[str, Any]:
    try:
        st = path.stat()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def run_cmd(cmd: Sequence[str], timeout: int = 300) -> Tuple[int, str, str]:
    try:
        cp = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return int(cp.returncode), cp.stdout or "", cp.stderr or ""
    except FileNotFoundError:
        return 127, "", f"executable not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {cmd[0]}"
    except OSError as exc:
        return 126, "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# dependencies / binaries (inspection only; nothing is installed)
# ---------------------------------------------------------------------------
def module_status(module: str, distribution: Optional[str] = None) -> Dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError) as exc:
        return {"module": module, "present": False, "error": str(exc)}
    if spec is None:
        return {"module": module, "present": False}
    version = None
    dist = distribution or module
    try:
        version = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception:
        version = None
    return {
        "module": module,
        "present": True,
        "version": version,
        "origin": getattr(spec, "origin", None),
    }


def binary_status(name: str) -> Dict[str, Any]:
    path = shutil.which(name)
    out: Dict[str, Any] = {"name": name, "path": path, "present": bool(path)}
    if path:
        code, stdout, stderr = run_cmd([path, "-version"], timeout=30)
        first = (stdout or stderr).splitlines()
        out["returncode"] = code
        out["version_line"] = first[0].strip() if first else None
    return out


def dependency_report() -> Dict[str, Any]:
    modules = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("torch", "torch"),
        ("smplx", "smplx"),
        ("pytorch3d", "pytorch3d"),
        ("rtmlib", "rtmlib"),
        ("onnxruntime", "onnxruntime"),
        ("ultralytics", "ultralytics"),
        ("scipy", "scipy"),
        ("sam2", "sam2"),
        ("trimesh", "trimesh"),
    ]
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "modules": {name: module_status(name, dist) for name, dist in modules},
        "binaries": {
            "ffprobe": binary_status("ffprobe"),
            "ffmpeg": binary_status("ffmpeg"),
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "CPU-only inventory; torch is not imported and no CUDA context is created.",
        },
    }


# ---------------------------------------------------------------------------
# video metadata / calibration
# ---------------------------------------------------------------------------
def parse_fraction(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        text = str(value)
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def ffprobe_stream(path: Path, ffprobe: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not shutil.which(ffprobe) and not Path(ffprobe).exists():
        return None, {"error": f"ffprobe not found: {ffprobe}"}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=index,codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "time_base,duration,nb_frames,start_time:format=duration,size,bit_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    code, stdout, stderr = run_cmd(cmd)
    if code != 0:
        return None, {"returncode": code, "stderr": stderr.strip()}
    try:
        payload = json.loads(stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None, {"error": "ffprobe returned no video stream"}
        stream = streams[0]
        fmt = payload.get("format") or {}
        stream["_format_duration"] = fmt.get("duration")
        stream["_format_size"] = fmt.get("size")
        stream["_format_bit_rate"] = fmt.get("bit_rate")
        return stream, {}
    except Exception as exc:
        return None, {"error": f"cannot parse ffprobe JSON: {exc}", "stdout_head": stdout[:500]}


def ffprobe_frames(path: Path, ffprobe: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not shutil.which(ffprobe) and not Path(ffprobe).exists():
        return [], {"error": f"ffprobe not found: {ffprobe}"}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,key_frame,pict_type",
        "-of",
        "json",
        str(path),
    ]
    code, stdout, stderr = run_cmd(cmd, timeout=600)
    if code != 0:
        return [], {"returncode": code, "stderr": stderr.strip()}
    try:
        payload = json.loads(stdout or "{}")
        frames = payload.get("frames") or []
        out: List[Dict[str, Any]] = []
        for f in frames:
            t = f.get("best_effort_timestamp_time", f.get("pts_time"))
            try:
                ts = float(t) if t not in (None, "", "N/A") else None
            except (TypeError, ValueError):
                ts = None
            out.append(
                {
                    "pts_sec": ts,
                    "duration_sec": parse_fraction(f.get("pkt_duration_time")),
                    "key_frame": f.get("key_frame"),
                    "pict_type": f.get("pict_type"),
                }
            )
        return out, {}
    except Exception as exc:
        return [], {"error": f"cannot parse ffprobe frame JSON: {exc}", "stdout_head": stdout[:500]}


def pts_summary(pts: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(pts, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    d = np.diff(a)
    positive = d[d > 0]
    return {
        "count": int(a.size),
        "first_sec": float(a[0]),
        "last_sec": float(a[-1]),
        "span_sec": float(a[-1] - a[0]),
        "mean_dt_sec": float(positive.mean()) if positive.size else None,
        "median_dt_sec": float(np.median(positive)) if positive.size else None,
        "std_dt_sec": float(positive.std()) if positive.size else None,
        "min_dt_sec": float(positive.min()) if positive.size else None,
        "max_dt_sec": float(positive.max()) if positive.size else None,
        "implied_fps_from_median": float(1.0 / np.median(positive)) if positive.size else None,
        "non_increasing_steps": int(np.sum(d <= 0)),
        "equal_pts_steps": int(np.sum(d == 0)),
    }


def parse_calibration(path: Path, wanted_serials: Iterable[str]) -> Dict[str, Any]:
    """Parse the project's blank-line separated OpenCV calibration text file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: List[List[str]] = []
    cur: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            cur.append(s)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    cameras: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    for bi, block in enumerate(blocks):
        if len(block) < 8:
            skipped.append({"block": bi, "reason": "fewer than 8 non-empty lines"})
            continue
        serial = block[0]
        try:
            K = np.array(
                [[float(x) for x in block[1 + r].split()] for r in range(3)],
                dtype=np.float64,
            )
            dist = np.array([float(x) for x in block[4].split()], dtype=np.float64)
            W = np.eye(4, dtype=np.float64)
            for r in range(3):
                vals = [float(x) for x in block[5 + r].split()]
                if len(vals) < 3:
                    raise ValueError("W row has fewer than 3 values")
                W[r, : len(vals)] = vals
            if K.shape != (3, 3):
                raise ValueError(f"K shape {K.shape}")
        except Exception as exc:
            skipped.append({"block": bi, "serial": serial, "reason": str(exc)})
            continue
        cameras[serial] = {
            "K_native": K.tolist(),
            "distortion": dist.tolist(),
            "W": W.tolist(),
            "W_convention": (
                "OpenCV world-to-camera assumed: Xc = W[:3,:3] @ Xw + W[:3,3]; "
                "P_undistorted = K_undistorted @ W[:3,:]"
            ),
        }

    wanted = list(dict.fromkeys(str(s) for s in wanted_serials))
    matched = [s for s in wanted if s in cameras]
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "sha256": sha256_file(path),
        "blocks": len(blocks),
        "parsed_cameras": sorted(cameras),
        "matched_serials": matched,
        "missing_serials": [s for s in wanted if s not in cameras],
        "coverage": f"{len(matched)}/{len(wanted)}",
        "skipped_blocks": skipped,
        "cameras": cameras,
    }


def calibration_candidates(
    root: Path, explicit: Optional[str], manifest: Optional[Dict[str, Any]]
) -> List[Path]:
    out: List[Path] = []

    def add(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        if resolved not in out:
            out.append(resolved)

    if explicit:
        add(Path(explicit))
        if not Path(explicit).is_absolute():
            add(root / explicit)
    if manifest:
        raw = manifest.get("calibration")
        if raw:
            add(Path(raw))
            add(root / "data" / Path(raw).name)
            add(root / Path(raw))
    data_dir = root / "data"
    try:
        for p in sorted(data_dir.glob("*.txt")):
            add(p)
    except OSError:
        pass
    cal_dir = data_dir / "calibration"
    try:
        for p in sorted(cal_dir.glob("*.txt")):
            add(p)
    except OSError:
        pass
    return [p for p in out if p.is_file()]


def select_calibration(
    root: Path,
    serials: Sequence[str],
    explicit: Optional[str],
    manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    parsed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for path in calibration_candidates(root, explicit, manifest):
        try:
            parsed.append(parse_calibration(path, serials))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})

    manifest_sha = (manifest or {}).get("calibration_sha256")
    for item in parsed:
        item["sha256_matches_manifest"] = bool(
            manifest_sha and item.get("sha256") == manifest_sha
        )

    if explicit:
        chosen = next(
            (p for p in parsed if Path(p["path"]).resolve() == Path(explicit).resolve()),
            None,
        )
        if chosen is None:
            return {
                "selected": None,
                "explicit": str(explicit),
                "errors": errors,
                "reason": "explicit calibration path could not be parsed",
            }
    elif parsed:
        parsed.sort(
            key=lambda p: (len(p["matched_serials"]), bool(p.get("sha256_matches_manifest"))),
            reverse=True,
        )
        chosen = parsed[0]
    else:
        chosen = None

    return {
        "selected": chosen,
        "explicit": str(explicit) if explicit else None,
        "candidates": parsed,
        "errors": errors,
        "reason": None if chosen else "no parseable calibration text file found",
    }


def load_manifest(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    facts = file_facts(path)
    if not path.is_file():
        return None, facts
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        facts["sha256"] = sha256_file(path)
        facts["json_ok"] = True
        return obj, facts
    except Exception as exc:
        facts["json_ok"] = False
        facts["error"] = str(exc)
        return None, facts


# ---------------------------------------------------------------------------
# source discovery and sequential decode
# ---------------------------------------------------------------------------
def discover_sources(root: Path, session: str, envs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    sources: List[Dict[str, Any]] = []
    for env_rank, env in enumerate(envs):
        cam_dir = root / "data" / "videos" / env / Path(session)
        if not cam_dir.is_dir():
            warnings.append(f"missing camera directory: {cam_dir}")
            continue
        try:
            mp4s = sorted(
                p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
            )
        except OSError as exc:
            warnings.append(f"cannot list {cam_dir}: {exc}")
            continue
        if len(mp4s) != 4:
            warnings.append(f"{cam_dir} contains {len(mp4s)} mp4 files, expected 4")
        for p in mp4s:
            sources.append(
                {
                    "env": env,
                    "env_rank": env_rank,
                    "serial_from_filename": p.stem,
                    "path": p,
                    "relative_path": str(p.relative_to(root)),
                }
            )
    sources.sort(key=lambda s: (s["env_rank"], s["serial_from_filename"]))
    seen: Dict[str, str] = {}
    for s in sources:
        serial = s["serial_from_filename"]
        if serial in seen:
            warnings.append(
                f"duplicate serial {serial} in {seen[serial]} and {s['relative_path']}"
            )
        seen[serial] = s["relative_path"]
    for i, s in enumerate(sources):
        s["camera_index"] = i
    if len(sources) != 8:
        warnings.append(f"discovered {len(sources)} videos, expected exactly 8 (4+4)")
    return sources, warnings


def make_thumbnail_jpeg(frame: np.ndarray, size: Tuple[int, int] = (320, 240)) -> bytes:
    thumb = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return b""
    return buf.tobytes()


def decode_view(
    source: Dict[str, Any],
    out_dir: Path,
    ffprobe: str,
    hash_video: bool,
) -> Dict[str, Any]:
    path: Path = source["path"]
    serial = source["serial_from_filename"]
    result: Dict[str, Any] = {
        "camera_index": source["camera_index"],
        "env": source["env"],
        "serial": serial,
        "path": str(path),
        "relative_path": source["relative_path"],
        "file": file_facts(path),
        "errors": [],
    }
    if hash_video:
        result["file"]["sha256"] = sha256_file(path)

    stream, meta_err = ffprobe_stream(path, ffprobe)
    frame_entries, frames_err = ffprobe_frames(path, ffprobe)
    ff_pts_values: List[Optional[float]] = []
    for f in frame_entries:
        t = f.get("pts_sec")
        ff_pts_values.append(float(t) if t is not None else None)
    ff_pts_numeric = [x for x in ff_pts_values if x is not None]
    result["ffprobe"] = {
        "stream": stream,
        "stream_error": meta_err,
        "frame_entries": len(frame_entries),
        "frame_error": frames_err,
        "pts_summary": pts_summary(np.asarray(ff_pts_numeric, dtype=np.float64)),
    }
    if stream:
        result["ffprobe"]["width"] = stream.get("width")
        result["ffprobe"]["height"] = stream.get("height")
        result["ffprobe"]["fps_reported"] = parse_fraction(
            stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        )
        try:
            result["ffprobe"]["nb_frames_reported"] = int(stream.get("nb_frames"))
        except (TypeError, ValueError):
            result["ffprobe"]["nb_frames_reported"] = None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        result["errors"].append(f"cv2.VideoCapture cannot open {path}")
        result["decode"] = {
            "actual_frames": 0,
            "status": "open_failed",
            "frames_retained": 0,
        }
        return result

    declared = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_hashes: List[str] = []
    cv_pos_ms: List[float] = []
    motion_scores: List[float] = []
    thumbs: List[bytes] = []
    pts_seen: List[float] = []
    prev_gray_small: Optional[np.ndarray] = None
    prev_hash: Optional[str] = None

    frames_jsonl = out_dir / "frames" / f"{serial}.jsonl"
    frames_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = frames_jsonl.with_name(frames_jsonl.name + ".tmp")
    fh = tmp_jsonl.open("w", encoding="utf-8")
    duplicate_groups: Dict[str, List[int]] = {}
    duplicate_runs: List[Dict[str, int]] = []
    run_start: Optional[int] = None
    run_hash: Optional[str] = None
    width = height = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            idx = len(frame_hashes)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv_pos_ms.append(pos_ms)
            pts_seen.append(pos_ms / 1000.0)
            h = hashlib.blake2b(frame.tobytes(), digest_size=8).hexdigest()
            frame_hashes.append(h)
            duplicate_groups.setdefault(h, []).append(idx)

            if h == prev_hash:
                if run_start is None:
                    run_start = idx - 1
                    run_hash = h
            else:
                if run_start is not None:
                    duplicate_runs.append(
                        {"start": run_start, "end": idx - 1, "count": idx - run_start}
                    )
                run_start = None
                run_hash = None
            prev_hash = h

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            motion = 0.0
            if prev_gray_small is not None:
                motion = float(
                    np.mean(cv2.absdiff(small, prev_gray_small), dtype=np.float64)
                )
            prev_gray_small = small
            motion_scores.append(motion)
            thumbs.append(make_thumbnail_jpeg(frame))

            record = {
                "frame": idx,
                "ffprobe_pts_sec": ff_pts_values[idx] if idx < len(ff_pts_values) else None,
                "cv2_pos_msec": pos_ms,
                "hash8": h,
                "motion": round(motion, 6),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        result["errors"].append(f"decode failed near frame {len(frame_hashes)}: {exc}\n{traceback.format_exc(limit=3)}")
    finally:
        fh.close()
        cap.release()
        if run_start is not None:
            duplicate_runs.append(
                {"start": run_start, "end": len(frame_hashes) - 1, "count": len(frame_hashes) - run_start}
            )
        os.replace(tmp_jsonl, frames_jsonl)

    n = len(frame_hashes)
    groups = [idxs for idxs in duplicate_groups.values() if len(idxs) > 1]
    duplicate_frame_events = int(sum(len(g) - 1 for g in groups))
    ff_pts_arr = np.full(n, np.nan, dtype=np.float64)
    m = min(n, len(ff_pts_values))
    if m:
        ff_pts_arr[:m] = np.asarray(
            [np.nan if x is None else x for x in ff_pts_values[:m]], dtype=np.float64
        )
    cv_pos_arr = np.asarray(cv_pos_ms, dtype=np.float64)
    timeline_npz = out_dir / "timeline" / f"{serial}.npz"
    atomic_savez(
        timeline_npz,
        frame=np.arange(n, dtype=np.int32),
        ffprobe_pts_sec=ff_pts_arr,
        cv2_pos_msec=cv_pos_arr,
        motion=np.asarray(motion_scores, dtype=np.float64),
        hash8=np.asarray(frame_hashes, dtype="U16"),
    )

    # Exact global image duplicates are informative but never grounds for
    # dropping a frame: every PTS above remains in the timeline.
    if ff_pts_arr.size:
        cv2_vs_ff = cv_pos_arr / 1000.0 - ff_pts_arr
        finite = cv2_vs_ff[np.isfinite(cv2_vs_ff)]
        offset_median = float(np.median(finite)) if finite.size else None
    else:
        offset_median = None

    result["decode"] = {
        "status": "ok" if n > 0 else "no_frames",
        "actual_frames": n,
        "frames_retained": n,
        "declared_frames_cv2": declared,
        "declared_frames_ffprobe": result["ffprobe"].get("nb_frames_reported"),
        "ffprobe_frame_entries": len(ff_pts_values),
        "width": width,
        "height": height,
        "fps_cv2": cv_fps if math.isfinite(cv_fps) else None,
        "early_eof_vs_cv2_declared": bool(declared > 0 and n < declared),
        "count_matches_cv2_declared": bool(declared > 0 and n == declared),
        "count_matches_ffprobe_entries": bool(len(ff_pts_values) == n),
        "cv2_pos_minus_ffprobe_median_sec": offset_median,
        "pts_summary_cv2_pos": pts_summary(cv_pos_arr / 1000.0),
        "pts_summary_ffprobe": pts_summary(ff_pts_arr),
        "unique_decoded_images": len(duplicate_groups),
        "exact_duplicate_frame_events": duplicate_frame_events,
        "exact_duplicate_fraction": (
            float(duplicate_frame_events / n) if n else None
        ),
        "duplicate_groups": len(groups),
        "max_duplicate_group": max((len(g) for g in groups), default=1),
        "consecutive_duplicate_runs": duplicate_runs[:200],
        "consecutive_duplicate_runs_truncated": len(duplicate_runs) > 200,
        "duplicate_frames_retained": True,
    }
    result["timeline_npz"] = str(timeline_npz)
    result["frames_jsonl"] = str(frames_jsonl)
    result["_thumbs"] = thumbs
    result["_motion"] = motion_scores
    return result


# ---------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------
def font_or_default(size: int) -> Any:
    for candidate in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def decode_thumb(buf: bytes) -> Optional[np.ndarray]:
    if not buf:
        return None
    arr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def make_contact_sheet(
    path: Path,
    rows: List[Dict[str, Any]],
    views: List[Dict[str, Any]],
    thumbs_by_camera: Dict[int, List[bytes]],
    title: str,
) -> None:
    tile_w, tile_h, label_h, left_w, header_h = 320, 240, 20, 128, 34
    cols = len(views)
    width = left_w + cols * tile_w
    height = header_h + len(rows) * (tile_h + label_h)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    f_header = font_or_default(15)
    f_label = font_or_default(14)
    f_small = font_or_default(12)
    draw.text((6, 6), title, fill="black", font=f_header)
    for ci, view in enumerate(views):
        text = f"cam{view['camera_index']} {view['serial']}"
        draw.text((left_w + ci * tile_w + 4, 17), text, fill="black", font=f_small)

    for ri, row in enumerate(rows):
        y0 = header_h + ri * (tile_h + label_h)
        color = (150, 20, 20) if row.get("highlight") else (20, 20, 20)
        draw.text((6, y0 + 4), row["label"], fill=color, font=f_label)
        if row.get("detail"):
            draw.text((6, y0 + 22), row["detail"], fill=(80, 80, 80), font=f_small)
        for ci, view in enumerate(views):
            cam = view["camera_index"]
            frame = int(row["frames"][ci])
            thumbs = thumbs_by_camera.get(cam) or []
            buf = thumbs[frame] if 0 <= frame < len(thumbs) else b""
            rgb = decode_thumb(buf)
            x0 = left_w + ci * tile_w
            if rgb is None:
                tile = Image.new("RGB", (tile_w, tile_h), (235, 235, 235))
            else:
                tile = Image.fromarray(rgb).resize((tile_w, tile_h))
            sheet.paste(tile, (x0, y0))
            if row.get("highlight"):
                draw.rectangle(
                    [x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(200, 0, 0), width=3
                )
            caption = row.get("captions", {}).get(view["camera_index"])
            if caption:
                draw.rectangle([x0, y0 + tile_h - 18, x0 + tile_w, y0 + tile_h], fill=(0, 0, 0))
                draw.text((x0 + 4, y0 + tile_h - 17), caption, fill="white", font=f_small)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def choose_motion_window(
    views: List[Dict[str, Any]], min_frames: int
) -> Tuple[int, int, Dict[str, Any]]:
    sample_fps = None
    fps_values = [
        v["decode"].get("fps_cv2")
        for v in views
        if v.get("decode", {}).get("fps_cv2")
    ]
    if fps_values:
        sample_fps = float(np.median(fps_values))
    window = max(3, int(round(sample_fps))) if sample_fps else max(3, min_frames // 10)
    window = min(window, min_frames)
    if min_frames <= 1:
        return 0, min_frames, {"window": window, "scores": [], "source": "insufficient_frames"}

    motions = []
    for v in views:
        arr = v.get("_motion")
        if arr is not None and len(arr) >= min_frames:
            motions.append(np.asarray(arr[:min_frames], dtype=np.float64))
    if not motions:
        return 0, min_frames, {"window": window, "scores": [], "source": "no_motion"}

    mm = np.vstack(motions)
    per_frame = np.nanmean(mm, axis=0)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(per_frame, kernel, mode="valid")
    if smooth.size == 0:
        return 0, min_frames, {"window": window, "scores": [], "source": "no_motion"}
    start = int(np.argmax(smooth))
    end = min(min_frames, start + window)
    top = [
        {"start": int(i), "mean_motion": float(smooth[i])}
        for i in np.argsort(smooth)[-5:][::-1]
    ]
    return start, end, {
        "window": window,
        "source": "motion_candidate_unverified",
        "top_windows": top,
        "mean_motion_in_window": float(smooth[start]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_rows(
    views: List[Dict[str, Any]],
    interaction_start: int,
    interaction_end: int,
    interaction_count: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_labels = ("first", "middle", "last")
    for label in base_labels:
        frames = []
        captions = {}
        for v in views:
            n = v["decode"]["actual_frames"]
            if n <= 0:
                frames.append(0)
                captions[v["camera_index"]] = "NO FRAME"
                continue
            if label == "first":
                f = 0
            elif label == "middle":
                f = (n - 1) // 2
            else:
                f = n - 1
            frames.append(f)
            captions[v["camera_index"]] = f"f{f:04d}"
        rows.append({"label": label, "frames": frames, "captions": captions})

    if interaction_count > 0:
        targets = np.linspace(
            interaction_start,
            max(interaction_start, interaction_end - 1),
            interaction_count,
        )
        for k, target in enumerate(targets):
            frames = []
            captions = {}
            for v in views:
                n = v["decode"]["actual_frames"]
                f = int(round(float(target)))
                if n > 0:
                    f = max(0, min(n - 1, f))
                else:
                    f = 0
                frames.append(f)
                captions[v["camera_index"]] = f"f{f:04d}" if n > 0 else "NO FRAME"
            rows.append(
                {
                    "label": f"interaction {k + 1}/{interaction_count}",
                    "detail": f"target f{int(round(float(target)))}",
                    "frames": frames,
                    "captions": captions,
                    "highlight": True,
                }
            )
    return rows


def extract_review_frames(
    view: Dict[str, Any], frames: Sequence[int], out_dir: Path
) -> List[str]:
    path = Path(view["path"])
    serial = view["serial"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    targets = sorted({int(f) for f in frames})
    if not targets:
        cap.release()
        return []
    max_target = max(targets)
    wanted = set(targets)
    out_paths: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    try:
        while idx <= max_target:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx in wanted:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280.0 / w
                    frame = cv2.resize(
                        frame,
                        (1280, int(round(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                p = out_dir / f"{serial}_f{idx:04d}.jpg"
                cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                out_paths.append(str(p))
            idx += 1
    finally:
        cap.release()
    return out_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="project root")
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--calibration", default=None, help="explicit calibration .txt")
    ap.add_argument("--manifest", default=None, help="default: <root>/assets/capture/manifest.json")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--no-video-hash", action="store_true")
    ap.add_argument("--interaction-start", type=int, default=None)
    ap.add_argument("--interaction-end", type=int, default=None)
    ap.add_argument("--interaction-count", type=int, default=3)
    ap.add_argument("--no-review-frames", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    warnings: List[str] = []
    fatal: List[str] = []

    manifest_path = Path(args.manifest) if args.manifest else root / "assets" / "capture" / "manifest.json"
    manifest, manifest_facts = load_manifest(manifest_path)

    sources, discovery_warnings = discover_sources(root, args.session, args.envs)
    warnings.extend(discovery_warnings)
    if not sources:
        fatal.append(
            "no .mp4 sources found under "
            + " and ".join(str(root / "data" / "videos" / env / Path(args.session)) for env in args.envs)
        )

    serials = [s["serial_from_filename"] for s in sources]
    calib_choice = select_calibration(root, serials, args.calibration, manifest)
    calib = calib_choice.get("selected")
    if calib is None:
        warnings.append(f"calibration unavailable: {calib_choice.get('reason')}")
    elif calib["missing_serials"]:
        warnings.append(
            "calibration missing serials: " + ", ".join(calib["missing_serials"])
        )

    manifest_views: Dict[str, Dict[str, Any]] = {}
    manifest_sequence = None
    manifest_sequence_matches = None
    if manifest:
        manifest_sequence = manifest.get("sequence")
        manifest_sequence_matches = bool(
            manifest_sequence and str(manifest_sequence) == Path(args.session).name
        )
        for v in manifest.get("views") or []:
            if v.get("serial") is not None:
                manifest_views[str(v["serial"])] = v
        if not manifest_sequence_matches:
            warnings.append(
                f"assets/capture/manifest.json sequence={manifest_sequence!r} is not the requested "
                f"{Path(args.session).name!r}; its per-view K/W are comparison evidence only"
            )
        if (manifest.get("calibration") or "") and not Path(str(manifest["calibration"])).is_file():
            warnings.append(
                "manifest calibration path is stale on disk: "
                + str(manifest.get("calibration"))
            )

    views: List[Dict[str, Any]] = []
    for source in sources:
        print(
            f"[inventory] decoding cam{source['camera_index']} {source['serial_from_filename']} ...",
            flush=True,
        )
        v = decode_view(source, out_dir, args.ffprobe, not args.no_video_hash)
        views.append(v)
        dv = v.get("decode", {})
        print(
            f"[inventory] cam{source['camera_index']} frames={dv.get('actual_frames')} "
            f"size={dv.get('width')}x{dv.get('height')} "
            f"dups={dv.get('exact_duplicate_fraction')}",
            flush=True,
        )

    # Attach calibration to each view and compare with the old manifest.
    for v in views:
        serial = v["serial"]
        cal = (calib or {}).get("cameras", {}).get(serial)
        if not cal:
            v["calibration"] = {"available": False, "reason": "serial absent from selected calibration"}
            continue
        dec = v.get("decode", {})
        w = int(dec.get("width") or 0)
        h = int(dec.get("height") or 0)
        K = np.asarray(cal["K_native"], dtype=np.float64)
        dist = np.asarray(cal["distortion"], dtype=np.float64)
        W = np.asarray(cal["W"], dtype=np.float64)
        if w > 0 and h > 0:
            K_und, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        else:
            K_und = K.copy()
        v["calibration"] = {
            "available": True,
            "source": calib["path"],
            "source_sha256": calib.get("sha256"),
            "K_native": K.tolist(),
            "distortion_opencv5": dist.tolist(),
            "K_undistorted": np.asarray(K_und).tolist(),
            "W": W.tolist(),
            "W_convention": cal["W_convention"],
            "native_width": w,
            "native_height": h,
        }
        mview = manifest_views.get(serial)
        if mview:
            checks: Dict[str, Any] = {}
            for key, actual in (
                ("K_native", K),
                ("distortion_opencv5", dist),
                ("W", W),
            ):
                if key in mview:
                    ref = np.asarray(mview[key], dtype=np.float64)
                    if ref.shape == actual.shape:
                        checks[key] = {
                            "max_abs_diff": float(np.max(np.abs(ref - actual))),
                            "matches": bool(np.allclose(ref, actual, rtol=0, atol=1e-9)),
                        }
                    else:
                        checks[key] = {"shape_ref": list(ref.shape), "shape_actual": list(actual.shape)}
            if "K_undistorted_native" in mview:
                ref = np.asarray(mview["K_undistorted_native"], dtype=np.float64)
                checks["K_undistorted_native"] = {
                    "max_abs_diff": float(np.max(np.abs(ref - K_und))),
                    "matches": bool(np.allclose(ref, K_und, rtol=0, atol=1e-6)),
                }
            v["calibration"]["manifest_comparison"] = checks
            v["calibration"]["manifest_source"] = mview.get("source")

    decoded = [v for v in views if v.get("decode", {}).get("actual_frames", 0) > 0]
    min_frames = min((v["decode"]["actual_frames"] for v in decoded), default=0)
    if min_frames <= 0:
        fatal.append("no view decoded any frame")

    # Interaction selection: user interval wins; otherwise a motion candidate.
    user_window = args.interaction_start is not None or args.interaction_end is not None
    if user_window:
        if args.interaction_start is None or args.interaction_end is None:
            fatal.append("--interaction-start and --interaction-end must be supplied together")
            istart, iend = 0, max(1, min_frames)
            iw = {"source": "invalid_user_args"}
        else:
            istart = int(args.interaction_start)
            iend = int(args.interaction_end)
            if not (0 <= istart < iend <= min_frames):
                fatal.append(
                    f"interaction window [{istart},{iend}) outside common decoded range [0,{min_frames})"
                )
            istart = max(0, min(istart, max(0, min_frames - 1)))
            iend = max(istart + 1, min(iend, min_frames))
            iw = {"source": "user_supplied", "start": istart, "end": iend}
    else:
        istart, iend, iw = choose_motion_window(decoded, min_frames)
        warnings.append(
            "interaction interval is an unverified motion-energy candidate; "
            "Codex must confirm it visually or pass --interaction-start/--interaction-end"
        )
    iw.update({"start": int(istart), "end": int(iend), "frames_used": int(iend - istart)})

    # Contact sheets.
    views_sorted = sorted(views, key=lambda v: v["camera_index"])
    rows = build_rows(views_sorted, istart, iend, max(0, args.interaction_count))
    thumbs_by_camera = {v["camera_index"]: v.pop("_thumbs", []) for v in views_sorted}
    title = (
        f"{Path(args.session).name} | all 8 native streams | "
        f"interaction f{istart}-f{iend - 1} ({iw['source']})"
    )
    sheet_path = out_dir / "contact_sheet.jpg"
    make_contact_sheet(sheet_path, rows, views_sorted, thumbs_by_camera, title)
    inter_rows = [r for r in rows if r.get("highlight")]
    inter_sheet = out_dir / "contact_sheet_interaction.jpg"
    make_contact_sheet(
        inter_sheet,
        inter_rows,
        views_sorted,
        thumbs_by_camera,
        f"{Path(args.session).name} | interaction candidate f{istart}-f{iend - 1}",
    )

    review_paths: List[str] = []
    if not args.no_review_frames and views_sorted:
        cam0 = views_sorted[0]
        picks = [0, max(0, min_frames - 1), min_frames // 2]
        picks.extend(
            int(round(x)) for x in np.linspace(
                istart, max(istart, iend - 1), max(0, args.interaction_count)
            )
        )
        review_paths = extract_review_frames(
            cam0, sorted(set(picks)), out_dir / "review_frames"
        )

    # Consistency facts that do not assume the people are present.
    resolutions = {(v["decode"].get("width"), v["decode"].get("height")) for v in decoded}
    counts = {v["decode"].get("actual_frames") for v in decoded}
    fps_values = {
        round(float(v["decode"]["fps_cv2"]), 6)
        for v in decoded
        if v["decode"].get("fps_cv2")
    }
    cal_matched = len((calib or {}).get("matched_serials", []))
    all_ok = (
        len(views) == 8
        and len(decoded) == 8
        and cal_matched == 8
        and all(v["decode"].get("actual_frames", 0) > 0 for v in views)
    )
    status = "complete" if all_ok else "partial"

    if len(counts) > 1:
        warnings.append(f"actual frame counts differ across cameras: {sorted(counts)}")
    if len(resolutions) > 1:
        warnings.append(f"native resolutions differ across cameras: {sorted(resolutions)}")
    if any((v.get("decode") or {}).get("exact_duplicate_frame_events", 0) for v in views):
        warnings.append(
            "exact duplicate decoded images exist; all frames and their distinct PTS were retained"
        )
    if any(
        (v.get("decode") or {}).get("actual_frames") != (v.get("decode") or {}).get("declared_frames_cv2")
        for v in views
    ):
        warnings.append("decoded frame count differs from cv2 container-declared count; decode count is authoritative")
    warnings.append(
        "two-person presence and interaction are NOT certified by this inventory; local visual review is pending"
    )
    for v in views:
        v.pop("_thumbs", None)
        v.pop("_motion", None)

    scene = {
        "schema": "fcs_two_person_scene_inventory/v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "elapsed_sec": round(time.time() - started, 3),
        "project_root": str(root),
        "output_dir": str(out_dir),
        "session": {
            "requested": args.session,
            "name": Path(args.session).name,
            "envs": list(args.envs),
            "expected_views": 8,
            "discovered_views": len(views),
        },
        "camera_order_note": "camera_index follows env order then filename serial; it is not a canonical rig id",
        "views": views,
        "calibration_selection": {
            "explicit": calib_choice.get("explicit"),
            "reason": calib_choice.get("reason"),
            "errors": calib_choice.get("errors"),
            "candidates": [
                {
                    "path": c["path"],
                    "sha256": c.get("sha256"),
                    "coverage": c.get("coverage"),
                    "matched_serials": c.get("matched_serials"),
                    "missing_serials": c.get("missing_serials"),
                    "sha256_matches_manifest": c.get("sha256_matches_manifest"),
                }
                for c in calib_choice.get("candidates", [])
            ],
            "selected": (
                {
                    "path": calib["path"],
                    "sha256": calib.get("sha256"),
                    "coverage": calib.get("coverage"),
                    "matched_serials": calib.get("matched_serials"),
                    "missing_serials": calib.get("missing_serials"),
                    "sha256_matches_manifest": calib.get("sha256_matches_manifest"),
                    "W_convention": next(
                        (c["W_convention"] for c in calib["cameras"].values()), None
                    ),
                }
                if calib
                else None
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "file": manifest_facts,
            "sequence": manifest_sequence,
            "sequence_matches_requested_session": manifest_sequence_matches,
            "view_serials": sorted(manifest_views),
            "requested_serials_present": sorted(set(serials) & set(manifest_views)),
            "calibration": (manifest or {}).get("calibration"),
            "calibration_sha256": (manifest or {}).get("calibration_sha256"),
            "world_normalization_note": (
                "manifest world_normalization belongs to the FCS006 single-person snapshot; "
                "it is not applied here and does not define the 008 second identity"
            ),
        },
        "scene_consistency": {
            "all_eight_videos_discovered": len(views) == 8,
            "all_eight_decoded": len(decoded) == 8,
            "native_resolutions": sorted([list(x) for x in resolutions]),
            "resolution_uniform": len(resolutions) == 1,
            "actual_frame_counts": {
                v["serial"]: v["decode"].get("actual_frames") for v in views
            },
            "frame_count_uniform": len(counts) == 1,
            "cv2_fps_values": sorted(fps_values),
            "calibration_serial_coverage": f"{cal_matched}/{len(serials)}",
            "same_session_dir_under_both_envs": not any(
                "missing camera directory" in w for w in discovery_warnings
            ),
            "same_calibration_file_for_all_views": bool(
                calib and len(calib.get("matched_serials", [])) == len(serials)
            ),
            "hardware_synchronization_certified": False,
            "synchronization_note": (
                "Same-index/PTS comparison only; equality does not prove hardware sync. "
                "All camera timelines are retained independently."
            ),
        },
        "two_person_verification": {
            "visual_two_person_confirmation": "pending_codex_local_review",
            "per_instance_keypoints_present": False,
            "second_identity_source": "not built by this script",
            "note": (
                "This script inventories the scene only. It does not run a person detector, "
                "does not assign identities, and must not be read as a completed two-person reconstruction."
            ),
        },
        "interaction_window": iw,
        "contact_sheets": {
            "all_views_0_mid_last_and_interaction": str(sheet_path),
            "interaction_only": str(inter_sheet),
            "review_frames_camera0": review_paths,
        },
        "dependencies": dependency_report(),
        "warnings": warnings,
        "fatal": fatal,
        "notes": [
            "Actual frame counts come from sequential cv2 decode, not from container metadata.",
            "ffprobe frame timestamps are stored per camera; cv2 position is cross-checked, not substituted.",
            "Exact duplicate images are counted but never removed; distinct PTS are preserved.",
            "No input asset, manifest, calibration file, or git-tracked file is modified.",
        ],
    }
    atomic_write_json(out_dir / "scene.json", scene)
    print(f"[inventory] wrote {out_dir / 'scene.json'}")
    print(f"[inventory] contact sheet: {sheet_path}")
    if warnings:
        print("[inventory] warnings:")
        for w in warnings:
            print("  -", w)
    if fatal:
        print("[inventory] fatal:")
        for f in fatal:
            print("  -", f)
        return 2
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
