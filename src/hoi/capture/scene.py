"""Adapt capture manifests into the shared video-observation input format."""

import json
from collections import deque
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from hoi.io import write_json
from hoi.runtime import PROJECT_ROOT, runtime_settings


def read_timeline(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read measured timestamps without deriving time from a nominal FPS."""
    with np.load(path, allow_pickle=False) as data:
        if "media_pts_sec" in data:
            pts = np.asarray(data["media_pts_sec"], np.float64)
            sources = np.asarray(data["pts_source"], dtype=str)
            if sources.ndim == 0:
                sources = np.full(pts.shape, str(sources), dtype=sources.dtype)
            if pts.ndim != 1 or sources.shape != pts.shape:
                raise ValueError(f"{path}: timestamps and sources must be matching 1D arrays")
            return pts, [str(tag) if np.isfinite(p) else "unknown" for p, tag in zip(pts, sources)]
        ff = np.asarray(data.get("ffprobe_pts_sec", []), np.float64)
        cv = np.asarray(data.get("cv2_pos_msec", []), np.float64) / 1000.0
        if ff.ndim != 1 or cv.ndim != 1:
            raise ValueError(f"{path}: legacy timestamps must be 1D arrays")
        if "ffprobe_pts_sec" not in data and "cv2_pos_msec" not in data:
            raise ValueError(f"{path}: no supported timestamp arrays")
        pts = np.full(max(ff.size, cv.size), np.nan, np.float64)
        tags = ["unknown"] * len(pts)
        for i in range(len(pts)):
            if i < ff.size and np.isfinite(ff[i]):
                pts[i], tags[i] = ff[i], "ffprobe_pts"
            elif i < cv.size and np.isfinite(cv[i]):
                pts[i], tags[i] = cv[i], "cv2_pos_msec"
        return pts, tags


def _probe_video_timeline(video: Path, runtime: dict) -> tuple[np.ndarray, list[str]]:
    configured = runtime_settings(runtime).get("ffmpeg")
    executable = (
        str(configured) if configured and Path(configured).is_file() else shutil.which("ffmpeg")
    )
    if not executable:
        raise FileNotFoundError("Configure ffmpeg in the shared runtime settings or PATH")
    command = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-threads",
        "1",
        "-copyts",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "showinfo",
        "-vsync",
        "0",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = deque(maxlen=20)
    pts, tags = [], []
    time_base = None
    try:
        for line in process.stderr:
            tail.append(line.rstrip()[-2000:])
            if "showinfo" not in line:
                continue
            base = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
            if base:
                numerator, denominator = map(int, base.groups())
                if numerator <= 0 or denominator <= 0:
                    raise RuntimeError(f"{video}: invalid decoder time base")
                time_base = (numerator, denominator)
            frame = re.search(r"\bn:\s*(\d+)\s+pts:\s*(\S+)", line)
            if not frame:
                if re.search(r"\bn:\s*\d+", line):
                    raise RuntimeError(f"{video}: cannot parse decoder frame timestamp")
                continue
            if int(frame[1]) != len(pts) or time_base is None:
                raise RuntimeError(f"{video}: invalid decoder frame sequence or missing time base")
            raw = frame[2]
            if raw == "NOPTS":
                pts.append(np.nan)
                tags.append("unknown")
            else:
                pts.append(int(raw) * time_base[0] / time_base[1])
                tags.append("ffmpeg_pts")
        returncode = process.wait()
        if returncode or not pts:
            raise RuntimeError(
                f"{video}: ffmpeg exited {returncode}, decoded {len(pts)} frames\n"
                + "\n".join(tail)
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        process.stderr.close()
    return np.asarray(pts, np.float64), tags


def _write_timeline(path: Path, pts: np.ndarray, tags: list[str], **metadata) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, media_pts_sec=pts, pts_source=np.asarray(tags), **metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _video_timeline(
    video: Path, path: Path, runtime: dict, cache_path: Path | None = None
) -> tuple[np.ndarray, list[str]]:
    stat = video.stat()
    signature = {
        "source_video": str(video.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }
    cache_path = path if cache_path is None else cache_path
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                matches = all(
                    key in cache and cache[key].item() == value for key, value in signature.items()
                )
            if matches:
                pts, tags = read_timeline(cache_path)
                if len(pts):
                    if cache_path != path:
                        _write_timeline(path, pts, tags, **signature)
                    return pts, tags
        except (OSError, ValueError, KeyError, EOFError):
            pass
    pts, tags = _probe_video_timeline(video, runtime)
    _write_timeline(path, pts, tags, **signature)
    return pts, tags


def prepare_capture(config, output: Path) -> Path:
    scene = json.loads(config.scene_path.read_text(encoding="utf-8-sig"))
    sequence = scene.get("sequence") or scene.get("session", {}).get("name")
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for index, view in enumerate(scene["views"]):
        serial = str(view["serial"])
        camera = dict(view.get("calibration", view))
        decode = dict(view.get("decode", {}))
        camera["K_undistorted"] = camera.get("K_undistorted", camera.get("K_undistorted_native"))
        camera["available"] = True
        width = camera.get("native_width", decode.get("width"))
        height = camera.get("native_height", decode.get("height"))
        decode.update(width=width, height=height)
        video = Path(view.get("path", view.get("source", "")))
        if not video.is_absolute():
            video = PROJECT_ROOT / video
        if not video.is_file():
            video_root = Path(config.preparation.get("videos_root", "data/videos"))
            if not video_root.is_absolute():
                video_root = PROJECT_ROOT / video_root
            candidates = list(video_root.glob(f"e_*/*/{sequence}/{serial}.mp4"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Provide the video path for camera {serial}: {video}")
            video = candidates[0]
        timing_path = view.get("timeline_npz")
        if timing_path and not Path(timing_path).is_absolute():
            timing_path = str(config.scene_path.parent / timing_path)
        if timing_path and Path(timing_path).is_file():
            explicit_path = Path(timing_path)
            with np.load(explicit_path, allow_pickle=False) as timing:
                generated = "source_video" in timing
            if generated:
                timing_path = str(output / f"{serial}.npz")
                points, _ = _video_timeline(
                    video, Path(timing_path), config.runtime, cache_path=explicit_path
                )
            else:
                points, _ = read_timeline(explicit_path)
        elif view.get("timeline"):
            timing = view["timeline"]
            points = np.asarray([entry["media_pts_ms"] for entry in timing], np.float64) / 1000.0
            timing_path = str(output / f"{serial}.npz")
            tags = ["manifest_pts" if np.isfinite(p) else "unknown" for p in points]
            _write_timeline(Path(timing_path), points, tags)
        else:
            timing_path = str(output / f"{serial}.npz")
            points, _ = _video_timeline(video, Path(timing_path), config.runtime)
        if not len(points):
            raise ValueError(f"{serial}: timeline contains no decoded frames")
        decode["actual_frames"] = len(points)
        views.append(
            {
                "camera_index": index,
                "serial": serial,
                "path": str(video),
                "calibration": camera,
                "decode": decode,
                "timeline_npz": timing_path,
            }
        )
    destination = output / "cameras.json"
    write_json(
        destination,
        {
            "session": {"name": sequence},
            "views": views,
            "calibration_selection": scene.get("calibration_selection", {}),
        },
    )
    return destination
