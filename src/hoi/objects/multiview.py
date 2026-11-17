"""Prepare real multiview images and retain their measured media clocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hoi.capture.scene import prepare_capture
from hoi.io import write_json
from hoi.observations.segmentation import extract_frames, timeline_pts
from hoi.runtime import PROJECT_ROOT, runtime_settings


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def prepare(config_path: str | Path) -> Path:
    """Replace this stage's own input cache, without changing the source capture."""
    config_path = resolve(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output = resolve(data["output"])
    preparation = dict(data["preparation"])
    start, end = int(preparation.get("start", 0)), int(preparation["end"])
    if start < 0 or end <= start:
        raise ValueError("preparation requires 0 <= start < end")
    if preparation.get("videos_root"):
        preparation["videos_root"] = str(resolve(preparation["videos_root"]))
    capture_config = SimpleNamespace(
        scene_path=resolve(data["scene_path"]),
        preparation=preparation,
        runtime=runtime_settings(data.get("runtime")),
    )
    camera_path = prepare_capture(capture_config, output / "capture")
    capture = json.loads(camera_path.read_text(encoding="utf-8"))
    views = capture["views"]
    serials = [str(view["serial"]) for view in views]
    if not views or len(set(serials)) != len(serials):
        raise ValueError("Capture must contain uniquely identified camera views")

    clocks, positive_steps = [], []
    for view in views:
        serial = str(view["serial"])
        if end > int(view["decode"]["actual_frames"]):
            raise ValueError(f"{serial}: requested frames exceed the decoded video")
        pts, tags = timeline_pts(view)
        selected = np.asarray(pts[start:end], dtype=np.float64)
        if selected.shape != (end - start,) or len(tags) < end:
            raise ValueError(f"{serial}: timeline does not cover requested frames")
        steps = np.diff(selected)
        if not np.isfinite(selected).all() or np.any(steps <= 0):
            raise ValueError(f"{serial}: PTS must be finite and strictly increasing")
        if steps.size:
            positive_steps.append(steps)
        clocks.append((view, pts, tags, selected))
    if not positive_steps:
        raise ValueError("Two selected timestamps are needed to measure a time step")
    dt = float(np.median(np.concatenate(positive_steps)))
    origin = min(float(selected[0]) for _, _, _, selected in clocks)
    end_exclusive = max(float(selected[-1]) for _, _, _, selected in clocks) + dt
    duration = end_exclusive - origin

    prepared_views = []
    for view, pts, tags, _ in clocks:
        serial = str(view["serial"])
        frames_dir = output / "input/images" / serial
        frames = extract_frames(view, frames_dir, start, end, pts, tags)
        for frame in frames:
            source_frame = int(frame["source_frame"])
            time_sec = float(pts[source_frame])
            frame.update(
                jpeg=str(Path(frame["jpeg"]).resolve()),
                time_normalized=(time_sec - origin) / duration,
            )
        video = Path(view["path"])
        stat = video.stat()
        prepared_views.append(
            {
                "serial": serial,
                "source_video": str(video),
                "source_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "calibration": view["calibration"],
                "decode": view["decode"],
                "timeline_npz": view["timeline_npz"],
                "selected_frame_count": len(frames),
                "frames": frames,
            }
        )
        print(f"[objects prepare] {serial}: {len(frames)} measured source frames", flush=True)

    destination = output / "input/scene.json"
    write_json(
        destination,
        {
            "schema": "hoi.multiview/v1",
            "id": data["id"],
            "config_path": str(config_path),
            "source_scene_path": str(capture_config.scene_path),
            "capture_path": str(camera_path),
            "calibration_selection": capture.get("calibration_selection", {}),
            "image_space": "undistorted_opencv_pixel_centers",
            "frame_range": [start, end],
            "selected_frame_count_per_view": end - start,
            "selected_image_count": sum(v["selected_frame_count"] for v in prepared_views),
            "time": {
                "source": "media_pts",
                "origin_sec": origin,
                "end_exclusive_sec": end_exclusive,
                "median_positive_dt_sec": dt,
                "normalization": "(pts_sec-origin_sec)/(end_exclusive_sec-origin_sec)",
                "physical_exposure_synchronization_verified": False,
            },
            "views": prepared_views,
        },
    )
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
