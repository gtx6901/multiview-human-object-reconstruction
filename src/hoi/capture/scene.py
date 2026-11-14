"""Adapt capture manifests into the shared video-observation input format."""

import json
from pathlib import Path

import numpy as np

from hoi.io import write_json
from hoi.runtime import PROJECT_ROOT


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
        decode["actual_frames"] = decode.get("actual_frames", view.get("frames"))
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
        if not timing_path and view.get("timeline"):
            timing = view["timeline"]
            points = np.asarray([entry["media_pts_ms"] for entry in timing], np.float64)
            timing_path = str(output / f"{serial}.npz")
            np.savez_compressed(timing_path, ffprobe_pts_sec=points / 1000.0, cv2_pos_msec=points)
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
