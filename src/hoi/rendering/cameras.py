"""Generate a reproducible orbit from the scene's actual motion and camera gauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoi.io import write_json
from hoi.rendering.scene import (
    load_scene_config,
    resolve_path,
    select_frames,
    select_render_frames,
    world_to_unit,
)
from hoi.runtime import PROJECT_ROOT


def orbit_cameras(frames, points, world_up, reference_eye, width, height, options):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    up = np.asarray(world_up, dtype=np.float64)
    if not len(points) or not np.isfinite(points).all() or np.linalg.norm(up) == 0:
        raise ValueError("orbit needs finite scene points and a nonzero world_up")
    up /= np.linalg.norm(up)
    origin = points.mean(axis=0)
    vertical = (points - origin) @ up
    target = origin + up * (vertical.min() + vertical.max()) / 2
    offset = points - target
    vertical = offset @ up
    horizontal = offset - np.outer(vertical, up)
    horizontal_radius = np.linalg.norm(horizontal, axis=1).max()
    vertical_radius = np.abs(vertical).max()
    fov = np.deg2rad(float(options.get("vertical_fov_degrees", 40)))
    margin = float(options.get("margin", 1.15))
    elevation = np.deg2rad(float(options.get("elevation_degrees", 8)))
    turns = float(options.get("turns", 1))
    if not 0 < fov < np.pi or margin <= 0 or not np.isfinite([fov, margin, elevation, turns]).all():
        raise ValueError("invalid orbit field of view, margin, elevation or turns")
    focal = height / (2 * np.tan(fov / 2))
    horizontal_tangent = width / (2 * focal)
    distance = margin * (
        max(vertical_radius / np.tan(fov / 2), horizontal_radius / horizontal_tangent)
        + horizontal_radius
    )
    radial = np.asarray(reference_eye, dtype=np.float64) - target
    radial -= up * (radial @ up)
    if np.linalg.norm(radial) == 0 or distance <= 0:
        raise ValueError("reference camera cannot define an orbit around the scene")
    radial /= np.linalg.norm(radial)
    tangent = np.cross(up, radial)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    span = max(1, frames[-1] - frames[0] + 1)
    matrices = []
    for frame in frames:
        angle = 2 * np.pi * turns * (frame - frames[0]) / span
        direction = np.cos(angle) * radial + np.sin(angle) * tangent
        eye = target + distance * (np.cos(elevation) * direction + np.sin(elevation) * up)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        W = np.eye(4)
        W[:3, :3] = np.stack([right, down, forward])
        W[:3, 3] = -W[:3, :3] @ eye
        matrices.append(W.tolist())
    return {
        "width": width,
        "height": height,
        "camera_space": "normalized",
        "frames": frames,
        "views": [{"label": "orbit", "K": K.tolist(), "W": matrices, "frames": frames}],
        "orbit": {
            "target": target.tolist(),
            "distance": float(distance),
            "world_up": up.tolist(),
            "vertical_fov_degrees": float(np.rad2deg(fov)),
            "elevation_degrees": float(np.rad2deg(elevation)),
            "margin": margin,
            "turns": turns,
            "framing": "joint bounds of the selected actual source poses; no motion interpolation",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--pose-frame", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    options = config["rendering"]["orbit"]
    pose_frame = args.pose_frame
    if pose_frame is None:
        pose_frame = config["rendering"].get("pose_frame")
    import torch

    motions = [
        torch.load(
            resolve_path(actor["motion"], PROJECT_ROOT), map_location="cpu", weights_only=False
        )
        for actor in config["actors"]
    ]
    if pose_frame is None:
        frames, indices = select_frames(motions, None)
    else:
        count = int(options.get("frame_count", 300))
        if count <= 0:
            raise ValueError("orbit.frame_count must be positive")
        frames, indices = select_render_frames(motions, [f"0:{count - 1}"], [], pose_frame)
    points = []
    for motion, index in zip(motions, indices):
        source_frames = frames if pose_frame is None else [pose_frame]
        points.append(torch.as_tensor(motion["joints55_world"])[[index[f] for f in source_frames]])
    points = torch.cat(points).double().numpy()
    scene, center, body_height, _ = load_scene_config(config, PROJECT_ROOT)
    reference = options.get("reference_serial", str(scene["views"][0]["serial"]))
    view = next(v for v in scene["views"] if str(v["serial"]) == str(reference))
    W = np.asarray(view.get("calibration", view)["W"], dtype=np.float64)
    if config.get("camera_space", "raw") == "raw":
        W = world_to_unit(W, center, body_height)
    eye = -W[:3, :3].T @ W[:3, 3]
    payload = orbit_cameras(
        frames, points, config["world_up"], eye, args.width, args.height, options
    )
    payload["orbit"].update(
        {"reference_serial": str(reference), "pose_frame": pose_frame, "config": str(args.config)}
    )
    write_json(args.out, payload)
    print(f"[cameras] wrote {len(frames)} exact output poses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
