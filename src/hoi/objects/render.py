"""Render the complete native three-branch checkpoint without loading Adam on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from hoi.io import write_json
from hoi.objects.multi4d import build_scene_info, parameters
from hoi.objects.multiview import resolve
from hoi.runtime import PROJECT_ROOT, runtime_settings


def load_models(author, payload, hidden):
    """Restore only the inference fields of the pinned author's capture formats."""
    import torch

    def tensor(value):
        value = value.detach().to("cuda")
        return torch.nn.Parameter(value, requires_grad=False)

    fg = author.GaussianModel_dynamic(3, hidden)
    state = payload["fg"]
    (
        fg.active_sh_degree,
        xyz,
        deformation,
        deformation_table,
        dc,
        rest,
        scaling,
        rotation,
        opacity,
        _radii,
        _gradient,
        _denom,
        _optimizer,
        fg.spatial_lr_scale,
    ) = state
    fg._deformation.load_state_dict(deformation, strict=True)
    fg._deformation.to("cuda").eval().requires_grad_(False)
    fg._deformation_table = deformation_table.detach().to("cuda")
    for name, value in zip(
        ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"),
        (xyz, dc, rest, scaling, rotation, opacity),
    ):
        setattr(fg, name, tensor(value))

    bg = author.GaussianModel(3)
    (
        bg.active_sh_degree,
        xyz,
        dc,
        rest,
        scaling,
        rotation,
        opacity,
        _radii,
        _gradient,
        _denom,
        _optimizer,
        bg.spatial_lr_scale,
    ) = payload["bg"]
    for name, value in zip(
        ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"),
        (xyz, dc, rest, scaling, rotation, opacity),
    ):
        setattr(bg, name, tensor(value))

    state = payload["tr"]
    constants = state["constants"]
    tr = author.GaussianModelTransient(
        constants["max_sh_degree"],
        gaussian_dim=constants["gaussian_dim"],
        time_duration=constants["time_duration"],
        rot_4d=constants["rot_4d"],
        sh_degree_t=constants["max_sh_degree_t"],
    )
    for name, value in constants.items():
        setattr(tr, name, value)
    for name, value in state["parameters"].items():
        setattr(tr, name, tensor(value))
    return fg, bg, tr


def midpoint_camera(camera_infos, frame_index, first, second):
    """Interpolate calibrated camera centers and rotations in the shared gauge."""
    from scipy.spatial.transform import Rotation, Slerp

    a = camera_infos.camera(frame_index, first)
    b = camera_infos.camera(frame_index, second, load_image=False)
    center = (-a.R @ a.T - b.R @ b.T) / 2
    r = Slerp([0, 1], Rotation.from_matrix(np.stack([a.R, b.R])))(0.5).as_matrix()
    return a._replace(R=r, T=-r.T @ center, image_name="midpoint", image_path=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--serial", default="all")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--novel", action="store_true")
    args = parser.parse_args(argv)
    author, config, options, groups = parameters(args.config)
    import torch
    from PIL import Image
    from scene.cameras import Camera
    from hoi.objects.checkpoint import Checkpoint
    from hoi.rendering.scene import write_video

    torch.set_num_threads(2)
    author.setup_seed(int(options["seed"]))
    output = resolve(config["output"])
    index = output / "input/scene.json"
    data = json.loads(index.read_text(encoding="utf-8"))
    info, gauge = build_scene_info(index, options)
    resolved = {
        key: vars(value)
        for key, value in zip(("dataset", "hidden", "optimization", "pipeline"), groups)
    }
    resolved.update(method=options, gauge=gauge)
    sources = json.loads((PROJECT_ROOT / "third_party/sources.json").read_text(encoding="utf-8"))
    implementation = (
        "third_party/multi4d.patch",
        "src/hoi/objects/multi4d.py",
        "src/hoi/objects/checkpoint.py",
    )
    signature = {
        "input": hashlib.sha256(index.read_bytes()).hexdigest(),
        "point_cloud": hashlib.sha256((index.parent / "fused.ply").read_bytes()).hexdigest(),
        "parameters": resolved,
        "author": sources["multi4d"]["revision"],
        "implementation": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in implementation
        },
    }
    checkpoint = Checkpoint(output / "model/checkpoint.pt", signature)
    payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False, mmap=True)
    if payload["metadata"] != checkpoint.metadata:
        raise ValueError("Checkpoint changed during render initialization")
    fg, bg, tr = load_models(author, payload, groups[1])
    del payload
    cameras = info.train_cameras
    frames = [int(frame["source_frame"]) for frame in cameras.views[0]["frames"]]
    selected = args.frames or [frames[0], frames[len(frames) // 2], frames[-1]]
    if set(selected) - set(frames):
        raise ValueError("Requested source frames are absent from this input")
    serials = [str(view["serial"]) for view in cameras.views]
    views = (
        list(enumerate(serials))
        if args.serial == "all"
        else [(serials.index(args.serial), args.serial)]
    )
    if args.novel:
        centers = np.stack(
            [
                -np.asarray(v["calibration"]["W"])[:3, :3].T
                @ np.asarray(v["calibration"]["W"])[:3, 3]
                for v in cameras.views
            ]
        )
        first = views[0][0]
        distances = np.linalg.norm(centers - centers[first], axis=1)
        distances[first] = np.inf
        second = int(np.argmin(distances))
        views.append(((first, second), f"midpoint_{serials[first]}_{serials[second]}"))
    root = output / "render"
    root.mkdir(parents=True, exist_ok=True)
    intervals = np.diff([frame["pts_sec"] for frame in cameras.views[0]["frames"]])
    fps = 1.0 / float(np.median(intervals[intervals > 0])) if len(intervals) else 1.0
    background = torch.zeros(3, device="cuda")
    for view_index, serial in views:
        destination = root / f"camera_{serial}"
        image_dir = destination / "frames"
        image_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".video-", dir=destination) as temporary:
            temporary = Path(temporary)
            for frame_index, frame in enumerate(frames):
                if not args.video and frame not in selected:
                    continue
                c = (
                    midpoint_camera(cameras, frame_index, *view_index)
                    if isinstance(view_index, tuple)
                    else cameras.camera(frame_index, view_index)
                )
                camera = Camera(
                    R=c.R,
                    T=c.T,
                    FoVx=c.FovX,
                    FoVy=c.FovY,
                    image=c.image,
                    image_name=c.image_name,
                    time=c.time,
                    cxr=c.cxr,
                    cyr=c.cyr,
                )
                with torch.no_grad():
                    result = author.render_rawall3pc(
                        camera,
                        fg,
                        bg,
                        tr,
                        groups[3],
                        background,
                        stage=checkpoint.metadata["stage"],
                        dropout_use=False,
                    )
                    rgb = (result["render"].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
                del result
                image = Image.fromarray(rgb)
                if frame in selected:
                    image.save(image_dir / f"{frame:04d}.png")
                if args.video:
                    image.save(temporary / f"{frame:04d}.png")
            if args.video:
                ffmpeg = Path(runtime_settings()["ffmpeg"])
                write_video(
                    ffmpeg,
                    destination / "preview.mp4",
                    frames,
                    temporary,
                    camera.image_width,
                    camera.image_height,
                    fps,
                )
        print(
            f"[objects render] {serial}; frames={len(frames) if args.video else len(selected)}",
            flush=True,
        )
    write_json(
        root / "report.json",
        {
            "checkpoint": str(checkpoint.path),
            "state": checkpoint.metadata,
            "source_frames": frames,
            "keyframes": selected,
            "views": [name for _, name in views],
            "video": args.video,
            "scope": "Native FG+BG+TR inference; no Adam on CUDA; midpoint is a qualitative novel view without held-out truth",
            "quality_claim": None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
