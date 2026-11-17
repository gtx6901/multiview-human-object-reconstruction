"""Calibrated image and training boundary for the pinned Multi4D implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from hoi.io import write_json
from hoi.objects.multiview import resolve
from hoi.runtime import PROJECT_ROOT, configure_runtime


def load_author():
    cache = PROJECT_ROOT / "runtime/soft_objects"
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["TORCH_HOME"] = str(PROJECT_ROOT / "runtime/clothing/torch")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache / "build")
    configure_runtime()
    paths = [
        cache / "python",
        cache / "extensions",
        cache / "extensions/hybrid",
        PROJECT_ROOT / "third_party/multi4d",
    ]
    sys.path[:0] = [str(p) for p in paths]
    return importlib.import_module("train")


class LazyCameraInfos:
    """Read one RGB image on demand in stable frame-major, view-minor order."""

    def __init__(self, views, origin, scale, max_image_size):
        self.views, self.origin, self.scale = views, np.asarray(origin), float(scale)
        self.max_image_size = int(max_image_size)
        self.frame_count = len(views[0]["frames"])
        if any(len(v["frames"]) != self.frame_count for v in views):
            raise ValueError("Views must contain the same selected frame count")

    def __len__(self):
        return self.frame_count * len(self.views)

    def camera(self, frame_index, view_index, load_image=True):
        import torch
        from scene.dataset_readers import CameraInfo
        from utils.graphics_utils import focal2fov

        view = self.views[view_index]
        frame = view["frames"][frame_index]
        width, height = int(view["decode"]["width"]), int(view["decode"]["height"])
        ratio = min(1.0, self.max_image_size / max(width, height))
        new_width, new_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        sx, sy = new_width / width, new_height / height
        k, w = np.asarray(view["calibration"]["K_undistorted"]), np.asarray(
            view["calibration"]["W"]
        )
        cx, cy = sx * (k[0, 2] + 0.5) - 0.5, sy * (k[1, 2] + 0.5) - 0.5
        image = None
        if load_image:
            bgr = cv2.imdecode(np.fromfile(frame["jpeg"], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (height, width):
                raise ValueError(f"Image does not match the calibrated dimensions: {frame['jpeg']}")
            if (new_width, new_height) != (width, height):
                bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        return CameraInfo(
            R=w[:3, :3].T.copy(),
            T=(w[:3, :3] @ self.origin + w[:3, 3]) * self.scale,
            FovX=focal2fov(k[0, 0] * sx, new_width),
            FovY=focal2fov(k[1, 1] * sy, new_height),
            image=image,
            image_path=frame["jpeg"],
            image_name=f"{view['serial']}/{Path(frame['jpeg']).stem}",
            width=new_width,
            height=new_height,
            time=float(frame["time_normalized"]),
            cxr=(cx + 0.5) / new_width - 0.5,
            cyr=(cy + 0.5) / new_height - 0.5,
        )

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.camera(*divmod(index, len(self.views)))


def build_scene_info(index: Path, options: dict):
    from scene.dataset_readers import SceneInfo, fetchPly, getNerfppNorm
    from utils.graphics_utils import BasicPointCloud
    from utils.sh_utils import SH2RGB

    data = json.loads(index.read_text(encoding="utf-8"))
    views = data["views"]
    matrices = [np.asarray(v["calibration"]["W"]) for v in views]
    centers = np.stack([-w[:3, :3].T @ w[:3, 3] for w in matrices])
    origin = centers.mean(0)
    radius = np.linalg.norm(centers - origin, axis=1).max()
    scale = float(options["camera_radius"] / radius)
    cameras = LazyCameraInfos(views, origin, scale, options["max_image_size"])
    normalization = getNerfppNorm([cameras.camera(0, i, False) for i in range(len(views))])
    source = fetchPly(str(index.parent / "fused.ply"))
    points = (np.asarray(source.points, np.float64) - origin) * scale
    background = BasicPointCloud(
        points=points.astype(np.float32), colors=source.colors, normals=source.normals
    )
    lo, hi = points.min(0), points.max(0)
    count = int(options["foreground_points"])
    xyz = np.hstack([np.random.random((count, 1)) * (hi[i] - lo[i]) + lo[i] for i in range(3)])
    sh = np.random.random((count, 3))
    foreground = BasicPointCloud(
        points=xyz.astype(np.float32), colors=SH2RGB(sh), normals=np.zeros((count, 3))
    )
    info = SceneInfo(
        point_cloud=foreground,
        point_cloud_second=background,
        train_cameras=cameras,
        test_cameras=[],
        nerf_normalization=normalization,
    )
    return info, {
        "origin": origin.tolist(),
        "scale": scale,
        "forward": "X_normalized=(X_source-origin)*scale",
        "inverse": "X_source=X_normalized/scale+origin",
        "source_ply": str(index.parent / "fused.ply"),
    }


def parameters(config_path: Path):
    author = load_author()
    import mmcv
    from arguments import ModelParams, ModelHiddenParams, OptimizationParams, PipelineParams
    from utils.params_utils import merge_hparams

    config = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    options = json.loads(resolve(config["method_config"]).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    groups = [
        ModelParams(parser),
        ModelHiddenParams(parser),
        OptimizationParams(parser),
        PipelineParams(parser),
    ]
    args = parser.parse_args([])
    args = merge_hparams(args, mmcv.Config.fromfile(str(resolve(options["author_config"]))))
    for key, value in options["overrides"].items():
        setattr(args, key, value)
    output = resolve(config["output"])
    args.source_path, args.model_path = str(output / "input"), str(output / "model")
    args.saving_folder = str(output / "render")
    author.args = args
    return author, config, options, [g.extract(args) for g in groups]


def train(config_path: Path) -> int:
    author, config, options, groups = parameters(config_path)
    import torch
    from hoi.objects.checkpoint import Checkpoint

    output = resolve(config["output"])
    index = output / "input/scene.json"
    author.setup_seed(int(options["seed"]))
    info, gauge = build_scene_info(index, options)
    names = ["dataset", "hidden", "optimization", "pipeline"]
    resolved = {key: vars(value) for key, value in zip(names, groups)}
    resolved.update(method=options, gauge=gauge)
    write_json(output / "parameters.json", resolved)
    sources = json.loads((PROJECT_ROOT / "third_party/sources.json").read_text(encoding="utf-8"))
    signature = {
        "input": hashlib.sha256(index.read_bytes()).hexdigest(),
        "point_cloud": hashlib.sha256((index.parent / "fused.ply").read_bytes()).hexdigest(),
        "parameters": resolved,
        "author": sources["multi4d"]["revision"],
        "implementation": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "third_party/multi4d.patch",
                "src/hoi/objects/multi4d.py",
                "src/hoi/objects/checkpoint.py",
            )
        },
    }
    checkpoint = Checkpoint(
        output / "model/checkpoint.pt", signature, options["checkpoint_interval"]
    )
    report = {
        "status": "running",
        "checkpoint": str(checkpoint.path),
        "quality_claim": None,
        "scope": "full native coarse/fine recipe; no intermediate PSNR selection",
    }
    write_json(output / "training.json", report)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        dataset, hidden, optimization, pipeline = groups
        author.training(
            dataset,
            hidden,
            optimization,
            pipeline,
            [],
            [],
            None,
            config["id"],
            scene_info=info,
            checkpoint_manager=checkpoint,
        )
        report["status"] = "completed"
        return 0
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted_or_failed"
        report.update(
            elapsed_sec=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            saved_state=checkpoint.metadata,
        )
        write_json(output / "training.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    raise SystemExit(train(parser.parse_args().config))
