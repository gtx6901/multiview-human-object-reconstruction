#!/usr/bin/env python
"""GaussianWardrobe reconstruction on explicitly bound native multiview data.

The native reader preserves the author SMPL-X, camera and layer contracts.
Production commands generate position maps and training masks, pretrain,
train, or render. All commands take explicit --data and --out paths; the
configuration selects native frames and cameras. This module does not
reconstruct a source template or provide independent diagnostic commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import random
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hoi.runtime import PROJECT_ROOT as PROJECT, runtime_settings
from hoi.clothing.prepare_inputs import expression_dim

HERE = Path(__file__).resolve().parent
CLOTHING_RUNTIME = PROJECT / "runtime/clothing"
AUTHOR = PROJECT / "third_party/gaussian_wardrobe"
CONFIG = PROJECT / "configs/clothing/mjj001.yaml"
DATA = PROJECT / "assets/scenes/mjj001/clothing"
RESULTS = PROJECT / "runs/mjj001/clothing"
TRAIN_CURRENT = RESULTS
TRAIN_CKPT = RESULTS / "checkpoint.pt"
LHM_ROOT = Path(runtime_settings()["lhm_root"])
LOCAL_SMPLX = LHM_ROOT / "pretrained_models/human_model_files/smplx"
LOCAL_DEPS = [CLOTHING_RUNTIME / "python", LHM_ROOT / "deps", LHM_ROOT]
RUNTIME_EXT = CLOTHING_RUNTIME / "extensions"
RUNTIME_WARDROBE = RUNTIME_EXT / "wardrobe"
RUNTIME_STYLE = RUNTIME_EXT / "styleunet"
LPIPS_CANDIDATES = [CLOTHING_RUNTIME / "weights/vgg.pth"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _find_lpips() -> Optional[Path]:
    for p in LPIPS_CANDIDATES:
        if p.is_file():
            return p
    return None


def _vgg_cache() -> Path:
    from torch.hub import get_dir

    return Path(get_dir()) / "checkpoints/vgg16-397923af.pth"


def _ffmpeg_exe() -> str:
    import shutil

    configured = runtime_settings().get("ffmpeg")
    if configured and Path(configured).is_file():
        return str(configured)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("Set ffmpeg in the shared runtime configuration or add it to PATH")


def _install_unicode_cv() -> Dict[str, Any]:
    # OpenEXR's memory decoder stages through a native temporary filename.
    # A relative ASCII path avoids the non-ASCII Windows user TEMP directory.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("OPENCV_TEMP_PATH", ".")
    import cv2 as cv

    if getattr(cv, "_wardrobe_unicode_io", False):
        return {"installed": True, "already": True}
    orig_read, orig_write = cv.imread, cv.imwrite

    def imread(path, flags=cv.IMREAD_COLOR):
        if not isinstance(path, (str, os.PathLike)):
            return orig_read(path, flags)
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv.imdecode(data, flags)
        except Exception:
            return orig_read(path, flags)

    def imwrite(path, image, params=None):
        if not isinstance(path, (str, os.PathLike)):
            return orig_write(path, image, params)
        ext = Path(path).suffix or ".png"
        if ext.lower() == ".exr":
            # OpenCV's EXR encoder has no in-memory writer; its imencode path
            # creates a temp file under the Unicode TEMP path and fails.  The
            # verified direct encoder accepts the Unicode destination.
            return orig_write(str(path), image, [] if params is None else params)
        ok, buf = cv.imencode(ext, image, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True

    cv.imread = imread
    cv.imwrite = imwrite
    cv._wardrobe_unicode_io = True
    return {"installed": True, "already": False}


def _ensure_wardrobe_rasterizer() -> None:
    """Require the isolated Wardrobe six-channel extension, never AG's 3ch pyd."""
    pkg = RUNTIME_WARDROBE / "diff_gaussian_rasterization_depth_alpha"
    pyds = sorted(pkg.glob("_C*.pyd"))
    if not pyds:
        raise FileNotFoundError(
            "Wardrobe six-channel rasterizer is not built at "
            f"{pkg}; run scripts\\build_clothing.ps1"
        )
    if not (pkg / "__init__.py").is_file():
        raise FileNotFoundError(f"Wardrobe rasterizer package init missing: {pkg}")
    mod = sys.modules.get("diff_gaussian_rasterization_depth_alpha")
    if mod is not None:
        c_file = getattr(mod, "_C", None)
        c_path = Path(c_file.__file__).resolve() if c_file is not None else None
        if c_path is None or RUNTIME_WARDROBE.resolve() not in c_path.parents:
            raise RuntimeError(
                "AG rasterizer was imported before Wardrobe bootstrap; "
                "start a fresh process and load runtime/extensions/wardrobe"
            )
        return
    mod = importlib.import_module("diff_gaussian_rasterization_depth_alpha")
    c_file = getattr(mod, "_C", None)
    c_path = Path(c_file.__file__).resolve() if c_file is not None else None
    if c_path is None or RUNTIME_WARDROBE.resolve() not in c_path.parents:
        raise RuntimeError(f"loaded rasterizer is not isolated Wardrobe extension: {c_path}")


def bootstrap(
    config_path: Path, device: str = "cpu", data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Author imports, local SMPL-X with input expression dimensions, local extensions."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    data_dir = Path(data_dir).resolve() if data_dir is not None else DATA
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    expr_dim = expression_dim(data_dir)

    for p in [RUNTIME_STYLE, RUNTIME_EXT, *LOCAL_DEPS, AUTHOR, RUNTIME_WARDROBE]:
        if p.is_dir():
            sys.path.insert(0, str(p))
    sys.path.insert(0, str(RUNTIME_WARDROBE))
    sys.path.insert(0, str(AUTHOR))  # author packages win
    if str(device).startswith("cuda"):
        _ensure_wardrobe_rasterizer()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(CLOTHING_RUNTIME / "torch_extensions"))
    os.environ.setdefault("TORCH_HOME", str(CLOTHING_RUNTIME / "torch"))
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    io = _install_unicode_cv()

    import torch
    import yaml

    # Must be set before importing author config: config.py builds
    # SURFACE_LABEL_COLOR on this device at import time.
    os.environ["WARDROBE_DEVICE"] = str(device)
    import config as author_config

    if AUTHOR not in Path(author_config.__file__).resolve().parents:
        raise RuntimeError(f"wrong Wardrobe config module: {author_config.__file__}")
    opt = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict):
        raise TypeError("config is not a mapping")
    for section in ("train", "test"):
        if isinstance(opt.get(section, {}).get("data"), dict):
            opt[section]["data"]["data_dir"] = str(data_dir)
    opt["train"]["net_ckpt_dir"] = str(TRAIN_CURRENT)
    author_config.opt = opt
    author_config.PROJ_DIR = str(AUTHOR)
    author_config.device = torch.device(device)
    os.chdir(AUTHOR)

    import smplx
    import smplxd

    for mod in (smplx, smplxd):
        if AUTHOR not in Path(mod.__file__).resolve().parents:
            raise RuntimeError(f"wrong SMPL-X module: {mod.__file__}")
        orig = mod.SMPLX

        class Patched(orig):
            def __init__(self, *args, **kwargs):
                if args:
                    args = (str(LOCAL_SMPLX),) + tuple(args[1:])
                else:
                    kwargs["model_path"] = str(LOCAL_SMPLX)
                kwargs["num_expression_coeffs"] = expr_dim
                super().__init__(*args, **kwargs)
                if self.num_expression_coeffs != expr_dim:
                    raise ValueError(
                        f"expression has {expr_dim} coefficients, but this SMPL-X model "
                        f"provides {self.num_expression_coeffs}; dimensions must match"
                    )

        mod.SMPLX = Patched

    import utils.smpl_util as smpl_util

    if AUTHOR not in Path(smpl_util.__file__).resolve().parents:
        raise RuntimeError(f"wrong smpl_util: {smpl_util.__file__}")

    def native_smplx_model(dataset_dir):
        # Native inputs use full 45-D hand poses; expression width comes from data.
        # The canonical and fitted roles are read-only
        # forwards over identical parameters, so one module owns both.
        model = smplx.SMPLX(use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1)
        return model, model

    smpl_util.smplx_model = native_smplx_model
    smpl_util.smpl_model = smplx.SMPLX(
        use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1
    )
    return {
        "torch": torch,
        "config": author_config,
        "opt": opt,
        "data_dir": data_dir,
        "expr_dim": expr_dim,
        "config_path": config_path,
        "author": str(AUTHOR),
        "smplx_module": str(Path(smplx.__file__).resolve()),
        "smplxd_module": str(Path(smplxd.__file__).resolve()),
        "smplx_model_dir": str(LOCAL_SMPLX),
        "unicode_io": io,
        "device": str(author_config.device),
    }


def native_layout(
    data: Path, config_path: Path, data_options=None, check_images=True
) -> Dict[str, Any]:
    """Resolve configured native indices from explicit binding metadata."""
    import yaml

    data = Path(data).resolve()
    binding_path, source_path = (
        data / "binding_manifest.json",
        data / "source_frames.json",
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_frames = [int(frame) for frame in source["source_frames"]]
    if not source_frames:
        raise ValueError("source_frames metadata must be nonempty")
    if int(source["T"]) != len(source_frames) or int(binding["T"]) != len(source_frames):
        raise ValueError("binding/source native timestep counts disagree")
    if [int(frame) for frame in binding["source_frames"]] != source_frames:
        raise ValueError("binding and source frame mappings disagree")
    # These are the binding's historical capture references, not inferred
    # sequential-video frame IDs. Every on-disk lookup uses native pose i.
    pose_to_source = {i: frame for i, frame in enumerate(source_frames)}
    for metadata in (binding, source):
        mapping = metadata.get("pose_to_source_frame")
        if mapping is not None and {int(k): int(v) for k, v in mapping.items()} != pose_to_source:
            raise ValueError("explicit native-to-source frame mapping disagrees")
    with (data / "cameras.pkl").open("rb") as stream:
        cameras = pickle.load(stream)
    camera_order = list(cameras)
    if not camera_order or list(binding["camera_order"]) != camera_order:
        raise ValueError("binding camera order differs from cameras.pkl")
    if data_options is None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        data_options = config["train"]["data"]
    selection = data_options.get("frame_range")
    if selection is None:
        poses = list(pose_to_source)
    elif isinstance(selection, list) and len(selection) in (2, 3):
        if len(selection) == 3 and int(selection[2]) <= 0:
            raise ValueError("native frame step must be positive")
        poses = list(range(*[int(value) for value in selection]))
    elif isinstance(selection, str):
        poses = np.loadtxt(data / selection, dtype=int).reshape(-1).tolist()
    else:
        raise ValueError("unsupported configured native frame_range")
    if (
        not poses
        or len(poses) != len(set(poses))
        or any(pose not in pose_to_source for pose in poses)
    ):
        raise ValueError("configured native poses must uniquely index source_frames")
    configured_cameras = data_options.get("used_cam_ids")
    camera_ids = (
        list(range(len(camera_order)))
        if configured_cameras is None
        else [int(value) for value in configured_cameras]
    )
    if (
        not camera_ids
        or len(camera_ids) != len(set(camera_ids))
        or any(i < 0 or i >= len(camera_order) for i in camera_ids)
    ):
        raise ValueError("configured camera IDs must uniquely index cameras.pkl")
    fps = float(source["fps"])
    if not np.isfinite(fps) or fps <= 0 or ("fps" in binding and float(binding["fps"]) != fps):
        raise ValueError("binding/source fps must agree and be positive")
    strides = np.diff(poses)
    stride = int(strides[0]) if len(strides) else 1
    video_fps = fps / stride if stride > 0 and bool(np.all(strides == stride)) else None
    sizes = {}
    for camera_id in camera_ids:
        name = camera_order[camera_id]
        width, height = [int(value) for value in cameras[name]["imgSize"]]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid native dimensions for {name}")
        if check_images:
            from PIL import Image

            image_path = next(
                (
                    data / "images" / name / f"{poses[0]:08d}{ext}"
                    for ext in (".jpg", ".png")
                    if (data / "images" / name / f"{poses[0]:08d}{ext}").is_file()
                ),
                None,
            )
            if image_path is None:
                raise FileNotFoundError(f"native image missing for {name}, pose {poses[0]}")
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise ValueError(
                        f"native image dimensions differ from cameras.pkl: {image_path}"
                    )
        sizes[camera_id] = (width, height)
    return {
        "poses": poses,
        "camera_ids": camera_ids,
        "camera_order": camera_order,
        "camera_names": [camera_order[i] for i in camera_ids],
        "camera_serials": binding.get("camera_serials", {}),
        "sizes": sizes,
        "pose_to_source": pose_to_source,
        "source_frames": source_frames,
        "fps": fps,
        "video_fps": video_fps,
        "native_total": len(source_frames),
        "expected_items": len(poses) * len(camera_ids),
        "binding_path": binding_path,
        "source_path": source_path,
    }


def make_native_reader():
    """Return an adapter subclassing the actual Wardrobe reader classes."""
    from dataset.dataset_mv_rgb import MvRgbDataset4DDress

    class MvRgbDatasetNative(MvRgbDataset4DDress):
        def load_cam_data(self):
            p = Path(self.data_dir) / "cameras.pkl"
            if not p.is_file():
                raise FileNotFoundError(p)
            with p.open("rb") as fp:
                cameras = pickle.load(fp)
            self.cam_names = list(cameras.keys())
            self.view_num = len(self.cam_names)
            self.extr_mats = []
            self.intr_mats = []
            self.img_widths = []
            self.img_heights = []
            for name in self.cam_names:
                c = cameras[name]
                e = np.eye(4, dtype=np.float32)
                e[:3, :] = np.asarray(c["extrinsics"], np.float32)
                self.extr_mats.append(e)
                self.intr_mats.append(np.asarray(c["intrinsics"], np.float32))
                self.img_widths.append(int(c["imgSize"][0]))
                self.img_heights.append(int(c["imgSize"][1]))

        def _native_file(self, folder, pose_idx, view_idx, exts=(".jpg", ".png")):
            cam = self.cam_names[view_idx]
            for ext in exts:
                p = Path(self.data_dir) / folder / cam / f"{pose_idx:08d}{ext}"
                if p.is_file():
                    return p
            return None

        def load_color_mask_images(self, pose_idx, view_idx):
            import cv2 as cv

            ci = self._native_file("images", pose_idx, view_idx)
            mi = self._native_file("masks", pose_idx, view_idx)
            if ci is None or mi is None:
                return None, None
            color = cv.imread(str(ci), cv.IMREAD_UNCHANGED)
            mask = cv.imread(str(mi), cv.IMREAD_UNCHANGED)
            if mask is not None and mask.ndim == 3:
                mask = mask[:, :, 0]
            return color, mask

        def load_image(self, pose_idx, view_idx, folder_name):
            import cv2 as cv

            p = self._native_file(folder_name, pose_idx, view_idx)
            if p is None:
                return None
            # The generated body prior is binary and identical in B, G, and R.
            # Read it as one channel; the native label loss broadcasts this
            # singleton channel across its three rendered label channels.
            flag = cv.IMREAD_GRAYSCALE if folder_name == "body_masks" else cv.IMREAD_UNCHANGED
            return cv.imread(str(p), flag)

        @staticmethod
        def get_boundary_mask(mask, kernel_size=3):
            """Author-equivalent boundary masks with bounded CPU temporaries."""
            import cv2 as cv

            original = mask.copy()
            threshold_lut = np.zeros((256, 1), dtype=np.uint8)
            threshold_lut[128, 0] = 128
            threshold_lut[129:, 0] = 1
            mask = cv.LUT(mask, threshold_lut)

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded = cv.erode(mask, kernel)
            dilated = cv.dilate(mask, kernel)
            cv.subtract(dilated, eroded, dst=dilated)
            cv.compare(dilated, 1, cv.CMP_EQ, dst=dilated)

            support = cv.inRange(original, 6, 249)
            cv.bitwise_or(dilated, support, dst=dilated)
            np.right_shift(dilated, 7, out=dilated)

            binary_lut = np.zeros((256, 1), dtype=np.uint8)
            binary_lut[1, 0] = 1
            mask = cv.LUT(mask, binary_lut)
            return dilated.view(np.bool_), mask.view(np.bool_)

        def getitem(self, index, training=True, **kwargs):
            item = super().getitem(index, training=training, **kwargs)
            if not (getattr(self, "use_label", False) and training):
                return item
            if training or kwargs.get("eval", False):
                pose_idx, view_idx = self.data_list[index]
                pose_idx = kwargs.get("pose_idx", pose_idx)
                view_idx = kwargs.get("view_idx", view_idx)
            else:
                pose_idx, view_idx = self.pose_list[index], None
            if view_idx is None:
                return item
            import cv2 as cv
            import torch

            sem_p = self._native_file("semantic_valid", pose_idx, view_idx, (".png",))
            if sem_p is None:
                raise FileNotFoundError(
                    f"semantic_valid missing for pose {pose_idx}, view {view_idx}; "
                    "refusing to treat unknown observations as background"
                )
            sem = cv.imread(str(sem_p), cv.IMREAD_UNCHANGED)
            if sem is None:
                raise FileNotFoundError(sem_p)
            if sem.ndim == 3:
                sem = sem[:, :, 0]
            instance_mask = np.asarray(item["mask_img"]).astype(bool)
            if sem.shape != instance_mask.shape:
                raise ValueError(
                    f"semantic_valid shape {sem.shape} != mask shape {instance_mask.shape}"
                )
            # Outside the instance is true background-valid; inside the instance
            # only observed non-unknown semantics are valid for label supervision.
            label_valid = np.logical_or(~instance_mask, sem > 0)
            item["label_valid_img"] = torch.from_numpy(label_valid)
            return item

    return MvRgbDatasetNative


def install_native_reader():
    cls = make_native_reader()
    import dataset.dataset_mv_rgb as ds

    ds.MvRgbDatasetNative = cls
    ds.MvRgbDatasetMJJNative = cls  # Legacy name in the unchanged checkpoint YAML.
    return cls


def _position_map_inputs(data: Path) -> Dict[str, Path]:
    return {
        "template_body_offset.ply": data / "template_body_offset.ply",
        "template_body_offset.pkl": data / "template_body_offset.pkl",
        "template_cloth_offset.ply": data / "template_cloth_offset.ply",
        "template_cloth_offset.pkl": data / "template_cloth_offset.pkl",
        "cano_weight_volume.npz": data / "cano_weight_volume.npz",
        "cano_weight_volume_shape_zero.npz": data / "cano_weight_volume_shape_zero.npz",
        "smpl_params.npz": data / "smpl_params.npz",
        "cameras.pkl": data / "cameras.pkl",
    }


def _expected_position_map_outputs(
    data: Path, layer: str, config_path: Path = CONFIG
) -> List[Path]:
    base = data / f"smpl_pos_map_offset_{layer}"
    paths = [
        base / "cano_smpl_pos_map.exr",
        base / "cano_smpl_nml_map.exr",
        base / "cano_smpl_offset_map.exr",
        base / "neighbor_idx.npy",
        base / "neighbor_weights.npy",
        base / "with_neighbor.npy",
        base / "valid_faces.npy",
        base / "with_face.npy",
        base / "init_pts_lbs.npy",
    ]
    if layer == "body":
        paths.append(base / "cano_smpl_smplx_fix_offset_map.exr")
    elif layer == "cloth":
        paths.append(base / "cano_smpl_segment_map.exr")
    else:
        raise ValueError(layer)
    paths.extend(
        base / f"{pose_idx:08d}.exr"
        for pose_idx in native_layout(data, config_path, check_images=False)["poses"]
    )
    return paths


def generate_position_maps(config_path: Path, layer_choice: str) -> int:
    """Run pinned author gen_pos_maps with back-face-ID alignment correction."""
    data = DATA.resolve()
    layers = ["body", "cloth"] if layer_choice == "both" else [layer_choice]
    inputs = _position_map_inputs(data)
    missing_inputs = [name for name, p in inputs.items() if not p.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "position-map production requires actual canonical inputs; "
            f"missing: {missing_inputs}; refusing author SMPL fallback"
        )

    old_argv = list(sys.argv)
    old_cwd = os.getcwd()
    old_path = list(sys.path)
    calls: List[Dict[str, Any]] = []
    try:
        ctx = bootstrap(config_path, device="cuda", data_dir=data)
        install_native_reader()
        source = AUTHOR / "gen_data/gen_pos_maps.py"
        for layer in layers:
            template_ply = data / f"template_{layer}_offset.ply"
            template_pkl = data / f"template_{layer}_offset.pkl"
            if not template_ply.is_file() or not template_pkl.is_file():
                raise FileNotFoundError(
                    f"template_{layer}_offset.ply/.pkl missing at invocation; "
                    "author would silently fall back to bare SMPL-X"
                )
            argv = [
                "gen_data/gen_pos_maps.py",
                "-c",
                str(config_path),
                "-d",
                str(data),
                "-o",
                f"smpl_pos_map_offset_{layer}",
                "-t",
                f"template_{layer}_offset",
                "-rc",
                "-ro",
                "-lw",
                "cano_weight_volume_shape_zero",
            ]
            calls.append({"layer": layer, "argv": argv})
            sys.argv[:] = argv
            os.chdir(AUTHOR)
            runpy.run_module("gen_data.gen_pos_maps", run_name="__main__")
    finally:
        sys.argv[:] = old_argv
        os.chdir(old_cwd)
        sys.path[:] = old_path

    outputs: Dict[str, Any] = {}
    missing: List[Dict[str, Any]] = []
    core_missing: List[str] = []
    mask_missing: List[str] = []
    for layer in layers:
        expected = _expected_position_map_outputs(data, layer, config_path)
        present = [str(p) for p in expected if p.is_file()]
        layer_missing = [str(p) for p in expected if not p.is_file()]
        core_missing.extend(layer_missing)
        outputs[layer] = {
            "directory": str(data / f"smpl_pos_map_offset_{layer}"),
            "expected_count": len(expected),
            "present_count": len(present),
            "pose_map_count": sum(
                1 for p in expected if p.suffix == ".exr" and p.name[:8].isdigit() and p.is_file()
            ),
            "missing": layer_missing,
        }
        for p in layer_missing:
            missing.append({"path": p, "reason": "author gen_pos_maps output missing"})

    body_dir = data / "smpl_pos_map_offset_body"
    hand_map = body_dir / "cano_smpl_hand_map.exr"
    manifest_path = data / "layer_geometry_manifest.json"
    prior_manifest = {}
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hand_training_record = prior_manifest.get("training_masks")
    hand_present = hand_map.is_file()
    hand_status = {
        "path": str(hand_map),
        "present": hand_present,
        "source": (
            "existing file / training_masks"
            if hand_present
            else "missing; use --gen-training-masks"
        ),
        "training_masks": hand_training_record,
    }
    if "body" in layers and not hand_present:
        mask_missing.append(str(hand_map))
        missing.append(
            {
                "path": str(hand_map),
                "reason": "body hand map missing; run --gen-training-masks",
            }
        )
    external_mask = data / "smplx_masks/cano_smpl_smplx_map.exr"
    external_mask_present = external_mask.is_file()
    # CombinedNetwork only; current MultiLAvatarNet neither reads nor requires it.

    model_file = Path(ctx["smplx_model_dir"]) / "SMPLX_NEUTRAL.npz"
    source = AUTHOR / "gen_data/gen_pos_maps.py"
    info = {
        "status": (
            "pos_maps_incomplete"
            if core_missing
            else ("pos_maps_generated_with_mask_gaps" if mask_missing else "pos_maps_generated")
        ),
        "generated_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "data": str(data),
        "layers": layers,
        "calls": calls,
        "call_count": len(calls),
        "runpy_module": "gen_data.gen_pos_maps",
        "runpy_run_name": "__main__",
        "source": str(source),
        "source_sha256": sha256(source),
        "integration_sha256": sha256(Path(__file__).resolve()),
        "model_file": str(model_file),
        "model_sha256": sha256(model_file) if model_file.is_file() else None,
        "input_sha256": {name: sha256(p) for name, p in inputs.items() if p.is_file()},
        "outputs": outputs,
        "core_missing": core_missing,
        "mask_missing": mask_missing,
        "missing": missing,
        "hand_status": hand_status,
        "external_smplx_mask": {
            "path": str(external_mask),
            "present": external_mask_present,
            "required_for_current_multil": False,
            "role": "read only by unused CombinedNetwork; not read by MultiLAvatarNet",
            "generated": False,
            "sha256": sha256(external_mask) if external_mask_present else None,
        },
        "mask_requirements": {
            "body": {
                "cano_smpl_smplx_fix_offset_map.exr": "generated",
                "cano_smpl_hand_map.exr": (
                    "present" if hand_present else "missing; --gen-training-masks"
                ),
            },
            "cloth": {"cano_smpl_segment_map.exr": "generated"},
        },
        "semantic_masks_verified": False,
        "no_fake_masks": True,
        "no_mask_substitution": True,
        "no_training": True,
        "quality_claim": None,
        "notes": "Actual author gen_pos_maps calls only; no fallback/copy/renderer rewrite. "
        "Position maps are not training or quality evidence.",
    }
    manifest = data / "layer_geometry_manifest.json"
    existing = {}
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    existing.setdefault("schema", "gaussian_wardrobe_layer_geometry/v1")
    existing.update(
        {
            "generated_at_utc": utc_now(),
            "position_maps": info,
        }
    )
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest)
    print(
        json.dumps(
            {
                "status": info["status"],
                "calls": len(calls),
                "present": {k: v["present_count"] for k, v in outputs.items()},
                "core_missing": len(core_missing),
                "mask_missing": mask_missing,
            },
            indent=2,
        )
    )
    return 0 if not core_missing else 2


def generate_training_masks(data: Path, config_path: Path = CONFIG) -> int:
    """Generate body hand/fix, canonical cloth segment, and corrected adjacency."""
    _install_unicode_cv()
    import cv2 as cv
    import torch

    ctx = bootstrap(config_path, device="cuda", data_dir=data)
    import trimesh
    import smplx
    from utils.renderer.renderer_pytorch3d import Renderer
    from gen_data.gen_pos_maps import get_neighbors

    cfg = ctx["config"]
    body_dir = data / "smpl_pos_map_offset_body"
    cloth_dir = data / "smpl_pos_map_offset_cloth"
    body_pos_path = body_dir / "cano_smpl_pos_map.exr"
    cloth_pos_path = cloth_dir / "cano_smpl_pos_map.exr"
    body_lbs_path = body_dir / "init_pts_lbs.npy"
    body_template_path = data / "template_body_offset.ply"
    cloth_template_path = data / "template_cloth_offset.ply"
    votes_path = data / "layer_geometry_votes.npz"
    manifest_path = data / "layer_geometry_manifest.json"
    required = (
        body_pos_path,
        cloth_pos_path,
        body_lbs_path,
        body_template_path,
        cloth_template_path,
        votes_path,
        manifest_path,
    )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "training masks require actual generated sources; missing: " + ", ".join(missing)
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    canon = manifest.get("canonicalization", {})
    if not canon.get("body_prior_matches_v_beta"):
        raise RuntimeError("body prior is not the verified canonical V_beta; refusing")
    body_template = trimesh.load(str(body_template_path), process=False)
    cloth_template = trimesh.load(str(cloth_template_path), process=False)
    if not hasattr(body_template, "faces") or not hasattr(cloth_template, "faces"):
        raise TypeError("template body/cloth must be triangle meshes")
    if len(body_template.faces) == 0 or len(cloth_template.faces) == 0:
        raise ValueError("empty body/cloth template faces")

    with np.load(votes_path, allow_pickle=True) as z:
        votes = {k: z[k] for k in z.files}
    for key in ("faces", "cloth_faces", "upper_votes", "lower_votes"):
        if key not in votes:
            raise RuntimeError(f"votes missing {key}; semantic map incomplete")
    all_faces = np.asarray(votes["faces"])
    cloth_mask = np.asarray(votes["cloth_faces"], bool)
    selected_faces = all_faces[cloth_mask]
    if not np.array_equal(np.asarray(cloth_template.faces), selected_faces):
        raise RuntimeError(
            "template_cloth_offset faces do not exactly equal " "votes.faces[votes.cloth_faces]"
        )
    upper = np.asarray(votes["upper_votes"])
    lower = np.asarray(votes["lower_votes"])
    if upper.shape != all_faces.shape[:1] or lower.shape != upper.shape:
        raise RuntimeError("upper/lower votes shape mismatch")
    upper_sel = upper[cloth_mask]
    lower_sel = lower[cloth_mask]
    total_sel = upper_sel + lower_sel
    unobserved = total_sel == 0
    tied = (upper_sel == lower_sel) & ~unobserved
    # Native categorical MAP order: upper first, lower second. Exact ties
    # therefore choose upper via ordinary numpy.argmax, not a new heuristic.
    labels = np.argmax(np.stack([upper_sel, lower_sel], axis=1), axis=1)

    with torch.no_grad():
        body_img = cv.imread(str(body_pos_path), cv.IMREAD_UNCHANGED)
        cloth_img = cv.imread(str(cloth_pos_path), cv.IMREAD_UNCHANGED)
        if body_img is None or cloth_img is None:
            raise FileNotFoundError("failed to read canonical position maps")
        if (
            body_img.ndim != 3
            or body_img.shape[2] != 3
            or cloth_img.ndim != 3
            or cloth_img.shape[2] != 3
        ):
            raise ValueError("canonical position maps must be 3-channel")
        body_t = torch.from_numpy(np.asarray(body_img)).to(torch.float32)
        cloth_t = torch.from_numpy(np.asarray(cloth_img)).to(torch.float32)
        body_support = (torch.linalg.norm(body_t, dim=-1) > 0).cpu().numpy()
        cloth_support = (torch.linalg.norm(cloth_t, dim=-1) > 0).cpu().numpy()
        if body_support.shape != body_t.shape[:2] or cloth_support.shape != cloth_t.shape[:2]:
            raise ValueError("canonical support shape mismatch")
        if int(body_support.sum()) == 0 or int(cloth_support.sum()) == 0:
            raise ValueError("empty canonical body/cloth support")
        if cloth_support.shape != (1024, 2048):
            raise ValueError(f"unexpected canonical cloth map shape {cloth_t.shape}")

        # gen_pos_maps.py is invoked with cano_weight_volume_shape_zero, which
        # zeroes betas. The atlas center must therefore use the same zero-beta
        # flat-hand zero-expression canonical SMPL-X body, not fitted betas.
        betas = torch.zeros((1, 10), dtype=torch.float32, device="cuda")
        model = (
            smplx.SMPLX(
                model_path=ctx["smplx_model_dir"],
                gender="neutral",
                use_pca=False,
                num_pca_comps=45,
                flat_hand_mean=True,
                batch_size=1,
                num_expression_coeffs=ctx["expr_dim"],
            )
            .to("cuda")
            .eval()
        )
        with torch.no_grad():
            body = model(
                betas=betas,
                global_orient=cfg.cano_smpl_global_orient[None].to("cuda"),
                transl=cfg.cano_smpl_transl[None].to("cuda"),
                body_pose=cfg.cano_smpl_body_pose[None].to("cuda"),
            ).vertices[0]
        model_faces = np.asarray(model.faces, np.int64)
        if not np.array_equal(np.asarray(body_template.faces), model_faces):
            raise RuntimeError(
                "template_body_offset faces do not exactly match zero-body model.faces"
            )
        cano_center = 0.5 * (body.min(0).values + body.max(0).values)
        center = cano_center.detach().cpu().numpy().astype(np.float32)

    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1
    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # Render canonical face-ID atlases with the same author camera/atlas layout.
    renderer = Renderer(1024, 1024, shader_name="vertex_attribute", device="cuda")

    def render_face_ids(template):
        vertices_dup = template.vertices[template.faces.reshape(-1)]
        renderer.set_model(vertices_dup, np.zeros((len(vertices_dup), 3), np.float32))
        renderer.set_camera(front_mv)
        _, front_faces = renderer.render()
        renderer.set_camera(back_mv)
        _, back_faces = renderer.render()
        return np.concatenate([front_faces, np.flip(back_faces, axis=1)], axis=1)

    def check_face_support(face_ids, support, label):
        if face_ids.shape != support.shape:
            raise RuntimeError(
                f"{label} rendered face-id shape {face_ids.shape} != support {support.shape}"
            )
        if not np.array_equal(face_ids >= 0, support):
            raise RuntimeError(
                f"{label} rendered face support differs from cano_smpl_pos_map support; "
                "refusing a mismatched canonical atlas"
            )

    def rebuild_adjacency(layer_dir, template, template_path, face_ids, support, pos_path):
        if not np.array_equal(face_ids >= 0, support):
            raise RuntimeError(f"adjacency support mismatch for {layer_dir}")
        pixel_idx = np.full(support.shape, -1, np.int64)
        pixel_idx[support] = np.arange(int(support.sum()), dtype=np.int64)
        neighbor_idx, neighbor_weights, with_neighbor = get_neighbors(
            pixel_idx, support, face_ids, template, neighbor_max_num=8
        )
        outputs = {}
        for name, array in (
            ("neighbor_idx.npy", neighbor_idx),
            ("neighbor_weights.npy", neighbor_weights),
            ("with_neighbor.npy", with_neighbor),
        ):
            path = layer_dir / name
            tmp = path.with_name(path.name + ".tmp.npy")
            np.save(tmp, array)
            os.replace(tmp, path)
            outputs[name] = {
                "path": str(path),
                "sha256": sha256(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        return {
            "status": "corrected",
            "layer": layer_dir.name,
            "canonical_pos": str(pos_path),
            "canonical_pos_sha256": sha256(pos_path),
            "template": str(template_path),
            "template_sha256": sha256(template_path),
            "back_face_ids_flipped": True,
            "face_id_atlas": "author Renderer face IDs; back IDs flipped with back attrs",
            "edge_rule": "author get_neighbors 8-neighbor; original algebra unchanged",
            "support_pixels": int(support.sum()),
            "neighbor_max_num": 8,
            "outputs": outputs,
        }

    body_face_ids = render_face_ids(body_template)
    cloth_face_ids = render_face_ids(cloth_template)
    check_face_support(body_face_ids, body_support, "body")
    check_face_support(cloth_face_ids, cloth_support, "cloth")
    if np.any((body_face_ids >= 0) & (body_face_ids >= len(body_template.faces))):
        raise RuntimeError("rendered body face id outside template face order")
    if np.any((cloth_face_ids >= 0) & (cloth_face_ids >= len(selected_faces))):
        raise RuntimeError("rendered face id outside selected cloth face order")
    body_adjacency = rebuild_adjacency(
        body_dir,
        body_template,
        body_template_path,
        body_face_ids,
        body_support,
        body_pos_path,
    )
    cloth_adjacency = rebuild_adjacency(
        cloth_dir,
        cloth_template,
        cloth_template_path,
        cloth_face_ids,
        cloth_support,
        cloth_pos_path,
    )

    face_ids = cloth_face_ids
    face_ids_safe = np.where(face_ids >= 0, face_ids, 0)
    face_unobserved = unobserved[face_ids_safe]
    face_tied = tied[face_ids_safe]
    unobserved_px = cloth_support & face_unobserved
    tied_px = cloth_support & face_tied
    lower_px = cloth_support & (labels[face_ids_safe] == 1)
    upper_px = cloth_support & (labels[face_ids_safe] == 0)
    voted_px = cloth_support & ~face_unobserved
    unobserved_face_ids = np.unique(face_ids[unobserved_px])
    unobserved_face_ids = unobserved_face_ids[unobserved_face_ids >= 0]
    tied_face_ids = np.unique(face_ids[tied_px])
    tied_face_ids = tied_face_ids[tied_face_ids >= 0]
    support_count = int(cloth_support.sum())
    unobserved_pixels = int(unobserved_px.sum())
    tied_pixels = int(tied_px.sum())
    segment_path = cloth_dir / "cano_smpl_segment_map.exr"
    previous_segment_sha = sha256(segment_path) if segment_path.is_file() else None
    common_segment = {
        "output": str(segment_path),
        "source_votes": str(votes_path),
        "source_votes_sha256": sha256(votes_path),
        "template": str(cloth_template_path),
        "template_sha256": sha256(cloth_template_path),
        "canonical_pos": str(cloth_pos_path),
        "canonical_pos_sha256": sha256(cloth_pos_path),
        "back_face_ids_flipped": True,
        "canonical_center": "zero-beta flat-hand zero-expression SMPL-X bbox center",
        "face_order_contract": "template_cloth_offset.faces == votes.faces[cloth_faces]",
        "canonical_support_contract": "canvas face-id support == cloth cano_smpl_pos_map norm>0",
        "class_order": ["upper", "lower"],
        "class_rule": "argmax([upper_votes, lower_votes], axis=1)",
        "tie_choice": "exact ties choose upper (first class in argmax)",
        "voted_face_count": int((~unobserved).sum()),
        "unobserved_face_count": int(unobserved.sum()),
        "tied_face_count": int(tied.sum()),
        "tied_visible_face_count": int(len(tied_face_ids)),
        "unobserved_visible_face_count": int(len(unobserved_face_ids)),
        "support_pixels": support_count,
        "voted_pixels": int(voted_px.sum()),
        "tied_pixels": tied_pixels,
        "unobserved_pixels": unobserved_pixels,
        "total_cloth_faces": int(len(selected_faces)),
    }
    if unobserved_pixels:
        segment_info = {
            **common_segment,
            "status": "incomplete",
            "reason": "canonical support contains faces with zero total votes",
            "output_sha256": previous_segment_sha,
            "map_preserved": True,
            "tied_are_uniquely_observed": False,
        }
        segment_status = "incomplete"
    else:
        segment = np.zeros((*cloth_support.shape, 3), np.float32)
        segment[lower_px] = 1.0
        segment[upper_px] = 0.0
        if not cv.imwrite(str(segment_path), segment, []):
            raise RuntimeError(f"failed to write cloth segment map: {segment_path}")
        segment_info = {
            **common_segment,
            "status": "generated",
            "output_sha256": sha256(segment_path),
            "map_preserved": False,
            "upper_pixels": int(upper_px.sum()),
            "lower_pixels": int(lower_px.sum()),
            "valid_voted_coverage": float(voted_px.sum() / support_count),
            "tied_are_uniquely_observed": False if int(tied.sum()) else True,
            "unobserved_equals_upper": False,
        }
        segment_status = "generated"

    body_fix_path = body_dir / "cano_smpl_smplx_fix_offset_map.exr"
    body_fix = np.zeros((*body_support.shape, 3), np.float32)
    body_fix[body_support] = 1.0
    if not cv.imwrite(str(body_fix_path), body_fix, []):
        raise RuntimeError(f"failed to write body fix map: {body_fix_path}")
    body_fix_info = {
        "status": "generated",
        "output": str(body_fix_path),
        "output_sha256": sha256(body_fix_path),
        "support_pixels": int(body_support.sum()),
        "body_template": str(body_template_path),
        "body_template_sha256": sha256(body_template_path),
        "body_template_faces": int(len(body_template.faces)),
        "body_zero_model_faces": int(len(model_faces)),
        "body_template_faces_equal_zero_body": True,
        "canonical_center": "zero-beta flat-hand zero-expression SMPL-X bbox center",
        "canonical_pos": str(body_pos_path),
        "canonical_pos_sha256": sha256(body_pos_path),
        "provenance": "entire explicit canonical SMPL-X body prior; not an observed "
        "skin mask; author fixed-input/Laplacian support",
        "body_prior_matches_v_beta": True,
    }

    # Hand map: preserve the existing anatomical LBS rule and map semantics.
    support_np = body_support
    support_count_body = int(support_np.sum())
    init = np.load(body_lbs_path)
    if init.ndim != 2 or init.shape != (support_count_body, 55) or not np.isfinite(init).all():
        raise ValueError("invalid init_pts_lbs for body support")
    hand = (init.argmax(axis=1) == 20) | (init.argmax(axis=1) == 21) | (init.argmax(axis=1) >= 25)
    hand_map = np.zeros((*support_np.shape, 3), np.float32)
    hand_map[support_np] = hand[:, None].astype(np.float32)
    hand_path = body_dir / "cano_smpl_hand_map.exr"
    if not cv.imwrite(str(hand_path), hand_map, []):
        raise RuntimeError(f"failed to write hand map: {hand_path}")
    hand_info = {
        "status": "generated",
        "output": str(hand_path),
        "output_sha256": sha256(hand_path),
        "canonical_pos": str(body_pos_path),
        "canonical_pos_sha256": sha256(body_pos_path),
        "init_pts_lbs": str(body_lbs_path),
        "init_pts_lbs_sha256": sha256(body_lbs_path),
        "support_count": support_count_body,
        "hand_support_count": int(hand.sum()),
        "hand_rule": "argmax(lbs, axis=1) == 20 or == 21 or >= 25",
        "output_dtype": "float32",
        "output_values": [0.0, 1.0],
    }
    manifest.setdefault("schema", "gaussian_wardrobe_layer_geometry/v1")
    manifest.update(
        {
            "generated_at_utc": utc_now(),
            "training_masks": {
                "hand": hand_info,
                "body_fix": body_fix_info,
                "cloth_segment": segment_info,
                "adjacency": {
                    "body": body_adjacency,
                    "cloth": cloth_adjacency,
                    "back_face_ids_flipped": True,
                    "source_bug": "author gen_pos_maps returned unflipped back pix_to_face",
                    "edge_rule": "author get_neighbors with original 8-neighbor algebra",
                },
                "status": segment_status,
                "generated_at_utc": utc_now(),
                "no_fake_masks": True,
                "no_observed_body_claim": True,
                "quality_claim": None,
            },
        }
    )
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(
        json.dumps(
            {
                "status": segment_status,
                "hand_support": int(hand.sum()),
                "body_fix_support": int(body_support.sum()),
                "cloth_support": support_count,
                "voted_face_count": int((~unobserved).sum()),
                "tied_face_count": int(tied.sum()),
                "unobserved_face_count": int(unobserved.sum()),
                "tied_pixels": tied_pixels,
                "tied_visible_faces": int(len(tied_face_ids)),
                "unobserved_pixels": unobserved_pixels,
                "unobserved_visible_faces": int(len(unobserved_face_ids)),
                "segment_output": str(segment_path),
                "segment_sha256": segment_info.get("output_sha256"),
            },
            indent=2,
        )
    )
    return 0 if segment_status == "generated" else 2


def _source_signature(config_path: Path) -> Dict[str, Any]:
    files = {
        "smpl_params": DATA / "smpl_params.npz",
        "cameras": DATA / "cameras.pkl",
    }
    for layer in ("body", "cloth"):
        base = DATA / f"smpl_pos_map_offset_{layer}"
        for name in (
            "cano_smpl_pos_map.exr",
            "cano_smpl_nml_map.exr",
            "cano_smpl_offset_map.exr",
            "init_pts_lbs.npy",
        ):
            files[f"{layer}/{name}"] = base / name
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"pretrain source assets missing: {missing}")
    return {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "files": {name: sha256(p) for name, p in files.items()},
    }


def _seed_all(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_snapshot(torch) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _rng_restore(torch, state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _epoch_order(epoch: int, n: int) -> List[int]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(31359 + int(epoch))
    return torch.randperm(n, generator=generator).tolist()


def _collate_one_sample(value: Any, torch):
    """Add the native batch-of-one axis without copying large CPU arrays."""
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, dict):
        return {key: _collate_one_sample(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        # default_collate transposes fixed-length tuples/lists into a list.
        return [_collate_one_sample(item, torch) for item in value]
    if isinstance(value, (bool, int, float, np.generic)):
        return torch.as_tensor([value])
    if value is None or isinstance(value, str):
        return [value]
    raise TypeError(f"unsupported batch-of-one value: {type(value).__name__}")


def _save_training_checkpoint(
    path: Path,
    trainer,
    epoch: int,
    offset: int,
    updates: int,
    signature: Dict[str, Any],
    phase: str = "author_pretrain",
    extra_signature: Optional[Dict[str, Any]] = None,
    iter_idx: Optional[int] = None,
    epoch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    import torch

    # Keep the live model tensors on their training device. torch.save streams
    # each storage into the archive; eagerly copying the entire 447M-parameter
    # model to CPU here overlaps that 1.8 GB snapshot with CPU Adam state and
    # caused a native memory failure at the first image-training checkpoint.
    # Readers use map_location="cpu", so the rolling checkpoint stays portable.
    model_state = trainer.avatar_net.state_dict()
    payload = {
        "phase": phase,
        "model": model_state,
        "optimizer": trainer.optm.state_dict(),
        "epoch": int(epoch),
        "offset": int(offset),
        "updates": int(updates),
        "rng": _rng_snapshot(torch),
        "source_signature": signature,
        "seed": 31359,
        "config": signature.get("config", str(CONFIG)),
        "training_memory": "CPUAdam+native_adam_cpu_state+unet_checkpointing",
    }
    if extra_signature is not None:
        payload["train_source_signature"] = extra_signature
    if getattr(trainer, "checkpoint_recovery", None):
        payload["recovery"] = trainer.checkpoint_recovery
    if iter_idx is not None:
        payload["iter_idx"] = int(iter_idx)
    if epoch_idx is not None:
        payload["epoch_idx"] = int(epoch_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def signature_equal(actual: Any, expected: Any) -> bool:
    """Compare input content while allowing relocation of the config file.

    Only a source-signature root (config SHA plus files) and its explicit
    common source root may omit an absolute config path. Input hashes,
    counts, training fields and nested input configuration stay untouched.
    Neither the saved checkpoint nor generated signatures are rewritten.
    """

    def normalize(value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        config = result.get("config")
        if (
            isinstance(result.get("config_sha256"), str)
            and isinstance(result.get("files"), dict)
            and isinstance(config, str)
            and Path(config).is_absolute()
        ):
            del result["config"]
        if "common" in result:
            result["common"] = normalize(result["common"])
        return result

    return normalize(actual) == normalize(expected)


def _load_training_checkpoint(path: Path, trainer, signature: Dict[str, Any]):
    import torch
    from hoi.clothing.training_memory import load_checkpoint

    payload = load_checkpoint(path)
    if payload.get("phase") == "author_train":
        raise RuntimeError(
            "pretrain refuses to load an author_train checkpoint; "
            "use --train to resume finished image training"
        )
    if not signature_equal(payload.get("source_signature"), signature):
        raise RuntimeError(
            "training checkpoint source signature mismatch; refusing to resume "
            "with changed geometry/pose/config inputs"
        )
    trainer.avatar_net.load_state_dict(payload["model"], strict=True)
    trainer.optm.sync_from_model()
    trainer.optm.load_state_dict(payload["optimizer"])
    _rng_restore(torch, payload["rng"])
    return int(payload["epoch"]), int(payload["offset"]), int(payload["updates"])


def _train_source_signature(config_path: Path) -> Dict[str, Any]:
    common = _source_signature(config_path)
    layout = native_layout(DATA, config_path)
    missing = []
    counts = {}
    root_specs = {
        "images": (".jpg", ".png"),
        "masks": (".jpg", ".png"),
        "labels_color": (".png",),
        "semantic_valid": (".png",),
        "body_masks": (".png",),
    }
    for name, exts in root_specs.items():
        root = DATA / name
        count = 0
        if root.is_dir():
            for cam in layout["camera_names"]:
                for pose_idx in layout["poses"]:
                    if any((root / cam / f"{pose_idx:08d}{ext}").is_file() for ext in exts):
                        count += 1
        counts[name] = count
        if count != layout["expected_items"]:
            missing.append(f"{name} ({count}/{layout['expected_items']})")

    manifest_paths = {
        "segmentation_manifest": DATA / "segmentation_manifest.json",
        "layer_geometry_manifest": DATA / "layer_geometry_manifest.json",
        "source_frames": DATA / "source_frames.json",
    }
    for name, path in manifest_paths.items():
        if not path.is_file():
            missing.append(name)
    maps = {
        "body_fix": DATA / "smpl_pos_map_offset_body/cano_smpl_smplx_fix_offset_map.exr",
        "body_hand": DATA / "smpl_pos_map_offset_body/cano_smpl_hand_map.exr",
        "cloth_segment": DATA / "smpl_pos_map_offset_cloth/cano_smpl_segment_map.exr",
    }
    body_pos = DATA / "smpl_pos_map_offset_body/cano_smpl_pos_map.exr"
    cloth_pos = DATA / "smpl_pos_map_offset_cloth/cano_smpl_pos_map.exr"
    body_template = DATA / "template_body_offset.ply"
    cloth_template = DATA / "template_cloth_offset.ply"
    for name, path in maps.items():
        if not path.is_file():
            missing.append(name)
    for name, path in (
        ("body_pos", body_pos),
        ("cloth_pos", cloth_pos),
        ("body_template", body_template),
        ("cloth_template", cloth_template),
    ):
        if not path.is_file():
            missing.append(name)
    votes_path = DATA / "layer_geometry_votes.npz"
    if not votes_path.is_file():
        missing.append("layer_geometry_votes.npz")
    else:
        with np.load(votes_path, allow_pickle=True) as z:
            required_votes = (
                "upper_votes",
                "lower_votes",
                "upper_majority_faces",
                "lower_majority_faces",
                "ambiguous_cloth_faces",
            )
            absent = [k for k in required_votes if k not in z.files]
            if absent:
                missing.append(f"semantic_votes arrays ({absent})")
    try:
        lg = (
            json.loads((DATA / "layer_geometry_manifest.json").read_text(encoding="utf-8"))
            if (DATA / "layer_geometry_manifest.json").is_file()
            else {}
        )
    except Exception as exc:
        missing.append(f"layer_geometry_manifest parse: {exc}")
        lg = {}
    sv = lg.get("semantic_votes") if isinstance(lg, dict) else None
    if (
        not isinstance(sv, dict)
        or not set(layout["poses"]).issubset(set(sv.get("frames", [])))
        or not set(layout["camera_names"]).issubset(set(sv.get("cameras", [])))
        or sv.get("used_items") != len(sv.get("frames", [])) * len(sv.get("cameras", []))
    ):
        missing.append("semantic_votes manifest/incomplete")
    if "body_mask_generation" not in lg:
        missing.append("body_mask_generation")
    canon = lg.get("canonicalization", {}) if isinstance(lg, dict) else {}
    if not canon.get("body_prior_matches_v_beta"):
        missing.append("body_prior_matches_v_beta")
    tm = lg.get("training_masks", {}) if isinstance(lg, dict) else {}
    segment = tm.get("cloth_segment", {}) if isinstance(tm, dict) else {}
    body_fix = tm.get("body_fix", {}) if isinstance(tm, dict) else {}
    hand = tm.get("hand", {}) if isinstance(tm, dict) else {}
    adjacency = tm.get("adjacency", {}) if isinstance(tm, dict) else {}
    if not isinstance(segment, dict) or segment.get("status") != "generated":
        missing.append("training_masks.cloth_segment.status")
    if not isinstance(body_fix, dict) or body_fix.get("status") != "generated":
        missing.append("training_masks.body_fix.status")
    if not isinstance(hand, dict) or hand.get("status") != "generated":
        missing.append("training_masks.hand.status")
    if isinstance(segment, dict):
        expected = {
            "output_sha256": maps["cloth_segment"],
            "source_votes_sha256": votes_path,
            "template_sha256": cloth_template,
            "canonical_pos_sha256": cloth_pos,
        }
        for key, path in expected.items():
            if segment.get(key) != (sha256(path) if path.is_file() else None):
                missing.append(f"training_masks.cloth_segment.{key} mismatch")
        if segment.get("unobserved_pixels") != 0:
            missing.append("training_masks.cloth_segment.unobserved_pixels")
    if isinstance(body_fix, dict):
        for key, path in {
            "output_sha256": maps["body_fix"],
            "canonical_pos_sha256": body_pos,
            "body_template_sha256": body_template,
        }.items():
            if body_fix.get(key) != (sha256(path) if path.is_file() else None):
                missing.append(f"training_masks.body_fix.{key} mismatch")
    if isinstance(hand, dict):
        for key, path in {
            "output_sha256": maps["body_hand"],
            "canonical_pos_sha256": body_pos,
            "init_pts_lbs_sha256": DATA / "smpl_pos_map_offset_body/init_pts_lbs.npy",
        }.items():
            if hand.get(key) != (sha256(path) if path.is_file() else None):
                missing.append(f"training_masks.hand.{key} mismatch")
    adjacency_paths = {
        "body": {
            name: DATA / "smpl_pos_map_offset_body" / name
            for name in (
                "neighbor_idx.npy",
                "neighbor_weights.npy",
                "with_neighbor.npy",
            )
        },
        "cloth": {
            name: DATA / "smpl_pos_map_offset_cloth" / name
            for name in (
                "neighbor_idx.npy",
                "neighbor_weights.npy",
                "with_neighbor.npy",
            )
        },
    }
    for layer, paths in adjacency_paths.items():
        rec = adjacency.get(layer, {}) if isinstance(adjacency, dict) else {}
        if not isinstance(rec, dict) or rec.get("status") != "corrected":
            missing.append(f"training_masks.adjacency.{layer}.status")
            continue
        if rec.get("back_face_ids_flipped") is not True:
            missing.append(f"training_masks.adjacency.{layer}.back_face_ids_flipped")
        pos = body_pos if layer == "body" else cloth_pos
        template = body_template if layer == "body" else cloth_template
        if rec.get("canonical_pos_sha256") != (sha256(pos) if pos.is_file() else None):
            missing.append(f"training_masks.adjacency.{layer}.canonical_pos_sha256 mismatch")
        if rec.get("template_sha256") != (sha256(template) if template.is_file() else None):
            missing.append(f"training_masks.adjacency.{layer}.template_sha256 mismatch")
        outputs = rec.get("outputs", {}) if isinstance(rec, dict) else {}
        for name, path in paths.items():
            out = outputs.get(name, {}) if isinstance(outputs, dict) else {}
            if not path.is_file() or out.get("sha256") != sha256(path):
                missing.append(f"training_masks.adjacency.{layer}.{name} mismatch")
    if missing:
        raise RuntimeError(
            "train prerequisites not established; refusing image training: " + ", ".join(missing)
        )

    input_files = {}
    input_files.update({f"manifest/{k}": v for k, v in manifest_paths.items()})
    input_files.update({f"mask/{k}": v for k, v in maps.items()})
    input_files["layer_geometry_votes"] = votes_path
    for layer, paths in adjacency_paths.items():
        for name, path in paths.items():
            input_files[f"adjacency/{layer}/{name}"] = path
    return {
        "common": common,
        "counts": counts,
        "files": {name: sha256(p) for name, p in input_files.items()},
        "labels_color_palette": "from segmentation_manifest.json (SHA above)",
        "semantic_mask_complete": True,
        "train_data_labels_enabled": True,
        "training_masks": {
            "hand_sha256": hand.get("output_sha256"),
            "body_fix_sha256": body_fix.get("output_sha256"),
            "cloth_segment_sha256": segment.get("output_sha256"),
            "cloth_segment_voted_pixels": segment.get("voted_pixels"),
            "cloth_segment_tied_pixels": segment.get("tied_pixels"),
            "cloth_segment_unobserved_pixels": segment.get("unobserved_pixels"),
            "cloth_segment_support_pixels": segment.get("support_pixels"),
            "cloth_segment_tied_face_count": segment.get("tied_face_count"),
            "cloth_segment_unobserved_face_count": segment.get("unobserved_face_count"),
            "adjacency_sha256": {
                layer: {
                    name: adjacency.get(layer, {}).get("outputs", {}).get(name, {}).get("sha256")
                    for name in paths
                }
                for layer, paths in adjacency_paths.items()
            },
        },
    }


def _prepare_lpips(trainer, opt: Dict[str, Any]) -> bool:
    weights = opt["train"]["loss_weight"]
    if not (weights.get("lpips", 0) > 0 or weights.get("face_lpips", 0) > 0):
        return False
    lpips_path = _find_lpips()
    if lpips_path is None:
        raise FileNotFoundError("LPIPS linear weights missing; refusing automatic download")
    vgg_cache = _vgg_cache()
    if not vgg_cache.is_file():
        raise FileNotFoundError(
            f"local VGG cache missing at {vgg_cache}; refusing automatic download"
        )
    from network.lpips import LPIPS

    trainer.lpips = LPIPS(net="vgg", model_path=str(lpips_path)).to(trainer.bg_color_cuda.device)
    for p in trainer.lpips.parameters():
        p.requires_grad = False
    return True


def run_author_pretrain(config_path: Path, target_updates: int) -> int:
    """Shared real author pretrain implementation for step and full modes."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    out = RESULTS / "wardrobe_training.json"
    report: Dict[str, Any] = {
        "schema": "gaussian_wardrobe_training/v1",
        "phase": "author_pretrain",
        "status": "failed",
        "config": str(config_path),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "data_dir": str(DATA),
        "target_updates": int(target_updates),
        "updates": 0,
        "step_only": False,
        "wandb_mode": os.environ["WANDB_MODE"],
        "wandb_disabled": os.environ["WANDB_DISABLED"],
        "optimizer": "CPUAdam(native torch.optim.Adam over CPU FP32 mirrors)",
        "activation_checkpointing": "6 DualStyleUNet forwards",
        "no_checkpoint": False,
        "no_mini_test": True,
        "no_full_training": False,
        "no_amp": True,
        "quality_claim": None,
    }
    torch = None
    trainer = None
    signature = None
    updates = 0
    epoch = 0
    offset = 0
    loss_sums: Dict[str, float] = {}
    loss_count = 0
    last_losses: Dict[str, float] = {}
    last_total = None
    checkpoint = None
    elapsed = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for pretrain but unavailable")
        torch.cuda.reset_peak_memory_stats()
        _seed_all(torch, 31359)
        ctx = bootstrap(config_path, device="cuda", data_dir=DATA)
        install_native_reader()
        cfg = ctx["config"]
        opt = ctx["opt"]
        opt["mode"] = "train"
        cfg.opt = opt
        signature = _source_signature(config_path)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "config_mode": opt.get("mode"),
                "source_signature": signature,
            }
        )
        from main_avatar import AvatarTrainer
        from hoi.clothing.training_memory import CPUAdam, enable_unet_checkpointing
        import utils.net_util as net_util
        import dataset.dataset_mv_rgb as dsmod

        started = time.time()
        trainer = AvatarTrainer(opt)
        checkpointed = enable_unet_checkpointing(trainer.avatar_net)
        # The author's geometry/opacity warmup never calls get_colors. Keep
        # those unchanged weights on CPU until the RGB training process loads
        # the checkpoint; do not occupy CUDA for an unused forward branch.
        for layer in trainer.avatar_net.layers_nn.values():
            layer.color_net.to("cpu")
        trainer.optm = CPUAdam(trainer.avatar_net.parameters(), lr=trainer.lr_init)
        data_opts = dict(opt["train"]["data"])
        layout = native_layout(DATA, config_path, data_opts)
        overrides = {}
        for key in ("use_label", "use_body_label", "use_normal"):
            if data_opts.get(key):
                data_opts[key] = False
                overrides[key] = False
        dataset_cls = getattr(dsmod, opt["train"].get("dataset", "MvRgbDatasetMJJNative"))
        dataset = dataset_cls(**data_opts)
        n = len(dataset)
        if n != layout["expected_items"]:
            raise RuntimeError(
                f"pretrain requires {layout['expected_items']} configured native items, got {n}"
            )
        param_elements = int(sum(p.numel() for p in trainer.avatar_net.parameters()))
        trainable_elements = int(
            sum(p.numel() for p in trainer.avatar_net.parameters() if p.requires_grad)
        )
        resumed = False
        if TRAIN_CKPT.is_file():
            epoch, offset, updates = _load_training_checkpoint(TRAIN_CKPT, trainer, signature)
            resumed = True
            print(
                f"pretrain resumed at update {updates}/{target_updates} "
                f"epoch={epoch} offset={offset}",
                flush=True,
            )
        order = _epoch_order(epoch, n)
        if offset >= n:
            epoch += 1
            offset = 0
            order = _epoch_order(epoch, n)
        report.update(
            {
                "checkpointing_instances": len(checkpointed),
                "parameter_elements": param_elements,
                "trainable_parameter_elements": trainable_elements,
                "dataset_class": dataset_cls.__name__,
                "dataset_len": n,
                "data_options": {
                    k: v
                    for k, v in data_opts.items()
                    if k
                    in (
                        "layers",
                        "frame_range",
                        "used_cam_ids",
                        "load_smpl_pos_map",
                        "subject_name",
                    )
                },
                "label_loading_overrides": overrides,
                "resume": bool(resumed),
                "seed": 31359,
            }
        )
        while updates < target_updates:
            if offset >= n:
                epoch += 1
                offset = 0
                order = _epoch_order(epoch, n)
            subset = torch.utils.data.Subset(dataset, order[offset:])
            loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
            for items in loader:
                if updates >= target_updates:
                    break
                items = net_util.to_cuda(items)
                total_loss, batch_losses = trainer.forward_one_pass_pretrain(items)
                total_value = float(total_loss.detach().cpu().item())
                loss_values = {k: float(v) for k, v in batch_losses.items()}
                if not bool(torch.isfinite(total_loss).item()) or not all(
                    bool(np.isfinite(v)) for v in loss_values.values()
                ):
                    raise FloatingPointError("non-finite author pretrain loss")
                updates += 1
                offset += 1
                loss_count += 1
                last_total = total_value
                last_losses = loss_values
                for key, value in loss_values.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + value
                if updates % 10 == 0:
                    print(
                        f"pretrain update {updates}/{target_updates} "
                        f"epoch={epoch} offset={offset} loss={total_value:.6f}",
                        flush=True,
                    )
                if updates % 100 == 0:
                    checkpoint = _save_training_checkpoint(
                        TRAIN_CKPT, trainer, epoch, offset, updates, signature
                    )
        elapsed = time.time() - started
        checkpoint = _save_training_checkpoint(
            TRAIN_CKPT, trainer, epoch, offset, updates, signature
        )
        peak = int(torch.cuda.max_memory_allocated())
        report.update(
            {
                "status": "ok",
                "phase": (
                    "pretrained"
                    if updates >= int(opt.get("train", {}).get("pretrain_iters", 5001))
                    else "author_pretrain"
                ),
                "training_phase": "author_pretrain",
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "last_total_loss": last_total,
                "last_batch_losses": last_losses,
                "mean_batch_losses": {k: v / loss_count for k, v in loss_sums.items()},
                "elapsed_sec": round(elapsed, 3),
                "peak_vram_bytes": peak,
                "checkpoint": checkpoint,
                "optimizer_steps": int(updates),
                "no_checkpoint": False,
                "no_full_training": False,
                "scope": "real author pretrain updates; not dynamic RGB/cloth "
                "training or quality",
            }
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 0
    except KeyboardInterrupt:
        if trainer is not None:
            try:
                checkpoint = _save_training_checkpoint(
                    TRAIN_CKPT, trainer, epoch, offset, updates, signature
                )
            except Exception:
                checkpoint = None
        report.update(
            {
                "status": "interrupted",
                "phase": "author_pretrain_interrupted",
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "checkpoint": checkpoint,
                "exception_type": "KeyboardInterrupt",
                "exception_traceback": traceback.format_exc(),
            }
        )
        if torch is not None:
            try:
                report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 130
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "phase": "author_pretrain_failed",
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exception_traceback": traceback.format_exc(),
            }
        )
        if torch is not None:
            try:
                report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 2


def _load_train_checkpoint(
    path: Path,
    trainer,
    common_signature: Dict[str, Any],
    train_signature: Dict[str, Any],
    warmup: int,
):
    import torch
    from hoi.clothing.training_memory import load_checkpoint

    payload = load_checkpoint(path)
    if not signature_equal(payload.get("source_signature"), common_signature):
        raise RuntimeError(
            "training checkpoint common geometry/source signature mismatch; "
            "refusing to start image training"
        )
    phase = payload.get("phase")
    trainer.checkpoint_recovery = payload.get("recovery")
    updates = int(payload.get("updates", 0))
    if phase == "author_train":
        if not signature_equal(payload.get("train_source_signature"), train_signature):
            raise RuntimeError("author_train input signature mismatch; refusing to resume")
        trainer.avatar_net.load_state_dict(payload["model"], strict=True)
        trainer.optm.sync_from_model()
        trainer.optm.load_state_dict(payload["optimizer"])
        _rng_restore(torch, payload["rng"])
        return (
            int(payload.get("epoch", 0)),
            int(payload.get("offset", 0)),
            updates,
            int(payload.get("iter_idx", updates)),
            int(payload.get("epoch_idx", 0)),
            True,
            False,
        )
    if phase in ("author_pretrain", "pretrained"):
        if updates < warmup:
            raise RuntimeError(
                f"pretrain checkpoint has {updates} updates < warmup {warmup}; "
                "refusing to start image training from incomplete pretrain"
            )
        trainer.avatar_net.load_state_dict(payload["model"], strict=True)
        trainer.optm.sync_from_model()
        return 0, 0, 0, 0, 0, False, True
    raise RuntimeError(f"unsupported training checkpoint phase: {phase}")


def run_author_train(config_path: Path, updates_limit: Optional[int]) -> int:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    out = RESULTS / "wardrobe_training.json"
    report: Dict[str, Any] = {
        "schema": "gaussian_wardrobe_training/v1",
        "phase": "author_train",
        "status": "failed",
        "config": str(config_path),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "data_dir": str(DATA),
        "wandb_mode": os.environ["WANDB_MODE"],
        "wandb_disabled": os.environ["WANDB_DISABLED"],
        "optimizer": "CPUAdam(native torch.optim.Adam over CPU FP32 mirrors)",
        "activation_checkpointing": "6 DualStyleUNet forwards",
        "no_amp": True,
        "no_epoch_copies": True,
        "no_mini_test": True,
        "quality_claim": None,
    }
    torch = None
    trainer = None
    checkpoint = None
    updates = 0
    epoch = 0
    offset = 0
    iter_idx = 0
    epoch_idx = 0
    loss_sums: Dict[str, float] = {}
    loss_count = 0
    last_losses: Dict[str, float] = {}
    last_total = None
    elapsed = None

    def progress(stage: str) -> None:
        # Flush stage boundaries so native crashes can be localized from the
        # same real training run without wrapping or tracing every Python line.
        print(f"train stage: {stage}", flush=True)

    try:
        import torch

        progress("torch imported")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for training but unavailable")
        torch.cuda.reset_peak_memory_stats()
        ctx = bootstrap(config_path, device="cuda", data_dir=DATA)
        progress("runtime bootstrapped")
        install_native_reader()
        cfg = ctx["config"]
        opt = ctx["opt"]
        opt["mode"] = "train"
        cfg.opt = opt
        common_signature = _source_signature(config_path)
        train_signature = _train_source_signature(config_path)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "config_mode": opt.get("mode"),
                "source_signature": common_signature,
                "train_source_signature": train_signature,
            }
        )
        from main_avatar import AvatarTrainer
        from hoi.clothing.training_memory import CPUAdam, enable_unet_checkpointing
        import utils.net_util as net_util
        import dataset.dataset_mv_rgb as dsmod

        started = time.time()
        _seed_all(torch, 31359)
        progress("avatar trainer initialization")
        trainer = AvatarTrainer(opt)
        progress("avatar trainer initialized")
        checkpointed = enable_unet_checkpointing(trainer.avatar_net)
        trainer.optm = CPUAdam(trainer.avatar_net.parameters(), lr=trainer.lr_init)
        progress("cpu Adam mirrors initialized")
        lpips_ready = _prepare_lpips(trainer, opt)
        progress("LPIPS prepared")
        data_opts = dict(opt["train"]["data"])
        layout = native_layout(DATA, config_path, data_opts)
        dataset_cls = getattr(dsmod, opt["train"].get("dataset", "MvRgbDatasetMJJNative"))
        dataset = dataset_cls(**data_opts)
        # Keep the decoded RGB/semantic images as uint8 in host memory. The
        # author losses still receive the same FP32 [0,1] values, but conversion
        # happens after transfer so large per-view float temporaries do not
        # compete with Adam state and the rest of the process for Windows commit.
        dataset.defer_float_image_conversion = True
        dataset.defer_canonical_pos_maps = True
        from utils.net_util import process_pos_map

        canonical_pos_maps = {}
        for layer in data_opts["layers"]:
            map_path = DATA / dataset.smpl_pos_map[layer] / "cano_smpl_pos_map.exr"
            pos_map = process_pos_map(str(map_path))
            canonical_pos_maps[layer] = torch.from_numpy(np.ascontiguousarray(pos_map)).to(
                cfg.device
            )
            del pos_map
        progress("dataset initialized")
        n = len(dataset)
        if n != layout["expected_items"]:
            raise RuntimeError(
                f"author_train requires {layout['expected_items']} configured native items, got {n}"
            )
        warmup = int(opt["train"].get("pretrain_iters", 5001))
        if not TRAIN_CKPT.is_file():
            raise FileNotFoundError(
                f"completed pretrain checkpoint missing: {TRAIN_CKPT}; "
                "refusing to start image training from random initialization"
            )
        progress("loading author pretrain checkpoint")
        epoch, offset, updates, iter_idx, epoch_idx, resumed, transition = _load_train_checkpoint(
            TRAIN_CKPT, trainer, common_signature, train_signature, warmup
        )
        progress("author pretrain checkpoint loaded")
        if transition:
            _seed_all(torch, 31359)
            epoch = offset = updates = iter_idx = epoch_idx = 0
        target = int(updates_limit) if updates_limit is not None else (trainer.iter_num + 1)
        target = min(target, trainer.iter_num + 1)
        order = _epoch_order(epoch, n)
        if offset >= n:
            epoch += 1
            offset = 0
            order = _epoch_order(epoch, n)
        report.update(
            {
                "checkpointing_instances": len(checkpointed),
                "parameter_elements": int(sum(p.numel() for p in trainer.avatar_net.parameters())),
                "trainable_parameter_elements": int(
                    sum(p.numel() for p in trainer.avatar_net.parameters() if p.requires_grad)
                ),
                "dataset_class": dataset_cls.__name__,
                "dataset_len": n,
                "data_options": {
                    k: v
                    for k, v in data_opts.items()
                    if k
                    in (
                        "layers",
                        "frame_range",
                        "used_cam_ids",
                        "load_smpl_pos_map",
                        "use_label",
                        "use_body_label",
                        "use_normal",
                        "subject_name",
                    )
                },
                "labels_enabled": bool(data_opts.get("use_label"))
                and bool(data_opts.get("use_body_label")),
                "lpips_initialized": bool(lpips_ready),
                "pretrain_warmup": warmup,
                "resume": bool(resumed),
                "pretrain_transition": bool(transition),
                "target_updates": target,
                "iter_num": int(trainer.iter_num),
            }
        )
        while updates < target:
            if offset >= n:
                epoch += 1
                offset = 0
                order = _epoch_order(epoch, n)
            # DataLoader's default batch_size=1 collator copies every large
            # EXR-derived position map into a second host tensor. Preserve the
            # same batch-of-one shapes with views, then transfer those arrays.
            for sample_index in order[offset:]:
                if updates >= target:
                    break
                items = _collate_one_sample(dataset[int(sample_index)], torch)
                items = net_util.to_cuda(items)
                for image_key in ("color_img", "label_img", "label_body_img"):
                    image = items.get(image_key)
                    if image is not None and image.dtype == torch.uint8:
                        items[image_key] = image.to(torch.float32).mul_(1.0 / 255.0)
                # These canonical maps do not vary by frame. Keep one device
                # copy instead of decoding and transferring them every sample.
                items["smpl_cano_pos_map"] = {
                    layer: pos_map.unsqueeze(0) for layer, pos_map in canonical_pos_maps.items()
                }
                trainer.epoch_idx = epoch
                trainer.iter_idx = updates
                trainer.update_lr()
                if (epoch + 1) % trainer.opt["train"]["dropout_interval"] == 0 and isinstance(
                    items["smpl_pos_map"], dict
                ):
                    fix_mask = trainer.avatar_net.cano_smpl_fix_offset_input_mask
                    items["smpl_pos_map"]["body"][:, :, fix_mask] = items["smpl_cano_pos_map"][
                        "body"
                    ][:, :, fix_mask]
                    items["smpl_pos_map"]["cloth"] = items["smpl_cano_pos_map"]["cloth"]
                total_loss, batch_losses = trainer.forward_one_pass(items)
                total_value = float(total_loss.detach().cpu().item())
                loss_values = {k: float(v) for k, v in batch_losses.items()}
                if not bool(torch.isfinite(total_loss).item()) or not all(
                    bool(np.isfinite(v)) for v in loss_values.values()
                ):
                    raise FloatingPointError("non-finite author train loss")
                updates += 1
                offset += 1
                iter_idx = updates
                epoch_idx = epoch
                loss_count += 1
                last_total = total_value
                last_losses = loss_values
                for key, value in loss_values.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + value
                if updates % 10 == 0:
                    print(
                        f"train update {updates}/{target} epoch={epoch} "
                        f"offset={offset} loss={total_value:.6f}",
                        flush=True,
                    )
                if updates % 100 == 0:
                    checkpoint = _save_training_checkpoint(
                        TRAIN_CKPT,
                        trainer,
                        epoch,
                        offset,
                        updates,
                        common_signature,
                        phase="author_train",
                        extra_signature=train_signature,
                        iter_idx=iter_idx,
                        epoch_idx=epoch_idx,
                    )
                # The iterator requests its next sample before rebinding items;
                # release the previous sample and backward graph now so they
                # do not overlap the next image's decode buffers.
                del items, total_loss, batch_losses
        elapsed = time.time() - started
        checkpoint = _save_training_checkpoint(
            TRAIN_CKPT,
            trainer,
            epoch,
            offset,
            updates,
            common_signature,
            phase="author_train",
            extra_signature=train_signature,
            iter_idx=iter_idx,
            epoch_idx=epoch_idx,
        )
        peak = int(torch.cuda.max_memory_allocated())
        completed = updates >= (trainer.iter_num + 1)
        report.update(
            {
                "status": "ok",
                "phase": "author_train_complete" if completed else "author_train",
                "completed": bool(completed),
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "iter_idx": int(iter_idx),
                "epoch_idx": int(epoch_idx),
                "last_total_loss": last_total,
                "last_batch_losses": last_losses,
                "mean_batch_losses": {k: v / loss_count for k, v in loss_sums.items()},
                "elapsed_sec": round(elapsed, 3),
                "peak_vram_bytes": peak,
                "checkpoint": checkpoint,
                "optimizer_steps": int(updates),
                "scope": "real author RGB/label image training; not a quality claim",
            }
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 0
    except KeyboardInterrupt:
        if trainer is not None:
            try:
                checkpoint = _save_training_checkpoint(
                    TRAIN_CKPT,
                    trainer,
                    epoch,
                    offset,
                    updates,
                    common_signature,
                    phase="author_train",
                    extra_signature=train_signature,
                    iter_idx=iter_idx,
                    epoch_idx=epoch_idx,
                )
            except Exception:
                checkpoint = None
        report.update(
            {
                "status": "interrupted",
                "phase": "author_train_interrupted",
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "checkpoint": checkpoint,
                "exception_type": "KeyboardInterrupt",
                "exception_traceback": traceback.format_exc(),
            }
        )
        if torch is not None:
            try:
                report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 130
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "phase": "author_train_failed",
                "updates": int(updates),
                "epoch": int(epoch),
                "offset": int(offset),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exception_traceback": traceback.format_exc(),
            }
        )
        if torch is not None:
            try:
                report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        print(json.dumps(report, indent=2))
        return 2


def run_render(
    config_path: Path, frames: Optional[List[int]], video: bool, video_camera: int
) -> int:
    """Render actual author RGB review PNGs and optional preview video."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    _install_unicode_cv()
    import cv2 as cv
    import subprocess
    import torch
    from hoi.clothing.training_memory import load_checkpoint

    if not TRAIN_CKPT.is_file():
        raise FileNotFoundError(f"render checkpoint missing: {TRAIN_CKPT}")
    payload = load_checkpoint(TRAIN_CKPT)
    phase = payload.get("phase")
    if phase != "author_train":
        raise RuntimeError(
            f"render requires phase author_train, got {phase!r}; "
            "pretrain/random RGB is not a dynamic result"
        )
    checkpoint_sha = sha256(TRAIN_CKPT)
    checkpoint_updates = int(payload.get("updates", 0))
    layout = native_layout(DATA, config_path)
    if frames is None:
        poses = layout["poses"]
        frames = list(dict.fromkeys([poses[0], poses[len(poses) // 2], poses[-1]]))
    if not frames or any(int(frame) not in layout["poses"] for frame in frames):
        raise ValueError("render frames must be selected native pose indices")
    if int(video_camera) not in layout["camera_ids"]:
        raise ValueError("video camera must be selected by used_cam_ids")
    if video and layout["video_fps"] is None:
        raise ValueError("video requires regularly spaced configured native pose indices")
    ffmpeg = _ffmpeg_exe() if video else None
    review = TRAIN_CURRENT / "review"
    review.mkdir(parents=True, exist_ok=True)
    report_path = review / "report.json"
    if report_path.is_file():
        report_path.unlink()

    ctx = bootstrap(config_path, device="cuda", data_dir=DATA)
    install_native_reader()
    cfg = ctx["config"]
    opt = ctx["opt"]
    opt["mode"] = "train"
    cfg.opt = opt
    model_cfg = opt["model"]
    model_module = importlib.import_module(model_cfg.get("module", "network.avatar"))
    model_cls = getattr(model_module, model_cfg.get("network", "AvatarNet"))
    layers = opt["train"]["data"].get("layers")
    _seed_all(torch, 31359)
    net = model_cls(model_cfg, layers).to(cfg.device)
    missing, unexpected = net.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict model load mismatch: {missing} {unexpected}")
    del payload
    net.eval()

    import utils.net_util as net_util
    import dataset.dataset_mv_rgb as dsmod

    data_opts = dict(opt["train"]["data"])
    dataset_cls = getattr(dsmod, opt["train"].get("dataset", "MvRgbDatasetMJJNative"))
    dataset = dataset_cls(**data_opts)
    if (
        len(dataset) != layout["expected_items"]
        or list(dataset.pose_list) != layout["poses"]
        or int(dataset.view_num) != len(layout["camera_order"])
        or list(dataset.used_cam_ids) != layout["camera_ids"]
    ):
        raise RuntimeError("render dataset does not match configured native binding layout")

    def render_frame(pose_idx: int, cam_idx: int):
        intr = dataset.intr_mats[cam_idx].copy()
        extr = dataset.extr_mats[cam_idx].copy()
        item = dataset.getitem(
            0,
            pose_idx=int(pose_idx),
            view_idx=int(cam_idx),
            training=True,
            eval=False,
            img_h=int(dataset.img_heights[cam_idx]),
            img_w=int(dataset.img_widths[cam_idx]),
            extr=extr,
            intr=intr,
            exact_hand_pose=True,
        )
        items = net_util.to_cuda(item, add_batch=False)
        with torch.no_grad():
            ret = net.render(items, bg_color=(0.0, 0.0, 0.0))
        rgb = ret["rgb_map"].detach().float().cpu()
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"unexpected render rgb shape: {tuple(rgb.shape)}")
        rgb.clip_(0.0, 1.0)
        # Native dataset RGB is BGR and the trained network output is the same
        # BGR convention. cv.imwrite / ffmpeg raw bgr24 consume it directly.
        return np.ascontiguousarray((rgb * 255).to(torch.uint8).numpy())

    png_paths: List[str] = []
    cached_video: Dict[int, np.ndarray] = {}
    selected = [int(v) for v in frames]
    for cam in layout["camera_ids"]:
        cam_dir = review / f"cam{cam:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for pose in selected:
            frame = render_frame(pose, cam)
            path = cam_dir / f"frame{pose:08d}.png"
            if not cv.imwrite(str(path), frame, []):
                raise RuntimeError(f"failed to write review PNG: {path}")
            png_paths.append(str(path.relative_to(review)).replace("\\", "/"))
            if video and cam == int(video_camera):
                cached_video[pose] = frame

    video_rel = None
    if video:
        cam = int(video_camera)
        raw_w = int(dataset.img_widths[cam])
        raw_h = int(dataset.img_heights[cam])
        display_w = min(1008, raw_w)
        display_w -= display_w % 2
        display_h = int(round(raw_h * display_w / raw_w))
        display_h -= display_h % 2
        from hoi.results import record_video, video_path as result_video_path

        video_path = result_video_path(RESULTS, f"cam{cam:02d}", collection="exploration")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{raw_w}x{raw_h}",
            "-r",
            str(layout["video_fps"]),
            "-i",
            "pipe:0",
            "-vf",
            f"scale={display_w}:{display_h}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(video_path),
        ]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            for pose in layout["poses"]:
                frame = cached_video.get(pose)
                if frame is None:
                    frame = render_frame(pose, cam)
                if frame.shape != (raw_h, raw_w, 3):
                    raise RuntimeError(f"video frame shape {frame.shape} != {(raw_h, raw_w, 3)}")
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            code = proc.wait()
            if code != 0:
                raise RuntimeError("ffmpeg failed: " + stderr.decode("utf-8", "replace")[-2000:])
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise RuntimeError("ffmpeg did not produce a valid preview")
            decoded = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-progress",
                    "pipe:1",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            counts = [
                int(line.split("=", 1)[1])
                for line in decoded.stdout.splitlines()
                if line.startswith("frame=")
            ]
            if decoded.returncode != 0 or not counts or counts[-1] != len(layout["poses"]):
                raise RuntimeError(f"preview decode failed or incomplete: {decoded.stderr[-2000:]}")
            video_rel = os.path.relpath(video_path, review).replace("\\", "/")
            record_video(
                video_path, report=report_path, frames=len(layout["poses"]),
                fps=layout["video_fps"], checkpoint_updates=checkpoint_updates,
            )
        except BaseException:
            if proc is not None:
                if proc.stdin is not None and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if video_path.exists():
                video_path.unlink()
            raise

    binding_path, source_path = layout["binding_path"], layout["source_path"]
    camera_order, camera_serials = layout["camera_order"], layout["camera_serials"]
    cameras = []
    for cam in layout["camera_ids"]:
        name = camera_order[cam]
        serial = camera_serials.get(name)
        cameras.append(
            {
                "index": cam,
                "output_dir": f"cam{cam:02d}",
                "name": name,
                "original_serial": serial,
                "frames": selected,
                "binding_frame_references": [layout["pose_to_source"][pose] for pose in selected],
                "preview_video": (video_rel if video and cam == int(video_camera) else None),
            }
        )
    source_t, source_fps = layout["native_total"], layout["fps"]
    first_camera = layout["camera_ids"][0]
    preview_width = min(1008, int(dataset.img_widths[first_camera]))
    preview_width -= preview_width % 2
    preview_height = int(
        round(
            int(dataset.img_heights[first_camera])
            * preview_width
            / int(dataset.img_widths[first_camera])
        )
    )
    preview_height -= preview_height % 2
    report = {
        "schema": "gaussian_wardrobe_render_review/v1",
        "generated_at_utc": utc_now(),
        "status": "ok",
        "checkpoint": {
            "path": str(TRAIN_CKPT),
            "sha256": checkpoint_sha,
            "phase": phase,
            "updates": checkpoint_updates,
            "model_state_load": "strict; optimizer not loaded",
        },
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "data_dir": str(DATA),
        "render_mode": "native author net.render eval; configured native pose/camera layout",
        "frames": selected,
        "cameras": cameras,
        "camera_mapping_source": [str(p) for p in (binding_path, source_path) if p.is_file()],
        "source_timesteps": source_t,
        "source_fps": source_fps,
        "native_timesteps": len(layout["poses"]),
        "frame_index_semantics": "files and pose_list use native indices; binding frame references are historical metadata, not inferred decoded-video indices",
        "fps": layout["video_fps"],
        "original_resolution": [
            int(dataset.img_widths[first_camera]),
            int(dataset.img_heights[first_camera]),
        ],
        "display_resize": [preview_width, preview_height],
        "video": video_rel,
        "video_frames": len(layout["poses"]) if video else None,
        "video_fps": layout["video_fps"] if video else None,
        "outputs": {
            "png": sorted(png_paths),
            "video": video_rel,
        },
        "no_metrics": True,
        "no_thresholds": True,
        "quality_claim": None,
    }
    report_path = review / "report.json"
    tmp = report_path.with_name(report_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, report_path)
    print(json.dumps(report, indent=2))
    return 0


def run_full_pretrain(config_path: Path, updates: Optional[int]) -> int:
    import yaml

    opt = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = (
        updates if updates is not None else int(opt.get("train", {}).get("pretrain_iters", 5001))
    )
    return run_author_pretrain(config_path, target_updates=target)


def main() -> int:
    global DATA, RESULTS, TRAIN_CURRENT, TRAIN_CKPT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gen-pos-maps", action="store_true")
    g.add_argument("--gen-training-masks", action="store_true")
    g.add_argument("--pretrain", action="store_true")
    g.add_argument("--train", action="store_true")
    g.add_argument("--render", action="store_true")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-camera", type=int, default=0)
    ap.add_argument("--layer", choices=("body", "cloth", "both"), default="both")
    a = ap.parse_args()
    path = Path(a.config).resolve()
    DATA = a.data.expanduser().resolve()
    RESULTS = a.out.expanduser().resolve()
    TRAIN_CURRENT = RESULTS
    TRAIN_CKPT = RESULTS / "checkpoint.pt"
    if a.render:
        return run_render(path, a.frames, a.video, a.video_camera)
    if a.train:
        return run_author_train(path, a.updates)
    if a.pretrain:
        return run_full_pretrain(path, a.updates)
    if a.gen_training_masks:
        return generate_training_masks(DATA.resolve(), path)
    if a.gen_pos_maps:
        return generate_position_maps(path, a.layer)
    raise RuntimeError("no production command selected")


if __name__ == "__main__":
    raise SystemExit(main())
