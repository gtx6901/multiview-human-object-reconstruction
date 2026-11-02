#!/usr/bin/env python
"""Runnable integration layer for the cloned AnimatableGaussians backend.

This file intentionally calls the author's own classes/modules; it does not
reimplement the template/avatar networks or the canonical transforms.  It only:

  * puts the author vendored `smplx` before any pip/project copy;
  * supplies the existing local SMPLX model and the actual expression dimension
    (100 here) through the author constructor;
  * resolves the true data path and runs with author CWD, using ASCII-relative
    data paths for the author's cv2 image reads;
  * adds the locally built CUDA extensions from `runtime/extensions` without
    changing the shared environment;
  * exposes doctor / weight-volume / gen-pos-maps / template-train / avatar-train
    and bounded one-item forward+backward smokes.

The smokes are interface evidence, not training or quality evidence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHOR = HERE / "author"
EXT_ROOT = HERE / "runtime/extensions"
STYLE_EXT = EXT_ROOT / "styleunet"
RESULTS = HERE / "runtime/results"
TORCH_EXT = HERE / "runtime/torch_extensions"
LOCAL_SMPLX_CANDIDATES = [
    ROOT / "lhmpp/pretrained_models/human_model_files/smplx",
    ROOT / "assets/runs/lhmpp/runtime/pretrained_models/human_model_files/smplx",
    ROOT / "assets/runs/lhmpp/runtime/data/pretrained_models/human_model_files/smplx",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _insert_first(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _resolve_data_dir(opt: Dict[str, Any], author: Path) -> Path:
    raw = Path(str(opt["train"]["data"]["data_dir"]))
    return (author / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _local_smplx_dir() -> Path:
    for p in LOCAL_SMPLX_CANDIDATES:
        if (p / "SMPLX_NEUTRAL.npz").is_file():
            return p
    raise FileNotFoundError(
        "local SMPLX_NEUTRAL.npz not found; expected one of "
        + ", ".join(str(p) for p in LOCAL_SMPLX_CANDIDATES)
    )


def bootstrap(config_path: Path, require_cuda: bool = False) -> Dict[str, Any]:
    """Set local paths, author import priority, model/expression adaptation."""
    config_path = Path(config_path).resolve()
    if not (AUTHOR / "dataset/dataset_mv_rgb.py").is_file():
        raise FileNotFoundError(f"author clone missing at {AUTHOR}")
    if not (AUTHOR / "smplx/body_models.py").is_file():
        raise FileNotFoundError(f"author vendored smplx missing under {AUTHOR / 'smplx'}")

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXT))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    # author smplx must win over any pip/project copy; extension dirs come next.
    for p in reversed([AUTHOR, EXT_ROOT, STYLE_EXT, HERE / "runtime/python", ROOT / "lhmpp",
                       ROOT / "lhmpp/deps", ROOT / "runtime/deps"]):
        _insert_first(p)

    import torch
    import config

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this smoke but torch.cuda.is_available() is false")
    config.PROJ_DIR = str(AUTHOR)
    opt = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict):
        raise TypeError(f"config is not a mapping: {config_path}")
    config.opt = opt

    data_dir = _resolve_data_dir(opt, AUTHOR)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"resolved data_dir does not exist: {data_dir}")
    if not (data_dir / "smpl_params.npz").is_file():
        raise FileNotFoundError(f"missing {data_dir / 'smpl_params.npz'}")

    model_dir = _local_smplx_dir()
    with np.load(data_dir / "smpl_params.npz", allow_pickle=True) as sp:
        expr_dim = int(sp["expression"].shape[1])
        betas_shape = tuple(sp["betas"].shape)
    if betas_shape != (1, 10):
        raise ValueError(f"author reader requires betas (1,10), got {betas_shape}")

    import smplx
    if AUTHOR not in Path(smplx.__file__).resolve().parents:
        raise RuntimeError(
            f"smplx imported from {smplx.__file__}, not author vendored; "
            "do not use a pip smplx without .A"
        )
    orig_smplx = smplx.SMPLX

    class patched_smplx(orig_smplx):
        def __init__(self, *args, **kwargs):
            if args:
                args = (str(model_dir),) + tuple(args[1:])
            else:
                kwargs["model_path"] = str(model_dir)
            kwargs["num_expression_coeffs"] = expr_dim
            super().__init__(*args, **kwargs)

    smplx.SMPLX = patched_smplx
    os.chdir(AUTHOR)  # author scripts use ./bins, debug/tmp, smpl_files
    torch.set_num_threads(4)

    import dataset.dataset_mv_rgb as ds
    return {
        "torch": torch, "config": config, "smplx": smplx, "ds": ds,
        "opt": opt, "config_path": str(config_path.resolve()),
        "data_dir": str(opt["train"]["data"]["data_dir"]),
        "data_dir_absolute": str(data_dir), "model_dir": str(model_dir),
        "expr_dim": expr_dim, "author": str(AUTHOR),
        "smplx_file": str(Path(smplx.__file__).resolve()),
        "smplx_model_patch": "local model_path + num_expression_coeffs",
    }


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} missing: {path}")


def _extensions_status() -> Dict[str, Any]:
    out = {}
    for name in ("posevocab_custom_ops", "root_finding", "upfirdn2d", "fused",
                 "diff_gaussian_rasterization_depth_alpha"):
        try:
            mod = importlib.import_module(name)
            out[name] = {"ok": True, "file": str(getattr(mod, "__file__", ""))}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def cmd_doctor(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config))
    data = Path(ctx["data_dir_absolute"])
    checks = {
        "author": str(AUTHOR),
        "author_reader": str(AUTHOR / "dataset/dataset_mv_rgb.py"),
        "smplx_file": ctx["smplx_file"],
        "local_smplx": ctx["model_dir"],
        "data_dir": str(data),
        "data_dir_relative": ctx["data_dir"],
        "expression_dim": ctx["expr_dim"],
        "files": {
            "template.ply": (data / "template.ply").is_file(),
            "cano_weight_volume.npz": (data / "cano_weight_volume.npz").is_file(),
            "cano_smpl_pos_map.exr": (data / "smpl_pos_map/cano_smpl_pos_map.exr").is_file(),
            "init_pts_lbs.npy": (data / "smpl_pos_map/init_pts_lbs.npy").is_file(),
            "PointInterpolant.exe": (AUTHOR / "bins/PointInterpolant.exe").is_file(),
            "smpl_files/mano": (AUTHOR / "smpl_files/mano").is_dir(),
        },
        "extensions": _extensions_status(),
        "python": sys.executable,
        "cuda": bool(ctx["torch"].cuda.is_available()),
        "torch": ctx["torch"].__version__,
        "torch_cuda": ctx["torch"].version.cuda,
    }
    missing = [k for k, v in checks["files"].items() if not v]
    checks["missing_files"] = missing
    checks["notes"] = [
        "template.ply must be produced by author main_template.py, not copied from SMPL-X.",
        "cano_weight_volume.npz is the author Poisson/PointInterpolant prerequisite.",
        "extensions in runtime/extensions are local; no global pip install is required.",
    ]
    write_json(RESULTS / "doctor.json", checks)
    print(json.dumps(checks, indent=2))
    return 0


def _run_author(module: str, config_path: Path) -> int:
    ctx = bootstrap(config_path)
    module_file = AUTHOR / (module.replace(".", "/") + ".py")
    if not module_file.is_file():
        raise FileNotFoundError(module_file)
    sys.argv = [str(module_file), "-c", str(ctx["config_path"])]
    runpy.run_module(module, run_name="__main__")
    return 0


def run_author_module(module: str, config_path: Path, author: Optional[Path] = None) -> int:
    """Public entry used by author_entry.py; keeps all adaptation in one place."""
    if author is not None and Path(author).resolve() != AUTHOR.resolve():
        raise ValueError(f"runtime_backend is pinned to {AUTHOR}; got --author {author}")
    return _run_author(module, Path(config_path))


def cmd_weight_volume(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config))
    _require(AUTHOR / "bins/PointInterpolant.exe", "PointInterpolant.exe (official GEN_DATA.md prerequisite)")
    try:
        import igl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"author gen_weight_volume requires igl: {exc}") from exc
    _run_author("gen_data.gen_weight_volume", Path(ctx["config_path"]))
    out = Path(ctx["data_dir_absolute"]) / "cano_weight_volume.npz"
    _require(out, "author cano_weight_volume.npz")
    print(f"[runtime] wrote {out}")
    return 0


def cmd_gen_pos_maps(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config))
    data = Path(ctx["data_dir_absolute"])
    _require(data / "template.ply", "author template.ply (run main_template.py first)")
    _require(data / "cano_weight_volume.npz", "author cano_weight_volume.npz")
    _run_author("gen_data.gen_pos_maps", Path(ctx["config_path"]))
    print(f"[runtime] author position maps written under {data / 'smpl_pos_map'}")
    return 0


def cmd_template_train(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config))
    _require(Path(ctx["data_dir_absolute"]) / "cano_weight_volume.npz",
             "author cano_weight_volume.npz")
    _run_author("main_template", Path(ctx["config_path"]))
    return 0


def cmd_avatar_train(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config))
    _require(Path(ctx["data_dir_absolute"]) / "smpl_pos_map/cano_smpl_pos_map.exr",
             "author canonical map")
    _run_author("main_avatar", Path(ctx["config_path"]))
    return 0


def _template_cuda_item(torch, config, item: Dict[str, Any]) -> Dict[str, Any]:
    """Batch one dataset item exactly once for TemplateNet.render."""
    from utils.net_util import to_cuda
    nerf = item.get("nerf_random")
    base = {k: v for k, v in item.items() if k != "nerf_random"}
    items = to_cuda(base, add_batch=True)
    if nerf is None:
        raise RuntimeError("author dataset item has no nerf_random")
    converted = {}
    for k, v in nerf.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).to(config.device)
        elif isinstance(v, torch.Tensor):
            v = v.to(config.device)
        else:
            converted[k] = v
            continue
        if v.dtype in (torch.float32, torch.float64):
            v = v.float()
        converted[k] = v.unsqueeze(0)
    items["nerf_random"] = converted
    items.update(converted)
    return items


def _template_smoke(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config), require_cuda=True)
    torch = ctx["torch"]
    data = Path(ctx["data_dir_absolute"])
    _require(data / "cano_weight_volume.npz",
             "author cano_weight_volume.npz; run runtime_backend.py weight-volume")
    from network.template import TemplateNet
    from utils.net_util import to_cuda

    ds = ctx["ds"]
    opt = ctx["opt"]
    dataset = ds.MvRgbDatasetTHuman4(**opt["train"]["data"])
    pose = int(a.pose_index)
    view = int(a.view_index)
    if (pose, view) not in dataset.data_list:
        raise ValueError(f"({pose},{view}) not in author data_list")
    item = dataset.getitem(dataset.data_list.index((pose, view)), training=True)
    if "nerf_random" not in item:
        raise RuntimeError("author dataset did not produce nerf_random; mode must be nerf")
    items = _template_cuda_item(torch, ctx["config"], item)

    net = TemplateNet(opt["model"]).to(ctx["config"].device)
    net.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    out = net.render(items, depth_guided_sampling=opt["train"]["depth_guided_sampling"])
    weights = opt["train"]["loss_weight"]
    color_loss = torch.nn.L1Loss()(out["rgb_map"], items["color_gt"])
    mask_loss = torch.nn.L1Loss()(out["acc_map"], items["mask_gt"])
    loss = weights.get("color", 1.0) * color_loss + weights.get("mask", 1.0) * mask_loss
    eik = None
    if "normal" in out and weights.get("eikonal", 0.0) > 0:
        eik = ((torch.linalg.norm(out["normal"], dim=-1) - 1.0) ** 2).mean()
        loss = loss + weights["eikonal"] * eik
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    report = {
        "schema": "runtime_backend/template_smoke/v1",
        "status": "ok",
        "author_reader": str(AUTHOR / "dataset/dataset_mv_rgb.py"),
        "author_template": str(AUTHOR / "network/template.py"),
        "data_dir": str(data),
        "data_dir_relative": ctx["data_dir"],
        "pose_index": pose, "view_index": view,
        "expr_dim": ctx["expr_dim"],
        "rgb_shape": list(out["rgb_map"].shape),
        "acc_shape": list(out["acc_map"].shape),
        "normal_shape": (list(out["normal"].shape) if "normal" in out else None),
        "color_loss": float(color_loss.detach()),
        "mask_loss": float(mask_loss.detach()),
        "eikonal_loss": (float(eik.detach()) if eik is not None else None),
        "total_loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "grad_tensors": len(grads),
        "grad_tensors_finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
        "elapsed_sec": round(time.time() - started, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "scope": "one real item, author TemplateNet forward+backward; NOT training and NOT quality",
    }
    write_json(RESULTS / "template_smoke.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["loss_finite"] and report["grad_tensors_finite"] else 1


def _avatar_smoke(a: argparse.Namespace) -> int:
    ctx = bootstrap(Path(a.config), require_cuda=True)
    torch = ctx["torch"]
    data = Path(ctx["data_dir_absolute"])
    for f in ("smpl_pos_map/cano_smpl_pos_map.exr", "smpl_pos_map/cano_smpl_nml_map.exr",
              "smpl_pos_map/init_pts_lbs.npy"):
        _require(data / f, f"author avatar prerequisite {f}")

    from torch.utils.data import DataLoader, Subset
    from main_avatar import AvatarTrainer
    from network.lpips import LPIPS
    from utils.net_util import to_cuda
    from train_avatar_control import resolve_lpips_weights, vgg_cache_file, sha256_file

    opt = ctx["opt"]
    opt["train"]["num_workers"] = 0
    ds = ctx["ds"]
    dataset = ds.MvRgbDatasetTHuman4(**opt["train"]["data"])
    pose, view = int(a.pose_index), int(a.view_index)
    if (pose, view) not in dataset.data_list:
        raise ValueError(f"({pose},{view}) not in author data_list")
    idx = dataset.data_list.index((pose, view))

    # Match AvatarTrainer.train's DataLoader batch semantics (B=1, default
    # collate).  A one-index Subset is not a reimplementation of the batch loop;
    # it supplies one actual author batch to the actual forward_one_pass.
    loader = DataLoader(Subset(dataset, [idx]), batch_size=1, shuffle=False,
                        num_workers=0, drop_last=True)
    items_batched = next(iter(loader))
    items = to_cuda(items_batched)  # author train() calls to_cuda without add_batch
    if any(isinstance(v, torch.Tensor) and v.shape[0] != 1 for v in items.values()):
        raise RuntimeError("author smoke item does not have batch size 1 before delete_batch_idx")

    lpips_path = resolve_lpips_weights(None)
    vgg = vgg_cache_file()
    if not vgg.is_file():
        raise FileNotFoundError(
            f"VGG16 trunk cache missing: {vgg}; no download is allowed")

    # Exactly one AvatarTrainer/AvatarNet allocation/renderer.
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = AvatarTrainer(opt)
    lpips = LPIPS(net="vgg", model_path=str(lpips_path)).to(ctx["config"].device)
    for p in lpips.parameters():
        p.requires_grad = False
    trainer.lpips = lpips
    trainer.iter_idx = 0
    trainer.avatar_net.train()

    render_info = {}
    grad_info = {}
    original_render = trainer.avatar_net.render
    original_step = trainer.optm.step

    def capture_render(render_items, bg_color):
        out = original_render(render_items, bg_color)
        render_info.update({k: list(v.shape) if hasattr(v, "shape") else None
                            for k, v in out.items()})
        return out

    def capture_step(*args, **kwargs):
        gs = [p.grad for p in trainer.avatar_net.parameters() if p.grad is not None]
        grad_info["tensors"] = len(gs)
        grad_info["finite"] = bool(gs) and all(bool(torch.isfinite(g).all()) for g in gs)
        grad_info["elements"] = int(sum(g.numel() for g in gs))
        return original_step(*args, **kwargs)

    trainer.avatar_net.render = capture_render
    trainer.optm.step = capture_step
    try:
        total_loss, batch_losses = trainer.forward_one_pass(items)
        torch.cuda.synchronize()
    finally:
        trainer.avatar_net.render = original_render
        trainer.optm.step = original_step

    grads = [p.grad for p in trainer.avatar_net.parameters() if p.grad is not None]
    params = [p for p in trainer.avatar_net.parameters()]
    report = {
        "schema": "runtime_backend/avatar_smoke/v2",
        "status": "ok",
        "author_avatar": str(AUTHOR / "network/avatar.py"),
        "author_avatar_sha256": sha256_file(AUTHOR / "network/avatar.py"),
        "author_main_avatar": str(AUTHOR / "main_avatar.py"),
        "author_main_avatar_sha256": sha256_file(AUTHOR / "main_avatar.py"),
        "config": ctx['config_path'],
        "config_sha256": sha256_file(Path(ctx['config_path'])),
        "data_dir": str(data), "pose_index": pose, "view_index": view,
        "expr_dim": ctx["expr_dim"],
        "batch_semantics": "DataLoader batch_size=1 -> to_cuda no add_batch -> author forward_one_pass delete_batch_idx",
        "render_keys_shapes": render_info,
        "batch_losses": {k: float(v) for k, v in batch_losses.items()},
        "total_loss": float(total_loss.detach()),
        "loss_finite": bool(torch.isfinite(total_loss)),
        "optimizer_steps": 1,
        "optimizer_step_is_author_forward_one_pass": True,
        "grad_tensors_before_step": int(grad_info.get("tensors", 0)),
        "grad_tensors_finite_before_step": bool(grad_info.get("finite", False)),
        "grad_elements_before_step": int(grad_info.get("elements", 0)),
        "grad_tensors_after_author_zero_grad": len(grads),
        "parameter_tensors": len(params),
        "parameter_elements": int(sum(p.numel() for p in params)),
        "trainable_parameter_elements": int(sum(p.numel() for p in params if p.requires_grad)),
        "weights_finite": bool(params) and all(bool(torch.isfinite(p).all()) for p in params),
        "patch_size": int(trainer.patch_size),
        "lpips_weights": str(lpips_path),
        "lpips_weights_sha256": sha256_file(lpips_path),
        "vgg_cache": str(vgg),
        "elapsed_sec": round(time.time() - started, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "scope": "one actual author AvatarTrainer forward_one_pass (includes its own backward+optimizer step); random state; NOT convergence or quality",
        "pretrained_used": False,
        "quality_claim": None,
    }
    write_json(RESULTS / "avatar_smoke.json", report)
    print(json.dumps(report, indent=2))
    return 0 if (report["loss_finite"] and report["grad_tensors_finite_before_step"]
                 and report["weights_finite"]) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_template.yaml"))
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("weight-volume", help="author gen_data.gen_weight_volume (PointInterpolant)")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_template.yaml"))
    p.set_defaults(func=cmd_weight_volume)

    p = sub.add_parser("gen-pos-maps", help="author gen_data.gen_pos_maps")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_template.yaml"))
    p.set_defaults(func=cmd_gen_pos_maps)

    p = sub.add_parser("template-train", help="author main_template.py (full, not a smoke)")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_template.yaml"))
    p.set_defaults(func=cmd_template_train)

    p = sub.add_parser("avatar-train", help="author main_avatar.py (full, not a smoke)")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_avatar.yaml"))
    p.set_defaults(func=cmd_avatar_train)

    p = sub.add_parser("template-smoke", help="one real item: author TemplateNet forward+backward")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_template.yaml"))
    p.add_argument("--pose-index", type=int, default=0)
    p.add_argument("--view-index", type=int, default=0)
    p.set_defaults(func=_template_smoke)

    p = sub.add_parser("avatar-smoke", help="one real item: author AvatarNet forward+backward")
    p.add_argument("--config", default=str(HERE / "configs/mjj001_avatar.yaml"))
    p.add_argument("--pose-index", type=int, default=0)
    p.add_argument("--view-index", type=int, default=0)
    p.set_defaults(func=_avatar_smoke)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
