#!/usr/bin/env python
"""Bounded controller around the real author AvatarTrainer.

It does not replace AvatarNet, its renderer, forward_one_pass, the three
StyleUNets, LPIPS, optimizer or scheduler.  It only:
  * resolves the author/local paths through runtime_backend.bootstrap;
  * supplies the existing local LPIPS linear weights and VGG trunk explicitly;
  * sets a per-run iteration stop (with the original 800000-update LR schedule
    denominator preserved) and saves net.pt + optm.pt after the author loop;
  * optionally invokes the author's own pretrain when no checkpoint exists, only
    when explicitly requested; the author stops at iter_idx==5000, which is 5001
    actual forward/update calls;

Default budget is one actual optimizer update.  The 792-update control is
8 cameras x 99 real times for one complete data epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHOR = HERE / "author"
DEFAULT_CONFIG = HERE / "configs/mjj001_avatar.yaml"
DEFAULT_OUT = HERE / "runs/mjj001/avatar_control"
RESULTS = HERE / "runtime/results"
LPIPS_CANDIDATES = [
    ROOT / "work/mjj/runtime/perceptual/deps/lpips/weights/v0.1/vgg.pth",
    AUTHOR / "network/lpips/weights/v0.1/vgg.pth",
]
DEFAULT_BUDGET = 1
DEFAULT_WORKERS = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def check_imports() -> List[Dict[str, str]]:
    modules = [
        "tensorboard", "torch.utils.tensorboard", "torchvision",
        "pytorch3d", "trimesh", "yaml", "numpy",
    ]
    errors = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return errors


def resolve_lpips_weights(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for p in LPIPS_CANDIDATES:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "LPIPS v0.1/vgg linear weights missing; expected "
        + " or ".join(str(p) for p in LPIPS_CANDIDATES)
    )


def vgg_cache_file() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"


def checkpoint_state(path: Path) -> Tuple[int, int]:
    net = path / "net.pt"
    if not net.is_file():
        raise FileNotFoundError(f"checkpoint net.pt missing: {net}")
    data = torch.load(net, map_location="cpu", weights_only=True)
    if "epoch_idx" not in data or "iter_idx" not in data:
        raise RuntimeError(f"checkpoint lacks epoch_idx/iter_idx: {net}")
    return int(data["epoch_idx"]), int(data["iter_idx"])


def state_paths(opt: Dict[str, Any], out: Path) -> Dict[str, Optional[Path]]:
    prev = opt["train"].get("prev_ckpt")
    prev_p = Path(prev).resolve() if prev else None
    pretrained_dir = opt["train"].get("pretrained_dir")
    pretrained_p = Path(pretrained_dir).resolve() if pretrained_dir else None
    return {
        "prev": prev_p,
        "epoch_latest": out / "epoch_latest",
        "pretrained_dir": pretrained_p,
        "pretrained_out": out / "pretrained",
    }


def state_status(paths: Dict[str, Optional[Path]]) -> Dict[str, bool]:
    return {
        "prev": bool(paths["prev"] and paths["prev"].is_dir()),
        "epoch_latest": bool(paths["epoch_latest"].is_dir()),
        "pretrained_dir": bool(paths["pretrained_dir"] and paths["pretrained_dir"].is_dir()),
        "pretrained_out": bool(paths["pretrained_out"].is_dir()),
    }


def start_info(opt: Dict[str, Any], out: Path) -> Dict[str, Any]:
    paths = state_paths(opt, out)
    if paths["prev"] and paths["prev"].is_dir():
        e, i = checkpoint_state(paths["prev"])
        return {"source": "prev_ckpt", "start_epoch": e + 1, "start_iter": i + 1}
    if paths["epoch_latest"].is_dir():
        e, i = checkpoint_state(paths["epoch_latest"])
        return {"source": "epoch_latest", "start_epoch": e + 1, "start_iter": i + 1}
    if (paths["pretrained_dir"] and paths["pretrained_dir"].is_dir()) or paths["pretrained_out"].is_dir():
        return {"source": "pretrained", "start_epoch": 0, "start_iter": 0}
    return {"source": "none", "start_epoch": None, "start_iter": None}


def optimizer_resume_status(opt: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Whether the author train() will load optimizer state (not a controller load)."""
    paths = state_paths(opt, out)
    if paths["prev"] and paths["prev"].is_dir():
        p = paths["prev"] / "optm.pt"
        return {"source": "prev_ckpt", "path": str(p), "present": p.is_file(),
                "author_loads_optm": p.is_file()}
    if paths["epoch_latest"].is_dir():
        p = paths["epoch_latest"] / "optm.pt"
        return {"source": "epoch_latest", "path": str(p), "present": p.is_file(),
                "author_loads_optm": p.is_file()}
    return {"source": "pretrained_or_none", "path": None, "present": False,
            "author_loads_optm": False}


class BoundedAvatarTrainer:
    """Mixin mixed into the real main_avatar.AvatarTrainer."""

    def configure_budget(self, max_updates: int, out_dir: Path, schedule_total: int,
                         lpips_path: Path) -> None:
        self.max_updates = int(max_updates)
        self.out_dir = Path(out_dir)
        self.schedule_total = int(schedule_total)
        self.lpips_path = Path(lpips_path)
        self._actual_updates = 0
        self._start_iter = 0
        self._last_iter = 0
        self._budget_saved = False
        self._pretrain_forward_calls = 0

    def update_lr(self):
        # Exact author formula, with the original 800000-update denominator even
        # though this control run stops after a smaller budget.
        alpha = 0.05
        progress = self.iter_idx / self.schedule_total
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        lr = self.lr_init * learning_factor
        for param_group in self.optm.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self):
        info = start_info(self.opt, self.out_dir)
        if info["start_iter"] is None:
            raise FileNotFoundError(
                "No avatar pretrained/prev checkpoint. The controller does not "
                "invent a pretrained state; pass --prev-ckpt or --pretrain-if-missing."
            )
        self._start_iter = int(info["start_iter"])
        self.iter_num = self._start_iter + self.max_updates - 1

        # The author AvatarTrainer looks up module-global LPIPS; pass only the
        # explicit local weights path, keeping the same class and loss.
        import main_avatar as ma
        orig_lpips = ma.LPIPS

        def patched_lpips(*args, **kwargs):
            kwargs.setdefault("net", "vgg")
            kwargs.setdefault("model_path", str(self.lpips_path))
            return orig_lpips(*args, **kwargs)

        if self.opt["train"]["loss_weight"].get("lpips", 0.0) > 0:
            ma.LPIPS = patched_lpips
        try:
            super().train()
        finally:
            if self.opt["train"]["loss_weight"].get("lpips", 0.0) > 0:
                ma.LPIPS = orig_lpips

        self._last_iter = int(self.iter_idx)
        self._actual_updates = max(0, self._last_iter - self._start_iter + 1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_ckpt(str(self.out_dir), save_optm=True)
        self._budget_saved = True

    def run_pretrain_if_requested(self, enabled: bool) -> int:
        status = state_status(state_paths(self.opt, self.out_dir))
        if any(status.values()):
            return 0
        if not enabled:
            return 0
        # Transparent counting wrapper only: delegate to the original bound
        # method unchanged.  The author loop stops at iter_idx == 5000, i.e.
        # 5000..0 inclusive = 5001 actual forward/update calls.
        original_forward = self.forward_one_pass_pretrain
        count = 0

        def counted_forward(items):
            nonlocal count
            count += 1
            return original_forward(items)

        self.forward_one_pass_pretrain = counted_forward
        try:
            self.pretrain()  # original author target index 5000; no shortening
        finally:
            self.forward_one_pass_pretrain = original_forward
            self._pretrain_forward_calls = count
        return count


def make_trainer() -> Any:
    from main_avatar import AvatarTrainer

    class BoundedAvatarTrainerImpl(BoundedAvatarTrainer, AvatarTrainer):
        pass

    return BoundedAvatarTrainerImpl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-updates", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--prev-ckpt", default=None)
    ap.add_argument("--pretrain-if-missing", action="store_true",
                    help="run the author pretrain if no state exists (author target index 5000 = 5001 actual calls)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="process-only dataloader override; default 0 for Windows")
    ap.add_argument("--lpips-weights", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.max_updates < 1:
        raise ValueError("--max-updates must be >= 1")
    config_path = Path(a.config).resolve()
    out_dir = Path(a.out).resolve()
    prev_ckpt = Path(a.prev_ckpt).resolve() if a.prev_ckpt else None
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not out_dir.is_relative_to(HERE):
        raise ValueError(f"--out must resolve inside the cloth_dynamics workspace: {out_dir}")
    if out_dir.is_relative_to((HERE / "data").resolve()):
        raise ValueError("--out must not be under data/")
    if out_dir == (HERE / "runs/mjj001/avatar").resolve():
        raise ValueError("--out must be a control directory, not the original baseline runs/mjj001/avatar")
    if prev_ckpt is not None and not prev_ckpt.is_dir():
        raise FileNotFoundError(prev_ckpt)

    sys.path.insert(0, str(HERE))
    from runtime_backend import bootstrap
    ctx = bootstrap(config_path, require_cuda=not a.dry_run)
    config = ctx["config"]
    opt = ctx["opt"]
    data_dir = Path(ctx["data_dir_absolute"])
    if prev_ckpt is not None:
        opt["train"]["prev_ckpt"] = str(prev_ckpt)
    opt["train"]["net_ckpt_dir"] = str(out_dir)
    opt["train"]["num_workers"] = int(a.workers)
    config.opt = opt

    # Keep author seeds from main_avatar.__main__.
    torch.manual_seed(31359)
    np.random.seed(31359)

    for rel in ("smpl_pos_map/cano_smpl_pos_map.exr",
                "smpl_pos_map/cano_smpl_nml_map.exr",
                "smpl_pos_map/init_pts_lbs.npy",
                "smpl_params.npz"):
        p = data_dir / rel
        if not p.is_file():
            raise FileNotFoundError(f"author avatar map/input missing: {p}")

    missing = check_imports()
    lpips_path = resolve_lpips_weights(a.lpips_weights)
    vgg = vgg_cache_file()
    if not vgg.is_file():
        raise FileNotFoundError(
            f"VGG16 trunk cache missing: {vgg}; no download is allowed. "
            "Provide compatible local LPIPS/VGG assets."
        )

    status = state_status(state_paths(opt, out_dir))
    info = start_info(opt, out_dir)
    if a.dry_run:
        # Import the actual module for API/import validation only; no model/GPU.
        if not missing:
            try:
                import main_avatar as ma  # noqa: F401
            except Exception as exc:
                missing.append({"module": "main_avatar", "error": f"{type(exc).__name__}: {exc}"})
        report = {
            "schema": "train_avatar_control_dry/v1",
            "status": "dry_run_ok" if not missing else "blocked_missing_imports",
            "config": str(config_path), "data_dir": str(data_dir),
            "out": str(out_dir), "prev_ckpt": (str(prev_ckpt) if prev_ckpt else None),
            "state_status": status, "start": info,
            "max_updates_requested": int(a.max_updates),
            "workers_override": int(a.workers),
            "lpips_weights": str(lpips_path), "vgg_cache": str(vgg),
            "pretrain_if_missing": bool(a.pretrain_if_missing),
            "expression_dim": ctx["expr_dim"], "missing_imports": missing,
            "scope": "CPU import/config dry run; not a GPU or quality test",
        }
        write_json(RESULTS / "train_avatar_control_dry.json", report)
        print(json.dumps(report, indent=2))
        return 0 if not missing else 2
    if missing:
        report = {"status": "blocked_missing_imports", "missing_imports": missing}
        write_json(RESULTS / "train_avatar_control_dry.json", report)
        print(json.dumps(report, indent=2))
        return 2

    Trainer = make_trainer()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = Trainer(opt)
    schedule_total = int(trainer.iter_num)  # author 800000 before budget override
    trainer.configure_budget(a.max_updates, out_dir, schedule_total, lpips_path)

    try:
        pretrain_updates = trainer.run_pretrain_if_requested(a.pretrain_if_missing)
        if pretrain_updates:
            info = start_info(opt, out_dir)
            if info["start_iter"] is None:
                raise RuntimeError("author pretrain did not leave out/pretrained")
        trainer.train()
        torch.cuda.synchronize()
    except Exception as exc:
        write_json(RESULTS / "train_avatar_control.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - started, 3),
            "config": str(config_path), "out": str(out_dir),
            "pretrain_if_missing": bool(a.pretrain_if_missing),
            "pretrain_target_index": 5000,
            "pretrain_actual_forward_calls": int(
                getattr(trainer, "_pretrain_forward_calls", 0)),
        })
        raise

    net_files = [out_dir / "net.pt", out_dir / "optm.pt"]
    finite = []
    for p in trainer.avatar_net.parameters():
        if p.numel() == 0:
            continue
        finite.append(bool(torch.isfinite(p).all()))
    report = {
        "schema": "train_avatar_control/v1",
        "status": "ok",
        "generated_at_utc": utc_now(),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "data_dir": str(data_dir), "out": str(out_dir),
        "author_avatar_source": str(AUTHOR / "network/avatar.py"),
        "author_avatar_source_sha256": sha256_file(AUTHOR / "network/avatar.py"),
        "author_main_avatar_source": str(AUTHOR / "main_avatar.py"),
        "author_main_avatar_source_sha256": sha256_file(AUTHOR / "main_avatar.py"),
        "smpl_params": str(data_dir / "smpl_params.npz"),
        "smpl_params_sha256": sha256_file(data_dir / "smpl_params.npz"),
        "prev_ckpt": (str(prev_ckpt) if prev_ckpt else None),
        "state_before": status,
        "start": info,
        "resume_optimizer": optimizer_resume_status(opt, out_dir),
        "max_updates_requested": int(a.max_updates),
        "actual_updates": int(trainer._actual_updates),
        "pretrain_target_index": 5000,
        "pretrain_actual_forward_calls": int(trainer._pretrain_forward_calls),
        "start_iter": int(trainer._start_iter),
        "last_iter": int(trainer._last_iter),
        "budget_iter_num": int(trainer.iter_num),
        "schedule_total_iter": int(trainer.schedule_total),
        "workers_override": int(a.workers),
        "num_workers_effective": int(opt["train"]["num_workers"]),
        "expression_dim": ctx["expr_dim"],
        "lpips_weights": str(lpips_path), "lpips_weights_sha256": sha256_file(lpips_path),
        "vgg_cache": str(vgg),
        "checkpoint_net": str(net_files[0]), "checkpoint_optm": str(net_files[1]),
        "checkpoint_exists": [p.is_file() for p in net_files],
        "checkpoint_sizes": [p.stat().st_size if p.is_file() else None for p in net_files],
        "weights_finite": bool(finite) and all(finite),
        "elapsed_sec": round(time.time() - started, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "lpips_path_injection": "module-global main_avatar.LPIPS wrapper with explicit local weights; class/objective unchanged",
        "resume_semantics": "author checkpoints are epoch/iter-granularity; no mid-epoch bitwise claim",
        "author_batch_cleanup": "AvatarTrainer.train deletes --out/batch_* at ckpt_interval.batch; --out is scoped to the control directory, never the baseline",
        "scope": "bounded actual author AvatarTrainer control; not quality or full-target evidence",
    }
    write_json(RESULTS / "train_avatar_control.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["weights_finite"] and all(report["checkpoint_exists"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
