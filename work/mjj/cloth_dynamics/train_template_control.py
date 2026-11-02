#!/usr/bin/env python
"""Bounded controller for the real author TemplateTrainer.

It does not replace TemplateNet, forward_one_pass, the objective, sampling,
optimizer or schedules.  It uses runtime_backend.bootstrap, instantiates the
author main_template.TemplateTrainer, sets only the iteration budget/output and
calls the author's actual BaseTrainer.train.  At the early-return point (which
does not save by itself in BaseTrainer), it saves net.pt + optm.pt to --out.
Canonical mesh extraction is disabled by default; if requested it calls the
author reconstructor and refuses to overwrite an existing data/template.ply.
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHOR = HERE / "author"
DEFAULT_CONFIG = HERE / "configs/mjj001_template.yaml"
DEFAULT_OUT = HERE / "runs/mjj001/template_control"
RESULTS = HERE / "runtime/results"
DEFAULT_MAX_UPDATES = 792


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
    """Keep the author's logging path; report missing deps instead of disabling it."""
    modules = [
        "tensorboard", "torch.utils.tensorboard", "yaml", "numpy",
        "trimesh", "pytorch3d",
    ]
    errors = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return errors


def checkpoint_epoch(path: Path) -> int:
    net = path / "net.pt"
    if not net.is_file():
        raise FileNotFoundError(f"checkpoint net.pt missing: {net}")
    data = torch.load(net, map_location="cpu", weights_only=True)
    if "epoch_idx" not in data:
        raise RuntimeError(f"checkpoint has no epoch_idx: {net}")
    return int(data["epoch_idx"])


class BudgetComplete(Exception):
    """Raised only after the budget checkpoint has been saved and the author run
    reaches canonical extraction while --extract-template is false."""


class BudgetTemplateTrainer:
    """Mix-in built around the actual TemplateTrainer/BaseTrainer methods."""

    def configure_budget(self, max_updates: int, out_dir: Path, extract: bool) -> None:
        self.max_updates = int(max_updates)
        self.out_dir = Path(out_dir)
        self.extract = bool(extract)
        self._budget_saved = False
        self._budget_done = False
        self._actual_updates = 0
        self._start_iter = 0
        self._last_iter = 0
        self._start_epoch = 0
        self._last_epoch = -1

    def _budget_start_epoch(self) -> int:
        prev = self.opt["train"].get("prev_ckpt")
        if prev:
            p = Path(prev)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            return checkpoint_epoch(p) + 1
        latest = self.out_dir / "epoch_latest"
        if latest.is_dir():
            return checkpoint_epoch(latest) + 1
        return int(self.opt["train"].get("start_epoch", 0))

    def train(self):
        batch_size = int(self.opt["train"].get("batch_size", 1))
        self.batch_num = len(self.dataset) // batch_size
        if self.batch_num <= 0:
            raise RuntimeError("empty author training dataset")
        start_epoch = self._budget_start_epoch()
        self._start_epoch = start_epoch
        self._start_iter = start_epoch * self.batch_num
        # BaseTrainer stops after executing the batch whose iter_idx == iter_num.
        self.iter_num = self._start_iter + self.max_updates - 1
        if self.iter_num < self._start_iter:
            raise ValueError("max_updates must be >= 1")
        self._last_epoch = start_epoch - 1

        # Actual author loop; its early return deliberately has no save, which
        # this wrapper repairs below.
        super().train()

        self._last_iter = int(getattr(self, "iter_idx", self._start_iter))
        self._actual_updates = max(0, self._last_iter - self._start_iter + 1)
        self._last_epoch = int(getattr(self, "epoch_idx", start_epoch))
        # BaseTrainer early return skips its end-of-epoch save; save both net and
        # optimizer state at the wrapper level.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_ckpt(str(self.out_dir), save_optm=True)
        self._budget_saved = True
        self._budget_done = True


def make_trainer() -> Any:
    from main_template import TemplateTrainer  # actual author class

    class BoundedTemplateTrainer(BudgetTemplateTrainer, TemplateTrainer):
        def test_geometry(self, *args, **kwargs):
            if not self.extract and self._budget_done:
                raise BudgetComplete(
                    "bounded template training reached checkpoint; canonical extraction disabled"
                )
            return super().test_geometry(*args, **kwargs)

    return BoundedTemplateTrainer


def expected_patch_evidence() -> Dict[str, Any]:
    return {
        "base_trainer": "BaseTrainer.train early return at iter_idx == iter_num skips epoch save",
        "controller": "saves net.pt and optm.pt immediately after super().train returns",
        "resume": "epoch-granularity checkpoints only; no mid-epoch bitwise resume claim",
        "forward_one_pass": "not overridden",
        "network": "not replaced",
        "sampling_loss_optimizer": "not overridden",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES,
                    help="number of actual optimizer updates in this controller run (default 792)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="checkpoint directory; net.pt and optm.pt are written here")
    ap.add_argument("--prev-ckpt", default=None)
    ap.add_argument("--extract-template", action="store_true",
                    help="after budget, run the real author canonical mesh extraction")
    ap.add_argument("--dry-run", action="store_true",
                    help="check imports/config/budget without training")
    a = ap.parse_args()

    config_path = Path(a.config).resolve()
    out_dir = Path(a.out).resolve()
    prev_ckpt = Path(a.prev_ckpt).resolve() if a.prev_ckpt else None
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if out_dir.is_relative_to((HERE / "data").resolve()):
        raise ValueError("--out must not be under data/; use runs/ or a checkpoint directory")

    # Import priority/model/100-D expression and author CWD.
    sys.path.insert(0, str(HERE))
    from runtime_backend import bootstrap
    ctx = bootstrap(config_path, require_cuda=not a.dry_run)
    config = ctx["config"]
    opt = ctx["opt"]
    data_dir = Path(ctx["data_dir_absolute"])
    if prev_ckpt is not None:
        if not prev_ckpt.is_dir():
            raise FileNotFoundError(prev_ckpt)
        opt["train"]["prev_ckpt"] = str(prev_ckpt)
    opt["train"]["net_ckpt_dir"] = str(out_dir)
    config.opt = opt

    # The author main_template.py sets these seeds in __main__; reproduce that
    # when invoking the class directly, without changing batch sampling.
    torch.manual_seed(31359)
    np.random.seed(31359)

    missing = check_imports()
    template_path = data_dir / "template.ply"
    if a.extract_template and template_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing author template: {template_path}"
        )
    volume = data_dir / "cano_weight_volume.npz"
    if not volume.is_file():
        raise FileNotFoundError(
            f"author cache missing: {volume}; run resume_weight_volume.py first"
        )

    report: Dict[str, Any] = {
        "schema": "train_template_control/v1",
        "generated_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_dir": str(data_dir),
        "data_dir_relative_author": ctx["data_dir"],
        "author_template_network": str(AUTHOR / "network/template.py"),
        "author_trainer": str(AUTHOR / "main_template.py"),
        "author_volume": str(volume),
        "author_volume_sha256": sha256_file(volume),
        "local_smplx": ctx["model_dir"],
        "smplx_file": ctx["smplx_file"],
        "expression_dim": ctx["expr_dim"],
        "out": str(out_dir),
        "prev_ckpt": (str(prev_ckpt) if prev_ckpt else None),
        "max_updates_requested": int(a.max_updates),
        "extract_template": bool(a.extract_template),
        "missing_imports": missing,
        "controller_patch": expected_patch_evidence(),
        "scope": "bounded actual author template training control; not quality evidence",
    }
    if missing:
        report["status"] = "blocked_missing_imports"
        write_json(RESULTS / "train_template_control.json", report)
        print(json.dumps(report, indent=2))
        return 2
    if a.max_updates < 1:
        raise ValueError("--max-updates must be >= 1")

    Trainer = make_trainer()
    trainer = Trainer(opt)
    if not isinstance(trainer, object) or not hasattr(trainer, "run"):
        raise RuntimeError("author TemplateTrainer construction failed")
    trainer.configure_budget(a.max_updates, out_dir, a.extract_template)

    if a.dry_run:
        batch_size = int(opt["train"].get("batch_size", 1))
        batch_num = len(trainer.dataset) // batch_size if trainer.dataset is not None else None
        report.update({
            "status": "dry_run_ok",
            "batch_size": batch_size,
            "dataset_length": len(trainer.dataset) if trainer.dataset is not None else None,
            "batch_num": batch_num,
            "start_epoch_would_be": trainer._budget_start_epoch() if trainer.dataset is not None else None,
        })
        write_json(RESULTS / "train_template_control_dry.json", report)
        print(json.dumps(report, indent=2))
        return 0

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.run()  # actual author setup + bounded BaseTrainer.train + extraction policy
    except BudgetComplete:
        pass
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_sec": round(time.time() - started, 3)})
        write_json(RESULTS / "train_template_control.json", report)
        raise
    torch.cuda.synchronize()

    net_files = [out_dir / "net.pt", out_dir / "optm.pt"]
    finite = []
    for p in trainer.network.parameters():
        if p.numel() == 0:
            continue
        finite.append(bool(torch.isfinite(p).all()))
    report.update({
        "status": "ok",
        "actual_updates": int(trainer._actual_updates),
        "start_epoch": int(trainer._start_epoch),
        "last_epoch": int(trainer._last_epoch),
        "start_iter": int(trainer._start_iter),
        "last_iter": int(trainer._last_iter),
        "batch_num": int(trainer.batch_num),
        "budget_saved": bool(trainer._budget_saved),
        "checkpoint_net": str(net_files[0]),
        "checkpoint_optm": str(net_files[1]),
        "checkpoint_exists": [p.is_file() for p in net_files],
        "checkpoint_sizes": [p.stat().st_size if p.is_file() else None for p in net_files],
        "weights_finite": bool(finite) and all(finite),
        "elapsed_sec": round(time.time() - started, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "template_ply": str(template_path) if a.extract_template else None,
        "template_ply_exists": template_path.is_file() if a.extract_template else None,
        "template_ply_sha256": (sha256_file(template_path)
                                if a.extract_template and template_path.is_file() else None),
        "resume_semantics": "epoch-granularity checkpoint only; no mid-epoch bitwise resume claim",
    })
    write_json(RESULTS / "train_template_control.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
