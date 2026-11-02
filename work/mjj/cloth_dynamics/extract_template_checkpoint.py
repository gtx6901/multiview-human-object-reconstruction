#!/usr/bin/env python
"""Extract the canonical template from an actual author TemplateTrainer checkpoint.

This is checkpoint-only extraction:
  * actual runtime_backend.bootstrap / author TemplateNet / TemplateTrainer;
  * actual net.pt and optional optm.pt state;
  * actual author test_geometry(space='cano', testing_res=(256, 256, 128));
  * actual author save_mesh_as_ply, written only to an explicit --out path.

There is no retraining, no optimizer step, no substitute SMPL mesh, and no
resolution/algorithm change. The current output is replaced atomically;
checkpoint hashes in the report identify the source without versioned copies.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/mjj001_template.yaml"
DEFAULT_CHECKPOINT = HERE / "runs/mjj001/template_convergence"
DEFAULT_OUT = HERE / "runs/mjj001/template_convergence/eval/template_current.ply"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_np(x: Any):
    import numpy as np
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    config_path = Path(a.config).resolve()
    checkpoint = Path(a.checkpoint).resolve()
    out = Path(a.out).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not (checkpoint / "net.pt").is_file():
        raise FileNotFoundError(checkpoint / "net.pt")
    if not out.is_relative_to(HERE):
        raise ValueError(f"--out must stay under {HERE}")

    sys.path.insert(0, str(HERE))
    from runtime_backend import bootstrap
    ctx = bootstrap(config_path, require_cuda=True)
    config = ctx["config"]
    opt = ctx["opt"]
    data_dir = Path(ctx["data_dir_absolute"])

    import importlib
    import numpy as np
    import torch
    from utils.net_util import to_cuda
    from utils.obj_io import save_mesh_as_ply
    import utils.lr_schedule as lr_schedule
    from main_template import TemplateNet, TemplateTrainer

    dataset_module = opt["train"].get("dataset", "MvRgbDatasetTHuman4")
    Dataset = importlib.import_module("dataset.dataset_mv_rgb").__getattribute__(dataset_module)
    dataset = Dataset(**opt["train"]["data"])

    trainer = TemplateTrainer(opt)
    trainer.set_dataset(dataset)
    trainer.set_network(TemplateNet(opt["model"]).to(config.device))
    trainer.set_net_dict({"network": trainer.network})
    trainer.set_optm_dict({"network": torch.optim.Adam(trainer.network.parameters(), lr=1e-3)})
    trainer.set_lr_schedule_dict({
        "network": lr_schedule.get_learning_rate_schedules(**opt["train"]["lr"]["network"])
    })
    trainer.set_update_keys(["network"])

    optm_path = checkpoint / "optm.pt"
    epoch_loaded = trainer.load_ckpt(str(checkpoint), load_optm=optm_path.is_file())
    ckpt_data = torch.load(checkpoint / "net.pt", map_location="cpu", weights_only=True)
    if "epoch_idx" not in ckpt_data:
        raise RuntimeError(f"checkpoint has no epoch_idx: {checkpoint / 'net.pt'}")
    epoch_saved = int(ckpt_data["epoch_idx"])

    test_src = inspect.getsource(trainer.test_geometry)
    save_src = inspect.getsource(save_mesh_as_ply)

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    items = to_cuda(dataset.getitem(0, training=False), add_batch=True)
    with torch.no_grad():
        trainer.network.eval()
        vertices, faces, normals = trainer.test_geometry(
            items, space="cano", testing_res=(256, 256, 128))
    vertices = to_np(vertices)
    faces = to_np(faces)
    normals = to_np(normals) if normals is not None else None
    temp_out = out.with_name(out.stem + ".tmp.ply")
    save_mesh_as_ply(str(temp_out), vertices, faces, normals)
    os.replace(temp_out, out)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated())

    import trimesh
    mesh = trimesh.load(str(out), process=False)
    report: Dict[str, Any] = {
        "schema": "extract_template_checkpoint/v1",
        "generated_at_utc": utc_now(),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_net": str(checkpoint / "net.pt"),
        "checkpoint_net_sha256": sha256_file(checkpoint / "net.pt"),
        "checkpoint_optm": str(optm_path) if optm_path.is_file() else None,
        "checkpoint_optm_sha256": sha256_file(optm_path) if optm_path.is_file() else None,
        "checkpoint_optm_loaded": bool(optm_path.is_file()),
        "epoch_idx_file": epoch_saved,
        "epoch_idx_loaded": epoch_loaded,
        "out": str(out), "out_sha256": sha256_file(out),
        "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
        "normals": (int(normals.shape[0]) if normals is not None else None),
        "bounds_min": np.asarray(mesh.vertices).min(0).tolist(),
        "bounds_max": np.asarray(mesh.vertices).max(0).tolist(),
        "watertight": bool(getattr(mesh, "is_watertight", False)),
        "elapsed_sec": round(elapsed, 3),
        "peak_vram_bytes": peak,
        "test_geometry": "TemplateTrainer.test_geometry(space='cano', testing_res=(256,256,128))",
        "test_geometry_source_sha256": sha256_text(test_src),
        "save_mesh_as_ply_source_sha256": sha256_text(save_src),
        "retrained": False,
        "optimizer_step": False,
        "substituted_smpl": False,
        "resolution_changed": False,
        "output_policy": "replace current output atomically; source checkpoint hash retained",
        "scope": "actual checkpoint extraction only; no quality claim",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
