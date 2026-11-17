"""Complete three-branch state in one atomic, rolling Multi4D checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import zipfile

import numpy as np
import torch


class Checkpoint:
    metadata_name = "hoi_checkpoint_metadata.json"
    fg_extra = ("mean_radii2D", "_deformation_accum", "sum_dx", "sum_dx_sq", "count_dx")
    tr_parameters = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_t",
        "_scaling_t",
        "_rotation_r",
    )
    tr_constants = (
        "active_sh_degree",
        "active_sh_degree_t",
        "max_sh_degree",
        "max_sh_degree_t",
        "gaussian_dim",
        "time_duration",
        "rot_4d",
        "percent_dense",
        "spatial_lr_scale",
    )
    tr_statistics = (
        "max_radii2D",
        "mean_radii2D",
        "xyz_gradient_accum",
        "t_gradient_accum",
        "denom",
    )

    def __init__(self, path: Path, signature: dict, interval: int = 500):
        self.path = Path(path)
        self.signature = json.loads(json.dumps(signature, sort_keys=True))
        self.interval = int(interval)
        self.metadata = None
        if self.path.is_file():
            # A small member avoids loading tensor storages merely to inspect the stage.
            with zipfile.ZipFile(self.path) as archive:
                members = [n for n in archive.namelist() if n.endswith("/" + self.metadata_name)]
                if len(members) != 1:
                    raise ValueError("Checkpoint lacks unambiguous rolling metadata")
                self.metadata = json.loads(archive.read(members[0]))
            self.validate_metadata(self.metadata)

    def validate_metadata(self, metadata):
        if metadata["schema"] != "hoi.multi4d.checkpoint/1":
            raise ValueError("Unsupported checkpoint schema")
        if metadata["stage"] not in ("coarse", "fine"):
            raise ValueError("Invalid checkpoint stage")
        if metadata["signature"] != self.signature:
            raise ValueError("Checkpoint inputs/configuration differ from the current run")

    def needs_stage(self, stage):
        return not (
            stage == "coarse" and self.metadata is not None and self.metadata["stage"] == "fine"
        )

    @staticmethod
    def capture_rng():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

    @staticmethod
    def restore_rng(rng):
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    @staticmethod
    def fields(model, names):
        return {name: getattr(model, name) for name in names}

    def save(self, stage, iteration, remaining_indices, models, opt):
        if stage not in ("coarse", "fine"):
            raise ValueError(f"Unknown stage: {stage}")
        metadata = {
            "schema": "hoi.multi4d.checkpoint/1",
            "signature": self.signature,
            "stage": stage,
            "iteration": int(iteration),
            "boundary": "completed native iteration; before next sampling",
            "point_counts": {key: int(model.get_xyz.shape[0]) for key, model in models.items()},
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        fg, bg, tr = models["fg"], models["bg"], models["tr"]
        payload = {
            "metadata": metadata,
            "remaining_indices": list(remaining_indices),
            "rng": self.capture_rng(),
            "fg": fg.capture(),
            "fg_extra": {n: getattr(fg, n) for n in self.fg_extra if hasattr(fg, n)},
            "bg": bg.capture(),
            "tr": {
                "parameters": self.fields(tr, self.tr_parameters),
                "constants": self.fields(tr, self.tr_constants),
                "statistics": self.fields(tr, self.tr_statistics),
                "optimizer": tr.optimizer.state_dict(),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        # Preserve native Parameter types and each storage's original device tag.
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with zipfile.ZipFile(temporary, "a") as archive:
            roots = {n.split("/", 1)[0] for n in archive.namelist()}
            if len(roots) != 1:
                raise ValueError("Unexpected torch.save archive layout")
            archive.writestr(
                f"{next(iter(roots))}/{self.metadata_name}",
                json.dumps(metadata, sort_keys=True).encode("utf-8"),
            )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        self.metadata = metadata

    def restore(self, stage, models, opt):
        if self.metadata is None or self.metadata["stage"] != stage:
            return None
        payload = torch.load(self.path, weights_only=False)
        self.validate_metadata(payload["metadata"])
        if payload["metadata"] != self.metadata:
            raise ValueError("Checkpoint changed after metadata inspection")
        fg, bg, tr = models["fg"], models["bg"], models["tr"]
        fg.restore(payload["fg"], opt)
        for name, value in payload["fg_extra"].items():
            setattr(fg, name, value)
        bg.restore(payload["bg"], opt)
        state = payload["tr"]
        for name, value in {**state["parameters"], **state["constants"]}.items():
            setattr(tr, name, value)
        tr.training_setup(opt)
        for name, value in state["statistics"].items():
            setattr(tr, name, value)
        tr.optimizer.load_state_dict(state["optimizer"])
        return (
            int(payload["metadata"]["iteration"]),
            list(payload["remaining_indices"]),
            payload["rng"],
        )
