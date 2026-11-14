"""Local third-party libraries and compiled rendering dependencies."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def runtime_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings: dict[str, Any] = {"project_root": str(PROJECT_ROOT), "lhm_root": "lhmpp"}
    config_path = Path(
        os.environ.get("HOI_RUNTIME_CONFIG", PROJECT_ROOT / "configs/runtime.local.json")
    )
    if config_path.is_file():
        settings.update(json.loads(config_path.read_text(encoding="utf-8")))
    settings.update(overrides or {})
    root = Path(settings["project_root"]).resolve()
    settings["project_root"] = str(root)
    for key in (
        "lhm_root",
        "gsplat_extension",
        "ffmpeg",
        "hamer_root",
        "nlf_model",
        "sam2_root",
        "sam2_checkpoint",
        "rtmw_model",
    ):
        if settings.get(key):
            path = Path(settings[key])
            settings[key] = str(path if path.is_absolute() else root / path)
    return settings


def configure_runtime(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = runtime_settings(settings)
    os.environ["SPCONV_DISABLE_JIT"] = "1"
    root, lhm = Path(settings["project_root"]), Path(settings["lhm_root"])
    paths = [
        str(path)
        for path in (lhm / "vendor", lhm / "deps", root / "runtime/deps", lhm)
        if path.is_dir()
    ]
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    from hoi.compat import configure_vendor

    configure_vendor(lhm)
    return settings


def load_rasterization(settings: dict[str, Any] | None = None):
    settings = configure_runtime(settings)
    if settings.get("gsplat_extension"):
        path = Path(settings["gsplat_extension"])
        spec = importlib.util.spec_from_file_location("gsplat_cuda", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Gaussian rasterizer: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gsplat.csrc"] = module
    from gsplat import rasterization

    return rasterization
