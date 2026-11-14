"""Defer unused upstream CUDA modules for the selected LHM++ architecture."""

import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

IMPORT_PATCHES = {
    "core.models.modeling_humana4o_lrm": (
        "core/models/modeling_humana4o_lrm.py",
        "from core.models.encoders.pointtransformer.point_transformer_warp import (\n    PointTransformerEncoder,\n)",
        "def PointTransformerEncoder(*args, **kwargs):\n    from core.models.encoders.pointtransformer.point_transformer_warp import PointTransformerEncoder as actual\n    return actual(*args, **kwargs)",
    ),
    "core.models.rendering.base_gs_render": (
        "core/models/rendering/base_gs_render.py",
        "from diff_gaussian_rasterization import (\n    GaussianRasterizationSettings,\n    GaussianRasterizer,\n)",
        "def GaussianRasterizationSettings(*args, **kwargs):\n    from diff_gaussian_rasterization import GaussianRasterizationSettings as actual\n    return actual(*args, **kwargs)\ndef GaussianRasterizer(*args, **kwargs):\n    from diff_gaussian_rasterization import GaussianRasterizer as actual\n    return actual(*args, **kwargs)",
    ),
}


class DeferredModelImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname in IMPORT_PATCHES:
            return importlib.util.spec_from_file_location(
                fullname, self.root / "vendor" / IMPORT_PATCHES[fullname][0], loader=self
            )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative, before, after = IMPORT_PATCHES[module.__name__]
        filename = self.root / "vendor" / relative
        source = filename.read_text(encoding="utf-8")
        if source.count(before) != 1:
            raise ImportError(f"Unsupported upstream module revision: {filename}")
        exec(compile(source.replace(before, after), str(filename), "exec"), module.__dict__)


def configure_vendor(root: Path) -> None:
    config = json.loads((root / "checkpoint_config.json").read_text(encoding="utf-8"))
    if (
        config["latent_query_points_type"] != "e2e_points"
        or config["gs_rendering"] != "featbacksplat"
    ):
        raise ValueError(
            "The selected checkpoint requires the e2e_points / featbacksplat architecture"
        )
    os.environ["HF_HOME"] = str(root / "hf_cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not any(
        isinstance(finder, DeferredModelImports) and finder.root == root for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, DeferredModelImports(root))
