"""Strict loading of the shared LHM++ model and its pretrained components."""

import inspect
import json
import os
from pathlib import Path

from hoi.runtime import configure_runtime


def load_model(runtime: dict | None = None):
    settings = configure_runtime(runtime)
    root = Path(settings["lhm_root"])
    import numpy as np
    import torch
    from accelerate import Accelerator, init_empty_weights
    from safetensors import safe_open

    torch.set_num_threads(int(settings.get("cpu_threads", 4)))

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in (
        ("bool", bool),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)
    os.environ["SPCONV_DISABLE_JIT"] = "1"
    from core.models import modeling_humana4o_lrm as upstream
    from core.models.encoders.dinov2_wrapper import Dinov2Wrapper

    config = json.loads((root / "checkpoint_config.json").read_text(encoding="utf-8"))
    config["transformer_decoder"]["point_backbone"]["enable_flash"] = False
    model_class = upstream.ModelHumanA4OLRM
    original_transformer = model_class.build_transformer
    original_renderer = model_class._build_neural_renderer
    original_dino = Dinov2Wrapper._build_dinov2

    def defer(function):
        def wrapped(*args, **kwargs):
            with init_empty_weights(include_buffers=False):
                return function(*args, **kwargs)

        return wrapped

    def dino_from_checkpoint(model_name, modulation_dim=None, pretrained=True):
        with init_empty_weights(include_buffers=False):
            return original_dino(model_name, modulation_dim=modulation_dim, pretrained=False)

    model_class.build_transformer = defer(original_transformer)
    model_class._build_neural_renderer = defer(original_renderer)
    Dinov2Wrapper._build_dinov2 = staticmethod(dino_from_checkpoint)
    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        Accelerator()
        model = model_class(**config)
        model.eval()
    finally:
        os.chdir(previous_directory)
        model_class.build_transformer = original_transformer
        model_class._build_neural_renderer = original_renderer
        Dinov2Wrapper._build_dinov2 = staticmethod(original_dino)
    with safe_open(
        root / "weights/LHMPP-700M/model.safetensors",
        framework="pt",
        device="cpu",
        # Windows copy-on-write mappings can exhaust commit during construction.
        # pread reads the same tensors without a file-wide PyTorch storage map.
        backend="pread",
    ) as checkpoint:
        state = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    if type(model.neural_renderer).__name__ != "PatchDPT4DecoderOnly":
        raise TypeError("This checkpoint requires PatchDPT4DecoderOnly")
    if any(key.startswith(("neural_renderer.transformer_block.", "id_face_net.")) for key in state):
        raise ValueError("Checkpoint component layout differs from the supported LHM++ release")
    model.neural_renderer.transformer_block = torch.nn.Module()
    state.update(
        {"id_face_net." + key: value for key, value in model.id_face_net.state_dict().items()}
    )
    model.load_state_dict(state, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("Identity checkpoint did not populate every parameter")
    return model
