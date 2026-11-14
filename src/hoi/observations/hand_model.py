"""Load the official image-conditioned HaMeR hand predictor."""

from pathlib import Path
import sys
import types

from hoi.runtime import configure_runtime, runtime_settings


def load_hand_predictor():
    settings = configure_runtime()
    root = Path(settings.get("hamer_root", Path(settings["project_root"]) / "runtime/hamer"))
    for name in (
        "hamer",
        "hamer.models",
        "hamer.models.heads",
        "hamer.models.components",
        "hamer.utils",
        "hamer.datasets",
    ):
        module = types.ModuleType(name)
        module.__path__ = [str((root / "source").joinpath(*name.split(".")))]
        sys.modules[name] = module
    import torch
    from yacs.config import CfgNode
    from accelerate import init_empty_weights
    from hamer.models.backbones import create_backbone
    from hamer.models.heads.mano_head import build_mano_head

    config = CfgNode.load_cfg(
        (root / "_DATA/hamer_ckpts/model_config.yaml").read_text(encoding="utf-8")
    )
    config.MODEL.BBOX_SHAPE = [192, 256]
    config.MANO.MEAN_PARAMS = str(root / "_DATA/data/mano_mean_params.npz")
    config.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    config.freeze()

    class HandPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = create_backbone(config)
            self.mano_head = build_mano_head(config)

        def forward(self, images):
            return self.mano_head(self.backbone(images[:, :, :, 32:-32]))

    with init_empty_weights(include_buffers=False):
        model = HandPredictor()
    checkpoint = root / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
    selected = {
        key: value for key, value in state.items() if key.startswith(("backbone.", "mano_head."))
    }
    model.load_state_dict(selected, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("Hand predictor checkpoint did not populate every parameter")
    model.eval().requires_grad_(False)
    return model, config, root
