"""Memory-bounded training utilities for the real Wardrobe author model.

CPUAdam keeps the FP32 model on CUDA while all Adam moments live on CPU.  It
intentionally delegates every optimizer detail to the native torch Adam over
same-shaped CPU mirror parameters; no Adam formula is reimplemented here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.checkpoint import checkpoint


class CPUAdam:
    """Native torch.optim.Adam over CPU mirrors of CUDA model parameters."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter], lr: float):
        self.model_params: List[torch.nn.Parameter] = [
            p for p in parameters if p.requires_grad
        ]
        self.mirror_params: List[torch.nn.Parameter] = [
            torch.nn.Parameter(
                p.detach().to(device="cpu", dtype=torch.float32).clone(),
                requires_grad=True,
            )
            for p in self.model_params
        ]
        # No extra options: native Adam defaults must match the author.
        self.optimizer = torch.optim.Adam(self.mirror_params, lr=lr)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, value):
        self.optimizer.state = value

    def sync_from_model(self) -> None:
        with torch.no_grad():
            for mirror, model in zip(self.mirror_params, self.model_params):
                mirror.copy_(model.detach().to(device="cpu", dtype=torch.float32))
                mirror.grad = None

    def step(self, closure: Optional[Any] = None):
        for mirror, model in zip(self.mirror_params, self.model_params):
            if model.grad is None:
                mirror.grad = None
            else:
                mirror.grad = model.grad.detach().to(
                    device="cpu", dtype=torch.float32).clone()
        result = self.optimizer.step(closure)
        with torch.no_grad():
            for mirror, model in zip(self.mirror_params, self.model_params):
                if model.grad is not None:
                    model.copy_(mirror)
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for model in self.model_params:
            if model.grad is None:
                continue
            if set_to_none:
                model.grad = None
            else:
                model.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        # Native state is CPU-only by construction; keep it that way.
        state = self.optimizer.state_dict()
        for item in state.get("state", {}).values():
            for key, value in list(item.items()):
                if torch.is_tensor(value):
                    item[key] = value.detach().to("cpu")
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        for item in self.optimizer.state.values():
            for key, value in list(item.items()):
                if torch.is_tensor(value):
                    item[key] = value.detach().to("cpu")


def enable_unet_checkpointing(net: torch.nn.Module) -> List[str]:
    """Wrap only the six DualStyleUNet forwards; preserve module/state keys."""
    unets = [m for m in net.modules() if type(m).__name__ == "DualStyleUNet"]
    if len(unets) != 6:
        raise RuntimeError(
            f"expected exactly 6 DualStyleUNet instances, found {len(unets)}")
    names: List[str] = []
    for unet in unets:
        name = unet.__class__.__name__
        if getattr(unet, "_wardrobe_checkpointed", False):
            names.append(name)
            continue
        original = unet.forward

        def wrapped(*args, __orig=original, __unet=unet, **kwargs):
            if __unet.training and torch.is_grad_enabled():
                return checkpoint(
                    __orig, *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **kwargs,
                )
            return __orig(*args, **kwargs)

        unet.forward = wrapped
        unet._wardrobe_checkpointed = True
        names.append(name)
    return names
