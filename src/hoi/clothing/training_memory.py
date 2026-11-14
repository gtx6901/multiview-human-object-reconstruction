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

    # Bound host memory used for device-to-host gradients. Adam moments remain
    # the same FP32 native state; only the transient gradient transfer is
    # windowed so newly initialized moments do not overlap with every gradient.
    _gradient_chunk_bytes = 16 * 1024 * 1024

    def __init__(self, parameters: Iterable[torch.nn.Parameter], lr: float):
        self.model_params: List[torch.nn.Parameter] = [p for p in parameters if p.requires_grad]
        self.mirror_params: List[torch.nn.Parameter] = [
            torch.nn.Parameter(
                torch.empty((0,), device="cpu", dtype=torch.float32),
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
        # Model weights remain canonical on their original device. CPU mirrors
        # are allocated only for the bounded chunk being updated.
        for mirror in self.mirror_params:
            mirror.grad = None

    def step(self, closure: Optional[Any] = None):
        # The public optimizer API permits closures, although the Wardrobe
        # training loop does not use them. Keep that path identical to the
        # previous implementation.
        if closure is not None:
            for mirror, model in zip(self.mirror_params, self.model_params):
                mirror.data = model.detach().to(device="cpu", dtype=torch.float32, copy=True)
                mirror.grad = (
                    None
                    if model.grad is None
                    else model.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
                )
            result = self.optimizer.step(closure)
            with torch.no_grad():
                for mirror, model in zip(self.mirror_params, self.model_params):
                    if model.grad is not None:
                        model.copy_(mirror)
                    mirror.grad = None
                    mirror.data = torch.empty((0,), device="cpu", dtype=torch.float32)
            return result

        pairs = list(zip(self.mirror_params, self.model_params))
        chunks = []
        chunk = []
        chunk_bytes = 0
        element_bytes = torch.empty((), dtype=torch.float32).element_size()
        for pair in pairs:
            mirror, model = pair
            size = model.numel() * element_bytes
            if chunk and chunk_bytes + size > self._gradient_chunk_bytes:
                chunks.append(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append(pair)
            chunk_bytes += size
        if chunk:
            chunks.append(chunk)

        # Adam's state is keyed by parameter, so native Adam can step a bounded
        # subset at a time without changing any parameter's moments or step.
        # Temporarily narrowing the existing group's parameter list also avoids
        # scanning the full network once per chunk.
        if len(self.optimizer.param_groups) != 1:
            raise RuntimeError("CPUAdam expects its single native Adam parameter group")
        group = self.optimizer.param_groups[0]
        all_mirrors = group["params"]
        result = None
        for active_chunk in chunks:
            active = [(mirror, model) for mirror, model in active_chunk if model.grad is not None]
            if not active:
                continue
            for mirror, model in active:
                # Stage only this chunk's parameter and gradient on CPU. The
                # model copy is discarded after its update; Adam moments remain
                # in native torch.optim.Adam state.
                mirror.data = model.detach().to(device="cpu", dtype=torch.float32, copy=True)
                mirror.grad = model.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
            group["params"] = [mirror for mirror, _ in active]
            try:
                result = self.optimizer.step()
                with torch.no_grad():
                    for mirror, model in active:
                        model.copy_(mirror)
            finally:
                group["params"] = all_mirrors
                for mirror, _ in active:
                    mirror.grad = None
                    mirror.data = torch.empty((0,), device="cpu", dtype=torch.float32)
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
        raise RuntimeError(f"expected exactly 6 DualStyleUNet instances, found {len(unets)}")
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
                    __orig,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **kwargs,
                )
            return __orig(*args, **kwargs)

        unet.forward = wrapped
        unet._wardrobe_checkpointed = True
        names.append(name)
    return names
