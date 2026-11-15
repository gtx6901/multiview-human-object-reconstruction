"""Memory-bounded training utilities for the real Wardrobe author model.

CPUAdam keeps the FP32 model on CUDA while all Adam moments live on CPU.  It
intentionally delegates every optimizer detail to the native torch Adam over
same-shaped CPU mirror parameters; no Adam formula is reimplemented here.
"""

from __future__ import annotations

import mmap
import os
import pickle
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.checkpoint import checkpoint

from hoi.runtime import PROJECT_ROOT, runtime_settings


def _load_windows_checkpoint(path: Any) -> Dict[str, Any]:
    # Windows torch.load(mmap=True) uses UntypedStorage.from_file(shared=False),
    # which is FILE_MAP_COPY for the whole view and reserves private commit even
    # when the bytes are untouched.  A Python FILE_MAP_READ view has no such
    # private commit.  The mapping is intentionally not closed: Torch storages
    # in the returned payload keep it alive until those tensors are destroyed.
    import torch.serialization as torch_serialization

    checkpoint_path = os.fspath(path)
    with open(checkpoint_path, "rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The given buffer is not writable, and PyTorch does not "
                    r"support non-writable tensors\..*"
                ),
                category=UserWarning,
            )
            buffer = torch.frombuffer(mapping, dtype=torch.uint8)
        storage = buffer.untyped_storage()
        with torch_serialization._open_zipfile_reader(stream) as zip_file:
            payload = torch_serialization._load(
                zip_file,
                map_location="cpu",
                pickle_module=pickle,
                overall_storage=storage,
                encoding="utf-8",
            )
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a mapping: {type(payload).__name__}")
    return payload


def load_checkpoint(path: Any) -> Dict[str, Any]:
    """Load a rolling checkpoint without modifying its bytes.

    On Windows the checkpoint is exposed through a read-only Python mmap and
    parsed with the installed torch serializer.  On other platforms PyTorch's
    own mmap loader is used unchanged.
    """

    if os.name == "nt":
        return _load_windows_checkpoint(path)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a mapping: {type(payload).__name__}")
    return payload


_ARENA_ALIGNMENT = 64


def _default_optimizer_state_dir() -> Path:
    settings = runtime_settings()
    configured = settings.get("optimizer_state_dir")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return (PROJECT_ROOT / "runtime/cache/tmp").resolve()


class _MappedStateArena:
    """One writable temporary-file mapping per optimizer state dtype."""

    def __init__(self, dtype: torch.dtype, capacity: int, state_dir: Path):
        self.dtype = dtype
        self.capacity = int(capacity)
        self.state_dir = state_dir
        self.used = 0
        self.base_ptr: Optional[int] = None
        self.mapping_ref: Optional[weakref.ReferenceType[mmap.mmap]] = None

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def align_elements(self) -> int:
        return (_ARENA_ALIGNMENT + self.element_size - 1) // self.element_size

    def _mapping(self) -> mmap.mmap:
        if self.mapping_ref is not None:
            mapping = self.mapping_ref()
            if mapping is not None and not mapping.closed:
                return mapping
        # No live state tensor owns the old file: its delete-on-close handle is
        # gone and a fresh arena can start at offset zero.
        self.used = 0
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="optimizer_", suffix=".bin", dir=self.state_dir, delete=True
        ) as stream:
            stream.truncate(self.capacity * self.element_size)
            stream.flush()
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
        self.mapping_ref = weakref.ref(mapping)
        self.base_ptr = (
            torch.frombuffer(mapping, dtype=self.dtype, count=self.capacity)
            .untyped_storage()
            .data_ptr()
        )
        return mapping

    def owns(self, tensor: torch.Tensor) -> bool:
        if tensor.dtype != self.dtype:
            return False
        mapping = self.mapping_ref() if self.mapping_ref is not None else None
        if mapping is None or mapping.closed:
            return False
        storage = tensor.untyped_storage()
        return (
            storage.data_ptr() == self.base_ptr
            and storage.nbytes() == self.capacity * self.element_size
        )

    def append(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.layout != torch.strided:
            return tensor.detach().clone()
        mapping = self._mapping()
        offset = self.used + (-self.used) % self.align_elements
        count = tensor.numel()
        if offset + count > self.capacity:
            raise RuntimeError("optimizer state arena capacity exhausted")
        arena = torch.frombuffer(mapping, dtype=self.dtype, count=self.capacity)
        target = arena.narrow(0, offset, count).view(tensor.shape)
        target.copy_(tensor.detach())
        self.used = offset + count
        return target

    def snapshot_view(self, tensor: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping_ref() if self.mapping_ref is not None else None
        if mapping is None or mapping.closed:
            raise RuntimeError("optimizer state mapping is no longer available")
        # Serialize each native state tensor through its own bounded storage.
        # PyTorch's Windows file writer truncates a single >INT_MAX write;
        # joining all moments into one multi-GB save storage corrupts the ZIP.
        view = torch.frombuffer(
            mapping,
            dtype=self.dtype,
            count=tensor.numel(),
            offset=tensor.storage_offset() * self.element_size,
        )
        return view.view(tensor.shape)


def _collect_state_tensors(value: Any, result: Dict[torch.dtype, List[torch.Tensor]]) -> None:
    if torch.is_tensor(value):
        result.setdefault(value.dtype, []).append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_state_tensors(item, result)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_state_tensors(item, result)


def _aligned_state_total(tensors: Iterable[torch.Tensor], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    align_elements = (_ARENA_ALIGNMENT + element_size - 1) // element_size
    total = 0
    for tensor in tensors:
        if tensor.layout != torch.strided:
            continue
        total += (-total) % align_elements
        total += tensor.numel()
    return total


class CPUAdam:
    """Native torch.optim.Adam over CPU mirrors of CUDA model parameters."""

    # Bound host memory used for device-to-host gradients. Adam moments remain
    # the same FP32 native state; only the transient gradient transfer is
    # windowed so newly initialized moments do not overlap with every gradient.
    _gradient_chunk_bytes = 16 * 1024 * 1024

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        lr: float,
        state_dir: Optional[Any] = None,
    ):
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
        self.state_dir = (
            Path(state_dir).expanduser().resolve()
            if state_dir is not None
            else _default_optimizer_state_dir()
        )
        self._state_arenas: Dict[torch.dtype, _MappedStateArena] = {}
        self._numel_total = int(sum(p.numel() for p in self.model_params))
        self._param_count = len(self.model_params)

    def _amsgrad_enabled(self) -> bool:
        return any(
            bool(group.get("amsgrad", self.optimizer.defaults.get("amsgrad", False)))
            for group in self.optimizer.param_groups
        )

    def _future_capacity(self, dtype: torch.dtype) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        align_elements = (_ARENA_ALIGNMENT + element_size - 1) // element_size
        if dtype == torch.float32:
            moments = 3 if self._amsgrad_enabled() else 2
            tensor_count = self._param_count * (moments + 1)
            base = self._numel_total * moments + self._param_count
        else:
            tensor_count = self._param_count
            base = self._param_count
        padding = (tensor_count + 1) * (align_elements - 1) + align_elements
        return base + padding

    def _ensure_arenas(self) -> None:
        by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
        for item in self.optimizer.state.values():
            _collect_state_tensors(item, by_dtype)
        for dtype, tensors in by_dtype.items():
            need = _aligned_state_total(tensors, dtype)
            arena = self._state_arenas.get(dtype)
            if arena is None:
                capacity = max(need, self._future_capacity(dtype), 1)
                self._state_arenas[dtype] = _MappedStateArena(dtype, capacity, self.state_dir)
            elif need > arena.capacity:
                raise RuntimeError("optimizer state arena capacity exhausted")

    def _map_value(self, value: Any, memo: Dict[int, torch.Tensor]) -> Any:
        if torch.is_tensor(value):
            key = id(value)
            if key in memo:
                return memo[key]
            for arena in self._state_arenas.values():
                if arena.owns(value):
                    memo[key] = value
                    return value
            arena = self._state_arenas.get(value.dtype)
            if arena is None:
                raise RuntimeError(f"no mapped optimizer arena for {value.dtype}")
            mapped = arena.append(value)
            memo[key] = mapped
            return mapped
        if isinstance(value, dict):
            return {key: self._map_value(item, memo) for key, item in value.items()}
        if isinstance(value, list):
            return [self._map_value(item, memo) for item in value]
        if isinstance(value, tuple):
            return tuple(self._map_value(item, memo) for item in value)
        return value

    def _map_items(self, items: Iterable[Dict[str, Any]], memo: Dict[int, torch.Tensor]) -> None:
        for item in items:
            for key, value in list(item.items()):
                item[key] = self._map_value(value, memo)

    def _map_all_state(self) -> None:
        if os.name != "nt":
            return
        self._ensure_arenas()
        memo: Dict[int, torch.Tensor] = {}
        self._map_items(self.optimizer.state.values(), memo)

    def _map_active_state(self, active: Iterable[Any]) -> None:
        if os.name != "nt":
            return
        self._ensure_arenas()
        memo: Dict[int, torch.Tensor] = {}
        for mirror, _ in active:
            item = self.optimizer.state.get(mirror)
            if item:
                self._map_items([item], memo)

    def _prefix_value(
        self,
        value: Any,
        views: Dict[int, torch.Tensor],
    ) -> Any:
        if torch.is_tensor(value):
            key = id(value)
            if key in views:
                return views[key]
            for arena in self._state_arenas.values():
                if not arena.owns(value):
                    continue
                if value.storage_offset() + value.numel() > arena.used:
                    return value
                view = arena.snapshot_view(value)
                views[key] = view
                return view
            return value
        if isinstance(value, dict):
            return {key: self._prefix_value(item, views) for key, item in value.items()}
        if isinstance(value, list):
            return [self._prefix_value(item, views) for item in value]
        if isinstance(value, tuple):
            return tuple(self._prefix_value(item, views) for item in value)
        return value

    def _prefix_state(self, payload: Dict[str, Any]) -> None:
        state = payload.get("state")
        if not isinstance(state, dict):
            return
        views: Dict[int, torch.Tensor] = {}
        payload["state"] = {key: self._prefix_value(item, views) for key, item in state.items()}

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
            self._map_all_state()
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
                self._map_active_state(active)
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
        # Native state is CPU-only by construction; keep it that way.  On
        # Windows, expose only each tensor's bytes to the serializer, excluding
        # reserved capacity and avoiding a single multi-GB storage write.
        state = self.optimizer.state_dict()
        if os.name == "nt" and self._state_arenas:
            self._prefix_state(state)
        casted: Dict[int, torch.Tensor] = {}
        for item in state.get("state", {}).values():
            for key, value in list(item.items()):
                if torch.is_tensor(value):
                    key_id = id(value)
                    if key_id not in casted:
                        casted[key_id] = value.detach().to("cpu")
                    item[key] = casted[key_id]
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        # A load can reference a previous snapshot of this same optimizer.
        # Use fresh storage rather than appending or overwriting live input.
        self._state_arenas = {}
        self._map_all_state()
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
