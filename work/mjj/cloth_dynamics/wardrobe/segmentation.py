#!/usr/bin/env python
"""SCHP LIP-20 semantic parsing for the bound GaussianWardrobe MJJ001 data.

This is measurement preparation, not multi-layer training, template
reconstruction, or quality acceptance.  It uses the pinned official SCHP
source and the local lip.pth, with a process-local inference-equivalent
InPlaceABNSync shim (the original CUDA extension is not rebuilt).

Outputs only:
  parsing/<cam>/<pose>.png          raw 20-class labels
  labels_color/<cam>/<pose>.png     disk BGR body gray / upper blue / lower pink
  semantic_valid/<cam>/<pose>.png   binary valid = SAM instance & raw>0
plus segmentation_manifest.json and one review contact sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("OPENCV_TEMP_PATH", ".")
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CLOTH = HERE.parent
PARSER = HERE / "parser"
DEFAULT_DATA = HERE / "data/mjj001"
DEFAULT_CKPT = CLOTH / "runtime/parser/lip.pth"
LIP_INPUT = (473, 473)
BODY_CLASSES = {2, 4, 13, 14, 15, 16, 17, 18, 19}
UPPER_CLASSES = {1, 3, 5, 6, 7, 10, 11}
LOWER_CLASSES = {8, 9, 12}
CLOTH_CLASSES = UPPER_CLASSES | LOWER_CLASSES
# Disk uint8 values; the author reader uses cv.imread -> BGR/255 directly.
BODY_LABEL_DISK = np.array([128, 128, 128], dtype=np.uint8)
UPPER_LABEL_DISK = np.array([0, 128, 255], dtype=np.uint8)
LOWER_LABEL_DISK = np.array([255, 0, 128], dtype=np.uint8)
LABEL_PALETTE = {
    "body": BODY_LABEL_DISK.tolist(),
    "upper": UPPER_LABEL_DISK.tolist(),
    "lower": LOWER_LABEL_DISK.tolist(),
    "background_unknown": [0, 0, 0],
}
MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32)
STD_BGR = np.array([0.225, 0.224, 0.229], dtype=np.float32)
LIP_CLASSES = {
    0: "background", 1: "hat", 2: "hair", 3: "glove", 4: "sunglasses",
    5: "upper-clothes", 6: "dress", 7: "coat", 8: "socks", 9: "pants",
    10: "jumpsuits", 11: "scarf", 12: "skirt", 13: "face", 14: "left-arm",
    15: "right-arm", 16: "left-leg", 17: "right-leg", 18: "left-shoe",
    19: "right-shoe",
}


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


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _install_unicode_cv() -> None:
    """Unicode-safe cv2 I/O for the Chinese workspace path."""
    if getattr(cv, "_wardrobe_seg_unicode_io", False):
        return
    orig_read, orig_write = cv.imread, cv.imwrite

    def imread(path, flags=cv.IMREAD_COLOR):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv.imdecode(data, flags)
        except Exception:
            return orig_read(path, flags)

    def imwrite(path, image, params=None):
        ext = Path(path).suffix or ".png"
        ok, buf = cv.imencode(ext, image, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True

    cv.imread = imread
    cv.imwrite = imwrite
    cv._wardrobe_seg_unicode_io = True


def _act(x: torch.Tensor, activation: str, slope: float) -> torch.Tensor:
    if activation == "relu":
        return torch.nn.functional.relu(x, inplace=True)
    if activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=slope, inplace=True)
    if activation == "elu":
        return torch.nn.functional.elu(x, inplace=True)
    if activation == "none":
        return x
    raise ValueError(f"unsupported InPlaceABN activation: {activation}")


class InferenceInPlaceABN(nn.Module):
    """Inference-equivalent to the author's CUDA eval branch.

    effective_weight = abs(weight) + eps
    mul = rsqrt(running_var + eps) * effective_weight
    y = (x - running_mean) * mul + bias
    followed by the original activation (none / leaky_relu slope .01 / elu).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 activation="leaky_relu", slope=0.01):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            raise RuntimeError("InferenceInPlaceABN supports eval only")
        if self.weight is None:
            raise RuntimeError("affine=False is not present in this parser checkpoint")
        eps = float(self.eps)
        effective_weight = torch.abs(self.weight) + eps
        mul = torch.rsqrt(self.running_var + eps) * effective_weight
        y = (x - self.running_mean[None, :, None, None]) * mul[None, :, None, None]
        y = y + self.bias[None, :, None, None]
        return _act(y, self.activation, self.slope)


def install_inplace_shim() -> Dict[str, Any]:
    """Provide modules.InPlaceABNSync without importing the CUDA extension."""
    shim = types.ModuleType("modules")
    shim.__file__ = str(PARSER / "modules" / "bn.py")
    shim.ABN = InferenceInPlaceABN
    shim.InPlaceABN = InferenceInPlaceABN
    shim.InPlaceABNSync = InferenceInPlaceABN
    sys.modules["modules"] = shim
    return {
        "replacement": "InferenceInPlaceABN",
        "formula": "effective_weight=abs(weight)+eps; mul=rsqrt(running_var+eps)*effective_weight; y=(x-running_mean)*mul+bias; activation original",
        "source_bn_sha256": sha256_file(PARSER / "modules" / "bn.py"),
        "source_cuda_sha256": sha256_file(PARSER / "modules" / "src" / "inplace_abn_cuda.cu"),
        "training_allowed": False,
    }


def load_strict(model: nn.Module, checkpoint: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state_dict mapping")
    stripped = OrderedDict()
    prefixed = 0
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        prefixed += int(name != k)
        if name in stripped:
            raise RuntimeError(f"duplicate state_dict key after module stripping: {name}")
        stripped[name] = v
    try:
        model.load_state_dict(stripped, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "strict SCHP load failed; no missing/unexpected keys were silently ignored: "
            + str(exc)
        ) from exc
    return {
        "state_dict_keys": len(stripped),
        "module_prefix_stripped": prefixed,
        "strict": True,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def box_to_center_scale(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty SAM instance foreground mask")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # Match SimpleFolderDataset._box2cs: box width/height are x1-x0/y1-y0.
    w = float(x1 - x0)
    h = float(y1 - y0)
    if w <= 0 or h <= 0:
        raise ValueError("degenerate SAM instance foreground bbox")
    center = np.array([x0 + w * 0.5, y0 + h * 0.5], dtype=np.float32)
    ratio = LIP_INPUT[1] * 1.0 / LIP_INPUT[0]
    if w > ratio * h:
        h = w / ratio
    elif w < ratio * h:
        w = h * ratio
    return center, np.array([w, h], dtype=np.float32)


def parser_input(image_bgr: np.ndarray, mask: np.ndarray, affinity, affine) -> Tuple[torch.Tensor, Dict[str, Any]]:
    center, scale = box_to_center_scale(mask)
    trans = affine(center, scale, 0, np.asarray(LIP_INPUT))
    warped = cv.warpAffine(
        image_bgr, trans, (LIP_INPUT[1], LIP_INPUT[0]),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    arr = warped.transpose((2, 0, 1)).astype(np.float32) / 255.0
    arr = (arr - MEAN_BGR[:, None, None]) / STD_BGR[:, None, None]
    return torch.from_numpy(arr)[None], {
        "center": center.tolist(), "scale": scale.tolist(),
        "affine": np.asarray(trans).tolist(), "input_hw": list(LIP_INPUT),
    }


def encode_labels_color(raw: np.ndarray, mask: np.ndarray,
                        valid: np.ndarray) -> np.ndarray:
    """Single palette encoder used by inference and recolor-only."""
    if raw.shape != mask.shape or raw.shape != valid.shape:
        raise ValueError(
            f"shape mismatch raw/mask/valid: {raw.shape}/{mask.shape}/{valid.shape}")
    if raw.ndim != 2:
        raise ValueError(f"raw must be 2-D class ids, got {raw.shape}")
    valid = np.asarray(valid, dtype=bool)
    labels = np.zeros((*raw.shape, 3), np.uint8)
    body = np.isin(raw, list(BODY_CLASSES)) & valid
    upper = np.isin(raw, list(UPPER_CLASSES)) & valid
    lower = np.isin(raw, list(LOWER_CLASSES)) & valid
    labels[body] = BODY_LABEL_DISK
    labels[upper] = UPPER_LABEL_DISK
    labels[lower] = LOWER_LABEL_DISK
    return labels


def save_outputs(out_root: Path, cam: str, pose_idx: int, raw: np.ndarray,
                 mask: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    pdir = out_root / "parsing" / cam
    ldir = out_root / "labels_color" / cam
    vdir = out_root / "semantic_valid" / cam
    for d in (pdir, ldir, vdir):
        d.mkdir(parents=True, exist_ok=True)
    p = pdir / f"{pose_idx:08d}.png"
    l = ldir / f"{pose_idx:08d}.png"
    v = vdir / f"{pose_idx:08d}.png"
    body = np.isin(raw, list(BODY_CLASSES)) & valid
    upper = np.isin(raw, list(UPPER_CLASSES)) & valid
    lower = np.isin(raw, list(LOWER_CLASSES)) & valid
    labels = encode_labels_color(raw, mask, valid)
    cv.imwrite(str(p), raw)
    cv.imwrite(str(l), labels)
    cv.imwrite(str(v), (valid.astype(np.uint8) * 255))
    # Exact disk BGR read-back contract; training reader divides by 255.
    back = cv.imread(str(l), cv.IMREAD_UNCHANGED)
    if back is None or back.shape != labels.shape:
        raise RuntimeError(f"labels_color read-back failed: {l}")
    for name, selector, expected in (
        ("body", body, BODY_LABEL_DISK),
        ("upper", upper, UPPER_LABEL_DISK),
        ("lower", lower, LOWER_LABEL_DISK),
    ):
        if selector.any() and not np.all(back[selector] == expected[None, :]):
            raise RuntimeError(f"{name} disk BGR read-back mismatch: {l}")
    allowed = np.stack([
        np.array([0, 0, 0], np.uint8), BODY_LABEL_DISK,
        UPPER_LABEL_DISK, LOWER_LABEL_DISK,
    ], 0)
    rows = back.reshape(-1, 3)
    row_ok = np.any(np.all(rows[:, None, :] == allowed[None, :, :], axis=2), axis=1)
    if not np.all(row_ok):
        raise RuntimeError(f"unexpected labels_color values: {l}")
    return {
        "parsing": str(p), "labels_color": str(l), "semantic_valid": str(v),
        "labels_color_disk_bgr_body": BODY_LABEL_DISK.tolist(),
        "labels_color_disk_bgr_upper": UPPER_LABEL_DISK.tolist(),
        "labels_color_disk_bgr_lower": LOWER_LABEL_DISK.tolist(),
        "labels_color_training_float_bgr_body": (BODY_LABEL_DISK.astype(np.float32) / 255.0).tolist(),
        "labels_color_training_float_bgr_upper": (UPPER_LABEL_DISK.astype(np.float32) / 255.0).tolist(),
        "labels_color_training_float_bgr_lower": (LOWER_LABEL_DISK.astype(np.float32) / 255.0).tolist(),
        "class_counts": {str(i): int((raw == i).sum()) for i in range(20)},
        "valid_pixels": int(valid.sum()),
        "unknown_foreground_pixels": int(((mask > 0) & (raw == 0)).sum()),
    }


def _fit_tile(image: np.ndarray, size: int = 220) -> np.ndarray:
    h, w = image.shape[:2]
    s = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv.resize(image, (nw, nh), interpolation=cv.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def make_contact(data: Path, out_root: Path, frames: List[int]) -> Optional[Path]:
    times = [t for t in (0, 49, 98) if t in frames]
    if not times:
        return None
    tiles = []
    for t in times:
        row = []
        for c in range(8):
            cam = f"cam{c:02d}"
            img_p = data / "images" / cam / f"{t:08d}.jpg"
            mask_p = data / "masks" / cam / f"{t:08d}.jpg"
            raw_p = out_root / "parsing" / cam / f"{t:08d}.png"
            valid_p = out_root / "semantic_valid" / cam / f"{t:08d}.png"
            if not all(p.is_file() for p in (img_p, mask_p, raw_p, valid_p)):
                row.append(None)
                continue
            img = cv.imread(str(img_p), cv.IMREAD_UNCHANGED)
            mask = cv.imread(str(mask_p), cv.IMREAD_UNCHANGED)
            raw = cv.imread(str(raw_p), cv.IMREAD_UNCHANGED)
            valid = cv.imread(str(valid_p), cv.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if raw is None or valid is None:
                row.append(None)
                continue
            person = mask > 0
            ys, xs = np.where(person)
            if len(xs) == 0:
                row.append(None)
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pad = 20
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.shape[1] - 1, x1 + pad), min(img.shape[0] - 1, y1 + pad)
            crop = img[y0:y1 + 1, x0:x1 + 1].copy()
            raw_c = raw[y0:y1 + 1, x0:x1 + 1]
            valid_c = valid[y0:y1 + 1, x0:x1 + 1] > 0
            mask_c = person[y0:y1 + 1, x0:x1 + 1]
            semantic = np.zeros((*raw_c.shape, 3), np.uint8)
            body = np.isin(raw_c, list(BODY_CLASSES)) & valid_c
            upper = np.isin(raw_c, list(UPPER_CLASSES)) & valid_c
            lower = np.isin(raw_c, list(LOWER_CLASSES)) & valid_c
            unknown = mask_c & (raw_c == 0)
            semantic[body] = BODY_LABEL_DISK
            semantic[upper] = UPPER_LABEL_DISK
            semantic[lower] = LOWER_LABEL_DISK
            semantic[unknown] = np.array([128, 128, 128], np.uint8)
            top = _fit_tile(crop)
            bottom = _fit_tile(semantic)
            canvas = np.zeros((460, 230, 3), np.uint8)
            canvas[0:220, 5:225] = top
            canvas[230:450, 5:225] = bottom
            cv.putText(canvas, f"{cam} t={t} RGB", (4, 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(canvas, "body gray / upper blue / lower pink / unknown gray", (4, 228),
                       cv.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 255), 1, cv.LINE_AA)
            row.append(canvas)
        tiles.append(row)
    if not tiles:
        return None
    sheet = np.zeros((len(tiles) * 460, 8 * 230, 3), np.uint8)
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            if tile is not None:
                sheet[r * 460:(r + 1) * 460, c * 230:(c + 1) * 230] = tile
    p = out_root / "segmentation_contact_sheet.png"
    cv.imwrite(str(p), sheet)
    return p


def recolor_existing(data: Path) -> int:
    """Rebuild labels_color from existing raw/masks/valid; never load parser."""
    frames = list(range(99))
    cams = [f"cam{c:02d}" for c in range(8)]
    paths = []
    for pose_idx in frames:
        for cam in cams:
            paths.append({
                "key": f"{cam}/{pose_idx:08d}",
                "raw": data / "parsing" / cam / f"{pose_idx:08d}.png",
                "mask": data / "masks" / cam / f"{pose_idx:08d}.jpg",
                "valid": data / "semantic_valid" / cam / f"{pose_idx:08d}.png",
                "labels": data / "labels_color" / cam / f"{pose_idx:08d}.png",
            })
    missing = [str(rec[k]) for rec in paths for k in ("raw", "mask", "valid")
               if not rec[k].is_file()]
    if missing:
        raise FileNotFoundError(
            "recolor-only requires all 8x99 existing raw/mask/semantic_valid; "
            f"missing {len(missing)}: {missing[:5]}")

    label_hash = hashlib.sha256()
    outputs = []
    for rec in paths:
        raw = cv.imread(str(rec["raw"]), cv.IMREAD_UNCHANGED)
        mask = cv.imread(str(rec["mask"]), cv.IMREAD_UNCHANGED)
        sem = cv.imread(str(rec["valid"]), cv.IMREAD_UNCHANGED)
        if raw is None or mask is None or sem is None:
            raise FileNotFoundError(f"unreadable existing input for {rec['key']}")
        if raw.ndim != 2:
            raise ValueError(f"raw parsing must be 2-D: {rec['raw']} {raw.shape}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if sem.ndim == 3:
            sem = sem[:, :, 0]
        if raw.shape != mask.shape or raw.shape != sem.shape:
            raise ValueError(f"shape mismatch for {rec['key']}")
        if int(raw.min()) < 0 or int(raw.max()) >= len(LIP_CLASSES):
            raise ValueError(f"raw class outside LIP-20 for {rec['key']}")
        mask_bool = mask > 0
        sem_bool = sem > 0
        expected_valid = mask_bool & (raw > 0)
        if not np.array_equal(sem_bool, expected_valid):
            raise ValueError(
                f"semantic_valid mismatch with mask/raw for {rec['key']}; "
                "recolor-only does not rewrite semantic_valid")
        labels = encode_labels_color(raw, mask_bool, sem_bool)
        rec["labels"].parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(rec["labels"]), labels):
            raise RuntimeError(f"failed to write labels_color: {rec['labels']}")
        back = cv.imread(str(rec["labels"]), cv.IMREAD_UNCHANGED)
        if back is None or back.shape != labels.shape or not np.array_equal(back, labels):
            raise RuntimeError(f"labels_color read-back mismatch: {rec['labels']}")
        label_hash.update(rec["labels"].read_bytes())
        outputs.append({"key": rec["key"], "path": str(rec["labels"]),
                        "sha256": sha256_file(rec["labels"])})

    palette = dict(LABEL_PALETTE)
    palette_float = {k: (np.asarray(v, np.float32) / 255.0).tolist()
                     for k, v in palette.items()}
    encoding = {
        "mode": "recolor_only",
        "source": "existing parsing raw LIP-20 + masks + semantic_valid; no parser load",
        "parser_loaded": False,
        "raw_source_unchanged": True,
        "mask_source_unchanged": True,
        "semantic_valid_unchanged": True,
        "palette": palette,
        "palette_float_bgr_after_reader_div255": palette_float,
        "body_classes": sorted(BODY_CLASSES),
        "upper_classes": sorted(UPPER_CLASSES),
        "lower_classes": sorted(LOWER_CLASSES),
        "cloth_classes": sorted(CLOTH_CLASSES),
        "no_rgb_bgr_flip_in_encoder": True,
        "unknown_excluded": True,
    }
    manifest_path = data / "segmentation_manifest.json"
    existing = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.setdefault("schema", "gaussian_wardrobe_schp_segmentation/v1")
    existing.update({
        "generated_at_utc": utc_now(),
        "body_classes": sorted(BODY_CLASSES),
        "upper_classes": sorted(UPPER_CLASSES),
        "lower_classes": sorted(LOWER_CLASSES),
        "cloth_classes": sorted(CLOTH_CLASSES),
        "labels_color_disk_bgr": palette,
        "labels_color_training_float_bgr_after_reader_div255": palette_float,
        "labels_color_encoding": encoding,
        "outputs": {
            "labels_color": {
                "root": str(data / "labels_color"),
                "count": len(outputs),
                "files": outputs,
                "files_sha256": label_hash.hexdigest(),
                "encoding": "recolor_only",
            },
            "parsing": "unchanged",
            "masks": "unchanged",
            "semantic_valid": "unchanged",
        },
    })
    items = existing.get("items")
    if isinstance(items, list):
        by_key = {str(it.get("key")): it for it in items if isinstance(it, dict)}
        for out in outputs:
            if out["key"] in by_key:
                by_key[out["key"]]["labels_color_sha256"] = out["sha256"]
                by_key[out["key"]]["labels_color_encoding"] = "recolor_only"
        existing["items"] = items
    atomic_write_json(manifest_path, existing)
    print(json.dumps({
        "status": "recolor_only",
        "labels_color_count": len(outputs),
        "labels_color_sha256": label_hash.hexdigest(),
        "palette": palette,
    }, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--recolor-only", action="store_true",
                    help="re-encode existing parsing/masks/semantic_valid; no parser load")
    ap.add_argument("--frames", default=None,
                    help="comma-separated real pose indices; default all 99")
    a = ap.parse_args()
    _install_unicode_cv()
    data = Path(a.data).resolve()
    checkpoint = Path(a.checkpoint).resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    if a.recolor_only:
        if a.frames is not None:
            raise ValueError("--recolor-only requires all existing 8x99 frames")
        return recolor_existing(data)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sf_p = data / "source_frames.json"
    if not sf_p.is_file():
        raise FileNotFoundError(sf_p)
    sf = json.loads(sf_p.read_text(encoding="utf-8"))
    source_frames = [int(x) for x in sf["source_frames"]]
    if len(source_frames) != 99:
        raise ValueError("expected 99 source frames")
    all_frames = list(range(99))
    frames = all_frames if a.frames is None else [int(x) for x in a.frames.split(",") if x.strip()]
    if any(t < 0 or t >= 99 for t in frames):
        raise ValueError("frame index outside 0..98")

    if str(PARSER) not in sys.path:
        sys.path.insert(0, str(PARSER))
    shim_info = install_inplace_shim()
    from networks import init_model
    from utils.transforms import get_affine_transform, transform_logits

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device(a.device)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = init_model("resnet101", num_classes=20, pretrained=None)
    load_info = load_strict(model, checkpoint)
    model = model.to(dev).eval()

    parser_hashes = {
        "simple_extractor.py": sha256_file(PARSER / "simple_extractor.py"),
        "datasets/simple_extractor_dataset.py": sha256_file(PARSER / "datasets/simple_extractor_dataset.py"),
        "utils/transforms.py": sha256_file(PARSER / "utils/transforms.py"),
        "networks/AugmentCE2P.py": sha256_file(PARSER / "networks/AugmentCE2P.py"),
        "modules/bn.py": shim_info["source_bn_sha256"],
        "modules/src/inplace_abn_cuda.cu": shim_info["source_cuda_sha256"],
    }
    segmentation_sha256 = sha256_file(Path(__file__).resolve())
    palette = {
        "background_unknown": [0, 0, 0],
        "body": BODY_LABEL_DISK.tolist(),
        "upper": UPPER_LABEL_DISK.tolist(),
        "lower": LOWER_LABEL_DISK.tolist(),
    }
    palette_float = {k: (np.asarray(v, np.float32) / 255.0).tolist()
                     for k, v in palette.items()}
    run_key = sha256_text(json.dumps({
        "segmentation_sha256": segmentation_sha256,
        "checkpoint": load_info["checkpoint_sha256"],
        "source_frames": source_frames,
        "parser_hashes": parser_hashes,
        "input_size": LIP_INPUT,
        "classes": LIP_CLASSES,
        "body_classes": sorted(BODY_CLASSES),
        "upper_classes": sorted(UPPER_CLASSES),
        "lower_classes": sorted(LOWER_CLASSES),
        "cloth_classes": sorted(CLOTH_CLASSES),
        "labels_color_palette": palette,
    }, sort_keys=True))
    manifest_path = data / "segmentation_manifest.json"
    old = {}
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old_key = old.get("run_key")
    old_items = {str(it["key"]): it for it in old.get("items", [])}
    resume_key_match = old_key == run_key
    out_root = data
    # Preserve earlier items only when this exact algorithm/model/input key is
    # unchanged.  Expanding --frames therefore does not discard completed work.
    items_by_key: Dict[str, Dict[str, Any]] = (
        dict(old_items) if resume_key_match else {}
    )
    started_count = 0
    manifest_core = {
        "schema": "gaussian_wardrobe_schp_segmentation/v1",
        "generated_at_utc": utc_now(),
        "run_key": run_key,
        "segmentation_sha256": segmentation_sha256,
        "resumed_key_match": resume_key_match,
        "overwrite_current": bool(old and not resume_key_match),
        "data": str(data),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": load_info["checkpoint_sha256"],
        "checkpoint_bytes": load_info["checkpoint_bytes"],
        "strict_load": load_info,
        "parser": str(PARSER),
        "parser_hashes": parser_hashes,
        "source_frames": source_frames,
        "selected_pose_indices": frames,
        "a_model_was_not_retrained": True,
        "input_size": list(LIP_INPUT),
        "mean_bgr": MEAN_BGR.tolist(), "std_bgr": STD_BGR.tolist(),
        "inplace_abn_shim": shim_info,
        "class_names": LIP_CLASSES,
        "body_classes": sorted(BODY_CLASSES),
        "upper_classes": sorted(UPPER_CLASSES),
        "lower_classes": sorted(LOWER_CLASSES),
        "cloth_classes": sorted(CLOTH_CLASSES),
        "labels_color_disk_bgr": palette,
        "labels_color_training_float_bgr_after_reader_div255": palette_float,
        "labels_color_encoding": {
            "mode": "parser_inference",
            "source": "raw LIP-20 class ids from the actual parser; mask/SAM instance; valid=mask & raw>0",
            "palette": palette,
            "palette_float_bgr_after_reader_div255": palette_float,
            "no_rgb_bgr_flip_in_encoder": True,
            "unknown_excluded": True,
        },
        "semantic_valid_rule": "SAM instance foreground AND raw class > 0; no confidence threshold",
        "unknown_rule": "class 0 inside SAM remains unknown and is not relabeled body/upper/lower",
        "output_layout": {
            "parsing": "parsing/<cam>/<pose>.png raw 20-class",
            "labels_color": "labels_color/<cam>/<pose>.png disk BGR body/upper/lower",
            "semantic_valid": "semantic_valid/<cam>/<pose>.png 0/255",
        },
        "device": a.device,
    }

    def write_progress(contact: Optional[Path] = None) -> None:
        merged = sorted(items_by_key.values(), key=lambda it: str(it["key"]))
        covered_keys = sorted({it["key"] for it in merged
                               if it.get("status") in ("written", "reused")})
        missing_keys = sorted({it["key"] for it in merged
                               if it.get("status") not in ("written", "reused")})
        elapsed_now = time.time() - started
        peak_now = int(torch.cuda.max_memory_allocated()) if a.device == "cuda" else None
        payload = {
            **manifest_core,
            "generated_at_utc": utc_now(),
            "elapsed_sec": round(elapsed_now, 3),
            "peak_vram_bytes": peak_now,
            "newly_processed": started_count,
            "covered_count": len(covered_keys), "covered": covered_keys,
            "missing_count": len(missing_keys), "missing": missing_keys,
            "items": merged,
            "contact_sheet": str(contact) if contact else None,
            "scope": "SCHP semantic measurement preparation only; not multi-layer training or quality acceptance",
        }
        atomic_write_json(manifest_path, payload)

    for pose_idx in frames:
        for c in range(8):
            cam = f"cam{c:02d}"
            img_p = data / "images" / cam / f"{pose_idx:08d}.jpg"
            mask_p = data / "masks" / cam / f"{pose_idx:08d}.jpg"
            key = f"{cam}/{pose_idx:08d}"
            if not img_p.is_file() or not mask_p.is_file():
                items_by_key[key] = {"key": key, "status": "missing_input",
                                     "image": str(img_p), "mask": str(mask_p)}
                write_progress()
                continue
            input_hash = hashlib.sha256(
                (hashlib.sha256(img_p.read_bytes()).digest() +
                 hashlib.sha256(mask_p.read_bytes()).digest())).hexdigest()
            prev = old_items.get(key, {})
            outputs = [out_root / "parsing" / cam / f"{pose_idx:08d}.png",
                       out_root / "labels_color" / cam / f"{pose_idx:08d}.png",
                       out_root / "semantic_valid" / cam / f"{pose_idx:08d}.png"]
            if (resume_key_match and prev.get("input_hash") == input_hash
                    and all(p.is_file() for p in outputs)):
                items_by_key[key] = {**prev, "status": "reused"}
                write_progress()
                continue
            image = cv.imread(str(img_p), cv.IMREAD_COLOR)
            mask = cv.imread(str(mask_p), cv.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise FileNotFoundError(f"unreadable input {img_p} / {mask_p}")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask > 0
            tensor, meta = parser_input(image, mask, None, get_affine_transform)
            tensor = tensor.to(dev)
            with torch.no_grad():
                output = model(tensor)
                logits = output[0][-1][0]
                up = torch.nn.Upsample(size=LIP_INPUT, mode="bilinear",
                                       align_corners=True)(logits.unsqueeze(0)).squeeze()
                up = up.permute(1, 2, 0).detach().cpu().numpy()
                mapped = transform_logits(up, np.asarray(meta["center"], np.float32),
                                          np.asarray(meta["scale"], np.float32),
                                          image.shape[1], image.shape[0],
                                          input_size=LIP_INPUT)
                raw = np.argmax(mapped, axis=2).astype(np.uint8)
            raw[~mask] = 0
            valid = mask & (raw > 0)
            saved = save_outputs(out_root, cam, pose_idx, raw, mask, valid)
            items_by_key[key] = {
                "key": key, "status": "written", "input_hash": input_hash,
                "pose_idx": pose_idx, "cam": cam, "source_frame": source_frames[pose_idx],
                "image": str(img_p), "mask": str(mask_p),
                "crop": meta, **saved,
            }
            started_count += 1
            # Atomic per-item save: interruption leaves all completed items.
            write_progress()
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated()) if a.device == "cuda" else None
    covered = sorted({it["key"] for it in items_by_key.values()
                      if it.get("status") in ("written", "reused")})
    missing = sorted({it["key"] for it in items_by_key.values()
                      if it.get("status") not in ("written", "reused")})
    contact = make_contact(data, out_root, frames) if all(
        (out_root / "semantic_valid" / f"cam{c:02d}" / f"{t:08d}.png").is_file()
        for c in range(8) for t in (0, 49, 98) if t in frames) else None
    write_progress(contact=contact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps({k: manifest[k] for k in (
        "checkpoint_sha256", "strict_load", "device", "newly_processed",
        "covered_count", "missing_count", "elapsed_sec", "peak_vram_bytes",
        "contact_sheet")}, indent=2))
    return 0 if not missing else 2



if __name__ == "__main__":
    raise SystemExit(main())
