#!/usr/bin/env python
"""Build the independent green partner identity with the existing LHM++ stack.

Input is the real SAM2 output from ``segment_instances.py``:
  * undistorted RGB frames from ``frames/<serial>/00000.jpg``
  * per-actor packed masks in ``masks/<serial>.npz``

Only cameras/frames where the partner mask is actually present are encoded.
A fully hidden camera is recorded as ``unknown`` and is not replaced by a
white-background person, a background person, or the FCS avatar.  The existing
LHM++ initialization, encoder, latent fusion, Gaussian decode and avatar export
are called directly; no substitute reconstruction algorithm is introduced.

All caches/inputs/latent/avatar are written below this script's own ``assets/``
and ``cache/``; the frozen ``lhmpp/runs`` tree is not written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
LHM = PROJECT_ROOT / "lhmpp"
DEFAULT_SEGMENT = THIS_DIR / "data" / "segment_instances" / "index.json"
DEFAULT_OUT = THIS_DIR / "assets" / "partner_identity"
DEFAULT_CACHE = THIS_DIR / "cache" / "partner_lhm"
ACTORS = ("fcs", "partner")


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def move(x: Any, device: str) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [move(v, device) for v in x]
    return x


def load_model(cache_dir: Path) -> Any:
    """Use lhmpp/load_model_probe.py with only its output path redirected.

    The probe's one ``exist_ok=False`` is changed to ``exist_ok=True`` exactly
    as the existing reconstruct.model_load does, so repeated identities do not
    need a new hard-coded run directory.  The model construction/checkpoint
    strict loading is unchanged.
    """
    source = LHM / "load_model_probe.py"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rel = str(cache_dir.resolve())
    text = source.read_text(encoding="utf-8")
    old = "out.mkdir(parents=True,exist_ok=False)"
    if text.count(old) != 1:
        raise RuntimeError("load_model_probe output line changed; refusing to guess")
    text = text.replace(old, "out.mkdir(parents=True,exist_ok=True)")
    sys.path.insert(0, str(LHM))
    sys.argv = [str(source), "--name", rel]
    scope: Dict[str, Any] = {"__name__": "__main__", "__file__": str(source)}
    exec(compile(text, str(source), "exec"), scope)
    if "model" not in scope:
        raise RuntimeError(f"LHM load failed; see {cache_dir / 'report.json'}")
    return scope["model"]


def unpack_mask(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=w).astype(bool).reshape(h, w)


def load_data_utils() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lhm_data_utils", LHM / "vendor/core/datasets/data_utils.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import LHM data_utils")
    du = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(du)
    return du


def build_rgba_inputs(
    segment_index: Path,
    actor: str,
    source_frame: int,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (input tensors entries, visible records, unknown records)."""
    import cv2

    idx = json.loads(segment_index.read_text(encoding="utf-8"))
    if str(idx.get("coordinate_space")) != "undistorted_native_2048x1536":
        raise ValueError("segment index must be in undistorted_native_2048x1536")
    if actor not in ACTORS:
        raise ValueError(actor)
    # Project-specific Unicode-safe cv2 I/O used by the running segment code.
    sys.path.insert(0, str(THIS_DIR.parent / "dual_track"))
    from image_io import read_image

    du = load_data_utils()

    crop_records: List[Dict[str, Any]] = []
    visible: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []
    for cam in idx.get("cameras") or []:
        serial = str(cam["serial"])
        mask_path = Path(cam["mask_npz"])
        frame_meta = next(
            (f for f in cam.get("frames") or [] if int(f["source_frame"]) == source_frame),
            None,
        )
        rec = {"serial": serial, "camera_index": cam.get("camera_index")}
        if frame_meta is None:
            unknown.append({**rec, "reason": "source_frame_not_in_segment"})
            continue
        if not mask_path.is_file():
            unknown.append({**rec, "reason": "mask_npz_missing"})
            continue
        with np.load(mask_path, allow_pickle=False) as z:
            packed = z["packed"]
            shape = [int(x) for x in z["mask_shape"].tolist()]
            actors = [str(x) for x in z["actors"].tolist()]
            frames = [int(x) for x in z["local_frames"].tolist()]
            present = z["present"]
            areas = z["areas"]
        if actor not in actors:
            unknown.append({**rec, "reason": "actor_not_in_mask_file"})
            continue
        ai = actors.index(actor)
        fi = frames.index(int(frame_meta["local_frame"]))
        if not bool(present[fi, ai]):
            unknown.append(
                {
                    **rec,
                    "reason": "sam2_mask_absent",
                    "source_frame": source_frame,
                    "raw_area": int(areas[fi, ai]),
                }
            )
            continue
        h, w = shape
        mask = unpack_mask(packed[fi, ai], h, w)
        area = int(mask.sum())
        if area < 200:
            unknown.append({**rec, "reason": "mask_too_small", "area": area})
            continue
        frame_path = Path(frame_meta["jpeg"])
        bgr = read_image(frame_path, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (h, w):
            unknown.append({**rec, "reason": "frame_read_failed_or_size_mismatch"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        alpha = mask.astype(np.float32)
        rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
        input_dir = out_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        rgba_path = input_dir / f"{serial}_f{source_frame:04d}.png"
        Image.fromarray(rgba, mode="RGBA").save(rgba_path)
        # Exact LHM preprocessing used by lhm_encode.py.
        rgb_white = rgb.astype(np.float32) * alpha[..., None] + 255 * (1 - alpha[..., None])
        crop, _crop_mask, left, top, _ = du.src_center_crop_according_to_mask(
            rgb_white, alpha, 5 / 3, [1.0, 1.0], None
        )
        crop_records.append(
            {
                "serial": serial,
                "source_frame": source_frame,
                "rgba": str(rgba_path),
                "crop_shape": list(crop.shape),
                "left": int(left),
                "top": int(top),
            }
        )
        visible.append(
            {
                **rec,
                "source_frame": source_frame,
                "mask_area": area,
                "rgba": str(rgba_path),
                "crop_shape": list(crop.shape),
            }
        )

    return crop_records, visible, unknown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment-index", default=str(DEFAULT_SEGMENT))
    ap.add_argument("--actor", default="partner", choices=list(ACTORS))
    ap.add_argument("--source-frame", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.actor != "partner":
        raise RuntimeError("this builder is for the unknown green partner; FCS already has its own frozen avatar")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    segment_index = Path(args.segment_index).resolve()
    out = Path(args.out).resolve()
    cache = Path(args.cache).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    configure_runtime_paths()
    crop_records, visible, unknown = build_rgba_inputs(
        segment_index, args.actor, args.source_frame, out
    )
    if len(crop_records) < args.min_views:
        report = {
            "success": False,
            "actor": args.actor,
            "visible": visible,
            "unknown": unknown,
            "error": f"only {len(crop_records)} visible views; need at least {args.min_views}",
            "not_reconstructed": True,
        }
        atomic_write_json(out / "identity.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    sys.path.insert(0, str(LHM))
    from chunk_attention import install

    install()
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch.manual_seed(0)
    du = load_data_utils()

    crops = []
    for r in crop_records:
        rgba = np.asarray(Image.open(r["rgba"]).convert("RGBA"))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        rgb_white = rgb * alpha[..., None] + 255 * (1 - alpha[..., None])
        crop, _m, _l, _t, _ = du.src_center_crop_according_to_mask(
            rgb_white, alpha, 5 / 3, [1.0, 1.0], None
        )
        crops.append(crop / 255.0)
    imgs = du.PadRatioWithScale(target_ratio=5 / 3, tgt_max_size_list=[840], val=True)(crops)
    tensors = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
    torch.save(tensors, out / "inputs.pt")
    n = tensors.shape[0]
    atomic_write_json(
        out / "input_manifest.json",
        {
            "source_segment_index": str(segment_index),
            "actor": args.actor,
            "source_frame": args.source_frame,
            "visible_views": visible,
            "unknown_views": unknown,
            "input_shape": list(tensors.shape),
            "input_order": [r["serial"] for r in crop_records],
            "preprocessing": "LHM src_center_crop_according_to_mask; PadRatioWithScale(5/3,840); real masks only",
        },
    )
    # Local review sheet of the actual (visible) encoder inputs.
    preview = np.full((280, 168 * n, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        p = np.uint8(np.clip(im, 0, 1) * 255)
        p = np.asarray(Image.fromarray(p).resize((168, 280)))
        preview[:, i * 168 : (i + 1) * 168] = p
    Image.fromarray(preview).save(out / "inputs.jpg", quality=92)

    print(f"[identity] loading LHM++ from {cache / 'model_load'}", flush=True)
    model = load_model(cache / "model_load")
    model.eval()
    model.encoder.to(args.device)
    torch.cuda.reset_peak_memory_stats()
    features = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n):
            f = model.forward_encode_image(tensors[i : i + 1].to(args.device))
            if not torch.isfinite(f).all():
                raise RuntimeError(f"non-finite encoder feature for view {i}")
            features.append(f.cpu())
            print(f"[identity] encoded view {i} {crop_records[i]['serial']}", flush=True)
    torch.cuda.synchronize()
    features = torch.cat(features)
    torch.save(features, out / "image_features.pt")
    model.encoder.cpu()
    del f
    torch.cuda.empty_cache()

    model.forward_encode_image = lambda image: features.to(image.device)
    model.transformer.to(args.device)
    model.motion_embed_mlp.to(args.device)
    model.renderer.smplx_model.to(args.device)
    betas = torch.zeros(1, 10, device=args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        query, params = model.renderer.get_query_points(None, {"betas": betas}, torch.device(args.device))
        result = model.forward_latent_points(
            tensors[None].to(args.device),
            None,
            query_points=query,
            ref_imgs_bool=torch.ones(1, n, dtype=torch.bool, device=args.device),
        )
    torch.cuda.synchronize()
    latent = {"result": move(result, "cpu"), "query": move(query, "cpu"), "smplx": move(params, "cpu")}
    torch.save(latent, out / "latent.pt")
    (cache / "latent_meta.json").write_text(
        json.dumps(
            {
                "n_views": n,
                "visible_serials": [r["serial"] for r in crop_records],
                "unknown_serials": [r["serial"] for r in unknown],
                "input_shape": list(tensors.shape),
                "feature_shape": list(features.shape),
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # As in the existing separate-stage runs, encoder/fusion weights need not
    # occupy the GPU while the renderer decodes the unchanged latent tensors.
    model.transformer.cpu()
    model.motion_embed_mlp.cpu()
    del result, query, params
    torch.cuda.empty_cache()
    # Decode exactly as lhm_decode.py.
    model.renderer.to(args.device)
    model.renderer.hyper_step(10000000)
    data = move(latent, args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        attrs, q2, p2 = model.renderer.forward_gs(
            data["result"][0], data["query"], data["smplx"]
        )
    import dataclasses

    serial_attrs = []
    for attr in attrs:
        item = {f.name: getattr(attr, f.name) for f in dataclasses.fields(attr)}
        for key, val in item.items():
            if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                raise RuntimeError(f"non-finite decoded Gaussian attribute: {key}")
        serial_attrs.append(move(item, "cpu"))
    torch.save(
        {"attrs": serial_attrs, "query": move(q2, "cpu"), "smplx": move(p2, "cpu")},
        out / "gaussian_attributes.pt",
    )

    # Export with the original reconstruct.export, not a rewritten exporter.
    # Import separately so the frozen reconstruct.py's MJJ constants do not
    # decide where this identity is written.
    sys.path.insert(0, str(PROJECT_ROOT / "work" / "mjj"))
    rec_spec = importlib.util.spec_from_file_location(
        "frozen_reconstruct", PROJECT_ROOT / "work" / "mjj" / "reconstruct.py"
    )
    if rec_spec is None or rec_spec.loader is None:
        raise RuntimeError("cannot import reconstruct.py")
    rec = importlib.util.module_from_spec(rec_spec)
    rec_spec.loader.exec_module(rec)
    rec.OUT = out / "avatar"
    rec.OUT.mkdir(parents=True, exist_ok=True)
    poses = {
        "root_pose": torch.zeros(1, 3),
        "body_pose": torch.zeros(1, 21, 3),
        "left_hand_pose": torch.zeros(1, 15, 3),
        "right_hand_pose": torch.zeros(1, 15, 3),
        "translation_world": torch.zeros(1, 3),
        "scale": torch.tensor(1.0),
        "betas": torch.zeros(1, 10),
        "frames": torch.tensor([0]),
    }
    bundle = rec.export(
        model,
        poses,
        destination=".",
        attributes_file=str(out / "gaussian_attributes.pt"),
        latent_file=str(out / "latent.pt"),
    )
    del bundle
    identity = {
        "schema": "partner_lhm_identity/v1",
        "created_at_utc": utc_now(),
        "actor": "partner",
        "source_segment_index": str(segment_index),
        "source_frame": args.source_frame,
        "visible_views": visible,
        "unknown_views": unknown,
        "n_visible_views": n,
        "input_shape": list(tensors.shape),
        "avatar": str(rec.OUT / "avatar.pt"),
        "avatar_sha256": sha256_file(rec.OUT / "avatar.pt"),
        "rig_check": str(rec.OUT / "rig_check.json"),
        "latent": str(out / "latent.pt"),
        "gaussian_attributes": str(out / "gaussian_attributes.pt"),
        "lhm_model_cache": str(cache / "model_load"),
        "fcs_avatar_used": False,
        "white_background_fake_view_used": False,
        "unknown_views_are_not_filled": True,
        "scope": "Independent partner LHM++ identity from real SAM2-visible views only.",
    }
    atomic_write_json(out / "identity.json", identity)
    print(f"[identity] wrote {rec.OUT / 'avatar.pt'}")
    print(f"[identity] visible={n} unknown={len(unknown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
