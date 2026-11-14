"""Build one identity with the shared LHM encoder, fusion and Gaussian decoder."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from hoi.runtime import configure_runtime, runtime_settings
from hoi.io import write_json as atomic_write_json
from hoi.identity.avatar import move, export_avatar
from hoi.identity.model import load_model

LHM = Path(runtime_settings()["lhm_root"])


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
    if str(idx.get("coordinate_space")) not in {
        "undistorted_pixels",
        "undistorted_native_2048x1536",
    }:
        raise ValueError("segment index must be in undistorted pixel coordinates")
    if actor not in idx["actors"]:
        raise ValueError(actor)
    # Project-specific Unicode-safe cv2 I/O used by the running segment code.
    from hoi.image_io import read_image

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


def build_identity(args: argparse.Namespace, out: Path, destination: Path) -> None:
    segment_index = Path(args.segment_index).resolve()
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
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise RuntimeError(report["error"])

    from hoi.identity.attention import install

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
    n = tensors.shape[0]
    print(f"[identity] loading model; visible views={n}", flush=True)
    model = load_model()
    model.eval()
    print("[identity] loading image encoder on device", flush=True)
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
    model.encoder.cpu()
    del f
    torch.cuda.empty_cache()

    model.forward_encode_image = lambda image: features.to(image.device)
    print("[identity] fusing multiview features", flush=True)
    model.transformer.to(args.device)
    model.motion_embed_mlp.to(args.device)
    model.renderer.smplx_model.to(args.device)
    betas = torch.zeros(1, 10, device=args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        query, params = model.renderer.get_query_points(
            None, {"betas": betas}, torch.device(args.device)
        )
        result = model.forward_latent_points(
            tensors[None].to(args.device),
            None,
            query_points=query,
            ref_imgs_bool=torch.ones(1, n, dtype=torch.bool, device=args.device),
        )
    torch.cuda.synchronize()
    latent = {
        "result": move(result, "cpu"),
        "query": move(query, "cpu"),
        "smplx": move(params, "cpu"),
    }
    torch.save(latent, out / "latent.pt")

    # As in the existing separate-stage runs, encoder/fusion weights need not
    # occupy the GPU while the renderer decodes the unchanged latent tensors.
    model.transformer.cpu()
    model.motion_embed_mlp.cpu()
    del result, query, params
    torch.cuda.empty_cache()
    # Decode exactly as lhm_decode.py.
    print("[identity] decoding Gaussian appearance", flush=True)
    model.renderer.to(args.device)
    model.renderer.hyper_step(10000000)
    data = move(latent, args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        attrs, q2, p2 = model.renderer.forward_gs(data["result"][0], data["query"], data["smplx"])
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

    poses = {
        "root_pose": torch.zeros(1, 3),
        "body_pose": torch.zeros(1, 21, 3),
        "left_hand_pose": torch.zeros(1, 15, 3),
        "right_hand_pose": torch.zeros(1, 15, 3),
        "translation_world": torch.zeros(1, 3),
        "scale": torch.tensor(1.0),
        "betas": torch.zeros(1, 10),
        "frames": torch.tensor([args.source_frame]),
    }
    if args.proportions:
        proportions = torch.load(args.proportions, map_location="cpu", weights_only=True)
        poses["betas"] = proportions["beta"].reshape(1, 10).float()
        poses["scale"] = torch.as_tensor(proportions["scale"]).float()
    export_avatar(model, poses, destination, out / "gaussian_attributes.pt", out / "latent.pt")
    atomic_write_json(
        destination / "identity.json",
        {
            "actor": args.actor,
            "source_frame": args.source_frame,
            "segment_index": str(segment_index),
            "input_order": [item["serial"] for item in crop_records],
            "unknown_views": [
                {"serial": item["serial"], "reason": item["reason"]} for item in unknown
            ],
            "avatar": str(destination / "avatar.pt"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--proportions")
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure_runtime()
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".identity-", dir=destination) as temporary:
        build_identity(args, Path(temporary), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
