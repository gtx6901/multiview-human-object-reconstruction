#!/usr/bin/env python
"""Beta-shaped layered template front-end from real multi-view semantics.

The input template is the existing AG beta-shaped canonical whole surface.  The
body layer is an explicit SMPL-X shape prior (not an observation of the body
under clothes).  Cloth faces are retained only when actual known cloth pixels
vote for them more than known body pixels across the selected real frames and
all eight cameras.  No colour threshold, joint partition, hole filling,
smoothing, or shape-zero conversion is performed.

Outputs are fixed under --data:
  beta_template_body.ply
  beta_template_cloth.ply
  layer_geometry_votes.npz
  layer_geometry_manifest.json

These are beta-shaped, not shape-zero.  No offset mesh, pkl, or zero-shape
volume is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CLOTH = HERE.parent
DEFAULT_CONFIG = HERE / "configs/mjj001.yaml"
DEFAULT_DATA = HERE / "data/mjj001"
DEFAULT_SOURCE_CONFIG = CLOTH / "author/config.py"
DEFAULT_TARGET_VOLUME = DEFAULT_DATA / "cano_weight_volume.npz"
from segmentation import (BODY_LABEL_DISK, UPPER_LABEL_DISK,
                         LOWER_LABEL_DISK, LABEL_PALETTE, BODY_CLASSES,
                         UPPER_CLASSES, LOWER_CLASSES, CLOTH_CLASSES)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_template(path: Path):
    import trimesh
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(str(path), process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(f"not a triangle mesh: {path}")
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"unexpected template arrays: {v.shape}, {f.shape}")
    if len(v) == 0 or len(f) == 0 or not np.isfinite(v).all():
        raise ValueError("empty/non-finite template")
    return v, f


def reader_dataset(config: Path, data: Path, frames: List[int], device: str,
                   use_label: bool = True):
    import sys
    sys.path.insert(0, str(HERE))
    from integration import bootstrap, install_native_reader
    ctx = bootstrap(config, device=device, data_dir=data)
    Native = install_native_reader()
    ds = Native(
        data_dir=str(data), frame_range=[0, 99, 1], used_cam_ids=list(range(8)),
        training=True, subject_name="mjj001", load_smpl_pos_map=False,
        layers=None, use_label=use_label, use_body_label=False,
        use_normal=False, mode="3dgs",
    )
    return ctx, ds, Native


def body_prior(ds) -> Tuple[np.ndarray, np.ndarray]:
    # dataset.cano_smpl is the zero-expression canonical body created by the
    # reader's actual SMPL-X model; item cano_smpl_v carries the frame expression.
    v = ds.cano_smpl["vertices"].detach().cpu().numpy().astype(np.float32)
    f = np.asarray(ds.smpl_faces, np.int64)
    if v.ndim != 2 or f.ndim != 2 or not np.isfinite(v).all():
        raise ValueError("invalid canonical body prior from reader")
    return v, f


def query_lbs(volume_path: Path, template_vertices: np.ndarray, device: torch.device) -> torch.Tensor:
    from network.volume import CanoBlendWeightVolume
    vol = CanoBlendWeightVolume(str(volume_path))
    try:
        pts = torch.from_numpy(template_vertices).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            lbs = vol.forward_weight(pts[None])[0]
        if lbs.ndim != 2 or lbs.shape[0] != template_vertices.shape[0]:
            raise ValueError(f"unexpected LBS shape {tuple(lbs.shape)}")
        if not torch.isfinite(lbs).all():
            raise ValueError("non-finite template LBS")
        return lbs
    finally:
        # Source and target volumes are never both resident.
        del vol
        gc.collect()


def live_vertices(template: torch.Tensor, lbs: torch.Tensor,
                  jnt_mats: torch.Tensor) -> torch.Tensor:
    # Same linear blend as author utils.smpl_util.skinning.
    pt_mats = torch.einsum("nj,jxy->nxy", lbs, jnt_mats)
    live = torch.einsum("nxy,ny->nx", pt_mats[..., :3, :3], template)
    return live + pt_mats[..., :3, 3]


def load_source_canonical_config(path: Path):
    """Read AG canonical definitions under an independent module name."""
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_ag_source_canonical_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source canonical config {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("cano_smpl_pose", "cano_smpl_global_orient",
                 "cano_smpl_transl", "cano_smpl_body_pose"):
        if not hasattr(module, attr):
            raise AttributeError(f"{path} missing {attr}")
    return module, {"path": str(path), "sha256": sha256_file(path), "module": name}


def vector_angle_deg(a, b) -> float:
    ta = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    tb = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    denom = torch.linalg.norm(ta) * torch.linalg.norm(tb)
    if float(denom) == 0.0:
        return 0.0
    c = torch.clamp(torch.dot(ta, tb) / denom, -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(c)))


def relative_canonical_vertices(ctx: Dict[str, Any], source_cfg, source_vertices: np.ndarray,
                               source_weights: torch.Tensor, data: Path,
                               device: torch.device):
    """Author-style relative LBS from source canonical A to target canonical A."""
    import smplx
    target_cfg = ctx["config"]
    with np.load(data / "smpl_params.npz", allow_pickle=True) as z:
        betas = torch.from_numpy(np.asarray(z["betas"][0], np.float32)).to(device)
    expr_dim = int(ctx["expr_dim"])
    model = smplx.SMPLX(
        model_path=ctx["smplx_model_dir"], gender="neutral", use_pca=False,
        num_pca_comps=45, flat_hand_mean=True, batch_size=1,
        num_expression_coeffs=expr_dim,
    ).to(device).eval()

    def forward(cfg):
        with torch.no_grad():
            return model(
                betas=betas[None],
                global_orient=cfg.cano_smpl_global_orient[None].to(device),
                transl=cfg.cano_smpl_transl[None].to(device),
                body_pose=cfg.cano_smpl_body_pose[None].to(device),
            )

    src = forward(source_cfg)
    tgt = forward(target_cfg)
    a_src = src.A[0].detach()
    a_tgt = tgt.A[0].detach()
    if a_src.shape != a_tgt.shape:
        raise ValueError(f"canonical A shape mismatch: {a_src.shape} vs {a_tgt.shape}")
    rel = torch.matmul(a_tgt, torch.linalg.inv(a_src))
    vh = torch.cat([
        torch.from_numpy(source_vertices).to(device=device, dtype=torch.float32),
        torch.ones((len(source_vertices), 1), dtype=torch.float32, device=device),
    ], dim=1)
    per_joint = torch.einsum("jab,nb->nja", rel, vh)
    v_target = (source_weights[:, :, None] * per_joint).sum(dim=1)[:, :3]
    if not torch.isfinite(v_target).all() or v_target.shape[0] != len(source_vertices):
        raise ValueError("invalid converted target canonical vertices")

    src_pose = torch.as_tensor(source_cfg.cano_smpl_pose).detach().cpu()
    tgt_pose = torch.as_tensor(target_cfg.cano_smpl_pose).detach().cpu()
    model_file = Path(ctx["smplx_model_dir"]) / "SMPLX_NEUTRAL.npz"
    info = {
        "method": "relative LBS: T_j=A_target_j @ inv(A_source_j); "
                  "V_target=sum_j w_source_j*(T_j*[V_source,1])",
        "source_config": source_cfg.__dict__.get("__file__", ""),
        "source_config_sha256": sha256_file(Path(source_cfg.__dict__["__file__"])),
        "target_config": str(ctx.get("config_path")),
        "target_config_sha256": sha256_file(Path(ctx["config_path"])),
        "source_canonical_pose": src_pose.reshape(-1).tolist(),
        "target_canonical_pose": tgt_pose.reshape(-1).tolist(),
        "source_canonical_global_orient": src_pose[3:6].tolist(),
        "target_canonical_global_orient": tgt_pose[3:6].tolist(),
        "source_canonical_transl": src_pose[:3].tolist(),
        "target_canonical_transl": tgt_pose[:3].tolist(),
        "source_canonical_body_pose": src_pose[6:69].tolist(),
        "target_canonical_body_pose": tgt_pose[6:69].tolist(),
        "canonical_body_pose_57_60_source_rad": [float(source_cfg.cano_smpl_body_pose[i]) for i in (57, 60)],
        "canonical_body_pose_57_60_target_rad": [float(target_cfg.cano_smpl_body_pose[i]) for i in (57, 60)],
        "joint_global_rotation_angle_deg": torch.rad2deg(torch.acos(((rel[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))).detach().cpu().tolist(),
        "transl_delta_maxabs": float(torch.max(torch.abs(src_pose[:3] - tgt_pose[:3]))),
        "smplx_model_file": str(model_file),
        "smplx_model_sha256": sha256_file(model_file) if model_file.is_file() else None,
        "flat_hand_mean": True,
        "expression_zero": True,
        "jaw_zero": True,
        "source_weight_vertex_count": int(source_vertices.shape[0]),
        "target_vertex_min_max": [float(v_target.min()), float(v_target.max())],
    }
    return v_target, info, rel.detach().cpu().numpy()


def canonical_body_vertices(ctx: Dict[str, Any], data: Path, device: torch.device):
    """Full canonical SMPL-X shape, beta vs all-zero beta, same canonical pose."""
    import smplx
    cfg = ctx["config"]
    expr_dim = int(ctx["expr_dim"])
    sp = data / "smpl_params.npz"
    if not sp.is_file():
        raise FileNotFoundError(sp)
    with np.load(sp, allow_pickle=True) as z:
        betas = torch.from_numpy(np.asarray(z["betas"][0], np.float32))
    betas = betas.to(device)
    zero = torch.zeros_like(betas)
    model = smplx.SMPLX(
        model_path=ctx["smplx_model_dir"], gender="neutral", use_pca=False,
        num_pca_comps=45, flat_hand_mean=True, batch_size=1,
        num_expression_coeffs=expr_dim,
    ).to(device).eval()
    go = cfg.cano_smpl_global_orient[None].to(device)
    tr = cfg.cano_smpl_transl[None].to(device)
    bp = cfg.cano_smpl_body_pose[None].to(device)
    expr = torch.zeros((1, expr_dim), dtype=torch.float32, device=device)

    def verts(b):
        with torch.no_grad():
            out = model(betas=b[None], global_orient=go, transl=tr, body_pose=bp,
                        expression=expr)
        return out.vertices[0].detach()

    v_beta = verts(betas)
    v_zero = verts(zero)
    if v_beta.shape != v_zero.shape or not torch.isfinite(v_beta).all():
        raise ValueError("invalid SMPL-X canonical vertices")
    return v_beta, v_zero, {
        "smplx_model_dir": ctx["smplx_model_dir"],
        "smplx_module": ctx["smplx_module"],
        "betas": betas.detach().cpu().numpy().tolist(),
        "expression_zero": True,
        "flat_hand_mean": True,
        "canonical_global_orient": cfg.cano_smpl_global_orient.tolist(),
        "canonical_transl": cfg.cano_smpl_transl.tolist(),
        "canonical_body_pose": cfg.cano_smpl_body_pose.tolist(),
        "expression_dim": expr_dim,
    }


def knn_shape_offset(query_vertices: np.ndarray, reference_beta: np.ndarray,
                     displacement: np.ndarray, device: torch.device,
                     chunk: int = 4096, K: int = 30) -> np.ndarray:
    """Author-style inverse-distance KNN displacement, no fixed grid."""
    from pytorch3d.ops import knn_points
    q = torch.from_numpy(query_vertices).to(device=device, dtype=torch.float32)
    ref = torch.from_numpy(reference_beta).to(device=device, dtype=torch.float32)[None]
    disp = torch.from_numpy(displacement).to(device=device, dtype=torch.float32)[None]
    outs = []
    for start in range(0, len(q), chunk):
        qc = q[start:start + chunk][None]
        dist, idx, _ = knn_points(qc, ref, K=K)
        dist = dist.sqrt().clamp(0.0001, 1.0)
        w = (1.0 / dist)
        w = w / w.sum(-1, keepdim=True)
        off = (w[..., None] * disp[0][idx[0]]).sum(-2).squeeze(0)
        outs.append(off)
    out = torch.cat(outs, 0) if outs else torch.zeros((0, 3), device=device)
    return out.detach().cpu().numpy().astype(np.float32)


def atomic_pickle(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def dataset_source_info() -> Dict[str, Any]:
    import dataset.dataset_mv_rgb as dsmod
    p = Path(dsmod.__file__).resolve()
    return {"path": str(p), "sha256": sha256_file(p)}


def canonicalize(data: Path, ctx: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    body_beta, body_faces = load_template(data / "beta_template_body.ply")
    cloth_beta, cloth_faces = load_template(data / "beta_template_cloth.ply")
    v_beta_t, v_zero_t, smpl_info = canonical_body_vertices(ctx, data, device)
    v_beta = v_beta_t.detach().cpu().numpy().astype(np.float32)
    v_zero = v_zero_t.detach().cpu().numpy().astype(np.float32)
    if body_beta.shape != v_beta.shape:
        raise ValueError(f"body prior vertices {body_beta.shape} != model {v_beta.shape}")
    prior_delta = float(np.max(np.abs(body_beta.astype(np.float64) - v_beta.astype(np.float64))))
    if prior_delta > 1e-5:
        raise RuntimeError(
            f"body prior does not match canonical SMPL-X V_beta: maxabs={prior_delta}; "
            "refusing to translate/scale silently")

    body_offset = (v_beta - v_zero).astype(np.float32)
    displacement = body_offset
    cloth_offset = knn_shape_offset(
        cloth_beta, v_beta, displacement, device, chunk=4096, K=30)
    cloth_zero = (cloth_beta - cloth_offset).astype(np.float32)

    body_roundtrip = float(np.max(np.abs(
        (v_zero.astype(np.float64) + body_offset.astype(np.float64)) - v_beta.astype(np.float64))))
    cloth_roundtrip = float(np.max(np.abs(
        (cloth_zero.astype(np.float64) + cloth_offset.astype(np.float64)) - cloth_beta.astype(np.float64))))
    if body_roundtrip > 1e-5 or cloth_roundtrip > 1e-5:
        raise RuntimeError(
            f"offset roundtrip too large: body={body_roundtrip}, cloth={cloth_roundtrip}")

    atomic_ply(data / "template_body_offset.ply", v_zero, body_faces)
    atomic_ply(data / "template_cloth_offset.ply", cloth_zero, cloth_faces)
    atomic_pickle(data / "template_body_offset.pkl", body_offset.astype(np.float32))
    atomic_pickle(data / "template_cloth_offset.pkl", cloth_offset.astype(np.float32))
    return {
        "method": "canonical SMPL-X V_beta/V_zero; full-pose shape difference; "
                  "cloth K30 inverse-distance displacement with no fixed grid",
        "beta_body_source": str(data / "beta_template_body.ply"),
        "beta_cloth_source": str(data / "beta_template_cloth.ply"),
        "beta_body_sha256": sha256_file(data / "beta_template_body.ply"),
        "beta_cloth_sha256": sha256_file(data / "beta_template_cloth.ply"),
        "smpl_model": smpl_info,
        "dataset_source": dataset_source_info(),
        "source_sha256": sha256_file(HERE / "integration.py"),
        "canonicalization_code_sha256": sha256_file(Path(__file__).resolve()),
        "K": 30, "chunk": 4096,
        "knn_reference": "V_beta vertices; D=V_beta-V_zero",
        "body_prior_delta_maxabs": prior_delta,
        "body_prior_matches_v_beta": True,
        "body_vertices": int(len(v_beta)),
        "body_offset_min_max": [float(body_offset.min()), float(body_offset.max())],
        "cloth_vertices": int(len(cloth_beta)),
        "cloth_faces": int(len(cloth_faces)),
        "cloth_offset_min_max": [float(cloth_offset.min()), float(cloth_offset.max())],
        "body_roundtrip_maxabs": body_roundtrip,
        "cloth_roundtrip_maxabs": cloth_roundtrip,
        "beta0_D_note": "V_beta-V_zero is mathematically zero when beta is all zero; "
                         "not separately executed in this run",
        "outputs": {
            "body_zero": str(data / "template_body_offset.ply"),
            "cloth_zero": str(data / "template_cloth_offset.ply"),
            "body_offset": str(data / "template_body_offset.pkl"),
            "cloth_offset": str(data / "template_cloth_offset.pkl"),
        },
        "output_hashes": {
            "body_zero": sha256_file(data / "template_body_offset.ply"),
            "cloth_zero": sha256_file(data / "template_cloth_offset.ply"),
            "body_offset": sha256_file(data / "template_body_offset.pkl"),
            "cloth_offset": sha256_file(data / "template_cloth_offset.pkl"),
        },
        "shape_gauge": "beta_template -> zero-shape body/cloth; beta = zero + offset; "
                       "not a measurement of occluded body under clothing",
    }


def generate_body_masks(data: Path, ctx: Dict[str, Any], ds, frames: List[int],
                        device: torch.device) -> Dict[str, Any]:
    """Rasterize the actual live SMPL-X body as a white geometric prior."""
    import cv2
    out_root = data / "body_masks"
    out_root.mkdir(parents=True, exist_ok=True)
    review = {}
    mask_paths = []
    mask_hash = hashlib.sha256()
    contact_times = [t for t in (0, 49, 98) if t in frames] or [frames[0]]
    for pose_idx in frames:
        for view_idx in range(8):
            cam = ds.cam_names[view_idx]
            item = ds.getitem(ds.data_list.index((pose_idx, view_idx)), training=True)
            color = np.asarray(item["color_img"], np.float32)
            if color.max() <= 1.0:
                color = color * 255.0
            color = np.clip(color, 0, 255).astype(np.uint8)
            if color.ndim != 3 or color.shape[2] != 3:
                raise ValueError(f"unexpected image shape {color.shape}")
            H, W = color.shape[:2]
            live = item["live_smpl_v"]
            if not isinstance(live, torch.Tensor):
                live = torch.as_tensor(live)
            live = live.to(device=device, dtype=torch.float32)
            faces = item["smpl_faces"]
            if not isinstance(faces, torch.Tensor):
                faces = torch.as_tensor(faces)
            faces = faces.to(device=device, dtype=torch.long)
            face_ids = raster_faces(live, faces, item["intr"], item["extr"], H, W)
            mask = (face_ids >= 0).astype(np.uint8) * 255
            cam_dir = out_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            path = cam_dir / f"{pose_idx:08d}.png"
            image = np.repeat(mask[:, :, None], 3, axis=2)
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"failed to write body mask {path}")
            mask_paths.append(str(path))
            mask_hash.update(path.read_bytes())
            if pose_idx in contact_times:
                review[(pose_idx, view_idx)] = (color.copy(), mask.copy())
        print(f"body masks {pose_idx + 1}/{len(frames)}", flush=True)

    cell_w, cell_h = 240, 180
    sheet = np.zeros((len(contact_times) * cell_h, 8 * cell_w, 3), np.uint8)
    for row, pose_idx in enumerate(contact_times):
        for col, view_idx in enumerate(range(8)):
            color, mask = review[(pose_idx, view_idx)]
            thumb = cv2.resize(color, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(thumb, contours, -1, (255, 255, 0), 1)
            cv2.putText(thumb, "PRIOR", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(thumb, f"{ds.cam_names[view_idx]} t{pose_idx}", (6, cell_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            sheet[row * cell_h:(row + 1) * cell_h,
                  col * cell_w:(col + 1) * cell_w] = thumb
    contact = data / "body_masks_contact_sheet.png"
    if not cv2.imwrite(str(contact), sheet):
        raise RuntimeError(f"failed to write {contact}")
    return {
        "mode": "body_masks_only",
        "frames": [int(t) for t in frames],
        "cameras": list(ds.cam_names),
        "mask_count": len(mask_paths),
        "mask_paths": mask_paths,
        "mask_files_sha256": mask_hash.hexdigest(),
        "contact_sheet": str(contact),
        "contact_sheet_sha256": sha256_file(contact),
        "contact_times": [int(t) for t in contact_times],
        "source": "actual Native reader live_smpl_v/smpl_faces; flat_hand_mean=True",
        "reader": "MvRgbDatasetMJJNative layers=None load_smpl_pos_map=False "
                  "use_label=False use_body_label=False mode=3dgs",
        "raster": "PyTorch3D actual R/t/K, native HxW, faces_per_pixel=1, "
                  "blur_radius=0, cull_backfaces=False; mask=(pix_to_face>=0)",
        "mask_format": "3-channel uint8 PNG; body prior 255 white, background 0",
        "white_target_source": "configured network.multi_layer_avatar.MultiLAvatarNet.render: "
                               "body Gaussian colors white 1.0, independent body render "
                               "background 0, label_body_rgb_map binary white; not "
                               "combined_network blue one-hot",
        "body_prior": "SMPL-X geometric full-body prior including body under clothing; "
                      "not an observed naked-body scan and not a SAM/skin segmentation",
        "not_sam": True,
        "not_observed": True,
        "hand_convention": "flat_hand_mean=True; Native actual live hand pose",
        "dataset_source": dataset_source_info(),
        "smplx_file": ctx.get("smplx_module"),
        "smplx_model_dir": ctx.get("smplx_model_dir"),
        "smplx_model_file": str(Path(ctx.get("smplx_model_dir")) / "SMPLX_NEUTRAL.npz"),
        "smplx_model_sha256": (
            sha256_file(Path(ctx.get("smplx_model_dir")) / "SMPLX_NEUTRAL.npz")
            if (Path(ctx.get("smplx_model_dir")) / "SMPLX_NEUTRAL.npz").is_file() else None),
        "geometry_code_sha256": sha256_file(Path(__file__).resolve()),
        "reader_code_sha256": sha256_file(HERE / "integration.py"),
        "no_training": True,
    }


def write_body_mask_manifest(data: Path, config: Path, info: Dict[str, Any]) -> None:
    path = data / "layer_geometry_manifest.json"
    existing = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("schema", "gaussian_wardrobe_layer_geometry/v1")
    existing.update({
        "generated_at_utc": utc_now(),
        "data": str(data),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "body_mask_generation": info,
    })
    atomic_json(path, existing)


def raster_faces(live: torch.Tensor, faces: torch.Tensor, K: np.ndarray,
                 E: np.ndarray, H: int, W: int) -> np.ndarray:
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.renderer.mesh.rasterize_meshes import kMaxFacesPerBin
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.structures import Meshes
    dev = live.device
    R = torch.from_numpy(np.asarray(E, np.float32)[:3, :3]).to(dev)[None]
    t = torch.from_numpy(np.asarray(E, np.float32)[:3, 3]).to(dev)[None]
    Km = torch.from_numpy(np.asarray(K, np.float32)).to(dev)[None]
    size = torch.tensor([[H, W]], dtype=torch.float32, device=dev)
    cams = cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=Km, image_size=size)
    settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        cull_backfaces=False,
        # A folded body can project many triangles into one tile.  The default
        # heuristic capacity silently drops faces at overflow; an all-face
        # capacity preserves the same nearest-face rasterization exactly.
        # Use the smallest tile allowed by this installed CUDA kernel.  It
        # requires ceil(max_image_size/bin_size) < kMaxFacesPerBin.
        # The all-face capacity and nearest-face rule stay unchanged.
        bin_size=(max(H, W) + kMaxFacesPerBin - 2) // (kMaxFacesPerBin - 1),
        max_faces_per_bin=int(faces.shape[0]),
    )
    rasterizer = MeshRasterizer(cameras=cams, raster_settings=settings)
    mesh = Meshes(verts=[live], faces=[faces])
    frag = rasterizer(meshes_world=mesh)
    return frag.pix_to_face[0, ..., 0].detach().cpu().numpy()


def label_colors(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.rint(np.asarray(label, np.float32) * 255.0).astype(np.uint8)
    body = np.all(rows == BODY_LABEL_DISK[None, None, :], axis=-1)
    upper = np.all(rows == UPPER_LABEL_DISK[None, None, :], axis=-1)
    lower = np.all(rows == LOWER_LABEL_DISK[None, None, :], axis=-1)
    return body, upper, lower


def add_votes(face_ids: np.ndarray, body_px: np.ndarray, cloth_px: np.ndarray,
              n_faces: int, body_votes: np.ndarray, cloth_votes: np.ndarray) -> None:
    visible = face_ids >= 0
    for mask, acc in ((body_px, body_votes), (cloth_px, cloth_votes)):
        sel = visible & mask
        if sel.any():
            acc += np.bincount(face_ids[sel].reshape(-1), minlength=n_faces)[:n_faces]


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def atomic_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import trimesh
    tmp = path.with_name(path.stem + ".tmp.ply")
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(tmp))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--template", default=None)
    ap.add_argument("--weight-volume", default=None)
    ap.add_argument("--source-canonical-config", default=str(DEFAULT_SOURCE_CONFIG))
    ap.add_argument("--target-weight-volume", default=str(DEFAULT_TARGET_VOLUME))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--canonicalize-only", action="store_true")
    mode.add_argument("--body-masks-only", action="store_true")
    mode.add_argument("--semantic-votes-only", action="store_true")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = ap.parse_args()

    config = Path(a.config).resolve()
    data = Path(a.data).resolve()
    template_path = Path(a.template).resolve() if a.template else None
    volume_path = Path(a.weight_volume).resolve() if a.weight_volume else None
    source_config_path = Path(a.source_canonical_config).resolve()
    target_volume_path = Path(a.target_weight_volume).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not data.is_dir():
        raise FileNotFoundError(data)
    if not (a.canonicalize_only or a.body_masks_only):
        if template_path is None or not template_path.is_file():
            raise FileNotFoundError("--template is required for voting mode")
        if volume_path is None or not volume_path.is_file():
            raise FileNotFoundError("--weight-volume is required for voting mode")
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        if not target_volume_path.is_file():
            raise FileNotFoundError(
                f"target canonical weight volume missing: {target_volume_path}; "
                "cannot reuse the source AG volume")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dev = torch.device(a.device)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    if a.canonicalize_only:
        import sys
        sys.path.insert(0, str(HERE))
        from integration import bootstrap
        ctx = bootstrap(config, device=a.device, data_dir=data)
        canon = canonicalize(data, ctx, dev)
        manifest_path = data / "layer_geometry_manifest.json"
        existing = {}
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing.update({
            "schema": "gaussian_wardrobe_layer_geometry/v1",
            "generated_at_utc": utc_now(),
            "data": str(data),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "canonicalization": canon,
            "canonicalization_only": True,
            "canonicalization_elapsed_sec": round(time.time() - started, 3),
            "canonicalization_peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated()) if a.device == "cuda" else None),
        })
        out = dict(existing.get("outputs", {}))
        out.update(canon["outputs"])
        existing["outputs"] = out
        hashes = dict(existing.get("output_hashes", {}))
        hashes.update(canon["output_hashes"])
        existing["output_hashes"] = hashes
        atomic_json(manifest_path, existing)
        print(json.dumps({k: canon[k] for k in (
            "body_prior_delta_maxabs", "body_roundtrip_maxabs",
            "cloth_roundtrip_maxabs", "body_vertices", "cloth_vertices")}, indent=2))
        return 0

    frames = list(range(99)) if a.frames is None else [
        int(x) for x in a.frames.split(",") if x.strip()]
    if any(t < 0 or t >= 99 for t in frames):
        raise ValueError("frame index outside 0..98")
    frames = sorted(set(frames))
    if not frames:
        raise ValueError("empty --frames")

    if a.body_masks_only:
        ctx, ds, Native = reader_dataset(
            config, data, frames, a.device, use_label=False)
        info = generate_body_masks(data, ctx, ds, frames, dev)
        write_body_mask_manifest(data, config, info)
        print(json.dumps({
            "mask_count": info["mask_count"],
            "contact_sheet": info["contact_sheet"],
            "mask_files_sha256": info["mask_files_sha256"],
        }, indent=2))
        return 0

    ctx, ds, Native = reader_dataset(config, data, frames, a.device)
    source_cfg, source_cfg_info = load_source_canonical_config(source_config_path)
    source_v, source_f = load_template(template_path)
    body_v, body_f = body_prior(ds)

    # Source AG coordinates: query the source AG volume only.
    w_source = query_lbs(volume_path, source_v, dev)
    target_v_t, conversion_info, rel = relative_canonical_vertices(
        ctx, source_cfg, source_v, w_source, data, dev)
    target_v = target_v_t.detach().cpu().numpy().astype(np.float32)
    del w_source, rel
    # Target canonical coordinates: target Wardrobe volume must supply LBS.
    w_target = query_lbs(target_volume_path, target_v, dev)

    template_t = torch.from_numpy(target_v).to(dev)
    faces_t = torch.from_numpy(source_f).to(dev)
    lbs = w_target

    conversion_info.update({
        "source_canonical_config_info": source_cfg_info,
        "source_template": str(template_path),
        "source_template_sha256": sha256_file(template_path),
        "source_weight_volume": str(volume_path),
        "source_weight_volume_sha256": sha256_file(volume_path),
        "target_weight_volume": str(target_volume_path),
        "target_weight_volume_sha256": sha256_file(target_volume_path),
        "source_weights_vertex_count": int(len(source_v)),
        "target_weights_vertex_count": int(len(target_v)),
        "source_faces": int(len(source_f)),
        "target_faces": int(len(source_f)),
    })

    n_faces = len(source_f)
    body_votes = np.zeros(n_faces, np.int64)
    cloth_votes = np.zeros(n_faces, np.int64)
    upper_votes = np.zeros(n_faces, np.int64)
    lower_votes = np.zeros(n_faces, np.int64)
    semantic_votes = bool(a.semantic_votes_only)
    progress_path = data / "semantic_votes_progress.npz"
    progress_signature = None
    progress_resumed = False
    progress_mismatch = False
    next_pose = 0
    if semantic_votes:
        seg_manifest = data / "segmentation_manifest.json"
        if not seg_manifest.is_file():
            raise FileNotFoundError(seg_manifest)
        signature_payload = {
            "template_sha256": sha256_file(template_path),
            "source_weight_volume_sha256": sha256_file(volume_path),
            "target_weight_volume_sha256": sha256_file(target_volume_path),
            "config_sha256": sha256_file(config),
            "source_canonical_config_sha256": sha256_file(source_config_path),
            "segmentation_manifest_sha256": sha256_file(seg_manifest),
            "frames": [int(x) for x in frames],
            "cameras": list(ds.cam_names),
        }
        progress_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
        if progress_path.is_file():
            with np.load(progress_path, allow_pickle=True) as p:
                stored_sig = str(p["source_signature"].item())
                if stored_sig == progress_signature:
                    for key, arr in (
                        ("body_votes", body_votes), ("cloth_votes", cloth_votes),
                        ("upper_votes", upper_votes), ("lower_votes", lower_votes),
                    ):
                        if key not in p.files or p[key].shape != arr.shape:
                            raise RuntimeError(
                                f"progress array shape/signature mismatch: {key}")
                        arr[:] = p[key]
                    next_pose = int(p["next_pose"].item())
                    if next_pose < 0 or next_pose > len(frames):
                        raise RuntimeError(f"invalid progress next_pose: {next_pose}")
                    progress_resumed = True
                else:
                    progress_mismatch = True
    used = []
    for pose_idx in frames[:next_pose]:
        for view_idx, cam in enumerate(ds.cam_names):
            used.append({
                "pose_idx": int(pose_idx), "view_idx": int(view_idx),
                "camera": cam,
                "height": int(ds.img_heights[view_idx]),
                "width": int(ds.img_widths[view_idx]),
            })

    def save_progress_if_needed() -> None:
        if not semantic_votes:
            return
        atomic_npz(
            progress_path,
            body_votes=body_votes,
            cloth_votes=cloth_votes,
            upper_votes=upper_votes,
            lower_votes=lower_votes,
            next_pose=np.int64(next_pose),
            source_signature=np.asarray(progress_signature),
            frames=np.asarray(frames, np.int64),
            cameras=np.asarray(ds.cam_names),
        )

    if progress_resumed:
        print(f"resumed semantic votes at pose {next_pose}/{len(frames)}",
              flush=True)
    if progress_mismatch:
        print("semantic progress signature mismatch; starting from zero",
              flush=True)
    label_hash = hashlib.sha256()
    semantic_hash = hashlib.sha256()
    for pose_idx in frames[:next_pose]:
        for cam in ds.cam_names:
            lp = data / "labels_color" / cam / f"{pose_idx:08d}.png"
            sp = data / "semantic_valid" / cam / f"{pose_idx:08d}.png"
            label_hash.update(lp.read_bytes())
            semantic_hash.update(sp.read_bytes())

    for pose_pos, pose_idx in enumerate(frames):
        if pose_pos < next_pose:
            continue
        live = None
        for view_idx in range(8):
            cam = ds.cam_names[view_idx]
            item = ds.getitem(ds.data_list.index((pose_idx, view_idx)), training=True)
            label = np.asarray(item["label_img"])
            valid = np.asarray(item["label_valid_img"]).astype(bool)
            inst = np.asarray(item["mask_img"]).astype(bool)
            body_px, upper_px, lower_px = label_colors(label)
            body_px &= valid & inst
            upper_px &= valid & inst
            lower_px &= valid & inst
            cloth_px = upper_px | lower_px
            H, W = inst.shape
            if live is None:
                # The pose's cano2live matrices are identical across cameras.
                mats = torch.as_tensor(item["cano2live_jnt_mats"],
                                       dtype=torch.float32, device=dev)
                live = live_vertices(template_t, lbs, mats)
            face_ids = raster_faces(live, faces_t, item["intr"], item["extr"], H, W)
            add_votes(face_ids, body_px, cloth_px, n_faces, body_votes, cloth_votes)
            if semantic_votes:
                add_votes(face_ids, upper_px, lower_px, n_faces, upper_votes, lower_votes)
            used.append({"pose_idx": int(pose_idx), "view_idx": int(view_idx), "camera": cam,
                         "height": int(H), "width": int(W)})
            lp = data / "labels_color" / cam / f"{pose_idx:08d}.png"
            sp = data / "semantic_valid" / cam / f"{pose_idx:08d}.png"
            label_hash.update(lp.read_bytes())
            semantic_hash.update(sp.read_bytes())
        next_pose = pose_pos + 1
        save_progress_if_needed()
        print(f"semantic votes pose {next_pose}/{len(frames)} complete",
              flush=True)

    body_dom = body_votes > cloth_votes
    cloth_faces = (cloth_votes > body_votes) & (cloth_votes > 0)
    unassigned = ~(body_dom | cloth_faces)
    if semantic_votes:
        if len(used) != 99 * 8:
            raise RuntimeError(
                f"semantic-votes-only requires complete 99x8, got {len(used)}; "
                "progress retained, refusing to write final votes")
        votes_path = data / "layer_geometry_votes.npz"
        if not votes_path.is_file():
            raise FileNotFoundError(votes_path)
        with np.load(votes_path, allow_pickle=True) as z:
            old = {k: z[k] for k in z.files}
        required = ("faces", "body_votes", "cloth_votes", "body_faces",
                    "cloth_faces", "unassigned_faces")
        missing = [k for k in required if k not in old]
        if missing:
            raise RuntimeError(f"existing votes missing arrays: {missing}")
        old_faces = np.asarray(old["faces"])
        old_cloth_faces = np.asarray(old["cloth_faces"], bool)
        faces_same = bool(np.array_equal(old_faces, source_f))
        if not faces_same:
            raise RuntimeError(
                "semantic-votes-only faces differ from existing votes; "
                "refusing to overwrite geometry/index contract")
        body_votes_diff = int(np.count_nonzero(
            np.asarray(old["body_votes"]) != body_votes))
        cloth_votes_diff = int(np.count_nonzero(
            np.asarray(old["cloth_votes"]) != cloth_votes))
        body_faces_diff = int(np.count_nonzero(
            np.asarray(old["body_faces"], bool) != body_dom))
        cloth_faces_diff = int(np.count_nonzero(old_cloth_faces != cloth_faces))
        unassigned_diff = int(np.count_nonzero(
            np.asarray(old["unassigned_faces"], bool) != unassigned))
        upper_majority = old_cloth_faces & (upper_votes > lower_votes)
        lower_majority = old_cloth_faces & (lower_votes > upper_votes)
        ambiguous = old_cloth_faces & ~(upper_majority | lower_majority)
        old_sha = sha256_file(votes_path)
        payload = dict(old)
        payload.update({
            "upper_votes": upper_votes,
            "lower_votes": lower_votes,
            "upper_majority_faces": upper_majority,
            "lower_majority_faces": lower_majority,
            "ambiguous_cloth_faces": ambiguous,
        })
        atomic_npz(votes_path, **payload)
        info = {
            "mode": "semantic_votes_only",
            "used_items": len(used),
            "used_99x8": bool(len(used) == 99 * 8),
            "frames": [int(x) for x in frames],
            "cameras": list(ds.cam_names),
            "template": str(template_path),
            "template_sha256": sha256_file(template_path),
            "source_weight_volume": str(volume_path),
            "source_weight_volume_sha256": sha256_file(volume_path),
            "target_weight_volume": str(target_volume_path),
            "target_weight_volume_sha256": sha256_file(target_volume_path),
            "source_canonical_config": str(source_config_path),
            "source_canonical_config_sha256": sha256_file(source_config_path),
            "config_sha256": sha256_file(config),
            "labels_color_files_sha256": label_hash.hexdigest(),
            "semantic_valid_files_sha256": semantic_hash.hexdigest(),
            "palette": LABEL_PALETTE,
            "body_classes": sorted(BODY_CLASSES),
            "upper_classes": sorted(UPPER_CLASSES),
            "lower_classes": sorted(LOWER_CLASSES),
            "cloth_classes": sorted(CLOTH_CLASSES),
            "faces_sha256": hashlib.sha256(source_f.tobytes()).hexdigest(),
            "old_votes_sha256": old_sha,
            "output_sha256": sha256_file(votes_path),
            "array_sha256": {
                key: hashlib.sha256(np.ascontiguousarray(payload[key]).tobytes()).hexdigest()
                for key in ("faces", "body_votes", "cloth_votes", "body_faces",
                            "cloth_faces", "unassigned_faces", "upper_votes",
                            "lower_votes", "upper_majority_faces",
                            "lower_majority_faces", "ambiguous_cloth_faces")
            },
            "faces_same": faces_same,
            "source_vote_diagnostics": {
                "body_votes_diff_count": body_votes_diff,
                "cloth_votes_diff_count": cloth_votes_diff,
                "body_faces_diff_count": body_faces_diff,
                "cloth_faces_diff_count": cloth_faces_diff,
                "unassigned_faces_diff_count": unassigned_diff,
                "old_body_cloth_arrays_preserved": True,
            },
            "progress": {
                "signature": progress_signature,
                "resumed": progress_resumed,
                "mismatch_started_from_zero": progress_mismatch,
                "completed_poses": next_pose,
                "path": str(progress_path),
                "removed_after_success": True,
            },
            "upper_vote_sum": int(upper_votes.sum()),
            "lower_vote_sum": int(lower_votes.sum()),
            "upper_majority_faces": int(upper_majority.sum()),
            "lower_majority_faces": int(lower_majority.sum()),
            "ambiguous_cloth_faces": int(ambiguous.sum()),
            "cloth_faces": int(old_cloth_faces.sum()),
            "no_geometry_write": True,
            "no_canonicalize": True,
            "no_label_rewrite": True,
            "quality_claim": None,
        }
        manifest = data / "layer_geometry_manifest.json"
        existing_manifest = {}
        if manifest.is_file():
            existing_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        existing_manifest.setdefault("schema", "gaussian_wardrobe_layer_geometry/v1")
        existing_manifest.update({
            "generated_at_utc": utc_now(),
            "semantic_votes": info,
        })
        tmp = manifest.with_name(manifest.name + ".tmp")
        tmp.write_text(json.dumps(existing_manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, manifest)
        progress_path.unlink(missing_ok=True)
        print(json.dumps({
            "mode": "semantic_votes_only",
            "used_items": len(used),
            "upper_votes": int(upper_votes.sum()),
            "lower_votes": int(lower_votes.sum()),
            "upper_majority_faces": int(upper_majority.sum()),
            "lower_majority_faces": int(lower_majority.sum()),
            "ambiguous_cloth_faces": int(ambiguous.sum()),
            "output_sha256": info["output_sha256"],
        }, indent=2))
        return 0
    if not cloth_faces.any():
        raise RuntimeError(
            "no observed cloth faces: cloth_votes never exceed body_votes; "
            "refusing to create a layered template without observed support")

    # Body is an explicit SMPL-X shape prior, not the occluded-body observation.
    atomic_ply(data / "beta_template_body.ply", body_v, body_f)
    # Preserve original template vertices/faces; only selected faces are exported.
    atomic_ply(data / "beta_template_cloth.ply", target_v, source_f[cloth_faces])
    atomic_npz(
        data / "layer_geometry_votes.npz",
        faces=source_f,
        body_votes=body_votes,
        cloth_votes=cloth_votes,
        body_faces=body_dom,
        cloth_faces=cloth_faces,
        unassigned_faces=unassigned,
    )
    # Same canonicalization function as --canonicalize-only.
    canon = canonicalize(data, ctx, dev)
    peak = int(torch.cuda.max_memory_allocated()) if a.device == "cuda" else None
    smplx_model_file = Path(ctx["smplx_model_dir"]) / "SMPLX_NEUTRAL.npz"
    config_hash = sha256_file(config)
    report = {
        "schema": "gaussian_wardrobe_layer_geometry/v1",
        "generated_at_utc": utc_now(),
        "data": str(data),
        "config": str(config),
        "config_sha256": config_hash,
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "weight_volume": str(volume_path),
        "weight_volume_sha256": sha256_file(volume_path),
        "source_weight_volume": str(volume_path),
        "source_weight_volume_sha256": sha256_file(volume_path),
        "target_weight_volume": str(target_volume_path),
        "target_weight_volume_sha256": sha256_file(target_volume_path),
        "source_canonical_config": str(source_config_path),
        "source_canonical_config_sha256": sha256_file(source_config_path),
        "labels_color_manifest": str(data / "segmentation_manifest.json"),
        "labels_color_manifest_sha256": (sha256_file(data / "segmentation_manifest.json")
                                          if (data / "segmentation_manifest.json").is_file() else None),
        "source_frames_file": str(data / "source_frames.json"),
        "source_frames_sha256": (sha256_file(data / "source_frames.json")
                                 if (data / "source_frames.json").is_file() else None),
        "smpl_params_sha256": (sha256_file(data / "smpl_params.npz")
                               if (data / "smpl_params.npz").is_file() else None),
        "cameras_sha256": (sha256_file(data / "cameras.pkl")
                           if (data / "cameras.pkl").is_file() else None),
        "labels_color_files_sha256": label_hash.hexdigest(),
        "semantic_valid_files_sha256": semantic_hash.hexdigest(),
        "reader_code_sha256": sha256_file(HERE / "integration.py"),
        "geometry_code_sha256": sha256_file(Path(__file__).resolve()),
        "smplx_file": ctx.get("smplx_module"),
        "smplx_model_dir": ctx.get("smplx_model_dir"),
        "smplx_model_sha256": (sha256_file(smplx_model_file)
                               if smplx_model_file.is_file() else None),
        "source_frames": frames,
        "cameras": ds.cam_names,
        "used_items": used,
        "reader": "MvRgbDatasetMJJNative layers=None use_label=True load_smpl_pos_map=False use_body_label=False mode=3dgs",
        "reader_class": "MvRgbDatasetMJJNative (thin subclass of MvRgbDataset4DDress/Base)",
        "dataset_source": dataset_source_info(),
        "visibility": "PyTorch3D cameras_from_opencv_projection + MeshRasterizer; "
                      "native HxW; faces_per_pixel=1; blur_radius=0; cull_backfaces=False; "
                      "actual R,t,K; no projected-vertex/backface-only voting",
        "vote_rule": "known foreground only; body BGR [0,0,255], cloth BGR [0,255,0]; "
                     "unknown/outside excluded; face bincount across all frames/cameras",
        "cloth_faces_rule": "cloth_votes > body_votes and cloth_votes > 0",
        "unassigned_rule": "equal/zero body-vs-cloth votes; no fill/smooth/hole repair",
        "shape_gauge": "source AG template is converted to target Wardrobe canonical "
                       "coordinates by relative LBS A_target@inv(A_source); beta templates, "
                       "target weights and zero-shape canonicalization are target-coordinate; "
                       "source AG volume is never used as target volume",
        "body_prior": "target reader.cano_smpl zero-expression canonical SMPL-X vertices/faces "
                      "(explicit prior; not observed body under clothing)",
        "canonicalization": canon,
        "canonical_space_conversion": conversion_info,
        "template_vertices": int(len(source_v)), "template_faces": int(len(source_f)),
        "body_vertices": int(len(body_v)), "body_faces": int(len(body_f)),
        "cloth_face_count": int(cloth_faces.sum()),
        "body_dominant_face_count": int(body_dom.sum()),
        "unassigned_face_count": int(unassigned.sum()),
        "body_vote_sum": int(body_votes.sum()), "cloth_vote_sum": int(cloth_votes.sum()),
        "outputs": {
            "body": str(data / "beta_template_body.ply"),
            "cloth": str(data / "beta_template_cloth.ply"),
            "votes": str(data / "layer_geometry_votes.npz"),
            **canon["outputs"],
        },
        "output_hashes": {
            "body": sha256_file(data / "beta_template_body.ply"),
            "cloth": sha256_file(data / "beta_template_cloth.ply"),
            "votes": sha256_file(data / "layer_geometry_votes.npz"),
            **canon["output_hashes"],
        },
        "elapsed_sec": round(time.time() - started, 3),
        "peak_vram_bytes": peak,
        "scope": "beta-shaped layered-template front-end; not shape-zero, not training, not quality",
    }
    atomic_json(data / "layer_geometry_manifest.json", report)
    print(json.dumps({k: report[k] for k in (
        "template_faces", "cloth_face_count", "body_dominant_face_count",
        "unassigned_face_count", "body_vote_sum", "cloth_vote_sum",
        "elapsed_sec", "peak_vram_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
