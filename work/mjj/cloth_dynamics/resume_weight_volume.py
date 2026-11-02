#!/usr/bin/env python
"""Shared real weight-volume production for AnimatableGaussians and Wardrobe.

The author Poisson solve is preserved:
55 LBS joints, depth 7, gradientWeight 0.05, clip/normalize, xyz transpose,
author bounds, real libigl signed distance and barycentric LBS, six-field NPZ.

Only three scratch files are used per joint:
  current_values.xyz, current_gradients.xyz, current_grid.grd
The grid is read immediately and accumulated in RAM; no grid_00..54 files are
created, and a failed solve is never allowed to reuse a previous joint grid.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
AUTHOR = HERE / "author"
WARDROBE_AUTHOR = HERE / "wardrobe/author"
DEFAULT_AG_CONFIG = HERE / "configs/mjj001_template.yaml"
DEFAULT_WARDROBE_CONFIG = HERE / "wardrobe/configs/mjj001.yaml"
DEFAULT_POINT = AUTHOR / "bins/PointInterpolant.exe"
DEFAULT_SCRATCH = HERE / "runtime/weight_volume_work"
RESULTS = HERE / "runtime/results"
DEPTH = 7
GRADIENT_WEIGHT = 0.05
SDF_CHUNK = 262144


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def backend_bootstrap(backend: str, config: Path) -> Dict[str, Any]:
    """CPU bootstrap only; no CUDA allocation is required."""
    if backend == "ag":
        from runtime_backend import bootstrap
        ctx = bootstrap(config, require_cuda=False)
        ctx["config"].device = __import__("torch").device("cpu")
        return ctx
    if backend == "wardrobe":
        os.environ["WARDROBE_DEVICE"] = "cpu"
        from wardrobe.integration import bootstrap
        return bootstrap(config, device="cpu")
    raise ValueError(f"unknown backend: {backend}")


def make_model(ctx: Dict[str, Any], backend: str):
    import torch
    import smplx
    if backend == "ag":
        mod = ctx["smplx"]
        model_dir = ctx["model_dir"]
        expr_dim = int(ctx["expr_dim"])
    else:
        mod = smplx
        model_dir = ctx["smplx_model_dir"]
        expr_dim = int(ctx["expr_dim"])
    model = mod.SMPLX(
        model_path=str(model_dir), gender="neutral", use_pca=False,
        num_pca_comps=45, flat_hand_mean=True, batch_size=1,
        num_expression_coeffs=expr_dim,
    ).to(torch.device("cpu")).eval()
    return model


def canonical_mesh(ctx: Dict[str, Any], backend: str, data: Path, zero_shape: bool):
    import torch
    import trimesh
    cfg = ctx["config"]
    with np.load(data / "smpl_params.npz", allow_pickle=True) as z:
        betas_np = np.asarray(z["betas"][0], np.float32)
    betas = torch.zeros((10,), dtype=torch.float32) if zero_shape else torch.from_numpy(betas_np)
    model = make_model(ctx, backend)
    with torch.no_grad():
        out = model(
            betas=betas[None],
            global_orient=cfg.cano_smpl_global_orient[None],
            transl=cfg.cano_smpl_transl[None],
            body_pose=cfg.cano_smpl_body_pose[None],
        )
    vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise ValueError("invalid canonical SMPL-X mesh")
    return model, vertices, faces, mesh, betas.detach().cpu().numpy()


def compute_lbs_grad(vertices: np.ndarray, faces: np.ndarray,
                     mesh, vertex_lbs: np.ndarray) -> np.ndarray:
    """Author's exact normal/tangent/neighbor finite difference."""
    normals = mesh.vertex_normals.copy().astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tx = np.cross(normals, np.array([[0, 0, 1]], np.float32))
    tx /= np.linalg.norm(tx, axis=1, keepdims=True)
    ty = np.cross(normals, tx)

    vnum = vertices.shape[0]
    jnum = vertex_lbs.shape[1]
    lbs_grad_tx = np.zeros((vnum, jnum), np.float32)
    lbs_grad_ty = np.zeros((vnum, jnum), np.float32)
    for vidx in range(vnum):
        v = vertices[vidx]
        for jidx in range(jnum):
            for neighbor_vidx in mesh.vertex_neighbors[vidx]:
                vn = vertices[neighbor_vidx]
                pos_diff = vn - v
                pos_diff_norm = np.linalg.norm(pos_diff)
                val_diff = vertex_lbs[neighbor_vidx, jidx] - vertex_lbs[vidx, jidx]
                val_diff /= pos_diff_norm
                pos_diff /= pos_diff_norm
                lbs_grad_tx[vidx, jidx] += val_diff * np.dot(pos_diff, tx[vidx])
                lbs_grad_ty[vidx, jidx] += val_diff * np.dot(pos_diff, ty[vidx])
            n = float(len(mesh.vertex_neighbors[vidx]))
            lbs_grad_tx[vidx, jidx] /= n
            lbs_grad_ty[vidx, jidx] /= n
    return (lbs_grad_tx[:, :, None] * tx[:, None] +
            lbs_grad_ty[:, :, None] * ty[:, None]).astype(np.float32)


def read_grid(path: Path) -> np.ndarray:
    res = 2 ** DEPTH
    payload = res ** 3 * 8
    raw = path.read_bytes()
    if len(raw) < payload:
        raise RuntimeError(f"grid too small: {path}, {len(raw)} < {payload}")
    header = len(raw) - payload
    values = np.frombuffer(raw[header:], dtype=np.float64, count=res ** 3).copy()
    return values.reshape(res, res, res)


def solve_joint(exe: Path, scratch: Path, vertices: np.ndarray, vals: np.ndarray,
                grads: np.ndarray, jid: int) -> np.ndarray:
    values = scratch / "current_values.xyz"
    gradients = scratch / "current_gradients.xyz"
    grid = scratch / "current_grid.grd"
    np.savetxt(values, np.concatenate([vertices, vals[:, None]], 1), fmt="%.8f")
    np.savetxt(gradients, np.concatenate([vertices, grads], 1), fmt="%.8f")
    # A failed process must not leave/consume the previous joint's grid.
    grid.unlink(missing_ok=True)
    cmd = [
        str(exe), "--inValues", values.name, "--inGradients", gradients.name,
        "--gradientWeight", str(GRADIENT_WEIGHT), "--dim", "3", "--verbose",
        "--grid", grid.name, "--depth", str(DEPTH),
    ]
    subprocess.run(cmd, check=True, cwd=str(scratch))
    if not grid.is_file():
        raise RuntimeError(f"PointInterpolant produced no grid for joint {jid}: {grid}")
    if grid.stat().st_size < (2 ** DEPTH) ** 3 * 8:
        raise RuntimeError(f"PointInterpolant grid too small for joint {jid}: {grid}")
    return read_grid(grid)


def get_grid_points(bounds: np.ndarray, res) -> np.ndarray:
    x = np.linspace(bounds[0, 0], bounds[1, 0], res[0])
    y = np.linspace(bounds[0, 1], bounds[1, 1], res[1])
    z = np.linspace(bounds[0, 2], bounds[1, 2], res[2])
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)


def sdf_and_lbs(vertices: np.ndarray, faces: np.ndarray, lbs_weights: np.ndarray,
                pts: np.ndarray, volume_bounds: np.ndarray, res):
    import trimesh
    from libigl_compat import signed_distance
    dists_all, ori_all = [], []
    for start in range(0, len(pts), SDF_CHUNK):
        chunk = pts[start:start + SDF_CHUNK]
        dists, face_id, closest = signed_distance(
            chunk, vertices, faces.astype(np.int32))
        triangles = vertices[faces[face_id]]
        weights = lbs_weights[faces[face_id]]
        bary = trimesh.triangles.points_to_barycentric(triangles, closest)
        ori = (bary[:, :, None] * weights).sum(1)
        dists_all.append(np.asarray(dists, np.float32))
        ori_all.append(np.asarray(ori, np.float32))
    dists = np.concatenate(dists_all, 0).reshape(res)
    ori = np.concatenate(ori_all, 0).reshape(list(res) + [lbs_weights.shape[1]])
    return dists, ori


def write_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("ag", "wardrobe"), default="ag")
    ap.add_argument("--config", default=None)
    ap.add_argument("--zero-shape", action="store_true")
    ap.add_argument("--point-interpolant", default=str(DEFAULT_POINT))
    ap.add_argument("--scratch", default=str(DEFAULT_SCRATCH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    config = Path(a.config).resolve() if a.config else (
        DEFAULT_AG_CONFIG if a.backend == "ag" else DEFAULT_WARDROBE_CONFIG)
    exe = Path(a.point_interpolant).resolve()
    scratch = Path(a.scratch).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not exe.is_file():
        raise FileNotFoundError(exe)
    scratch.mkdir(parents=True, exist_ok=True)

    started = time.time()
    ctx = backend_bootstrap(a.backend, config)
    from libigl_compat import signed_distance
    import torch
    torch.set_num_threads(4)
    data = Path(ctx.get("data_dir_absolute") or ctx["data_dir"])
    out_name = "cano_weight_volume_shape_zero.npz" if a.zero_shape else "cano_weight_volume.npz"
    out_path = data / out_name
    model, vertices, faces, mesh, betas = canonical_mesh(
        ctx, a.backend, data, a.zero_shape)
    import torch
    lbs_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    if lbs_weights.shape != (vertices.shape[0], 55):
        raise ValueError(f"unexpected LBS shape {lbs_weights.shape}")
    joint_num = int(lbs_weights.shape[1])

    if a.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "backend": a.backend,
            "config": str(config),
            "data": str(data),
            "output": str(out_path),
            "point_interpolant": str(exe),
            "scratch": str(scratch),
            "joints": joint_num,
            "depth": DEPTH,
            "gradientWeight": GRADIENT_WEIGHT,
            "zero_shape": bool(a.zero_shape),
            "device": "cpu",
            "runs_55_solves": False,
        }, indent=2))
        return 0

    lbs_grad = compute_lbs_grad(vertices, faces, mesh, lbs_weights)
    grids = []
    solve_log = []
    for jid in range(joint_num):
        grid = solve_joint(exe, scratch, vertices, lbs_weights[:, jid],
                           lbs_grad[:, jid], jid)
        grids.append(grid)
        solve_log.append({"joint": jid, "grid_shape": list(grid.shape),
                          "grid_bytes": int(grid.nbytes)})
        print(f"solved joint {jid + 1}/{joint_num}", flush=True)

    grids_all = np.stack(grids, 0)
    grids_all = np.clip(grids_all, 0.0, 1.0)
    sums = grids_all.sum(0)
    if not np.isfinite(sums).all() or np.any(sums == 0):
        raise RuntimeError("invalid zero/non-finite joint grid normalization")
    grids_all = grids_all / sums[None]
    diff_weights = grids_all.transpose((3, 2, 1, 0)).astype(np.float32)

    min_xyz = vertices.min(0).astype(np.float32)
    max_xyz = vertices.max(0).astype(np.float32)
    max_len = 1.1 * (max_xyz - min_xyz).max()
    center = 0.5 * (min_xyz + max_xyz)
    volume_bounds = np.stack([center - 0.5 * max_len, center + 0.5 * max_len], 0)
    min_xyz[:2] -= 0.05
    max_xyz[:2] += 0.05
    min_xyz[2] -= 0.15
    max_xyz[2] += 0.15
    smpl_bounds = np.stack([min_xyz, max_xyz], 0)

    res = diff_weights.shape[:3]
    pts = get_grid_points(volume_bounds, res).reshape(-1, 3)
    dists, ori_weights = sdf_and_lbs(
        np.asarray(mesh.vertices, np.float64), faces, lbs_weights, pts, volume_bounds, res)
    sdf_volume = -dists.astype(np.float32)

    write_npz(
        out_path,
        diff_weight_volume=diff_weights,
        ori_weight_volume=ori_weights.astype(np.float32),
        sdf_volume=sdf_volume,
        volume_bounds=volume_bounds,
        smpl_bounds=smpl_bounds,
        center=center,
    )
    expected = {"diff_weight_volume", "ori_weight_volume", "sdf_volume",
                "volume_bounds", "smpl_bounds", "center"}
    with np.load(out_path, allow_pickle=True) as z:
        keys = set(z.files)
        shapes = {k: list(z[k].shape) for k in z.files}
        finite = all(bool(np.isfinite(z[k]).all()) for k in z.files)
    if not expected <= keys:
        raise RuntimeError(f"output keys missing: {sorted(expected - keys)}")
    source = (AUTHOR if a.backend == "ag" else WARDROBE_AUTHOR) / "gen_data/gen_weight_volume.py"
    model_dir = Path(ctx.get("model_dir") or ctx.get("smplx_model_dir"))
    model_file = model_dir / "SMPLX_NEUTRAL.npz"
    norm_sum = diff_weights.sum(axis=-1)
    report = {
        "schema": "weight_volume_production/v2",
        "status": "ok",
        "generated_at_utc": utc_now(),
        "backend": a.backend,
        "config": str(config), "config_sha256": sha256_file(config),
        "data": str(data),
        "smpl_params_sha256": sha256_file(data / "smpl_params.npz"),
        "source": str(source), "source_sha256": sha256_file(source),
        "smplx_file": ctx.get("smplx_file") or ctx.get("smplx_module"),
        "smplx_model_dir": str(model_dir),
        "smplx_model_file": str(model_file),
        "smplx_model_sha256": (sha256_file(model_file)
                               if model_file.is_file() else None),
        "point_interpolant": str(exe), "point_interpolant_sha256": sha256_file(exe),
        "scratch": str(scratch),
        "depth": DEPTH, "gradientWeight": GRADIENT_WEIGHT,
        "zero_shape": bool(a.zero_shape),
        "betas": betas.reshape(-1).tolist(),
        "canonical_pose": "config.cano_smpl_pose/global/transl/body_pose; zero expression",
        "device": "cpu",
        "solves": joint_num, "solve_log": solve_log,
        "diff_weight_volume_shape": list(diff_weights.shape),
        "ori_weight_volume_shape": shapes.get("ori_weight_volume"),
        "sdf_volume_shape": shapes.get("sdf_volume"),
        "grid_res": int(res[0]), "joints": joint_num,
        "normalization_max_abs_deviation": float(np.max(np.abs(norm_sum - 1.0))),
        "finite": bool(finite),
        "npz_keys": sorted(keys), "npz_shapes": shapes,
        "npz_sha256": sha256_file(out_path),
        "elapsed_sec": round(time.time() - started, 3),
        "quality_claim": None,
        "scope": "real author Poisson volume; finite/normalization are interface diagnostics only",
    }
    atomic_json(RESULTS / "weight_volume.json", report)
    print(json.dumps({
        "output": str(out_path), "solves": joint_num,
        "normalization_max_abs_deviation": report["normalization_max_abs_deviation"],
        "finite": report["finite"], "npz_sha256": report["npz_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
