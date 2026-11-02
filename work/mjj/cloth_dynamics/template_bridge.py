#!/usr/bin/env python
"""Template / canonical-map bridge for AnimatableGaussians (CPU only).

What this file does
-------------------
The official reader (dataset/dataset_mv_rgb.py, commit 2b0f6e3) expects:

    data_dir/calibration.json          cam00..camNN, R (3x3), T (3), K (3x3), imgSize [w,h]
    data_dir/images/camXX/%08d.jpg     pose_idx, not source capture index
    data_dir/masks/camXX/%08d.jpg      grayscale; >128 is foreground
    data_dir/smpl_params.npz           betas, global_orient, transl, body_pose,
                                       jaw_pose, expression, left_hand_pose, right_hand_pose
    data_dir/smpl_pos_map/cano_smpl_pos_map.exr   1024x2048 float RGB (front|back)
    data_dir/smpl_pos_map/cano_smpl_nml_map.exr   1024x2048 float RGB
    data_dir/smpl_pos_map/%08d.exr                512x1024  float RGB
    data_dir/smpl_pos_map/init_pts_lbs.npy        (N,55) float32, row order = mask row-major

The official generator (gen_data/gen_pos_maps.py) builds a canonical *template*
mesh, renders front/back position/normal maps with an OpenGL renderer, transfers
LBS weights from that template, and deforms the canonical pixels with
`live_smpl_woRoot.A @ inv(cano_smpl.A)`.  That matrix excludes root translation,
hands, and expression?  It includes body + jaw + expression, excludes global
orientation / transl / hand pose.  This file reproduces the same algebra on CPU.

Important honest limits
-----------------------
* Without `--template-ply` the "template" is the subject's SMPL-X body under the
  author canonical pose.  It is an *initial* body template, NOT a reconstructed
  clothed template.  Loose clothing must be optimized from this data later.
* For a custom template mesh, LBS is transferred from the nearest body surface
  (IDW of nearest SMPL-X vertices).  It is not the author's Poisson weight volume
  (`cano_weight_volume.npz`).  Loose-cloth weight transfer is an open init issue.
* The canonical front/back parameterization is our orthographic convention,
  documented and determined at runtime from the SMPL-X nose/landmark side.  It
  is not the author's OpenGL bitstream.  The maps are internally consistent (same
  convention for canonical and live maps) and can be read by the official
  dataset reader; the landmark check is pending actual execution.
* `expression` is preserved at its actual dimension.  If it is not 10, the author
  reader and vendored SMPL-X must be instantiated with
  `num_expression_coeffs=<actual>`; no silent truncation to the author default 10
  is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import cv2 as cv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dual_track'))
from image_io import read_image, write_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np(x: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _safe_under(path: Path, root: Path) -> Path:
    p = path.resolve()
    r = root.resolve()
    if not p.is_relative_to(r):
        raise ValueError(f"Refusing to write outside {r}: {p}")
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Motion contract
# --------------------------------------------------------------------------- #

@dataclass
class MotionContract:
    path: str
    T: int
    root: np.ndarray            # (T,3) axis-angle, SMPL-X global_orient
    body: np.ndarray            # (T,63) axis-angle
    transl_model: np.ndarray    # (T,3) SMPL-X model units (before world scale)
    betas: np.ndarray           # (1,10)
    jaw: np.ndarray             # (T,3)
    expr: np.ndarray            # (T,10)
    lh: np.ndarray              # (T,45)
    rh: np.ndarray              # (T,45)
    scale: float
    fps: Optional[float]        # actual motion fps if present; never synthesized
    expr_dim: int               # actual SMPL-X expression dimension
    source_frames: np.ndarray   # (T,) indices into assets/mjj/dynamic/<serial>/####.jpg
    key_report: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _find(raw: Dict[str, Any], *names: str) -> Optional[Any]:
    for n in names:
        if n in raw and raw[n] is not None:
            return raw[n]
    return None


def extract_motion(path: Path) -> MotionContract:
    """Read the project's motion.pt and normalize to the official SMPL-X contract.

    Supported layouts include the base `motion.pt` written by work/mjj/motion.py
    (`root_pose` (T,3), `body_pose` (T,21,3), `translation_world`, `scale`,
    `betas`, `frames`) and a `poses` (T,165) SMPL-X array.  Unknown shapes fail
    loudly rather than silently fabricating a pose.
    """
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Local project file with possibly custom classes; not a remote input.
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"motion must be a dict, got {type(raw)}")
    keys = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
            for k, v in raw.items() if not isinstance(v, (list, tuple)) or len(str(v)) < 200}
    report: List[Dict[str, Any]] = []
    warnings: List[str] = []

    scale = float(_np(raw.get("scale", 1.0)).reshape(-1)[0]) if "scale" in raw else 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid motion scale {scale}")

    poses = _find(raw, "poses", "smplx_poses", "full_pose")
    poses_a = None
    if poses is not None:
        poses_a = _np(poses)
        if poses_a.ndim != 2 or poses_a.shape[1] < 66:
            poses_a = None
    T = None
    for cand in ("root_pose", "body_pose", "translation_world", "transl", "poses", "frames"):
        if cand in raw and isinstance(raw[cand], torch.Tensor):
            T = int(raw[cand].shape[0])
            break
    if T is None:
        for cand in ("root_pose", "body_pose", "translation_world", "transl", "poses", "frames"):
            if cand in raw:
                arr = np.asarray(raw[cand])
                if arr.ndim >= 1:
                    T = int(arr.shape[0])
                    break
    if T is None:
        raise ValueError("cannot infer frame count T from motion.pt")
    if poses_a is not None and poses_a.shape[0] != T:
        raise ValueError(f"poses T={poses_a.shape[0]} != {T}")

    root = _find(raw, "root_pose", "global_orient", "root_orient")
    if root is None and poses_a is not None:
        root = poses_a[:, :3]
    if root is None:
        raise ValueError("motion has no root_pose/global_orient")
    root = _np(root).reshape(T, 3)

    body = _find(raw, "body_pose", "body_poses")
    if body is None and poses_a is not None:
        body = poses_a[:, 3:66]
    if body is None:
        raise ValueError("motion has no body_pose")
    body = _np(body)
    if body.ndim == 3:
        if body.shape[1] == 55:          # full 55-joint axis-angle with root
            body = body[:, 1:22]
        elif body.shape[1] == 21:
            pass
        else:
            raise ValueError(f"unexpected body_pose shape {body.shape}")
        body = body.reshape(T, -1)
    if body.ndim != 2 or body.shape[1] != 63:
        if body.ndim == 2 and body.shape[1] > 63:
            warnings.append(f"body_pose has {body.shape[1]} cols; using first 63")
            body = body[:, :63]
        else:
            raise ValueError(f"body_pose must become (T,63), got {body.shape}")
    if body.shape[0] != T:
        raise ValueError(f"body_pose T={body.shape[0]} != {T}")

    tw = _find(raw, "translation_world", "trans_world", "translation")
    if tw is None:
        tr = _find(raw, "transl", "trans")
        if tr is None:
            raise ValueError("motion has neither translation_world nor transl")
        tw = _np(tr) * scale
        warnings.append("used `transl` as SMPL-X model units and multiplied by scale; "
                        "verify this if the source actually stored world translation")
    else:
        tw = _np(tw)
    tw = tw.reshape(T, 3)
    transl_model = tw / scale

    betas = _find(raw, "betas", "shape", "shapes", "beta")
    if betas is None:
        betas = np.zeros((1, 10), np.float32)
        warnings.append("no betas in motion; using zeros (will not match subject shape)")
    betas = _np(betas)
    if betas.ndim == 1:
        betas = betas[None]
    if betas.shape[0] != 1:
        warnings.append(f"betas has {betas.shape[0]} rows; using row 0")
        betas = betas[:1]
    if betas.shape[1] > 10:
        warnings.append(f"betas has {betas.shape[1]} dims; author SMPL-X default uses 10, truncating")
        betas = betas[:, :10]
    elif betas.shape[1] < 10:
        betas = np.pad(betas, ((0, 0), (0, 10 - betas.shape[1])))

    jaw = _find(raw, "jaw_pose", "jaw")
    jaw_missing = jaw is None
    jaw = np.zeros((T, 3), np.float32) if jaw_missing else _np(jaw).reshape(T, 3)
    if jaw_missing:
        warnings.append("no jaw_pose; using zeros")

    expr = _find(raw, "expression", "expr", "expressions", "expression_params")
    if expr is None:
        expr = np.zeros((T, 10), np.float32)
        warnings.append("no expression in motion; writing zeros at 10-D")
    expr = _np(expr).reshape(T, -1)
    if expr.shape[1] == 0:
        expr = np.zeros((T, 10), np.float32)
    elif expr.shape[1] < 10:
        # Keep the actual coefficients and pad the unused tail; this does not
        # change the meaning of the present coefficients.
        expr = np.pad(expr, ((0, 0), (0, 10 - expr.shape[1])))
    expr_dim = int(expr.shape[1])
    if expr_dim > 10:
        warnings.append(
            f"expression has {expr_dim} coefficients; values are preserved verbatim. "
            f"The author reader must instantiate SMPL-X with num_expression_coeffs={expr_dim} "
            f"(minimal vendored-SMPLX patch; do not truncate to 10)."
        )

    lh = _find(raw, "left_hand_pose", "left_hands", "left_hand")
    rh = _find(raw, "right_hand_pose", "right_hands", "right_hand")
    hands = _find(raw, "hands")
    if hands is not None and lh is None and rh is None:
        hands = _np(hands)
        if hands.ndim == 3 and hands.shape[1] == 2:
            lh, rh = hands[:, 0], hands[:, 1]
        else:
            raise ValueError(f"cannot split hands shape {hands.shape}")
    lh = np.zeros((T, 45), np.float32) if lh is None else _np(lh).reshape(T, -1)
    rh = np.zeros((T, 45), np.float32) if rh is None else _np(rh).reshape(T, -1)
    for name, h in (("left_hand_pose", lh), ("right_hand_pose", rh)):
        if h.shape[1] != 45:
            raise ValueError(f"{name} must be 45-D axis-angle, got {h.shape}")
    if lh.shape[0] != T or rh.shape[0] != T:
        raise ValueError("hand pose length does not match T")
    if _find(raw, "left_hand_pose", "left_hands", "hands") is None:
        warnings.append("no left hand pose; using zeros (flat hand, not a grip prior)")

    sf = _find(raw, "source_frames", "source_frame", "capture_frames")
    if sf is None:
        sf = _find(raw, "frames")
        if sf is not None:
            warnings.append("no source_frames; used `frames` as source image indices")
    if sf is None:
        raise ValueError(
            "motion has neither source_frames nor frames; refusing to invent an "
            "identity mapping.  The source capture index is required for MASKS/images."
        )
    sf = _np(sf, np.int64).reshape(-1)
    if sf.shape[0] != T:
        # The historical 100-index capture has one duplicate 66/67.  If the
        # source array is longer, keep the first occurrence of each index.
        uniq = []
        seen = set()
        for v in sf.tolist():
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        if len(uniq) == T:
            warnings.append(f"source_frames length {sf.shape[0]} with duplicate(s); kept first occurrences")
            sf = np.asarray(uniq, np.int64)
        else:
            raise ValueError(f"source_frames length {sf.shape[0]} != T={T}")
    if len(set(sf.tolist())) != T:
        raise ValueError("source_frames has duplicate indices")
    if (sf < 0).any():
        raise ValueError("source_frames has negative indices")

    fps = None
    if "fps" in raw:
        fps = float(raw["fps"])
    elif "capture_fps" in raw:
        fps = float(raw["capture_fps"])
    else:
        warnings.append("motion has no fps/capture_fps; fps is recorded as unknown, not synthesized")
    if fps is not None and abs(float(fps) - 100.0 / 11.0) < 1e-4:
        warnings.append(
            "motion fps field is the legacy 100/11 value; MJJ001 is 99 real times at 9 fps. "
            "This field is metadata only and is not used to build or reorder frames."
        )

    for k in ("root_pose", "body_pose", "translation_world", "betas", "jaw_pose",
              "expression", "left_hand_pose", "right_hand_pose", "source_frames",
              "frames", "scale", "fps"):
        if k in raw:
            v = raw[k]
            report.append({"key": k,
                           "shape": list(v.shape) if isinstance(v, torch.Tensor) else None,
                           "dtype": str(v.dtype) if isinstance(v, torch.Tensor) else type(v).__name__})

    return MotionContract(
        path=str(path), T=T, root=root.astype(np.float32), body=body.astype(np.float32),
        transl_model=transl_model.astype(np.float32), betas=betas.astype(np.float32),
        jaw=jaw.astype(np.float32), expr=expr.astype(np.float32),
        lh=lh.astype(np.float32), rh=rh.astype(np.float32), scale=scale,
        fps=fps, expr_dim=expr_dim, source_frames=sf.astype(np.int64),
        key_report=report, warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# SMPL-X + template
# --------------------------------------------------------------------------- #

CANO_BODY_POSE = np.zeros(63, np.float32)
CANO_BODY_POSE[2] = math.radians(25.0)
CANO_BODY_POSE[5] = math.radians(-25.0)


def resolve_smplx_dir(smplx_dir: Optional[str]) -> Path:
    cands: List[Path] = []
    if smplx_dir:
        p = Path(smplx_dir)
        cands.append(p if p.is_dir() else p.parent)
    cands += [
        ROOT / "lhmpp/pretrained_models/human_model_files/smplx",
        ROOT / "assets/runs/lhmpp/runtime/pretrained_models/human_model_files/smplx",
        ROOT / "assets/runs/lhmpp/runtime/data/pretrained_models/human_model_files/smplx",
        ROOT / "assets/runs/lhmpp/pretrained_models/human_model_files/smplx",
        ROOT / "pretrained_models/human_model_files/smplx",
    ]
    for p in cands:
        if (p / "SMPLX_NEUTRAL.npz").exists():
            return p
    raise FileNotFoundError(
        "SMPLX_NEUTRAL.npz not found. Pass --smplx-dir pointing to the directory "
        "that contains SMPLX_NEUTRAL.npz (the author loads it via smplx.SMPLX)."
    )


@dataclass
class SmplxBundle:
    model: Any
    cano_verts: np.ndarray
    cano_faces: np.ndarray
    cano_joints: np.ndarray
    cano_A: np.ndarray
    lbs_weights: np.ndarray
    module_file: str


def import_author_smplx(author_root: Optional[str] = None):
    """Use the author vendored SMPLX package.

    The author's `smplx` exposes `SMPLX.forward(...).A`; a pip smplx without `.A`
    is not an acceptable substitute.  Fail explicitly rather than silently
    falling back to the installed package.
    """
    if not author_root:
        raise FileNotFoundError(
            "author_root is required for the author vendored smplx (.A); "
            "pass --author-root or place the shallow clone at ./author"
        )
    root = Path(author_root).resolve()
    if not (root / "smplx/body_models.py").exists():
        raise FileNotFoundError(
            f"author vendored smplx not found under {root / 'smplx'}; "
            "refusing to substitute a pip smplx that may lack .A"
        )
    sys.path.insert(0, str(root))
    import smplx
    mod_file = Path(getattr(smplx, "__file__", "")).resolve()
    if root not in mod_file.parents:
        raise RuntimeError(
            f"imported smplx from {mod_file}, not from author root {root}; "
            "another smplx was already imported. Run in a fresh process."
        )
    return smplx


def make_smplx_bundle(smplx_dir: Path, betas: np.ndarray, expr_dim: int = 10,
                      author_root: Optional[str] = None) -> SmplxBundle:
    smplx = import_author_smplx(author_root)
    model = smplx.SMPLX(
        model_path=str(smplx_dir), gender="neutral", use_pca=False,
        num_pca_comps=45, flat_hand_mean=True, batch_size=1,
        num_expression_coeffs=int(expr_dim),
    )
    b = torch.from_numpy(betas.astype(np.float32))
    body = torch.from_numpy(CANO_BODY_POSE[None])
    with torch.no_grad():
        out = model(betas=b, global_orient=torch.zeros(1, 3),
                    transl=torch.zeros(1, 3), body_pose=body)
    cano_verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    cano_faces = model.faces.astype(np.int64)
    cano_joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    cano_A = out.A[0].detach().cpu().numpy().astype(np.float32)
    lbs = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    return SmplxBundle(model, cano_verts, cano_faces, cano_joints, cano_A, lbs,
                       getattr(smplx, "__file__", "unknown"))


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for j in range(3):
        np.add.at(vn, faces[:, j], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.clip(n, 1e-12, None)


def _nearest_vertex_lbs(verts: np.ndarray, ref_verts: np.ndarray,
                        ref_lbs: np.ndarray, k: int = 4) -> Tuple[np.ndarray, str]:
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_verts)
        d, idx = tree.query(verts, k=min(k, len(ref_verts)))
        d = np.atleast_2d(d)
        idx = np.atleast_2d(idx)
        w = 1.0 / np.clip(d, 1e-8, None)
        w /= w.sum(1, keepdims=True)
        lbs = np.einsum("nk,nkj->nj", w, ref_lbs[idx])
        method = f"scipy_cKDTree_idw_k{k}"
    except Exception:
        # Fallback: nearest vertex, chunked.  Slow but dependency-free.
        lbs = np.empty((len(verts), ref_lbs.shape[1]), np.float32)
        for s in range(0, len(verts), 512):
            d = ((verts[s:s + 512, None, :] - ref_verts[None, :, :]) ** 2).sum(-1)
            lbs[s:s + 512] = ref_lbs[np.argmin(d, axis=1)]
        method = "nearest_vertex_chunked"
    lbs = np.clip(lbs, 0, None)
    lbs /= np.clip(lbs.sum(1, keepdims=True), 1e-8, None)
    return lbs.astype(np.float32), method


def load_template(bundle: SmplxBundle, template_ply: Optional[str],
                  template_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if not template_ply:
        return (bundle.cano_verts, bundle.cano_faces, bundle.lbs_weights,
                "smplx_cano_body_initial_not_clothed", "exact_smplx_topology")
    import trimesh
    mesh = trimesh.load(str(template_ply), process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(f"{template_ply} is not a mesh")
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.int64)
    if template_scale != 1.0:
        verts = verts * float(template_scale)
    if verts.shape == bundle.cano_verts.shape and np.array_equal(faces, bundle.cano_faces):
        return verts, faces, bundle.lbs_weights, "supplied_template_ply", "exact_smplx_topology"
    lbs, method = _nearest_vertex_lbs(verts, bundle.cano_verts, bundle.lbs_weights)
    return verts, faces, lbs, "supplied_template_ply", method


# --------------------------------------------------------------------------- #
# CPU orthographic rasterizer
# --------------------------------------------------------------------------- #

def _rasterize_view(verts: np.ndarray, faces: np.ndarray, lbs: np.ndarray,
                    res: int, center: np.ndarray, half: float,
                    depth_sign: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render a mirrored orthographic view.

    Both front and back use u increasing with +X (the official generator
    horizontally flips the back image).  `depth_sign=+1` sees +Z as nearer;
    `depth_sign=-1` sees -Z as nearer.
    """
    vn = _vertex_normals(verts, faces) * depth_sign
    attr = np.concatenate([verts, vn, lbs], axis=1).astype(np.float32)
    H = W = int(res)
    zbuf = np.full((H, W), -np.inf, np.float32)
    out = np.zeros((H, W, attr.shape[1]), np.float32)

    px = (verts[:, 0] - center[0]) / (2.0 * half) * W + W / 2.0 - 0.5
    py = -(verts[:, 1] - center[1]) / (2.0 * half) * H + H / 2.0 - 0.5
    pz = depth_sign * verts[:, 2]

    # face normals, oriented toward the viewing side
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    flip = (fn[:, 2] * depth_sign) < 0
    fn[flip] *= -1.0
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12, None)

    yy, xx = np.mgrid[0:H, 0:W]
    for f in range(len(faces)):
        i0, i1, i2 = faces[f]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        X = xx[miny:maxy + 1, minx:maxx + 1].astype(np.float32)
        Y = yy[miny:maxy + 1, minx:maxx + 1].astype(np.float32)
        l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
        l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        z = l0 * pz[i0] + l1 * pz[i1] + l2 * pz[i2]
        local = zbuf[miny:maxy + 1, minx:maxx + 1]
        upd = inside & (z > local)
        if not upd.any():
            continue
        local[upd] = z[upd]
        a = (l0[..., None] * attr[i0] + l1[..., None] * attr[i1] +
             l2[..., None] * attr[i2])
        tile = out[miny:maxy + 1, minx:maxx + 1]
        tile[upd] = a[upd]
        out[miny:maxy + 1, minx:maxx + 1] = tile

    mask = zbuf > -np.inf
    pos = out[..., :3]
    nml = out[..., 3:6]
    n = np.linalg.norm(nml, axis=-1, keepdims=True)
    nml = nml / np.clip(n, 1e-8, None)
    w = out[..., 6:]
    pos[~mask] = 0
    nml[~mask] = 0
    w[~mask] = 0
    return pos, nml, w, mask


def _front_sign(bundle: SmplxBundle) -> float:
    try:
        import smplx
        from smplx.vertex_ids import vertex_ids
        nose = int(vertex_ids["smplx"]["nose"])
        z = float(bundle.cano_verts[nose, 2] - bundle.cano_verts[:, 2].mean())
    except Exception:
        z = float(bundle.cano_joints[15, 2] - bundle.cano_joints[0, 2])
    sign = 1.0 if z >= 0 else -1.0
    return sign


def build_canonical_maps(verts: np.ndarray, faces: np.ndarray, lbs: np.ndarray,
                         res: int = 1024, front_sign: float = 1.0) -> Dict[str, Any]:
    center = 0.5 * (verts.min(0) + verts.max(0))
    extent = verts.max(0) - verts.min(0)
    half = 0.5 * float(max(extent[0], extent[1])) * 1.02
    sign = float(front_sign)
    # Determine front from the mesh itself: nose/head is on the front side.
    # `build_template_and_maps` passes the real sign; this fallback is only for
    # direct calls.
    front_pos, front_nml, front_w, front_m = _rasterize_view(
        verts, faces, lbs, res, center, half, depth_sign=sign)
    back_pos, back_nml, back_w, back_m = _rasterize_view(
        verts, faces, lbs, res, center, half, depth_sign=-sign)
    cano_pos = np.concatenate([front_pos, back_pos], axis=1)
    cano_nml = np.concatenate([front_nml, back_nml], axis=1)
    mask = np.concatenate([front_m, back_m], axis=1)
    # Match the official reader: the canonical Gaussian set is selected by
    # norm(position)>0, not by the z-buffer alone.
    mask = mask & (np.linalg.norm(cano_pos, axis=-1) > 0)
    pts = cano_pos[mask].astype(np.float32)
    both_w = np.concatenate([front_w, back_w], axis=1)
    pt_lbs = both_w[mask].astype(np.float32)
    return {"pos": cano_pos, "nml": cano_nml, "mask": mask,
            "pts": pts, "lbs": pt_lbs, "center": center, "half": half,
            "front_sign": sign}


# --------------------------------------------------------------------------- #
# EXR / PLY writers
# --------------------------------------------------------------------------- #

def write_exr(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"EXR expects HxWx3, got {arr.shape}")
    if not cv.haveImageWriter(str(path)):
        raise RuntimeError(
            f"OpenCV cannot write EXR ({path}). Rebuild opencv with OpenEXR, or set "
            "OPENCV_IO_ENABLE_OPENEXR=1 before importing cv2. The official reader "
            "requires .exr, so do not silently fall back to .npy."
        )
    ok = write_image(str(path), arr)
    if not ok or not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for t in faces:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


# --------------------------------------------------------------------------- #
# Skinning + live maps
# --------------------------------------------------------------------------- #

def lbs_points(pts: np.ndarray, lbs: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Official algebra: pt_mats = einsum('nj,jxy->nxy', lbs, mats)."""
    t = np.einsum("nj,jxy->nxy", lbs.astype(np.float32), mats.astype(np.float32))
    return np.einsum("nxy,ny->nx", t[..., :3, :3], pts) + t[..., :3, 3]


def smplx_live(bundle: SmplxBundle, mc: MotionContract) -> Dict[str, np.ndarray]:
    T = mc.T
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    verts, joints, mats = [], [], []
    # The author's vendored SMPLX uses self.batch_size when repeating landmark
    # barycentric coordinates, so batch>T risks an inconsistent landmark tensor.
    # gen_data/gen_pos_maps.py itself loops frames; mirror that per-frame call.
    with torch.no_grad():
        for i in range(T):
            out = bundle.model(
                betas=t(mc.betas),
                global_orient=torch.zeros(1, 3),
                transl=torch.zeros(1, 3),
                body_pose=t(mc.body[i:i + 1]),
                jaw_pose=t(mc.jaw[i:i + 1]),
                expression=t(mc.expr[i:i + 1]),
                # Official gen_pos_maps.py omits hand pose from live_smpl_woRoot.
                # Hand pose is applied later by AvatarNet.transform_cano2live.
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
            )
            verts.append(out.vertices[0].detach().cpu().numpy())
            joints.append(out.joints[0].detach().cpu().numpy())
            mats.append(out.A[0].detach().cpu().numpy())
    return {
        "vertices": np.stack(verts).astype(np.float32),
        "joints": np.stack(joints).astype(np.float32),
        "A": np.stack(mats).astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def _project_one(K: np.ndarray, W: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pc = X @ W[:3, :3].T + W[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / np.clip(uv[:, 2:], 1e-8, None), pc[:, 2]


def _load_camera_json(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[List[int]]]:
    cal = json.loads((data_dir / "calibration.json").read_text(encoding="utf-8"))
    names = sorted(cal.keys())
    W = np.stack([np.array(cal[n]["R"], np.float32).reshape(3, 3) for n in names])
    T = np.stack([np.array(cal[n]["T"], np.float32).reshape(3) for n in names])
    K = np.stack([np.array(cal[n]["K"], np.float32).reshape(3, 3) for n in names])
    W4 = np.tile(np.eye(4, dtype=np.float32), (len(names), 1, 1))
    W4[:, :3, :3] = W
    W4[:, :3, 3] = T
    sizes = [list(map(int, cal[n]["imgSize"])) for n in names]
    return K, W4, names, sizes


def _qa_projection(data_dir: Path, mc: MotionContract, live: Dict[str, np.ndarray],
                   frames: Sequence[int], scale: float) -> List[str]:
    K, W, names, sizes = _load_camera_json(data_dir)
    out_dir = data_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    for t in frames:
        X = live["joints"][t] + mc.transl_model[t]
        img_paths = []
        for ci, name in enumerate(names):
            img = read_image(str(data_dir / "images" / name / f"{t:08d}.jpg"), cv.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            # draw at half resolution; K scales by 0.5 exactly
            K2 = K[ci].copy()
            K2[:2] *= 0.5
            uv, z = _project_one(K2, W[ci], X)
            small = cv.resize(img, (w // 2, h // 2), interpolation=cv.INTER_AREA)
            for j in range(22):
                if z[j] <= 1e-6 or not np.isfinite(uv[j]).all():
                    continue
                u, v = int(round(uv[j, 0])), int(round(uv[j, 1]))
                if 0 <= u < small.shape[1] and 0 <= v < small.shape[0]:
                    cv.circle(small, (u, v), 3, (0, 255, 0), -1, cv.LINE_AA)
            mask = read_image(str(data_dir / "masks" / name / f"{t:08d}.jpg"), cv.IMREAD_GRAYSCALE)
            if mask is not None:
                ms = cv.resize(mask, (w // 2, h // 2), interpolation=cv.INTER_NEAREST)
                ms = np.where(ms > 128, 255, 0).astype(np.uint8)
                cnt, _ = cv.findContours(ms, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(small, cnt, -1, (0, 0, 255), 1)
            p = out_dir / f"projection_{t:03d}_{name}.jpg"
            write_image(str(p), small)
            img_paths.append(small)
            made.append(str(p.relative_to(data_dir)))
        if img_paths:
            cols = 4
            rows = (len(img_paths) + cols - 1) // cols
            th, tw = img_paths[0].shape[:2]
            sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
            for i, im in enumerate(img_paths):
                r, c = divmod(i, cols)
                sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
            write_image(str(out_dir / f"contact_{t:03d}.jpg"), sheet)
            made.append(str((out_dir / f"contact_{t:03d}.jpg").relative_to(data_dir)))
    return made


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_template_and_maps(
    data_dir: Path,
    motion: Path,
    smplx_dir: Path,
    template_ply: Optional[str] = None,
    template_scale: float = 1.0,
    map_size: int = 1024,
    qa: bool = True,
    mc: Optional[MotionContract] = None,
    author_root: Optional[str] = None,
    template_kind_note: str = "",
) -> Dict[str, Any]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    mc = mc or extract_motion(motion)
    bundle = make_smplx_bundle(smplx_dir, mc.betas, mc.expr_dim, author_root)
    verts, faces, lbs, kind, lbs_method = load_template(bundle, template_ply, template_scale)
    if template_kind_note:
        kind = kind + ":" + template_kind_note

    sign = _front_sign(bundle) if verts.shape == bundle.cano_verts.shape else 1.0
    maps = build_canonical_maps(verts, faces, lbs, res=map_size, front_sign=sign)
    if maps["mask"].sum() == 0:
        raise RuntimeError("canonical rasterization produced an empty mask; check template bounds/front sign")
    if not np.isfinite(maps["pts"]).all():
        raise RuntimeError("canonical position map contains non-finite values")

    pos_dir = data_dir / "smpl_pos_map"
    pos_dir.mkdir(parents=True, exist_ok=True)
    write_exr(pos_dir / "cano_smpl_pos_map.exr", maps["pos"])
    write_exr(pos_dir / "cano_smpl_nml_map.exr", maps["nml"])
    np.save(pos_dir / "init_pts_lbs.npy", maps["lbs"].astype(np.float32))
    np.save(pos_dir / "cano_pts.npy", maps["pts"].astype(np.float32))
    np.save(pos_dir / "cano_mask.npy", maps["mask"].astype(bool))
    write_ply(data_dir / "template.ply", verts, faces)

    live = smplx_live(bundle, mc)
    cano_A = bundle.cano_A[None].astype(np.float32)
    M = live["A"] @ np.linalg.inv(cano_A)      # (T,55,4,4), excludes root/hands
    H, W2 = maps["mask"].shape
    mask = maps["mask"]
    pts = maps["pts"]
    lbs_pts = maps["lbs"]

    for t in range(mc.T):
        posed = lbs_points(pts, lbs_pts, M[t])
        lm = np.zeros((H, W2, 3), np.float32)
        lm[mask] = posed
        small = lm[::2, ::2]                   # torch nearest, scale=0.5
        write_exr(pos_dir / f"{t:08d}.exr", small)

    # Round-trip on cano -> live -> cano for three real times.
    times = sorted(set([0, mc.T // 2, mc.T - 1]))
    roundtrip = []
    for t in times:
        posed = lbs_points(pts, lbs_pts, M[t])
        back = lbs_points(posed, lbs_pts, np.linalg.inv(M[t]))
        e = np.linalg.norm(back - pts, axis=1)
        roundtrip.append({"t": int(t), "max_model_units": float(e.max()),
                          "mean_model_units": float(e.mean()),
                          "p99_model_units": float(np.percentile(e, 99))})
    # For SMPL-X topology, LBS(template) must equal live SMPL-X vertices.
    live_match = None
    if verts.shape == bundle.cano_verts.shape and np.array_equal(faces, bundle.cano_faces):
        errs = []
        for t in times:
            p = lbs_points(bundle.cano_verts, bundle.lbs_weights, M[t])
            errs.append(float(np.abs(p - live["vertices"][t]).max()))
        live_match = {"max_model_units": max(errs), "per_time": errs}

    # Camera projection invariance: model frame = world / scale, R unchanged,
    # T_model = T_world / scale; K unchanged.
    K, Wm, names, sizes = _load_camera_json(data_dir)
    Ww = Wm.copy()
    Ww[:, :3, 3] *= mc.scale
    proj = []
    for t in times:
        X = live["joints"][t] + mc.transl_model[t]
        uv_m, z_m = _project_one(K[0], Wm[0], X)
        uv_w, z_w = _project_one(K[0], Ww[0], X * mc.scale)
        ok = (z_m > 1e-6) & (z_w > 1e-6)
        d = np.linalg.norm(uv_m[ok] - uv_w[ok], axis=1) if ok.any() else np.array([np.nan])
        proj.append({"t": int(t), "camera": names[0], "max_px": float(np.nanmax(d)),
                     "median_px": float(np.nanmedian(d))})

    qa_files = _qa_projection(data_dir, mc, live, times, mc.scale) if qa else []

    meta = {
        "schema": "animatable_gaussians_template_bridge/v1",
        "motion": str(motion),
        "motion_sha256": sha256_file(motion) if Path(motion).exists() else None,
        "template_kind": kind,
        "template_is_clothed": False if not template_ply else None,
        "template_vertices": int(len(verts)),
        "template_faces": int(len(faces)),
        "lbs_weight_source": lbs_method,
        "smplx_module_file": bundle.module_file,
        "expression_dim": int(mc.expr_dim),
        "canonical_body_pose_rad": {"body_pose[2]": float(CANO_BODY_POSE[2]),
                                    "body_pose[5]": float(CANO_BODY_POSE[5]),
                                    "source": "config.py cano_smpl_pose[8]/[11]; cano_smpl_body_pose=cano_smpl_pose[6:69], hence body indices 2/5"},
        "map_convention": {
            "resolution": [int(map_size), int(2 * map_size)],
            "front_back": "left=front, right=back; both u increases with canonical +X; back is mirrored like official cv.flip(back,1)",
            "front_sign_z": float(maps["front_sign"]),
            "canonical_units": "SMPL-X model units (official); world = scale * model",
            "live_map_resolution": [int(map_size // 2), int(map_size)],
            "live_map_algebra": "M = live.A @ inv(cano.A), body+jaw+expression, no global_orient/transl/hand pose",
            "normative": False,
            "differences": "source/MAP_DIFFERENCES.md",
        },
        "author_normative_generator": {
            "entry": "author_entry.py --module gen_data.gen_pos_maps --config <author avatar.yaml>",
            "requires": ["author vendored smplx with .A", "pytorch3d", "igl", "trimesh",
                         "OpenGL/pyrender renderer", "gaussians CUDA rasterizer",
                         "network/styleunet CUDA extension", "cano_weight_volume.npz",
                         "PointInterpolant.exe if template.ply exists"],
            "warning": "Use the author generator for final maps; this CPU bridge is initialization/diagnostic only.",
        },
        "files": {
            "template_ply": "template.ply",
            "cano_pos_exr": "smpl_pos_map/cano_smpl_pos_map.exr",
            "cano_nml_exr": "smpl_pos_map/cano_smpl_nml_map.exr",
            "init_pts_lbs": "smpl_pos_map/init_pts_lbs.npy",
            "cano_pts": "smpl_pos_map/cano_pts.npy",
            "cano_mask": "smpl_pos_map/cano_mask.npy",
            "live_pos_maps": "smpl_pos_map/%08d.exr (0..T-1)",
        },
        "checks": {
            "canonical_mask_pixels": int(mask.sum()),
            "canonical_pts_finite": bool(np.isfinite(pts).all()),
            "canonical_lbs_row_sum_min": float(lbs_pts.sum(1).min()),
            "canonical_lbs_row_sum_max": float(lbs_pts.sum(1).max()),
            "roundtrip": roundtrip,
            "smplx_lbs_vs_live": live_match,
            "projection_model_vs_world": proj,
        },
        "warnings": list(mc.warnings) + [
            "template.ply without --template-ply is the fitted SMPL-X body under the author canonical pose; it is NOT a reconstructed clothing template.",
            "cano_weight_volume.npz is not built; official gen_data/gen_pos_maps.py would request it when template.ply exists. These maps are generated by this CPU bridge instead.",
            "For a custom template, LBS uses nearest-body IDW, not Poisson volumetric weights; loose clothing needs optimization/verification.",
            "Masks are the existing MJJ dynamic SAM2 masks at work/mjj/reconstruction/diagnosis/surface/masks/<serial>/<source_frame:04d>.png; they are detector masks, not exact anatomy.",
        ],
        "qa_files": qa_files,
    }
    write_json(data_dir / "bridge_report.json", meta)
    write_json(data_dir / "template_meta.json", {
        "template_kind": kind, "is_clothed_template": False if not template_ply else None,
        "note": "initial template only; clothing surface must be optimized from captured data",
        "source_motion": str(motion), "source_template_ply": template_ply,
        "vertices": int(len(verts)), "faces": int(len(faces)),
    })
    return meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Build AnimatableGaussians canonical/live position maps on CPU.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--smplx-dir", default=None)
    ap.add_argument("--template-ply", default=None)
    ap.add_argument("--template-scale", type=float, default=1.0)
    ap.add_argument("--map-size", type=int, default=1024)
    ap.add_argument("--no-qa", action="store_true")
    ap.add_argument("--author-root", default=str(HERE / "author"))
    a = ap.parse_args()
    report = build_template_and_maps(
        Path(a.data_dir), Path(a.motion), resolve_smplx_dir(a.smplx_dir),
        template_ply=a.template_ply, template_scale=a.template_scale,
        map_size=a.map_size, qa=not a.no_qa,
        author_root=a.author_root if Path(a.author_root).exists() else None,
    )
    print(json.dumps({"template_kind": report["template_kind"],
                      "checks": report["checks"]}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
