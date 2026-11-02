#!/usr/bin/env python
"""Bind the existing MJJ001 multiview assets into the GaussianWardrobe layout.

No RGB/mask/model is copied.  Images/masks/SMPL params and explicitly supplied
labels/body masks/templates are NTFS hardlinked; the original data directory is
only read.  Missing semantic/body labels, templates, layer position maps and
weight volumes are reported, never synthesized.

Run from the wardrobe directory (or pass absolute paths):
  python prepare_inputs.py --source ../data/mjj001_animatable \
      --out data/mjj001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
CLOTH = HERE.parent
DEFAULT_SOURCE = CLOTH / "data/mjj001_animatable"
DEFAULT_OUT = HERE / "data/mjj001"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hardlink(src: Path, dst: Path, ident: List[Dict[str, Any]]) -> None:
    """Hardlink only; never silently copy. Records inode identity."""
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            ident.append({"src": str(src), "dst": str(dst), "already_linked": True})
            return
        raise FileExistsError(f"refusing to replace existing path: {dst}")
    os.link(src, dst)
    if not os.path.samefile(src, dst):
        raise RuntimeError(f"hardlink identity check failed: {src} -> {dst}")
    st = dst.stat()
    ident.append({
        "src": str(src), "dst": str(dst), "size": st.st_size,
        "src_sha256": sha256_file(src), "dst_sha256": sha256_file(dst),
        "same_inode": True,
    })


def find_file(root: Path, kind: str, cam: str, pose_idx: int, source_frame: int,
              exts: Tuple[str, ...]) -> Optional[Path]:
    # Prepared MJJ images are author-index named; retain source-frame fallback.
    for stem in (pose_idx, source_frame):
        for ext in exts:
            p = root / kind / cam / f"{stem:08d}{ext}"
            if p.is_file():
                return p
    return None


def load_meta(source: Path) -> Dict[str, Any]:
    p = source / "dataset_meta.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    meta = json.loads(p.read_text(encoding="utf-8"))
    t = meta.get("time", {})
    frames = t.get("source_frames")
    if not frames or len(frames) != 99 or len(set(int(x) for x in frames)) != 99:
        raise ValueError("dataset_meta must provide 99 unique actual source frames")
    if int(t.get("T", -1)) != 99:
        raise ValueError("dataset_meta T must be 99")
    if float(t.get("fps", -1.0)) != 9.0:
        raise ValueError("dataset_meta fps must be 9.0")
    cams = meta.get("cameras")
    if not cams or len(cams) != 8:
        raise ValueError("dataset_meta must provide 8 cameras")
    return meta


def projection_invariance(calib: Dict[str, Any], cameras: Dict[str, Any],
                          transl: np.ndarray) -> Dict[str, Any]:
    """Float64 serialization invariance check, no image interpolation."""
    probes = []
    center = transl.reshape(-1, 3).mean(0) if transl.size else np.zeros(3)
    for dx, dy, dz in ((0, 0, 0), (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25),
                       (-0.25, 0.25, -0.25), (0.5, -0.5, 0.25)):
        probes.append(center + np.array([dx, dy, dz], np.float64))
    probes = np.stack(probes, 0)
    out = {}
    for cam, src in calib.items():
        R = np.asarray(src["R"], np.float64)
        T = np.asarray(src["T"], np.float64)
        K = np.asarray(src["K"], np.float64)
        R2 = np.asarray(cameras[cam]["extrinsics"], np.float64)[:, :3]
        T2 = np.asarray(cameras[cam]["extrinsics"], np.float64)[:, 3]
        K2 = np.asarray(cameras[cam]["intrinsics"], np.float64)
        def proj(R_, T_, K_):
            cam_xyz = probes @ R_.T + T_
            hom = cam_xyz @ K_.T
            uv = hom[:, :2] / hom[:, 2:3]
            return uv, cam_xyz[:, 2]
        uv1, z1 = proj(R, T, K)
        uv2, z2 = proj(R2, T2, K2)
        out[cam] = {
            "n": int(len(probes)),
            "max_px": float(np.max(np.abs(uv1 - uv2))),
            "max_z": float(np.max(np.abs(z1 - z2))),
            "finite": bool(np.isfinite(uv1).all() and np.isfinite(uv2).all()),
            "positive_depth": bool((z1 > 0).all()),
        }
    return out


def link_optional_tree(src_root: Optional[Path], dst_root: Path, kind: str,
                       pose_to_source: Dict[int, int], ident: List[Dict[str, Any]],
                       missing: List[str]) -> bool:
    if src_root is None:
        missing.append(f"{kind}/ (no --{kind.replace('_', '-')} supplied)")
        return False
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    found = 0
    for pose_idx in sorted(pose_to_source):
        source_frame = pose_to_source[pose_idx]
        src = find_file(src_root, "", "cam00", pose_idx, source_frame, (".png",))
        # The generic helper above is camera-specific; resolve explicitly below.
        del src
        for cam in [f"cam{i:02d}" for i in range(8)]:
            p = find_file(src_root, "", cam, pose_idx, source_frame, (".png",))
            if p is None:
                missing.append(f"{kind}/{cam}/{pose_idx:08d}.png")
                continue
            hardlink(p, dst_root / cam / f"{pose_idx:08d}.png", ident)
            found += 1
    return found > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--body-masks", default=None)
    ap.add_argument("--body-template", default=None)
    ap.add_argument("--cloth-template", default=None)
    a = ap.parse_args()

    source = Path(a.source).resolve()
    out = Path(a.out).resolve()
    labels = Path(a.labels).resolve() if a.labels else None
    body_masks = Path(a.body_masks).resolve() if a.body_masks else None
    body_tpl = Path(a.body_template).resolve() if a.body_template else None
    cloth_tpl = Path(a.cloth_template).resolve() if a.cloth_template else None

    if not source.is_dir():
        raise FileNotFoundError(source)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is not empty; refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)

    meta = load_meta(source)
    src_frames = [int(x) for x in meta["time"]["source_frames"]]
    pose_to_source = {i: f for i, f in enumerate(src_frames)}
    cams = [c["cam"] for c in meta["cameras"]]
    if cams != [f"cam{i:02d}" for i in range(8)]:
        raise ValueError("unexpected native camera order in metadata")

    calib_all = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    calib = {c: calib_all[c] for c in cams}
    cameras: Dict[str, Any] = {}
    for c in cams:
        R = np.asarray(calib[c]["R"], np.float64)
        T = np.asarray(calib[c]["T"], np.float64)
        K = np.asarray(calib[c]["K"], np.float64)
        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R
        E[:3, 3] = T
        cameras[c] = {
            "extrinsics": E[:3, :].tolist(),
            "intrinsics": K.tolist(),
            "imgSize": [int(calib[c]["imgSize"][0]), int(calib[c]["imgSize"][1])],
            "serial": next(x["serial"] for x in meta["cameras"] if x["cam"] == c),
            "source_camera": c,
        }

    sp_path = source / "smpl_params.npz"
    if not sp_path.is_file():
        raise FileNotFoundError(sp_path)
    with np.load(sp_path, allow_pickle=True) as sp:
        if sp["body_pose"].shape[0] != 99 or sp["betas"].shape != (1, 10):
            raise ValueError("expected 99 times, betas (1,10)")
        if sp["expression"].shape[1] != 100:
            raise ValueError("expected 100-D expression")
        transl = np.asarray(sp["transl"], np.float64)
        body_pose = np.asarray(sp["body_pose"], np.float64)

    with (out / "cameras.pkl").open("wb") as f:
        pickle.dump(cameras, f, protocol=pickle.HIGHEST_PROTOCOL)
    ident: List[Dict[str, Any]] = []
    hardlink(sp_path, out / "smpl_params.npz", ident)

    missing: List[str] = []
    for pose_idx, source_frame in pose_to_source.items():
        for cam in cams:
            for kind, exts in (("images", (".jpg", ".png")), ("masks", (".jpg", ".png"))):
                src = find_file(source, kind, cam, pose_idx, source_frame, exts)
                if src is None:
                    missing.append(f"{kind}/{cam}/{pose_idx:08d}")
                    continue
                hardlink(src, out / kind / cam / f"{pose_idx:08d}{src.suffix}", ident)

    labels_ok = link_optional_tree(labels, out / "labels_color", "labels_color",
                                   pose_to_source, ident, missing)
    body_masks_ok = link_optional_tree(body_masks, out / "body_masks", "body_masks",
                                       pose_to_source, ident, missing)

    tpl_report: Dict[str, Any] = {}
    for key, src, dst_name in (
        ("body_template", body_tpl, "template_body.ply"),
        ("cloth_template", cloth_tpl, "template_cloth_offset.ply"),
    ):
        if src is not None:
            if not src.is_file():
                raise FileNotFoundError(src)
            hardlink(src, out / dst_name, ident)
            tpl_report[key] = str(src)
        else:
            missing.append(dst_name)
            tpl_report[key] = None

    required_missing = [
        "labels_color/ (independent semantic body/cloth labels; --labels)",
        "body_masks/ (independent rendered/segmented body masks; --body-masks)",
        "template_body.ply (real body template; --body-template)",
        "template_cloth_offset.ply (real cloth template; --cloth-template)",
        "template_body_offset.ply",
        "template_body_offset.pkl",
        "template_cloth_offset.pkl",
        "cano_weight_volume.npz",
        "cano_weight_volume_shape_zero.npz",
        "smpl_pos_map_offset_body/",
        "smpl_pos_map_offset_cloth/",
    ]
    # Do not create these; report exactly which specific assets are absent.
    for rel in required_missing:
        p = out / rel
        if not p.exists():
            missing.append(rel)

    projection = projection_invariance(calib, cameras, transl)
    reader_result: Dict[str, Any] = {"attempted": bool((out / "cameras.pkl").is_file()
                                                      and (out / "smpl_params.npz").is_file())}
    if reader_result["attempted"]:
        try:
            sys.path.insert(0, str(HERE))
            from integration import reader_check
            rc = reader_check(HERE / "configs/mjj001.yaml", data_dir=out)
            reader_result.update({
                "status": "ok" if rc == 0 else "error",
                "report": str(out / "preflight_reader.json"),
            })
        except Exception as exc:
            reader_result.update({"status": "unavailable",
                                  "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "schema": "gaussian_wardrobe_stage1_binding/v1",
        "generated_at_utc": utc_now(),
        "source": str(source),
        "out": str(out),
        "source_dataset_meta_sha256": sha256_file(source / "dataset_meta.json"),
        "source_calibration_sha256": sha256_file(source / "calibration.json"),
        "source_smpl_params_sha256": sha256_file(sp_path),
        "binding": "NTFS hardlink; no RGB/mask/model copy",
        "source_frames": src_frames,
        "pose_to_source_frame": {str(k): int(v) for k, v in pose_to_source.items()},
        "T": len(src_frames),
        "fps": 9.0,
        "native_image_size": [2048, 1536],
        "camera_order": cams,
        "camera_serials": {c: cameras[c]["serial"] for c in cams},
        "world_to_model_scale": float(meta["units"]["motion_world_scale"]),
        "no_resampling": True,
        "no_rectification": True,
        "no_person_recenter": True,
        "smpl_expression_dim": 100,
        "labels_linked": labels_ok,
        "body_masks_linked": body_masks_ok,
        "templates_linked": tpl_report,
        "projection_invariance_float64": projection,
        "projection_max_px": max(v["max_px"] for v in projection.values()),
        "reader_check": reader_result,
        "hardlink_identity": ident,
        "hardlink_count": len(ident),
        "missing": sorted(set(missing)),
        "available": {
            "images": True, "masks": True, "smpl_params": True,
            "cameras": True, "labels_color": labels_ok, "body_masks": body_masks_ok,
        },
        "semantics_note": "foreground masks are not cloth labels; no all-foreground cloth or bare-SMPL cloth proxy was written",
    }
    (out / "binding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
    source_frames = {
        "T": len(src_frames), "fps": 9.0, "source_frames": src_frames,
        "pose_to_source_frame": {str(k): int(v) for k, v in pose_to_source.items()},
        "no_resampling": True,
    }
    (out / "source_frames.json").write_text(json.dumps(source_frames, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({
        "out": str(out), "hardlinked": len(ident), "missing_count": len(manifest["missing"]),
        "projection_max_px": manifest["projection_max_px"],
        "labels": labels_ok, "body_masks": body_masks_ok,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
