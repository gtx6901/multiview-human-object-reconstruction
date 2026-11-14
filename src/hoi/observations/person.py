"""Build per-identity observations in a shared scene coordinate system."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

RAW_SPACE = "raw_distorted_pixels"
UND_SPACE = "undistorted_pixels"


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def undistort_points(cam: Dict[str, Any], pts: np.ndarray) -> np.ndarray:
    import cv2

    K = np.asarray(cam["K_native"], np.float64)
    dist = np.asarray(cam["distortion_opencv5"], np.float64)
    K_und = np.asarray(cam["K_undistorted"], np.float64)
    q = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if dist.size:
        out = cv2.undistortPoints(q, K, dist, P=K_und)
    else:
        out = cv2.undistortPoints(q, K, np.zeros(5), P=K_und)
    return out.reshape(-1, 2)


def load_gauge(manifest_path: Path, scene: Dict[str, Any]) -> Dict[str, Any]:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    wn = man.get("world_normalization")
    if not wn:
        raise RuntimeError(f"{manifest_path} has no world_normalization; shared gauge is undefined")
    center = np.asarray(wn["center_raw"], np.float64)
    height = float(wn["body_height_raw_units"])
    if (
        center.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(height)
        or height <= 0
    ):
        raise RuntimeError("invalid shared world_normalization")
    selected = (scene.get("calibration_selection") or {}).get("selected") or {}
    scene_cal_sha = selected.get("sha256")
    manifest_cal_sha = man.get("calibration_sha256")
    if scene_cal_sha and manifest_cal_sha and scene_cal_sha != manifest_cal_sha:
        raise RuntimeError(
            "cannot apply shared normalization: scene calibration SHA "
            f"{scene_cal_sha} != manifest calibration SHA {manifest_cal_sha}"
        )
    return {
        "source": str(manifest_path),
        "sequence": man.get("sequence"),
        "center_raw": center.tolist(),
        "body_height_raw_units": height,
        "transform": "X_unit=(X_raw-center_raw)/body_height_raw_units; R unchanged; t_unit=(R@center_raw+t_raw)/body_height_raw_units",
        "z_up_or_floor_assumed": False,
        "calibration_sha256": scene_cal_sha or manifest_cal_sha,
        "calibration_sha_match": bool(
            scene_cal_sha and manifest_cal_sha and scene_cal_sha == manifest_cal_sha
        ),
        "note": "All identities share one world transform.",
    }


def world_to_unit(W: np.ndarray, center: np.ndarray, height: float) -> np.ndarray:
    W = np.asarray(W, np.float64)
    R, t = W[:3, :3], W[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = (R @ center + t) / height
    return out


def load_instance_package(index_path: Path, scene: Dict[str, Any], identity: str) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    npz_path = Path(index.get("observations_npz") or (index_path.parent / "observations.npz"))
    with np.load(npz_path, allow_pickle=False) as z:
        obs = {k: z[k] for k in z.files}
    ids = [str(x) for x in obs["identities"].tolist()]
    if identity not in ids:
        raise ValueError(f"identity {identity!r} not in {ids}")
    k = ids.index(identity)
    coord = str(index.get("coordinate_space") or RAW_SPACE)
    cams = index["cameras"]
    frames = [int(x) for x in obs["frames"].tolist()]
    return {
        "index": index,
        "obs": obs,
        "identity_index": k,
        "ids": ids,
        "coord": coord,
        "cams": cams,
        "frames": frames,
    }


def build(args: argparse.Namespace) -> int:
    scene_path = Path(args.scene).resolve()
    index_path = Path(args.instances).resolve()
    if index_path.is_dir():
        index_path = index_path / "index.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    pkg = load_instance_package(index_path, scene, args.identity)
    obs, ids, k, coord, cams, frames = (
        pkg["obs"],
        pkg["ids"],
        pkg["identity_index"],
        pkg["coord"],
        pkg["cams"],
        pkg["frames"],
    )
    if coord not in (RAW_SPACE, UND_SPACE):
        raise ValueError(f"unsupported observation coordinate space: {coord}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gauge = load_gauge(Path(args.manifest).resolve(), scene)
    center = np.asarray(gauge["center_raw"], np.float64)
    height = float(gauge["body_height_raw_units"])
    gauge_path = out.parent / "gauge.json"
    if gauge_path.is_file():
        old = json.loads(gauge_path.read_text(encoding="utf-8"))
        if not (
            np.allclose(old.get("center_raw"), gauge["center_raw"])
            and abs(float(old.get("body_height_raw_units")) - height) < 1e-12
        ):
            raise RuntimeError(
                f"{gauge_path} already contains a different gauge; refusing per-person scaling"
            )
    atomic_write_json(gauge_path, gauge)

    T, V, J = len(frames), len(cams), 133
    views = []
    for cam in cams:
        W_unit = (
            np.asarray(cam["W"], np.float64)
            if args.camera_space == "normalized"
            else world_to_unit(np.asarray(cam["W"], np.float64), center, height)
        )
        views.append(
            {
                "serial": str(cam["serial"]),
                "K_undistorted_native": np.asarray(cam["K_undistorted"]).tolist(),
                "W": W_unit.tolist(),
                "width": int(cam["width"]),
                "height": int(cam["height"]),
                "source": cam.get("path"),
            }
        )
    manifest = {
        "sequence": scene.get("session", {}).get("name"),
        "identity": args.identity,
        "views": views,
        "coordinate_space": UND_SPACE,
        "world_normalization": gauge,
        "scope": (
            "Per-person adapter package for shared triangulation and motion fitting; "
            "shared scene world gauge."
        ),
    }
    atomic_write_json(out / "manifest.json", manifest)

    key_dir = out / "keypoints"
    key_dir.mkdir(parents=True, exist_ok=True)
    for vi, cam in enumerate(cams):
        source_xy = np.asarray(obs["kp"][vi, :, k], np.float64)  # T,133,2
        sc = np.asarray(obs["kp_score"][vi, :, k], np.float64)  # T,133
        if coord == RAW_SPACE:
            xy = (
                np.stack([undistort_points(cam, x) for x in source_xy])
                if T
                else np.zeros((0, J, 2))
            )
        else:
            xy = source_xy.copy()
        in_image = (
            np.asarray(obs["kp_present"][vi, :, k], bool)[:, None]
            & np.isfinite(xy).all(axis=2)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] < int(cam["width"]))
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < int(cam["height"]))
        )
        atomic_savez(
            key_dir / f"{cam['serial']}.npz",
            xy=xy.astype(np.float32),
            score=sc.astype(np.float32),
            in_image=in_image,
            frames=np.asarray(frames, dtype=np.int32),
        )

    from hoi.observations.triangulation import triangulate_keypoints

    screen = triangulate_keypoints(key_dir, out / "manifest.json")
    xyz = np.asarray(screen["xyz"], np.float32)
    valid = np.asarray(screen["valid"], bool)
    inlier_by_view = np.asarray(screen["inlier"], bool)  # V,T,J from screen_keypoints
    if xyz.shape != (T, J, 3) or valid.shape != (T, J) or inlier_by_view.shape != (V, T, J):
        raise RuntimeError(
            "frozen screen_keypoints returned unexpected shapes: "
            f"xyz={xyz.shape} valid={valid.shape} inlier={inlier_by_view.shape}; "
            f"expected {(T,J,3)} {(T,J)} {(V,T,J)}"
        )
    # fit_motion permutes this V,T,J array to T,V,J to match stacked 2D scores.
    inlier_for_fit = np.ascontiguousarray(inlier_by_view)
    xyz_raw = xyz.astype(np.float64) * height + center[None, None, :]
    atomic_savez(
        out / "observations.npz",
        xyz=xyz,
        xyz_raw=xyz_raw.astype(np.float32),
        valid=valid,
        inlier=inlier_for_fit,
        support=np.asarray(screen.get("support"), np.int16),
        median_error_px=np.asarray(screen.get("median_error_px"), np.float32),
        frames=np.asarray(frames, dtype=np.int32),
        identity=np.asarray(args.identity),
        world_normalization_center=center.astype(np.float32),
        world_normalization_height=np.asarray(height, np.float32),
        screen_inlier_layout=np.asarray("V,T,J; fit_motion permutes to T,V,J"),
    )
    person = {
        "schema": "pipeline_person_observations/v1",
        "identity": args.identity,
        "identity_index_in_instance_run": k,
        "instance_index": str(index_path),
        "instance_coordinate_space": coord,
        "out": str(out),
        "frames": [int(x) for x in frames],
        "valid_joint_count": int(valid.sum()),
        "valid_frame_joint_fraction": float(valid.mean()) if valid.size else 0.0,
        "inlier_shape_for_fit_motion": list(inlier_for_fit.shape),
        "inlier_shape_screen_native": list(inlier_by_view.shape),
        "screen_parameters": {
            "SCORE_MIN": 0.5,
            "MAX_ERROR_PX": 10.0,
            "MIN_VIEWS": 3,
            "MIN_RAY_ANGLE_DEG": 1.5,
            "pair_ransac": "28 view pairs",
        },
        "gauge": gauge,
        "world_normalization_applied_once": True,
        "manifest": str(out / "manifest.json"),
        "observations": str(out / "observations.npz"),
        "keypoints": str(key_dir),
        "raw2d_coordinate_space": coord,
        "warnings": [],
    }
    if int(valid.sum()) == 0:
        person["warnings"].append("screen_keypoints produced no valid 3D joint")
    atomic_write_json(out / "person.json", person)
    print(f"[adapter] person package: {out}")
    print(
        f"[adapter] screen valid joints: {int(valid.sum())}, inlier fit shape={inlier_for_fit.shape}"
    )
    print(f"[adapter] shared gauge: center={gauge['center_raw']} height={height}")
    return 0 if int(valid.sum()) > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--camera-space", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--out", required=True)
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
