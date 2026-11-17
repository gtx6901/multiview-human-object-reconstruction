"""Use existing calibrated cameras for first-frame COLMAP geometry."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from hoi.io import write_json
from hoi.objects.multiview import resolve
from hoi.runtime import PROJECT_ROOT


def run_colmap(executable: Path, work: Path, report: dict, stage: str, arguments: list):
    command = [str(executable), stage, *map(str, arguments)]
    report["commands"].append(command)
    write_json(work / "report.json", report)
    with (work / f"{stage}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"[objects geometry] {stage} completed", flush=True)


def first_images(scene_index: Path) -> list[tuple[dict, str]]:
    scene_index = Path(scene_index).resolve()
    scene = json.loads(scene_index.read_text(encoding="utf-8"))
    image_root = scene_index.parent / "images"
    return [
        (view, Path(view["frames"][0]["jpeg"]).relative_to(image_root).as_posix())
        for view in scene["views"]
    ]


def export_known_cameras(scene_index: Path, database_path: Path, model_path: Path) -> dict:
    """Map database IDs to measured cameras; preserve their R,t and world units."""
    database_path, model_path = Path(database_path), Path(model_path)
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    camera_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    image_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    mappings, camera_specs = [], {}
    with closing(sqlite3.connect(str(database_path))) as connection, connection:
        images = {
            name: (int(image_id), int(camera_id))
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images"
            )
        }
        for view, name in first_images(scene_index):
            image_id, camera_id = images[name]
            width, height = int(view["decode"]["width"]), int(view["decode"]["height"])
            calibration = view["calibration"]
            k = np.asarray(calibration["K_undistorted"], dtype=np.float64)
            w = np.asarray(calibration["W"], dtype=np.float64)
            if k.shape != (3, 3) or w.shape != (4, 4):
                raise ValueError(f"{name}: expected K(3,3) and W(4,4)")
            if not np.isfinite(k).all() or not np.isfinite(w).all():
                raise ValueError(f"{name}: non-finite camera matrices")
            if k[0, 1] != 0 or k[1, 0] != 0:
                raise ValueError(f"{name}: COLMAP PINHOLE does not represent skew")
            # OpenCV integer pixel centers become COLMAP half-integer centers.
            params = np.asarray([k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5], "<f8")
            if camera_id in camera_specs:
                ow, oh, op = camera_specs[camera_id]
                if width != ow or height != oh or not np.array_equal(params, op):
                    raise ValueError(f"Database camera {camera_id} shares different intrinsics")
            else:
                camera_specs[camera_id] = width, height, params
                numbers = " ".join(format(float(x), ".17g") for x in params)
                camera_lines.append(f"{camera_id} PINHOLE {width} {height} {numbers}")
            xyzw = Rotation.from_matrix(w[:3, :3]).as_quat()
            pose = np.concatenate([xyzw[[3, 0, 1, 2]], w[:3, 3]])
            numbers = " ".join(format(float(x), ".17g") for x in pose)
            image_lines.extend([f"{image_id} {numbers} {camera_id} {name}", ""])
            mappings.append(
                {
                    "serial": view["serial"],
                    "source_frame": view["frames"][0]["source_frame"],
                    "image_name": name,
                    "image_id": image_id,
                    "camera_id": camera_id,
                }
            )
        for camera_id, (width, height, params) in camera_specs.items():
            changed = connection.execute(
                "UPDATE cameras SET model=?, width=?, height=?, params=?, "
                "prior_focal_length=? WHERE camera_id=?",
                (1, width, height, sqlite3.Binary(params.tobytes()), 1, camera_id),
            )
            if changed.rowcount != 1:
                raise ValueError(f"Missing database camera {camera_id}")
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
        (model_path / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        (model_path / "points3D.txt").write_text("", encoding="utf-8")
    return {
        "database": str(database_path),
        "model": str(model_path),
        "mapping": mappings,
        "intrinsics_conversion": "OpenCV cx,cy -> COLMAP cx+0.5,cy+0.5",
        "extrinsics": "Original OpenCV world-to-camera R,t; quaternion wxyz",
        "units": "Original world coordinates and translation units unchanged",
    }


def build_sparse(config_path: Path) -> Path:
    """Extract and match first-frame features on CPU; triangulate with fixed cameras."""
    import hashlib

    config = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    options = config["geometry"]
    executable = resolve(options["colmap"])
    output = resolve(config["output"])
    index = output / "input/scene.json"
    work = output / "geometry"
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "report.json"
    signature = {
        "input_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "colmap": options["colmap"],
        "sift_max_image_size": options["sift_max_image_size"],
    }
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError(
                "Geometry inputs changed; the existing geometry cache must be replaced"
            )
    report = {"signature": signature, "options": options, "status": "running", "commands": []}
    write_json(report_path, report)
    image_list = work / "image_list.txt"
    image_list.write_text(
        "\n".join(name for _, name in first_images(index)) + "\n", encoding="utf-8"
    )
    database = work / "database.db"
    threads = str(options["num_threads"])

    def run(stage, arguments):
        run_colmap(executable, work, report, stage, arguments)

    run(
        "feature_extractor",
        [
            "--database_path",
            database,
            "--image_path",
            index.parent / "images",
            "--image_list_path",
            image_list,
            "--ImageReader.single_camera_per_folder",
            "1",
            "--FeatureExtraction.use_gpu",
            "0",
            "--FeatureExtraction.num_threads",
            threads,
            "--SiftExtraction.max_image_size",
            options["sift_max_image_size"],
        ],
    )
    report["known_cameras"] = export_known_cameras(index, database, work / "known")
    run(
        "exhaustive_matcher",
        [
            "--database_path",
            database,
            "--FeatureMatching.use_gpu",
            "0",
            "--FeatureMatching.num_threads",
            threads,
        ],
    )
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    run(
        "point_triangulator",
        [
            "--database_path",
            database,
            "--image_path",
            index.parent / "images",
            "--input_path",
            work / "known",
            "--output_path",
            sparse,
            "--refine_intrinsics",
            "0",
            "--Mapper.fix_existing_frames",
            "1",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_refine_sensor_from_rig",
            "0",
            "--Mapper.tri_ignore_two_view_tracks",
            str(int(options["ignore_two_view_tracks"])),
            "--Mapper.num_threads",
            threads,
        ],
    )
    report.update(status="sparse_complete", sparse_model=str(sparse), dense_point_cloud=None)
    write_json(report_path, report)
    return report_path


def prepare_dense_workspace(config_path: Path) -> Path:
    config = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    output = resolve(config["output"])
    options = config["geometry"]
    work = output / "geometry"
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    run_colmap(
        resolve(options["colmap"]),
        work,
        report,
        "image_undistorter",
        [
            "--image_path",
            output / "input/images",
            "--input_path",
            work / "sparse",
            "--output_path",
            work / "dense",
            "--output_type",
            "COLMAP",
            "--max_image_size",
            options["max_image_size"],
        ],
    )
    report["status"] = "dense_workspace_prepared"
    write_json(work / "report.json", report)
    return work / "dense"


def build_dense(config_path: Path) -> Path:
    """Run native CUDA PatchMatch and CPU fusion; caller schedules the sole GPU task."""
    config = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    output = resolve(config["output"])
    options = config["geometry"]
    executable = resolve(options["colmap"])
    work = output / "geometry"
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    run_colmap(
        executable,
        work,
        report,
        "patch_match_stereo",
        [
            "--workspace_path",
            work / "dense",
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "1",
            "--PatchMatchStereo.gpu_index",
            options["gpu_index"],
            "--PatchMatchStereo.max_image_size",
            options["max_image_size"],
            "--PatchMatchStereo.cache_size",
            options["cache_size_gb"],
        ],
    )
    write_json(work / "report.json", report)
    return fuse_dense(config_path)


def fuse_dense(config_path: Path) -> Path:
    config = json.loads(resolve(config_path).read_text(encoding="utf-8"))
    output = resolve(config["output"])
    options = config["geometry"]
    work = output / "geometry"
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    report["options"] = options
    executable = resolve(options["colmap"])
    fused = output / "input/fused.ply"
    run_colmap(
        executable,
        work,
        report,
        "stereo_fusion",
        [
            "--workspace_path",
            work / "dense",
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            fused,
            "--StereoFusion.num_threads",
            options["num_threads"],
            "--StereoFusion.cache_size",
            options["cache_size_gb"],
            "--StereoFusion.min_num_pixels",
            options["min_fusion_views"],
        ],
    )
    vertex_count = None
    with fused.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Fused PLY header is incomplete")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
    report.update(status="dense_complete", dense_point_cloud=str(fused), vertex_count=vertex_count)
    write_json(work / "report.json", report)
    return fused


def build(config_path: Path) -> Path:
    build_sparse(config_path)
    prepare_dense_workspace(config_path)
    return build_dense(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(build(parser.parse_args().config))
