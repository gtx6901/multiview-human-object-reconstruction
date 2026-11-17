"""Observe faces on full source images with the locally pinned 3DDFA-V2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

from hoi.image_io import read_image
from hoi.io import write_json
from hoi.runtime import PROJECT_ROOT, configure_runtime


def install_cpu_nms(root: Path):
    """Use upstream NumPy hard NMS with the compiled implementation's boundary."""
    path = root / "FaceBoxes/utils/nms/py_cpu_nms.py"
    source = path.read_text(encoding="utf-8")
    if source.count("ovr <= thresh") != 1:
        raise ImportError(f"Unsupported FaceBoxes Python NMS revision: {path}")
    # cpu_nms.pyx suppresses overlap >= threshold. Keep that predicate, including
    # its NaN-overlap behavior, without rounding the threshold to float32.
    namespace = {}
    exec(
        compile(
            source.replace("ovr <= thresh", "~(ovr.astype(np.float64) >= float(thresh))"),
            str(path),
            "exec",
        ),
        namespace,
    )
    module = types.ModuleType("FaceBoxes.utils.nms_wrapper")
    module.nms = namespace["py_cpu_nms"]
    sys.modules[module.__name__] = module


class FaceObserver:
    """Return detections and 68 image-space landmarks without pre-cropping RGB."""

    def __init__(self, root: Path, checkpoint: Path):
        import torch

        configure_runtime()
        self.root, self.checkpoint = root.resolve(), checkpoint.resolve()
        sys.path.insert(0, str(self.root))
        np.__dict__.setdefault("long", np.int64)
        install_cpu_nms(self.root)
        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA

        # Upstream constructors disable grad globally; contain that side effect.
        with torch.inference_mode():
            self.detector = FaceBoxes()
            self.aligner = TDDFA(
                arch="mobilenet",
                widen_factor=1.0,
                size=120,
                num_params=62,
                checkpoint_fp=str(self.checkpoint),
            )

    def observe(self, bgr):
        import torch

        with torch.inference_mode():
            boxes = self.detector(bgr)
            parameters, rois = self.aligner(bgr, boxes)
            vertices = self.aligner.recon_vers(parameters, rois, dense_flag=False)
        return [
            {
                "box": np.asarray(box, np.float32),
                "xy": vertex[:2].T.copy(),
                "parameters": parameter,
                "roi": np.asarray(roi, np.float32),
            }
            for box, vertex, parameter, roi in zip(boxes, vertices, parameters, rois)
        ]

    def provenance(self):
        names = ("TDDFA.py", "FaceBoxes/FaceBoxes.py", "FaceBoxes/utils/nms/py_cpu_nms.py")
        return {
            "vendor_root": str(self.root),
            "source_sha256": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in names
            },
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "nms": "author NumPy hard NMS, overlap equality matches author cpu_nms.pyx",
            "rgb_context": "full original BGR image; author expands and samples each ROI",
            "landmarks": "68 image-space points; jaw contour is view-dependent; no calibrated depth inferred",
        }


def associate_face_detections(records, record_ids, boxes, people):
    """Assign detections globally to the already tracked people's observed heads."""
    from scipy.optimize import linear_sum_assignment

    identities = np.full(len(record_ids), "", dtype=object)
    for record_id in np.unique(record_ids):
        record = records[int(record_id)]
        candidates = np.flatnonzero(record_ids == record_id)
        heads = []
        for person in people:
            camera = person["cameras"].get(str(record["camera"]))
            if camera is None:
                continue
            positions = np.flatnonzero(camera["frames"] == int(record["frame"]))
            if not len(positions):
                continue
            t = int(positions[0])
            xy, score = camera["xy"][t, :5], camera["score"][t, :5]
            usable = (
                camera["in_image"][t, :5]
                & np.isfinite(xy).all(-1)
                & np.isfinite(score)
                & (score >= 0.5)
            )
            if usable.any():
                points = xy[usable]
                heads.append((person["identity"], points, np.median(points, axis=0)))
        if not heads:
            continue
        distances = np.zeros((len(heads), len(candidates)))
        eligible = np.zeros_like(distances, bool)
        for actor, (_, points, center) in enumerate(heads):
            for column, row in enumerate(candidates):
                box = boxes[row, :4]
                if not np.isfinite(box).all():
                    continue
                if ((points >= box[:2]) & (points <= box[2:])).all(-1).any():
                    eligible[actor, column] = True
                    distances[actor, column] = np.linalg.norm((box[:2] + box[2:]) / 2 - center)
        if not eligible.any():
            continue
        # Dummy costs encode legal matching cardinality, not image quality.
        dummy = (len(heads) + 1) * (float(distances[eligible].max()) + 1)
        costs = np.full((len(heads), len(candidates) + len(heads)), dummy)
        costs[:, : len(candidates)] = np.where(eligible, distances, dummy * (len(heads) + 1))
        rows, columns = linear_sum_assignment(costs)
        for actor, column in zip(rows, columns):
            if column < len(candidates) and eligible[actor, column]:
                identities[candidates[column]] = heads[actor][0]
    return np.asarray(identities, dtype=str)


def associate_observations(directory: Path, person_paths):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    with np.load(directory / "observations.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    people, sources = [], {}
    for path in person_paths:
        path = Path(path).resolve()
        person = json.loads(path.read_text(encoding="utf-8"))
        cameras = {}
        for filename in Path(person["keypoints"]).glob("*.npz"):
            with np.load(filename, allow_pickle=False) as archive:
                cameras[filename.stem] = {
                    name: archive[name] for name in ("frames", "xy", "score", "in_image")
                }
            sources[str(filename)] = hashlib.sha256(filename.read_bytes()).hexdigest()
        people.append({"identity": person["identity"], "cameras": cameras})
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    data["identities"] = associate_face_detections(
        report["records"], data["record_index"], data["boxes"], people
    )
    np.savez_compressed(directory / "observations.npz", **data)
    report["association"] = {
        "sources": sources,
        "method": "global one-to-one head-point/face-box association; no triangulation selection",
        "unassigned_detections": int((data["identities"] == "").sum()),
    }
    write_json(directory / "report.json", report)
    return report["association"]


def assemble_face_observations(directory: Path, frames, views, landmark_indices, identity=None):
    """Join actual detections to a person's source-frame and camera order."""
    directory = Path(directory).resolve()
    report_path, data_path = directory / "report.json", directory / "observations.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    frames = np.asarray(frames, np.int64)
    landmarks = np.asarray(landmark_indices, np.int64)
    serials = [str(view["serial"]) for view in views]
    frame_index = {int(frame): i for i, frame in enumerate(frames)}
    camera_index = {serial: i for i, serial in enumerate(serials)}
    if len(frame_index) != len(frames) or len(camera_index) != len(views):
        raise ValueError("Face frame and camera identities must be unique")
    record_keys = [(str(record["camera"]), int(record["frame"])) for record in records]
    if len(set(record_keys)) != len(record_keys):
        raise ValueError("Duplicate camera/source-frame face records")
    with np.load(data_path, allow_pickle=False) as data:
        record_ids, boxes, detected_xy = data["record_index"], data["boxes"], data["xy"]
        if "identities" in data.files:
            if identity is None:
                raise ValueError("Associated face observations require a person identity")
            selected = np.asarray(data["identities"]) == identity
            record_ids, boxes, detected_xy = (
                record_ids[selected],
                boxes[selected],
                detected_xy[selected],
            )
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("Multiple detected faces require explicit instance association")
    t, v, n = len(frames), len(views), len(landmarks)
    xy = np.full((t, v, n, 2), np.nan, np.float64)
    score = np.full((t, v, n), np.nan, np.float64)
    valid = np.zeros((v, t, n), bool)
    for row, record_id in enumerate(record_ids):
        serial, frame = record_keys[int(record_id)]
        if serial not in camera_index or frame not in frame_index:
            continue
        ti, vi = frame_index[frame], camera_index[serial]
        points = detected_xy[row, landmarks].astype(np.float64)
        face_score = float(boxes[row, 4])
        xy[ti, vi], score[ti, vi] = points, face_score
        valid[vi, ti] = (
            np.isfinite(face_score)
            & (face_score >= 0.5)
            & np.isfinite(points).all(-1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < views[vi]["width"])
            & (points[:, 1] >= 0)
            & (points[:, 1] < views[vi]["height"])
        )
    return {
        "xy": xy,
        "score": score,
        "valid": valid,
        "frames": frames,
        "cameras": serials,
        "landmark_indices": landmarks,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (report_path, data_path)
        },
        "score_semantics": "FaceBoxes detection score repeated per landmark; not point probability",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="JSON list of image records with a path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--people",
        type=Path,
        nargs="+",
        help="Associate existing detections to these tracked people",
    )
    parser.add_argument("--vendor", type=Path, default=PROJECT_ROOT / "runtime/vendor/3DDFA_V2")
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "runtime/weights/3ddfa_v2/mb1_120x120.pth"
    )
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(2)
    if args.people:
        print(associate_observations(args.out, args.people))
        return 0
    if args.images is None:
        parser.error("Supply --images for observation or --people for identity association")
    observer = FaceObserver(args.vendor, args.checkpoint)
    records = json.loads(args.images.read_text(encoding="utf-8"))
    rows, counts = [], []
    for index, record in enumerate(records):
        image = read_image(Path(record["path"]))
        if image is None:
            raise FileNotFoundError(record["path"])
        faces = observer.observe(image)
        counts.append(len(faces))
        rows.extend((index, face) for face in faces)
        if (index + 1) % 100 == 0:
            print(f"[faces] observed {index + 1}/{len(records)} images", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "observations.npz",
        record_index=np.asarray([index for index, _ in rows], np.int64),
        counts=np.asarray(counts, np.int32),
        boxes=np.stack([face["box"] for _, face in rows]) if rows else np.empty((0, 5), np.float32),
        xy=np.stack([face["xy"] for _, face in rows]) if rows else np.empty((0, 68, 2), np.float32),
        parameters=(
            np.stack([face["parameters"] for _, face in rows])
            if rows
            else np.empty((0, 62), np.float32)
        ),
        roi=np.stack([face["roi"] for _, face in rows]) if rows else np.empty((0, 4), np.float32),
    )
    write_json(
        args.out / "report.json",
        {
            "records": records,
            "counts": counts,
            "provenance": observer.provenance(),
            "scope": "Image observations only; no instance assignment, 3D triangulation, motion or appearance changes",
        },
    )
    print(f"[faces] {len(rows)} detections across {len(records)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
