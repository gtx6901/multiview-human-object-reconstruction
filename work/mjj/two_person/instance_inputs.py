#!/usr/bin/env python
"""Instance-level two-person input preparation for the real FCS scene.

The script deliberately separates three things that are easy to conflate:

1. *Detection*: proposing anonymous person boxes (local YOLO weights or an
   externally supplied boxes file).  RTMW is a top-down pose model and is only
   run on an already-proposed per-person box; a full-frame RTMW call is never
   treated as person detection or as two identities.
2. *Identity*: global labels come from a Codex-authored seed file.  This script
   never guesses who is person A or person B.  If the seed file is absent it can
   export numbered candidate boxes and a blank template for Codex to label.
3. *Observation*: a missing/occluded person is stored as ``present=false`` with
   NaN coordinates and an explicit reason.  It is never stored as a valid
   all-zero observation.

Outputs live below this script's ``data/`` directory.  Inputs (videos,
calibration, manifest, checkpoints) are only read.

Subcommands
-----------
preflight
    Inspect scene/seeds/dependencies/local checkpoints without running models.
candidates
    Run a local person detector on chosen frames, write numbered overlays and a
    blank ``seed_template.json`` for Codex to label.
prepare
    Track the two seeded identities per camera, optionally run per-instance
    RTMW, write trajectories/overlays, triangulation diagnostics, validation.
validate
    Re-check an existing run's structural/numeric invariants.

Example short segment (after Codex has labelled seeds):

  D:/Document/3dgs/.tmp/rebuild_20261014_runtime/venv/Scripts/python.exe ^
    instance_inputs.py prepare ^
      --scene data/scene_inventory/scene.json ^
      --seeds data/instance_inputs/seeds.json ^
      --detector yolo --weights <local-yolo-person.pt> ^
      --rtmw D:/Document/大四/暑校--数字人重建/runtime/wholebody/rtmw.onnx ^
      --frames 0:60:1 ^
      --out data/instance_inputs/run
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dual_track'))
    from image_io import read_image, write_image
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"OpenCV (cv2) is required: {exc}")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_OUT = THIS_DIR / "data" / "instance_inputs"
BODY_JOINTS = (5, 6, 11, 12, 13, 14)  # COCO-133 shoulders, hips, knees
HAND_JOINTS = (9, 10) + tuple(range(91, 133))  # wrists + COCO-133 hand blocks
COLORS = ((40, 190, 40), (0, 140, 255))  # BGR for person_a / person_b
RAW_SPACE = "raw_distorted_native_2048x1536"
UND_SPACE = "undistorted_native_2048x1536"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def module_status(module: str, distribution: Optional[str] = None) -> Dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError) as exc:
        return {"module": module, "present": False, "error": str(exc)}
    if spec is None:
        return {"module": module, "present": False}
    version = None
    try:
        version = importlib.metadata.version(distribution or module)
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    return {
        "module": module,
        "present": True,
        "version": version,
        "origin": getattr(spec, "origin", None),
    }


# ---------------------------------------------------------------------------
# scene / camera access
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    index: int
    serial: str
    path: Path
    width: int
    height: int
    frame_count: int
    K: np.ndarray
    dist: np.ndarray
    W: np.ndarray
    K_und: np.ndarray
    pts: np.ndarray

    def projection(self) -> np.ndarray:
        return np.asarray(self.K_und, dtype=np.float64) @ np.asarray(self.W, dtype=np.float64)[:3, :]

    def as_json(self) -> Dict[str, Any]:
        return {
            "camera_index": self.index,
            "serial": self.serial,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "K_native": self.K.tolist(),
            "distortion_opencv5": self.dist.tolist(),
            "K_undistorted": self.K_und.tolist(),
            "W": self.W.tolist(),
            "projection": self.projection().tolist(),
            "pts_first_sec": float(self.pts[0]) if self.pts.size and np.isfinite(self.pts[0]) else None,
            "pts_last_sec": float(self.pts[-1]) if self.pts.size and np.isfinite(self.pts[-1]) else None,
        }


def load_scene(path: Path) -> Tuple[Dict[str, Any], List[Camera], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scene = json.loads(path.read_text(encoding="utf-8"))
    warnings: List[str] = []
    cameras: List[Camera] = []
    for v in sorted(scene.get("views") or [], key=lambda x: x.get("camera_index", 0)):
        serial = str(v.get("serial"))
        cal = v.get("calibration") or {}
        if not cal.get("available"):
            warnings.append(f"{serial}: calibration unavailable; camera omitted from projection")
            continue
        try:
            K = np.asarray(cal["K_native"], dtype=np.float64)
            dist = np.asarray(cal.get("distortion_opencv5", []), dtype=np.float64)
            W = np.asarray(cal["W"], dtype=np.float64)
            K_und = np.asarray(cal.get("K_undistorted", K), dtype=np.float64)
        except Exception as exc:
            warnings.append(f"{serial}: bad calibration ({exc}); camera omitted")
            continue
        if K.shape != (3, 3) or W.shape != (4, 4):
            warnings.append(f"{serial}: unexpected K/W shape; camera omitted")
            continue
        dec = v.get("decode") or {}
        width = int(dec.get("width") or cal.get("native_width") or 0)
        height = int(dec.get("height") or cal.get("native_height") or 0)
        frame_count = int(dec.get("actual_frames") or 0)
        if frame_count <= 0:
            warnings.append(f"{serial}: no decoded frames in scene; camera omitted")
            continue
        pts = np.full(frame_count, np.nan, dtype=np.float64)
        timeline = v.get("timeline_npz")
        if timeline:
            tp = Path(timeline)
            if not tp.is_absolute():
                tp = path.parent / tp
            if tp.is_file():
                try:
                    with np.load(tp, allow_pickle=False) as z:
                        ff = np.asarray(z["ffprobe_pts_sec"], dtype=np.float64)
                        cv = np.asarray(z["cv2_pos_msec"], dtype=np.float64) / 1000.0
                        n = min(frame_count, ff.size, cv.size)
                        pts[:n] = np.where(np.isfinite(ff[:n]), ff[:n], cv[:n])
                except Exception as exc:
                    warnings.append(f"{serial}: cannot read timeline ({exc}); PTS marked unknown")
            else:
                warnings.append(f"{serial}: timeline missing: {tp}")
        cameras.append(
            Camera(
                index=int(v.get("camera_index", len(cameras))),
                serial=serial,
                path=Path(v["path"]),
                width=width,
                height=height,
                frame_count=frame_count,
                K=K,
                dist=dist,
                W=W,
                K_und=K_und,
                pts=pts,
            )
        )
    if len(cameras) != 8:
        warnings.append(f"scene exposes {len(cameras)} calibrated cameras, expected 8")
    return scene, cameras, warnings


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
@dataclass
class Identity:
    id: str
    display_name: str = ""
    notes: str = ""

    def as_json(self) -> Dict[str, Any]:
        return {"id": self.id, "display_name": self.display_name, "notes": self.notes}


@dataclass
class Seed:
    camera_index: int
    serial: str
    frame: int
    identity: str
    box: np.ndarray
    score: float = 1.0
    source: str = "codex_visual_seed"
    keypoints_enabled: bool = True


@dataclass
class SeedSet:
    path: Path
    identities: List[Identity]
    seeds: List[Seed]
    facts: Dict[str, Any] = field(default_factory=dict)


def load_seeds(path: Path, cameras: Sequence[Camera]) -> SeedSet:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("identities") or []
    identities: List[Identity] = []
    for item in ids:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError("each identities[] entry needs a non-empty id")
        identities.append(
            Identity(
                id=str(item["id"]),
                display_name=str(item.get("display_name") or ""),
                notes=str(item.get("notes") or ""),
            )
        )
    if len(identities) != 2:
        raise ValueError(f"seed file must define exactly two identities, found {len(identities)}")
    if len({x.id for x in identities}) != 2:
        raise ValueError("identity ids must be unique")
    by_serial = {c.serial: c for c in cameras}
    seeds: List[Seed] = []
    seen_keys = set()
    for view in payload.get("views") or []:
        serial = str(view.get("serial") or "")
        cam = by_serial.get(serial)
        if cam is None:
            raise ValueError(f"seed view serial {serial!r} is not a calibrated scene camera")
        for person in view.get("persons") or []:
            ident = person.get("identity")
            if ident in (None, "", "null", "unlabelled", "unknown"):
                raise ValueError(
                    f"{serial}: a seed person has no identity; label it in seeds.json, "
                    "do not let the runner guess"
                )
            if str(ident) not in {x.id for x in identities}:
                raise ValueError(f"{serial}: unknown identity {ident!r}")
            frame = int(person.get("frame", view.get("frame", 0)))
            if not (0 <= frame < cam.frame_count):
                raise ValueError(f"{serial}: seed frame {frame} outside [0,{cam.frame_count})")
            box = np.asarray(person.get("box"), dtype=np.float64).reshape(-1)
            if box.size != 4 or not np.isfinite(box).all():
                raise ValueError(f"{serial} f{frame}: invalid box {person.get('box')}")
            x0, y0, x1, y1 = box.tolist()
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"{serial} f{frame}: box has non-positive area {box.tolist()}")
            if x0 < -2 or y0 < -2 or x1 > cam.width + 2 or y1 > cam.height + 2:
                raise ValueError(
                    f"{serial} f{frame}: box {box.tolist()} outside {cam.width}x{cam.height}"
                )
            box[[0, 2]] = np.clip(box[[0, 2]], 0, cam.width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, cam.height)
            key = (serial, frame, str(ident))
            if key in seen_keys:
                raise ValueError(f"duplicate seed for {key}")
            seen_keys.add(key)
            seeds.append(
                Seed(
                    camera_index=cam.index,
                    serial=serial,
                    frame=frame,
                    identity=str(ident),
                    box=box.astype(np.float32),
                    score=float(person.get("score", 1.0)),
                    source=str(person.get("source") or "codex_visual_seed"),
                    keypoints_enabled=bool(person.get("keypoints_enabled", True)),
                )
            )
    if not seeds:
        raise ValueError("seed file contains no labelled person boxes")
    counts = {x.id: 0 for x in identities}
    for s in seeds:
        counts[s.identity] += 1
    if any(v == 0 for v in counts.values()):
        raise ValueError(f"both identities need at least one seed; counts={counts}")

    per_cam = {}
    for c in cameras:
        per_cam[c.serial] = {
            x.id: sum(1 for s in seeds if s.serial == c.serial and s.identity == x.id)
            for x in identities
        }
    facts = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "session": payload.get("session"),
        "identity_counts": counts,
        "per_camera_identity_counts": per_cam,
        "coordinate_space": payload.get("coordinate_space"),
        "cameras_with_both_at_some_seed_frame": sum(
            1
            for c in cameras
            if all(v > 0 for v in per_cam[c.serial].values())
        ),
    }
    return SeedSet(path=path, identities=identities, seeds=seeds, facts=facts)


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------
def parse_frame_spec(spec: str, frame_count: int) -> List[int]:
    text = str(spec).strip()
    if text.lower() in ("all", "*", ""):
        return list(range(frame_count))
    if "," in text:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
        out = sorted({v for v in values if 0 <= v < frame_count})
        if not out:
            raise ValueError(f"frame list {spec!r} selects no valid frame")
        return out
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"frame spec must be START:END[:STEP], ALL, or a comma list: {spec!r}")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    if end < 0:
        end = frame_count
    if step <= 0:
        raise ValueError("frame step must be positive")
    start = max(0, min(start, frame_count))
    end = max(0, min(end, frame_count))
    if start >= end:
        raise ValueError(f"frame range {spec!r} is empty for frame_count={frame_count}")
    return list(range(start, end, step))


def timestamp(cam: Camera, frame: int) -> Optional[float]:
    if 0 <= frame < cam.pts.size and np.isfinite(cam.pts[frame]):
        return float(cam.pts[frame])
    return None


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    box: np.ndarray
    score: float
    candidate_id: Optional[str] = None
    source: str = "detector"


class SeedOnlyDetector:
    name = "none"

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        return []

    def config(self) -> Dict[str, Any]:
        return {"name": self.name}


class ExternalDetections:
    """Read anonymous person boxes from a JSON file (no model download)."""

    name = "external"

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.coordinate_space = str(
            payload.get("coordinate_space") or "raw_distorted_native_2048x1536"
        )
        self.views: Dict[str, Dict[int, List[Detection]]] = {}
        views = payload.get("views") or payload
        if isinstance(views, list):
            for view in views:
                serial = str(view.get("serial") or view.get("camera") or "")
                frames = view.get("frames") or []
                if isinstance(frames, dict):
                    iterable = [
                        {"frame": int(k), "detections": v} for k, v in frames.items()
                    ]
                else:
                    iterable = frames
                for item in iterable:
                    frame = int(item.get("frame", item.get("index", 0)))
                    dets = item.get("detections", item.get("boxes", []))
                    self.views.setdefault(serial, {})[frame] = self._normalize(dets)
        elif isinstance(views, dict):
            for serial, frames in views.items():
                if isinstance(frames, dict):
                    for frame, dets in frames.items():
                        self.views.setdefault(str(serial), {})[int(frame)] = self._normalize(dets)
                elif isinstance(frames, list):
                    for item in frames:
                        frame = int(item.get("frame", 0))
                        dets = item.get("detections", item.get("boxes", []))
                        self.views.setdefault(str(serial), {})[frame] = self._normalize(dets)
        else:
            raise ValueError("external detections must contain a views list or object")
        self.facts = {"path": str(path), "sha256": sha256_file(path), "serials": sorted(self.views)}
        self.facts["coordinate_space"] = self.coordinate_space

    @staticmethod
    def _normalize(dets: Any) -> List[Detection]:
        out: List[Detection] = []
        for item in dets or []:
            if isinstance(item, dict):
                box = item.get("box", item.get("bbox"))
                score = float(item.get("score", item.get("confidence", 1.0)))
            else:
                box = item
                score = 1.0
            arr = np.asarray(box, dtype=np.float64).reshape(-1)
            if arr.size == 4 and np.isfinite(arr).all():
                x0, y0, x1, y1 = arr.tolist()
                if x1 > x0 and y1 > y0:
                    out.append(Detection(box=arr.astype(np.float32), score=score, source="external"))
        return out

    def detect_for(self, serial: str, frame: int) -> List[Detection]:
        return list(self.views.get(serial, {}).get(frame, []))

    def detect(self, bgr: np.ndarray) -> List[Detection]:  # not frame-aware
        raise RuntimeError("ExternalDetections needs detect_for(serial, frame)")

    def config(self) -> Dict[str, Any]:
        return {"name": self.name, **self.facts}


class YoloPersonDetector:
    """Local Ultralytics YOLO person detector; never downloads weights."""

    name = "yolo"

    def __init__(self, weights: Path, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640):
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}. Refusing to let Ultralytics download."
            )
        try:
            from ultralytics import YOLO  # lazy: no import / CUDA context in preflight
        except Exception as exc:
            raise RuntimeError(f"ultralytics is not importable: {exc}") from exc
        self.weights = weights
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.model = YOLO(str(weights))

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=bgr,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        for b, s in zip(xyxy, conf):
            arr = np.asarray(b, dtype=np.float64)
            if arr.size == 4 and np.isfinite(arr).all():
                x0, y0, x1, y1 = arr.tolist()
                if x1 > x0 and y1 > y0:
                    out.append(
                        Detection(
                            box=arr.astype(np.float32),
                            score=float(s),
                            source="yolo",
                        )
                    )
        return out

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weights": str(self.weights),
            "weights_size": int(self.weights.stat().st_size),
            "weights_mtime_utc": datetime.fromtimestamp(
                self.weights.stat().st_mtime, timezone.utc
            ).isoformat(timespec="seconds"),
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "classes": [0],
            "device": "cpu",
        }


def build_detector(args: argparse.Namespace) -> Any:
    if args.detector == "none":
        return SeedOnlyDetector()
    if args.detector == "external":
        if not args.detections:
            raise ValueError("--detector external requires --detections boxes.json")
        return ExternalDetections(Path(args.detections))
    if args.detector == "yolo":
        if not args.weights:
            raise ValueError("--detector yolo requires a local --weights checkpoint")
        return YoloPersonDetector(
            Path(args.weights), conf=args.conf, iou=args.iou, imgsz=args.imgsz
        )
    raise ValueError(args.detector)


class WholebodyPose:
    """Per-box RTMW wholebody (133 keypoints), CPU.  Top-down, not a detector."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from rtmlib import RTMPose
        except Exception as exc:
            raise RuntimeError(f"rtmlib is not importable: {exc}") from exc
        self.path = model_path
        self.model = RTMPose(
            str(model_path),
            model_input_size=(288, 384),
            backend="onnxruntime",
            device="cpu",
            to_openpose=False,
        )

    def predict(
        self, bgr: np.ndarray, boxes: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not boxes:
            return (
                np.full((0, 133, 2), np.nan, dtype=np.float32),
                np.full((0, 133), np.nan, dtype=np.float32),
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bboxes = [
            [float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes
        ]
        k, s = self.model(rgb, bboxes=bboxes)
        k = np.asarray(k, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        if k.ndim == 2:
            k = k[None]
        if s.ndim == 1:
            s = s[None]
        if k.shape[0] != len(boxes):
            raise RuntimeError(
                f"RTMW returned {k.shape[0]} poses for {len(boxes)} boxes; refusing to misassign"
            )
        return k, s

    def config(self) -> Dict[str, Any]:
        return {
            "model": str(self.path),
            "size": int(self.path.stat().st_size),
            "input_size": [288, 384],
            "backend": "onnxruntime",
            "device": "cpu",
            "keypoint_schema": "wholebody_133",
        }


# ---------------------------------------------------------------------------
# geometry / matching helpers
# ---------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def center_norm(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    cb = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    scale = max(1.0, math.sqrt(max(1.0, (a[2] - a[0]) * (a[3] - a[1]))))
    return float(np.linalg.norm(ca - cb) / scale)


def predict_box(box: np.ndarray, velocity: np.ndarray, gap: int, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float64).copy()
    shift = np.asarray(velocity, dtype=np.float64) * max(0, gap)
    out[[0, 2]] += shift[0]
    out[[1, 3]] += shift[1]
    out[[0, 2]] = np.clip(out[[0, 2]], 0, width)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, height)
    if out[2] <= out[0] or out[3] <= out[1]:
        return np.asarray(box, dtype=np.float64).copy()
    return out


def assign_cost(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        pairs: List[Tuple[int, int]] = []
        used_r, used_c = set(), set()
        flat = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in flat:
            r, c = int(r), int(c)
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs


def triangulate_dlt(Ps: Sequence[np.ndarray], uvs: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if len(Ps) < 2 or len(Ps) != len(uvs):
        return None
    uv_arr = np.asarray(uvs, dtype=np.float64)
    if not np.isfinite(uv_arr).all():
        return None
    # Hartley-style normalization of the pixel observations improves the
    # conditioning of the multi-view DLT without changing the solution.
    centroid = uv_arr.mean(axis=0)
    mean_dist = float(np.mean(np.linalg.norm(uv_arr - centroid, axis=1)))
    if mean_dist < 1e-9:
        return None
    scale = math.sqrt(2.0) / mean_dist
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    A = []
    for P, uv in zip(Ps, uvs):
        P = np.asarray(P, dtype=np.float64)
        Pn = T @ P
        un = T @ np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
        A.append(un[0] * Pn[2] - Pn[0])
        A.append(un[1] * Pn[2] - Pn[1])
    A = np.asarray(A, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X / X[3]
    if not np.isfinite(X[:3]).all() or X[2] <= 0:
        return None
    return X[:3]


def project_point(P: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    h = np.asarray(P, dtype=np.float64) @ np.array(
        [X[0], X[1], X[2], 1.0], dtype=np.float64
    )
    if not np.isfinite(h).all() or abs(h[2]) < 1e-12:
        return None
    return h[:2] / h[2]


def undistort_keypoints(cam: Camera, kp: np.ndarray) -> np.ndarray:
    if cam.dist.size == 0:
        return kp.astype(np.float64)
    pts = kp.reshape(-1, 1, 2).astype(np.float64)
    try:
        out = cv2.undistortPoints(pts, cam.K, cam.dist, P=cam.K_und)
    except cv2.error:
        return kp.astype(np.float64)
    return out.reshape(-1, 2)


def rectify_maps(cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    return cv2.initUndistortRectifyMap(
        cam.K, cam.dist, None, cam.K_und, (cam.width, cam.height), cv2.CV_32FC1
    )


def rectify_frame(cam: Camera, bgr: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rectify_box(cam: Camera, box: Sequence[float]) -> np.ndarray:
    """Map a raw box to the bbox of its rectified 7x7 boundary lattice."""
    x0, y0, x1, y1 = [float(v) for v in box]
    xs = np.linspace(x0, x1, 7, dtype=np.float64)
    ys = np.linspace(y0, y1, 7, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    if cam.dist.size:
        uv = cv2.undistortPoints(pts, cam.K, cam.dist, P=cam.K_und).reshape(-1, 2)
    else:
        uv = cv2.undistortPoints(
            pts, cam.K, np.zeros(5, np.float64), P=cam.K_und
        ).reshape(-1, 2)
    lo = np.min(uv, axis=0)
    hi = np.max(uv, axis=0)
    out = np.array(
        [
            np.clip(lo[0], 0, cam.width - 1),
            np.clip(lo[1], 0, cam.height - 1),
            np.clip(hi[0], 0, cam.width - 1),
            np.clip(hi[1], 0, cam.height - 1),
        ],
        dtype=np.float32,
    )
    if out[2] - out[0] < 8 or out[3] - out[1] < 8:
        raise ValueError(f"rectified seed box too small on {cam.serial}: {out.tolist()}")
    return out


# ---------------------------------------------------------------------------
# overlays
# ---------------------------------------------------------------------------
def overlay_frame(
    bgr: np.ndarray,
    cam: Camera,
    frame: int,
    pts: Optional[float],
    identities: Sequence[Identity],
    boxes: np.ndarray,
    present: np.ndarray,
    scores: np.ndarray,
    reasons: Sequence[str],
    out_path: Path,
) -> None:
    h, w = bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(1, w))
    img = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    label = f"cam{cam.index} {cam.serial} f{frame:04d}"
    if pts is not None:
        label += f"  pts={pts:.3f}s"
    cv2.rectangle(img, (0, 0), (min(img.shape[1] - 1, 640), 26), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    missing = []
    for k, ident in enumerate(identities):
        color = COLORS[k % len(COLORS)]
        if bool(present[k]):
            b = boxes[k]
            if np.isfinite(b).all():
                p0 = (int(round(b[0] * scale)), int(round(b[1] * scale)))
                p1 = (int(round(b[2] * scale)), int(round(b[3] * scale)))
                cv2.rectangle(img, p0, p1, color, 3)
                text = f"{ident.id} {float(scores[k]):.2f}"
                cv2.putText(img, text, (p0[0] + 4, max(16, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        else:
            missing.append(ident.id)
    if missing:
        cv2.rectangle(img, (0, 28), (min(img.shape[1] - 1, 720), 56), (0, 0, 80), -1)
        cv2.putText(
            img,
            "MISSING (not treated as valid): " + ", ".join(missing),
            (6, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def overlay_candidates(
    bgr: np.ndarray, cam: Camera, frame: int, pts: Optional[float], dets: Sequence[Detection], out_path: Path
) -> None:
    h, w = bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(1, w))
    img = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    title = f"cam{cam.index} {cam.serial} f{frame:04d}"
    if pts is not None:
        title += f" pts={pts:.3f}s"
    cv2.rectangle(img, (0, 0), (min(img.shape[1] - 1, 720), 26), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for i, det in enumerate(dets):
        b = det.box
        p0 = (int(round(b[0] * scale)), int(round(b[1] * scale)))
        p1 = (int(round(b[2] * scale)), int(round(b[3] * scale)))
        color = (0, 215, 255) if i % 2 == 0 else (255, 120, 0)
        cv2.rectangle(img, p0, p1, color, 3)
        cv2.putText(
            img,
            f"cand_{i:02d} {det.score:.2f}",
            (p0[0] + 4, max(16, p0[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# ---------------------------------------------------------------------------
# per-camera tracking
# ---------------------------------------------------------------------------
def process_camera(
    cam: Camera,
    identities: Sequence[Identity],
    seeds: Sequence[Seed],
    detector: Any,
    pose: Optional[WholebodyPose],
    frames: Sequence[int],
    coord_space: str,
    args: argparse.Namespace,
    out_dir: Path,
    signature: str,
) -> Dict[str, Any]:
    cache_npz = out_dir / "cache" / f"{cam.serial}.npz"
    cache_meta_path = out_dir / "cache" / f"{cam.serial}.json"
    if args.resume and cache_npz.is_file() and cache_meta_path.is_file():
        try:
            meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == signature:
                with np.load(cache_npz, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                meta["resumed"] = True
                return {"arrays": arrays, "meta": meta, "resumed": True}
        except Exception:
            pass

    ids = [x.id for x in identities]
    id_index = {x: i for i, x in enumerate(ids)}
    cam_seeds = [s for s in seeds if s.serial == cam.serial]
    seeds_at: Dict[int, List[Seed]] = {}
    for s in cam_seeds:
        seeds_at.setdefault(s.frame, []).append(s)
    seed_frames = sorted(seeds_at)
    first_seed = {x.id: min((s.frame for s in cam_seeds if s.identity == x.id), default=None) for x in identities}
    kp_enabled = {
        x.id: any(s.keypoints_enabled for s in cam_seeds if s.identity == x.id)
        for x in identities
    }

    F, K = len(frames), len(identities)
    boxes = np.full((F, K, 4), np.nan, dtype=np.float32)
    present = np.zeros((F, K), dtype=bool)
    scores = np.full((F, K), np.nan, dtype=np.float32)
    reasons = [["" for _ in range(K)] for _ in range(F)]
    sources = [["none" for _ in range(K)] for _ in range(F)]
    kp = np.full((F, K, 133, 2), np.nan, dtype=np.float32)
    kp_score = np.full((F, K, 133), np.nan, dtype=np.float32)
    kp_present = np.zeros((F, K), dtype=bool)
    pose_mismatch = 0

    cap = cv2.VideoCapture(str(cam.path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {cam.path}")
    current = -1
    last_box: Dict[int, np.ndarray] = {}
    last_frame: Dict[int, int] = {}
    velocity: Dict[int, np.ndarray] = {}
    active = set()
    overlay_every = max(1, int(args.overlay_every))
    last_status: Optional[Tuple[bool, ...]] = None
    overlay_frames = set(seed_frames) | {frames[0], frames[-1]}
    maps = rectify_maps(cam) if coord_space == UND_SPACE else None

    try:
        for fi, frame in enumerate(frames):
            bgr = None
            while current < frame:
                ok, got = cap.read()
                if not ok or got is None:
                    break
                current += 1
                bgr = got
            if bgr is None or current != frame:
                for k in range(K):
                    reasons[fi][k] = "decode_ended"
                continue
            if maps is not None:
                bgr = rectify_frame(cam, bgr, maps)

            pts = timestamp(cam, frame)
            dets: List[Detection] = []
            try:
                if isinstance(detector, ExternalDetections):
                    dets = detector.detect_for(cam.serial, frame)
                else:
                    dets = detector.detect(bgr)
            except Exception as exc:
                for k in range(K):
                    reasons[fi][k] = f"detector_error: {exc}"
                dets = []
            dets = [d for d in dets if d.box.size == 4 and np.isfinite(d.box).all() and d.box[2] > d.box[0] and d.box[3] > d.box[1]]
            used = set()
            seeded_now = set()

            for seed in seeds_at.get(frame, []):
                k = id_index[seed.identity]
                best_i, best_iou = None, 0.0
                for di, det in enumerate(dets):
                    if di in used:
                        continue
                    v = box_iou(seed.box, det.box)
                    if v > best_iou:
                        best_i, best_iou = di, v
                if best_i is not None and best_iou >= 0.5:
                    used.add(best_i)
                    b = dets[best_i].box
                    sc = dets[best_i].score
                    source = "seed+yolo" if dets[best_i].source == "yolo" else "seed+external"
                else:
                    b = seed.box
                    sc = seed.score
                    source = "seed_manual"
                boxes[fi, k] = b
                present[fi, k] = True
                scores[fi, k] = sc
                sources[fi][k] = source
                reasons[fi][k] = ""
                active.add(k)
                seeded_now.add(k)
                last_box[k] = np.asarray(b, dtype=np.float64)
                last_frame[k] = frame
                velocity[k] = np.zeros(2, dtype=np.float64)

            matchable = [
                k for k in range(K)
                if k not in seeded_now
                and k in active
                and k in last_box
                and (frame - last_frame.get(k, frame)) <= int(args.max_gap)
            ]
            remaining_dets = [di for di in range(len(dets)) if di not in used]
            if matchable and remaining_dets:
                cost = np.zeros((len(matchable), len(remaining_dets)), dtype=np.float64)
                for ri, k in enumerate(matchable):
                    gap = frame - last_frame.get(k, frame)
                    pred = predict_box(last_box[k], velocity.get(k, np.zeros(2)), gap, cam.width, cam.height)
                    for ci, di in enumerate(remaining_dets):
                        det = dets[di]
                        cost[ri, ci] = 1.0 - box_iou(pred, det.box) + 0.15 * center_norm(pred, det.box)
                for ri, ci in assign_cost(cost):
                    k = matchable[int(ri)]
                    det = dets[remaining_dets[int(ci)]]
                    gap = frame - last_frame.get(k, frame)
                    pred = predict_box(last_box[k], velocity.get(k, np.zeros(2)), gap, cam.width, cam.height)
                    iou = box_iou(pred, det.box)
                    cn = center_norm(pred, det.box)
                    if iou < float(args.iou_gate) and cn > float(args.center_gate):
                        reasons[fi][k] = "unmatched_after_gate"
                        velocity[k] = velocity.get(k, np.zeros(2)) * 0.5
                        continue
                    old_center = np.array([(last_box[k][0] + last_box[k][2]) / 2.0, (last_box[k][1] + last_box[k][3]) / 2.0])
                    new_center = np.array([(det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0])
                    velocity[k] = (new_center - old_center) / max(1, gap) * 0.5
                    boxes[fi, k] = det.box
                    present[fi, k] = True
                    scores[fi, k] = det.score
                    sources[fi][k] = det.source
                    reasons[fi][k] = ""
                    last_box[k] = np.asarray(det.box, dtype=np.float64)
                    last_frame[k] = frame
                    used.add(remaining_dets[int(ci)])

            unmatched_dets = [di for di in range(len(dets)) if di not in used]
            for k in range(K):
                if present[fi, k]:
                    continue
                if reasons[fi][k]:
                    continue
                if k not in active:
                    reasons[fi][k] = "before_first_seed"
                elif not unmatched_dets:
                    reasons[fi][k] = "no_detection"
                elif k not in matchable:
                    reasons[fi][k] = "gap_exceeds_max"
                else:
                    reasons[fi][k] = "unmatched_detection"

            # Per-instance keypoints only after a box exists and only when the
            # seed's visibility/keypoints_enabled flag permits it.
            if pose is not None:
                present_ks = [
                    k for k in range(K)
                    if present[fi, k] and kp_enabled.get(identities[k].id, False)
                ]
                if present_ks:
                    try:
                        kp_arr, ks_arr = pose.predict(bgr, [boxes[fi, k] for k in present_ks])
                        if kp_arr.shape[0] == len(present_ks):
                            for j, k in enumerate(present_ks):
                                kp[fi, k] = kp_arr[j]
                                kp_score[fi, k] = ks_arr[j]
                                kp_present[fi, k] = True
                        else:
                            pose_mismatch += 1
                    except Exception as exc:
                        for k in present_ks:
                            reasons[fi][k] = f"pose_error: {exc}"

            status = tuple(bool(present[fi, k]) for k in range(K))
            if (
                frame in overlay_frames
                or (frame % overlay_every == 0)
                or (last_status is not None and status != last_status)
            ):
                overlay_path = out_dir / "overlays" / f"{cam.serial}_f{frame:04d}.jpg"
                overlay_frame(
                    bgr,
                    cam,
                    frame,
                    pts,
                    identities,
                    boxes[fi],
                    present[fi],
                    scores[fi],
                    reasons[fi],
                    overlay_path,
                )
            last_status = status
    finally:
        cap.release()

    # Explicitly fill any frames not reached; no zero coordinates are valid.
    for fi in range(F):
        for k in range(K):
            if not reasons[fi][k] and not present[fi][k]:
                reasons[fi][k] = "not_processed"
            if not present[fi][k]:
                boxes[fi, k] = np.nan
                scores[fi, k] = np.nan
                kp[fi, k] = np.nan
                kp_score[fi, k] = np.nan

    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "boxes": boxes,
        "present": present,
        "scores": scores,
        "kp": kp,
        "kp_score": kp_score,
        "kp_present": kp_present,
    }
    meta = {
        "schema": "two_person_camera_tracks/v1",
        "signature": signature,
        "serial": cam.serial,
        "camera_index": cam.index,
        "width": cam.width,
        "height": cam.height,
        "frames": list(map(int, frames)),
        "identity_order": ids,
        "reasons": reasons,
        "sources": sources,
        "first_seed_frame": first_seed,
        "keypoints_enabled": kp_enabled,
        "seed_frames": seed_frames,
        "pose_model": pose.config() if pose else None,
        "pose_count_mismatch": pose_mismatch,
        "coordinate_space": coord_space,
        "resumed": False,
    }
    atomic_savez(cache_npz, **arrays)
    atomic_write_json(cache_meta_path, meta)
    return {"arrays": arrays, "meta": meta, "resumed": False}


# ---------------------------------------------------------------------------
# geometry diagnostics
# ---------------------------------------------------------------------------
def compute_geometry(
    cameras: Sequence[Camera],
    identities: Sequence[Identity],
    frames: Sequence[int],
    obs: Dict[str, Any],
    out_dir: Path,
    reproj_threshold: float,
    kp_score_threshold: float,
    coord_space: str,
    hand_contact_px: float = 80.0,
) -> Dict[str, Any]:
    id_count = len(identities)
    serial_to_ci = {c.serial: ci for ci, c in enumerate(cameras)}
    records: List[Dict[str, Any]] = []
    swap_flags: List[Dict[str, Any]] = []
    hand_records: List[Dict[str, Any]] = []
    per_identity_errors: Dict[str, List[float]] = {x.id: [] for x in identities}
    frame_both = 0

    for gi, frame in enumerate(frames):
        per_identity_views: Dict[int, List[Tuple[Camera, np.ndarray]]] = {k: [] for k in range(id_count)}
        for ci, cam in enumerate(cameras):
            if not bool(obs["present"][ci, gi].any()):
                continue
            for k in range(id_count):
                if not bool(obs["present"][ci, gi, k]) or not bool(obs["kp_present"][ci, gi, k]):
                    continue
                kp_raw = obs["kp"][ci, gi, k]
                ks = obs["kp_score"][ci, gi, k]
                valid = np.isfinite(kp_raw).all(axis=1) & np.isfinite(ks) & (ks >= kp_score_threshold)
                if valid.any():
                    kp_for_tri = (
                        kp_raw.astype(np.float64)
                        if coord_space == UND_SPACE
                        else undistort_keypoints(cam, kp_raw)
                    )
                    per_identity_views[k].append((cam, kp_for_tri))
        valid_views = [k for k in range(id_count) if len(per_identity_views[k]) >= 2]
        if len(valid_views) == id_count:
            frame_both += 1

        direct_costs: Dict[int, float] = {}
        for k in valid_views:
            errs: List[float] = []
            for j in BODY_JOINTS:
                Ps, uvs = [], []
                for cam, kp_und in per_identity_views[k]:
                    uv = kp_und[j]
                    if np.isfinite(uv).all():
                        Ps.append(cam.projection())
                        uvs.append(uv)
                if len(Ps) < 2:
                    continue
                X = triangulate_dlt(Ps, uvs)
                if X is None:
                    continue
                for P, uv in zip(Ps, uvs):
                    p = project_point(P, X)
                    if p is not None:
                        errs.append(float(np.linalg.norm(p - uv)))
            if errs:
                median = float(np.median(errs))
                direct_costs[k] = median
                per_identity_errors[identities[k].id].append(median)
                records.append(
                    {
                        "frame": int(frame),
                        "identity": identities[k].id,
                        "views": len(per_identity_views[k]),
                        "joints": len(errs),
                        "median_reprojection_px": median,
                        "high_reprojection": bool(median > reproj_threshold),
                    }
                )

        # Diagnostic only: compare direct and swapped identity labels using the
        # same per-view boxes/keypoints.  Never auto-relabel.
        if id_count == 2 and all(k in valid_views for k in range(2)):
            direct = [direct_costs.get(k) for k in range(2)]
            if all(x is not None for x in direct):
                swapped_cost = 0.0
                swapped_ok = True
                for k in range(2):
                    other = 1 - k
                    errs: List[float] = []
                    for j in BODY_JOINTS:
                        Ps, uvs = [], []
                        for cam, kp_und in per_identity_views[other]:
                            uv = kp_und[j]
                            if np.isfinite(uv).all():
                                Ps.append(cam.projection())
                                uvs.append(uv)
                        if len(Ps) < 2:
                            continue
                        X = triangulate_dlt(Ps, uvs)
                        if X is None:
                            continue
                        for P, uv in zip(Ps, uvs):
                            p = project_point(P, X)
                            if p is not None:
                                errs.append(float(np.linalg.norm(p - uv)))
                    if not errs:
                        swapped_ok = False
                        break
                    swapped_cost += float(np.median(errs))
                if swapped_ok:
                    direct_sum = float(sum(direct))
                    if swapped_cost + 2.0 < direct_sum:
                        swap_flags.append(
                            {
                                "frame": int(frame),
                                "direct_sum_px": direct_sum,
                                "swapped_sum_px": swapped_cost,
                                "note": "swapped labels fit geometry better; review seeds manually",
                            }
                        )

        # Shoulder/hand overlap diagnostic.  This only records evidence around
        # the handshake; it never adds a hand-contact loss or locks fingers.
        views_by_identity: Dict[int, Dict[str, np.ndarray]] = {k: {} for k in range(id_count)}
        for k in range(id_count):
            for cam, kp_und in per_identity_views[k]:
                views_by_identity[k][cam.serial] = kp_und
        per_camera_2d: Dict[str, float] = {}
        if id_count == 2:
            common = set(views_by_identity[0]) & set(views_by_identity[1])
            for serial in sorted(common):
                a = views_by_identity[0][serial][list(HAND_JOINTS)]
                b = views_by_identity[1][serial][list(HAND_JOINTS)]
                a = a[np.isfinite(a).all(axis=1)]
                b = b[np.isfinite(b).all(axis=1)]
                if a.size == 0 or b.size == 0:
                    continue
                d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
                per_camera_2d[serial] = float(d.min())
        wrist3d: Dict[int, List[np.ndarray]] = {k: [] for k in range(id_count)}
        for k in range(id_count):
            for j in (9, 10):
                Ps, uvs = [], []
                for cam, kp_und in per_identity_views[k]:
                    uv = kp_und[j]
                    if np.isfinite(uv).all():
                        Ps.append(cam.projection())
                        uvs.append(uv)
                if len(Ps) >= 2:
                    X = triangulate_dlt(Ps, uvs)
                    if X is not None:
                        wrist3d[k].append(X)
        min3d = None
        if id_count == 2 and len(wrist3d[0]) == 2 and len(wrist3d[1]) == 2:
            da = np.linalg.norm(np.asarray(wrist3d[0])[:, None, :] - np.asarray(wrist3d[1])[None, :, :], axis=2)
            min3d = float(da.min())
        min2d = min(per_camera_2d.values()) if per_camera_2d else None
        if min2d is not None or min3d is not None:
            hand_records.append(
                {
                    "frame": int(frame),
                    "camera_min_hand_2d_px": per_camera_2d,
                    "min_hand_2d_px": min2d,
                    "min_3d_wrist_distance": min3d,
                    "is_2d_contact_candidate": bool(
                        min2d is not None and min2d <= float(hand_contact_px)
                    ),
                }
            )

    flat_errors = [x for vals in per_identity_errors.values() for x in vals]
    summary = {
        "schema": "two_person_geometry_diagnostics/v1",
        "status": "ok" if records else "no_triangulable_keypoints",
        "joints_used": list(BODY_JOINTS),
        "reprojection_threshold_px": float(reproj_threshold),
        "keypoint_score_threshold": float(kp_score_threshold),
        "coordinate_space": coord_space,
        "frames": len(frames),
        "frames_with_both_identities_in_2plus_views": frame_both,
        "frame_identity_records": len(records),
        "median_reprojection_px_overall": float(np.median(flat_errors)) if flat_errors else None,
        "p95_reprojection_px_overall": float(np.percentile(flat_errors, 95)) if flat_errors else None,
        "per_identity": {
            ident.id: {
                "frames": len(per_identity_errors[ident.id]),
                "median_px": float(np.median(per_identity_errors[ident.id])) if per_identity_errors[ident.id] else None,
                "p95_px": float(np.percentile(per_identity_errors[ident.id], 95)) if per_identity_errors[ident.id] else None,
            }
            for ident in identities
        },
        "possible_identity_swap_frames": swap_flags[:50],
        "possible_identity_swap_count": len(swap_flags),
        "handshake": {
            "status": "diagnostic_only",
            "hand_joints": list(HAND_JOINTS),
            "contact_2d_threshold_px": float(hand_contact_px),
            "frames_with_2d_contact_candidate": int(
                sum(1 for r in hand_records if r["is_2d_contact_candidate"])
            ),
            "frames_with_3d_wrist_pair": int(
                sum(1 for r in hand_records if r["min_3d_wrist_distance"] is not None)
            ),
            "median_min_3d_wrist_distance": (
                float(np.median([r["min_3d_wrist_distance"] for r in hand_records if r["min_3d_wrist_distance"] is not None]))
                if any(r["min_3d_wrist_distance"] is not None for r in hand_records)
                else None
            ),
            "frames": hand_records[:80],
            "note": (
                "Overlap is recorded as evidence only. No hand point lock, no non-penetration "
                "constraint, and no identity decision is made from hand proximity."
            ),
        },
        "note": (
            "Geometry is a diagnostic of seed/label consistency, not a quality gate. "
            "A high value can come from keypoint noise, occlusion, or calibration, not only a swap."
        ),
    }
    atomic_write_json(out_dir / "geometry.json", summary)
    with (out_dir / "geometry_frames.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return summary


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_arrays(
    cameras: Sequence[Camera],
    identities: Sequence[Identity],
    frames: Sequence[int],
    obs: Dict[str, Any],
    geometry: Optional[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    id_count = len(identities)

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("two_identities", id_count == 2, [x.id for x in identities])
    check("camera_count", len(cameras) == 8, len(cameras))
    expected_shape = (len(cameras), len(frames), id_count)
    check("present_shape", tuple(obs["present"].shape) == expected_shape, list(obs["present"].shape))
    check("boxes_shape", tuple(obs["boxes"].shape) == expected_shape + (4,), list(obs["boxes"].shape))

    invalid_present = 0
    invalid_absent = 0
    for ci, cam in enumerate(cameras):
        for gi in range(len(frames)):
            for k in range(id_count):
                p = bool(obs["present"][ci, gi, k])
                b = obs["boxes"][ci, gi, k]
                if p:
                    if not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                        invalid_present += 1
                    elif b[0] < -1 or b[1] < -1 or b[2] > cam.width + 1 or b[3] > cam.height + 1:
                        invalid_present += 1
                else:
                    if np.isfinite(b).any():
                        invalid_absent += 1
                    if np.isfinite(obs["kp"][ci, gi, k]).any():
                        invalid_absent += 1
    check("present_boxes_valid", invalid_present == 0, invalid_present)
    check("missing_observations_are_nan", invalid_absent == 0, invalid_absent)

    pts_ok = True
    for cam in cameras:
        finite = cam.pts[np.isfinite(cam.pts)]
        if finite.size >= 2:
            if np.any(np.diff(finite) < 0):
                pts_ok = False
                warnings.append(f"{cam.serial}: decreasing PTS in timeline")
            if np.any(np.diff(finite) == 0):
                warnings.append(
                    f"{cam.serial}: duplicate/equal PTS retained; not treated as a dropped frame"
                )
    check("pts_monotonic_per_camera", pts_ok, "timeline ffprobe/PTS")

    calib_ok = True
    for cam in cameras:
        if not np.isfinite(cam.K).all() or not np.isfinite(cam.W).all():
            calib_ok = False
        elif abs(float(cam.W[3, 3]) - 1.0) > 1e-8:
            calib_ok = False
    check("calibration_finite", calib_ok, "K/W")

    coverage: Dict[str, Dict[str, int]] = {}
    for ident_i, ident in enumerate(identities):
        coverage[ident.id] = {}
        for ci, cam in enumerate(cameras):
            n = int(obs["present"][ci, :, ident_i].sum())
            coverage[ident.id][cam.serial] = n
    total_present = {ident.id: int(sum(v.values())) for ident, v in zip(identities, coverage.values())}
    check("both_identities_have_observations", all(v > 0 for v in total_present.values()), total_present)

    both_in_two_views = 0
    for gi in range(len(frames)):
        for ci in range(len(cameras)):
            if bool(obs["present"][ci, gi].all()):
                both_in_two_views += 1
                break
    check(
        "at_least_one_frame_has_both_identities",
        both_in_two_views > 0,
        {"frames": both_in_two_views},
    )
    if both_in_two_views == 0:
        warnings.append(
            "no frame has both identities present in the same camera; two-person interaction is not evidenced yet"
        )

    if geometry:
        if geometry.get("status") != "ok":
            warnings.append(
                "geometry diagnostics did not produce triangulable keypoints; "
                "per-instance RTMW or keypoint confidence is missing/low"
            )
        if geometry.get("possible_identity_swap_count", 0):
            warnings.append(
                f"geometry flags {geometry['possible_identity_swap_count']} possible identity-swap frames; review seeds"
            )
        if (geometry.get("median_reprojection_px_overall") or 0) > 30:
            warnings.append(
                "median triangulation reprojection is above 30 px; inspect calibration/keypoints before fitting"
            )
    else:
        warnings.append("geometry diagnostics not run (no per-instance keypoints)")

    structural_ok = all(c["ok"] for c in checks)
    result = {
        "schema": "two_person_instance_validation/v1",
        "generated_at_utc": utc_now(),
        "status": "passed" if structural_ok else "failed",
        "checks": checks,
        "coverage_present_frames": coverage,
        "total_present": total_present,
        "warnings": warnings,
        "quality_thresholds_used": False,
        "note": "These are structural/numeric checks, not a reconstruction-quality acceptance test.",
    }
    atomic_write_json(out_dir / "validation.json", result)
    return result


# ---------------------------------------------------------------------------
# candidate export
# ---------------------------------------------------------------------------
def command_candidates(args: argparse.Namespace) -> int:
    scene, cameras, scene_warnings = load_scene(Path(args.scene))
    if not cameras:
        raise RuntimeError("no calibrated cameras in scene")
    detector = build_detector(args)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_spec = args.frames
    payload: Dict[str, Any] = {
        "schema": "two_person_seed_candidates/v1",
        "generated_at_utc": utc_now(),
        "scene": {"path": str(Path(args.scene).resolve()), "sha256": sha256_file(Path(args.scene))},
        "detector": detector.config(),
        "frame_spec": frame_spec,
        "views": [],
        "warnings": list(scene_warnings),
    }
    for cam in cameras:
        frames = parse_frame_spec(frame_spec, cam.frame_count)
        cap = cv2.VideoCapture(str(cam.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {cam.path}")
        current = -1
        view_frames: List[Dict[str, Any]] = []
        try:
            for frame in frames:
                bgr = None
                while current < frame:
                    ok, got = cap.read()
                    if not ok or got is None:
                        break
                    current += 1
                    bgr = got
                if bgr is None or current != frame:
                    continue
                try:
                    if isinstance(detector, ExternalDetections):
                        dets = detector.detect_for(cam.serial, frame)
                    else:
                        dets = detector.detect(bgr)
                except Exception as exc:
                    view_frames.append({"frame": frame, "error": str(exc), "candidates": []})
                    continue
                dets = sorted(
                    [d for d in dets if d.box.size == 4 and np.isfinite(d.box).all()],
                    key=lambda d: (float(d.box[0]), float(d.box[1])),
                )
                for i, d in enumerate(dets):
                    d.candidate_id = f"cand_{i:02d}"
                overlay_path = out_dir / "seed_candidates" / "overlays" / f"{cam.serial}_f{frame:04d}.jpg"
                overlay_candidates(bgr, cam, frame, timestamp(cam, frame), dets, overlay_path)
                view_frames.append(
                    {
                        "frame": int(frame),
                        "pts_sec": timestamp(cam, frame),
                        "n_candidates": len(dets),
                        "candidates": [
                            {
                                "candidate_id": d.candidate_id,
                                "box": [round(float(x), 3) for x in d.box.tolist()],
                                "score": round(float(d.score), 5),
                            }
                            for d in dets
                        ],
                        "overlay": str(overlay_path),
                    }
                )
        finally:
            cap.release()
        payload["views"].append(
            {
                "camera_index": cam.index,
                "serial": cam.serial,
                "width": cam.width,
                "height": cam.height,
                "frames": view_frames,
            }
        )
        print(f"[candidates] cam{cam.index} {cam.serial}: {len(view_frames)} frame(s)", flush=True)

    atomic_write_json(out_dir / "seed_candidates" / "candidates.json", payload)
    template = {
        "schema": "two_person_identity_seeds/v1",
        "session": scene.get("session", {}).get("name"),
        "identities": [
            {"id": "person_a", "display_name": "REPLACE_WITH_VISUAL_ID", "notes": ""},
            {"id": "person_b", "display_name": "REPLACE_WITH_VISUAL_ID", "notes": ""},
        ],
        "views": [],
        "instructions": (
            "This is a BLANK template. Codex must inspect the local numbered overlays, "
            "assign each chosen box an identity (person_a or person_b), delete boxes that are "
            "not the target people, and save as seeds.json. The runner never guesses identities."
        ),
    }
    for view in payload["views"]:
        persons = []
        for fr in view["frames"]:
            for cand in fr.get("candidates", []):
                persons.append(
                    {
                        "identity": None,
                        "frame": fr["frame"],
                        "box": cand["box"],
                        "score": cand["score"],
                        "source": f"candidate:{cand['candidate_id']}",
                    }
                )
        if persons:
            template["views"].append(
                {
                    "serial": view["serial"],
                    "frame": persons[0]["frame"],
                    "persons": persons,
                }
            )
    atomic_write_json(out_dir / "seed_candidates" / "seed_template.json", template)
    print(f"[candidates] wrote {out_dir / 'seed_candidates' / 'candidates.json'}")
    print(f"[candidates] blank label template: {out_dir / 'seed_candidates' / 'seed_template.json'}")
    return 0


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def command_prepare(args: argparse.Namespace) -> int:
    scene_path = Path(args.scene).resolve()
    seeds_path = Path(args.seeds).resolve()
    out_dir = Path(args.out).resolve()
    scene, cameras, scene_warnings = load_scene(scene_path)
    if len(cameras) != 8:
        raise RuntimeError(f"need 8 calibrated cameras, scene provides {len(cameras)}")
    seed_set = load_seeds(seeds_path, cameras)
    detector = build_detector(args)
    pose = WholebodyPose(Path(args.rtmw)) if args.rtmw else None
    identities = seed_set.identities
    if len(identities) != 2:
        raise RuntimeError("this runner is intentionally two-person only")

    coord = args.coordinate_space
    detector_space = None
    if isinstance(detector, ExternalDetections):
        detector_space = detector.coordinate_space
        if coord == "auto":
            coord = "undistorted" if detector_space.startswith("undistorted") else "raw"
        else:
            detector_und = detector_space.startswith("undistorted")
            if (coord == "undistorted") != detector_und:
                raise ValueError(
                    f"external boxes are in {detector_space}, but --coordinate-space={coord}"
                )
    elif coord == "auto":
        coord = "raw"
    coord_space = UND_SPACE if coord == "undistorted" else RAW_SPACE

    # Seeds are authored in raw distorted coordinates unless they explicitly
    # declare otherwise. The SAM2 export declares undistorted coordinates, so
    # infer it rather than silently rectifying twice.
    declared = seed_set.facts.get("coordinate_space")
    seed_coord = args.seed_coordinate_space
    if declared:
        declared_space = "undistorted" if str(declared).startswith("undistorted") else "raw"
        if args.seed_coordinate_space == "raw" and declared_space == "undistorted":
            seed_coord = "undistorted"
        elif args.seed_coordinate_space != declared_space:
            raise ValueError(
                f"seeds declare {declared}, but --seed-coordinate-space={args.seed_coordinate_space}"
            )

    # Convert seeds exactly once when observation frames are rectified, so
    # seed boxes, detector boxes, RTMW keypoints and overlays share one space.
    if coord_space == UND_SPACE:
        if seed_coord == "raw":
            cam_by_serial = {c.serial: c for c in cameras}
            for seed in seed_set.seeds:
                cam = cam_by_serial.get(seed.serial)
                if cam is None:
                    raise ValueError(f"seed serial not in scene: {seed.serial}")
                seed.box = rectify_box(cam, seed.box)
            seed_set.facts["coordinate_space"] = UND_SPACE
            seed_set.facts["seed_coordinate_input"] = RAW_SPACE
        else:
            seed_set.facts["coordinate_space"] = UND_SPACE
            seed_set.facts["seed_coordinate_input"] = UND_SPACE
    else:
        if seed_coord == "undistorted":
            raise ValueError(
                "undistorted seeds cannot be used with raw observation frames; "
                "use --coordinate-space undistorted"
            )
        seed_set.facts["coordinate_space"] = RAW_SPACE
        seed_set.facts["seed_coordinate_input"] = RAW_SPACE

    signature_payload = {
        "scene_sha256": sha256_file(scene_path),
        "seeds_sha256": seed_set.facts.get("sha256"),
        "detector": detector.config(),
        "pose": pose.config() if pose else None,
        "frames": args.frames,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "iou_gate": args.iou_gate,
        "center_gate": args.center_gate,
        "max_gap": args.max_gap,
        "rtmw": str(args.rtmw) if args.rtmw else None,
        "coordinate_space": coord_space,
        "detector_coordinate_space": detector_space,
        "seed_coordinate_space": seed_coord,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=json_default).encode("utf-8")
    ).hexdigest()
    atomic_write_json(out_dir / "run_signature.json", {"signature": signature, "inputs": signature_payload})

    # Global frame index union; per-camera ranges are allowed to differ.
    frames_by_cam: Dict[str, List[int]] = {}
    for cam in cameras:
        frames_by_cam[cam.serial] = parse_frame_spec(args.frames, cam.frame_count)
    global_frames = sorted({f for v in frames_by_cam.values() for f in v})
    if not global_frames:
        raise RuntimeError("frame selection is empty")
    frame_to_g = {f: i for i, f in enumerate(global_frames)}
    C, G, K = len(cameras), len(global_frames), len(identities)
    boxes = np.full((C, G, K, 4), np.nan, dtype=np.float32)
    present = np.zeros((C, G, K), dtype=bool)
    scores = np.full((C, G, K), np.nan, dtype=np.float32)
    kp = np.full((C, G, K, 133, 2), np.nan, dtype=np.float32)
    kp_score = np.full((C, G, K, 133), np.nan, dtype=np.float32)
    kp_present = np.zeros((C, G, K), dtype=bool)
    reason_vocab: List[str] = [""]
    source_vocab: List[str] = ["none"]
    reason_codes = np.zeros((C, G, K), dtype=np.int16)
    source_codes = np.zeros((C, G, K), dtype=np.int8)
    frame_present = np.zeros((C, G), dtype=bool)
    pts_out = np.full((C, G), np.nan, dtype=np.float64)
    camera_results = []

    for ci, cam in enumerate(cameras):
        frames = frames_by_cam[cam.serial]
        cam_seed_counts = {
            ident.id: sum(
                1 for s in seed_set.seeds
                if s.serial == cam.serial and s.identity == ident.id
            )
            for ident in identities
        }
        for ident_id, count in cam_seed_counts.items():
            if count == 0:
                scene_warnings.append(
                    f"{cam.serial}: no seed for {ident_id}; that camera's track for this identity "
                    "will be present=false until a seed is supplied (cross-view bootstrap is not implemented)"
                )
        print(f"[prepare] cam{ci} {cam.serial}: {len(frames)} selected frames", flush=True)
        result = process_camera(
            cam, identities, seed_set.seeds, detector, pose, frames, coord_space,
            args, out_dir, signature
        )
        arrays, meta = result["arrays"], result["meta"]
        cam_results_frames = [int(x) for x in arrays["frames"].tolist()]
        for fi, f in enumerate(cam_results_frames):
            gi = frame_to_g.get(f)
            if gi is None:
                continue
            frame_present[ci, gi] = True
            pts_out[ci, gi] = timestamp(cam, f) if timestamp(cam, f) is not None else np.nan
            boxes[ci, gi] = arrays["boxes"][fi]
            present[ci, gi] = arrays["present"][fi]
            scores[ci, gi] = arrays["scores"][fi]
            kp[ci, gi] = arrays["kp"][fi]
            kp_score[ci, gi] = arrays["kp_score"][fi]
            kp_present[ci, gi] = arrays["kp_present"][fi]
            for k in range(K):
                reason = str(meta["reasons"][fi][k])
                if reason not in reason_vocab:
                    reason_vocab.append(reason)
                reason_codes[ci, gi, k] = reason_vocab.index(reason)
                source = str(meta["sources"][fi][k])
                if source not in source_vocab:
                    source_vocab.append(source)
                source_codes[ci, gi, k] = source_vocab.index(source)
        camera_results.append(
            {
                "serial": cam.serial,
                "camera_index": cam.index,
                "selected_frames": len(frames),
                "seed_counts": cam_seed_counts,
                "present_counts": {
                    identities[k].id: int(arrays["present"][:, k].sum()) for k in range(K)
                },
                "resumed": result["resumed"],
                "pose_count_mismatch": meta.get("pose_count_mismatch"),
                "keypoints_enabled": meta.get("keypoints_enabled"),
            }
        )

    obs = {
        "present": present,
        "boxes": boxes,
        "scores": scores,
        "kp": kp,
        "kp_score": kp_score,
        "kp_present": kp_present,
    }
    atomic_savez(
        out_dir / "observations.npz",
        frames=np.asarray(global_frames, dtype=np.int32),
        serials=np.asarray([c.serial for c in cameras], dtype="U32"),
        identities=np.asarray([x.id for x in identities], dtype="U32"),
        present=present,
        boxes=boxes,
        scores=scores,
        kp=kp,
        kp_score=kp_score,
        kp_present=kp_present,
        frame_present=frame_present,
        pts_sec=pts_out,
        reason_codes=reason_codes,
        reason_vocab=np.asarray(reason_vocab, dtype="U160"),
        source_codes=source_codes,
        source_vocab=np.asarray(source_vocab, dtype="U32"),
    )

    for ci, cam in enumerate(cameras):
        track = {
            "schema": "two_person_track/v1",
            "serial": cam.serial,
            "camera_index": cam.index,
            "identity_order": [x.id for x in identities],
            "frames": [],
        }
        for gi, frame in enumerate(global_frames):
            if not frame_present[ci, gi]:
                continue
            entry = {"frame": int(frame), "pts_sec": float(pts_out[ci, gi]) if np.isfinite(pts_out[ci, gi]) else None, "persons": {}}
            for k, ident in enumerate(identities):
                p = bool(present[ci, gi, k])
                entry["persons"][ident.id] = {
                    "present": p,
                    "box": [round(float(x), 3) for x in boxes[ci, gi, k].tolist()] if p else None,
                    "score": round(float(scores[ci, gi, k]), 5) if p and np.isfinite(scores[ci, gi, k]) else None,
                    "source": str(source_vocab[int(source_codes[ci, gi, k])]),
                    "reason": str(reason_vocab[int(reason_codes[ci, gi, k])]),
                }
            track["frames"].append(entry)
        atomic_write_json(out_dir / "tracks" / f"{cam.serial}.json", track)

    geometry = compute_geometry(
        cameras,
        identities,
        global_frames,
        obs,
        out_dir,
        reproj_threshold=float(args.reproj_threshold),
        kp_score_threshold=float(args.kp_score_threshold),
        coord_space=coord_space,
        hand_contact_px=float(args.hand_contact_px),
    )

    validation = validate_arrays(cameras, identities, global_frames, obs, geometry, out_dir)
    total_present = {
        ident.id: int(present[:, :, k].sum()) for k, ident in enumerate(identities)
    }
    index = {
        "schema": "two_person_instance_inputs/v1",
        "generated_at_utc": utc_now(),
        "status": "prepared" if validation["status"] == "passed" else "prepared_with_errors",
        "scene": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "seeds": seed_set.facts,
        "detector": detector.config(),
        "pose": pose.config() if pose else None,
        "frame_spec": args.frames,
        "frames": [int(x) for x in global_frames],
        "frame_count": len(global_frames),
        "first_frame": int(global_frames[0]),
        "last_frame": int(global_frames[-1]),
        "coordinate_space": coord_space,
        "detector_coordinate_space": detector_space,
        "seed_coordinate_input": seed_set.facts.get("seed_coordinate_input"),
        "time_index_note": (
            "Frame indices and per-camera scene PTS are retained. Equal indices/PTS are a "
            "provisional correspondence, not proof of hardware synchronization."
        ),
        "identities": [x.as_json() for x in identities],
        "cameras": [c.as_json() for c in cameras],
        "present_counts": total_present,
        "camera_results": camera_results,
        "geometry": {
            "status": geometry.get("status"),
            "median_reprojection_px_overall": geometry.get("median_reprojection_px_overall"),
            "possible_identity_swap_count": geometry.get("possible_identity_swap_count"),
            "file": str(out_dir / "geometry.json"),
        },
        "validation": {
            "status": validation["status"],
            "file": str(out_dir / "validation.json"),
            "warnings": validation["warnings"],
        },
        "observations_npz": str(out_dir / "observations.npz"),
        "overlays_dir": str(out_dir / "overlays"),
        "warnings": scene_warnings,
        "not_claimed": [
            "This is not a completed two-person reconstruction.",
            "Identity labels are those supplied visually by Codex; geometry only flags possible swaps.",
            "Missing/occluded observations are present=false with NaN, never valid zero scores.",
            "No SMPL-X fit, motion, avatar reconstruction, or render was run by this CPU preparation step.",
        ],
    }
    atomic_write_json(out_dir / "index.json", index)
    print(f"[prepare] wrote {out_dir / 'index.json'}")
    print(f"[prepare] present counts: {total_present}")
    print(f"[prepare] validation: {validation['status']}")
    print(f"[prepare] geometry: {geometry.get('status')}")
    if validation["warnings"]:
        print("[prepare] warnings:")
        for w in validation["warnings"]:
            print("  -", w)
    return 0 if validation["status"] == "passed" else 2


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------
def command_validate(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    index = json.loads((run / "index.json").read_text(encoding="utf-8"))
    scene, cameras, _ = load_scene(Path(args.scene).resolve())
    identities = [Identity(x["id"], x.get("display_name", ""), x.get("notes", "")) for x in index["identities"]]
    with np.load(run / "observations.npz", allow_pickle=False) as z:
        frames = z["frames"].astype(int).tolist()
        obs = {
            "present": z["present"],
            "boxes": z["boxes"],
            "scores": z["scores"],
            "kp": z["kp"],
            "kp_score": z["kp_score"],
            "kp_present": z["kp_present"],
        }
    geometry_path = run / "geometry.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.is_file() else None
    result = validate_arrays(cameras, identities, frames, obs, geometry, run)
    print(f"[validate] {result['status']}")
    for c in result["checks"]:
        print(f"  {'OK ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
    for w in result["warnings"]:
        print("  warning:", w)
    return 0 if result["status"] == "passed" else 2


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def find_model_candidates(root: Path, max_depth: int = 2) -> List[Dict[str, Any]]:
    bases = [
        root / "runtime" / "wholebody",
        root / "runtime" / "weights",
        root / "runtime" / "deps",
        root / "lhmpp" / "pretrained_models",
        root / "lhmpp" / "weights",
        root / "lhmpp" / "deps",
    ]
    keywords = ("yolo", "person", "det", "sam2", "rtmw")
    out: List[Dict[str, Any]] = []
    for base in bases:
        if not base.is_dir():
            continue
        stack = [(base, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                continue
            for p in entries:
                try:
                    if p.is_dir():
                        if depth < max_depth:
                            stack.append((p, depth + 1))
                        continue
                except OSError:
                    continue
                if p.suffix.lower() not in (".pt", ".pth", ".onnx", ".safetensors"):
                    continue
                if any(k in p.name.lower() for k in keywords):
                    try:
                        out.append(
                            {
                                "path": str(p),
                                "size": int(p.stat().st_size),
                                "name": p.name,
                            }
                        )
                    except OSError:
                        pass
    out.sort(key=lambda x: x["path"])
    return out


def command_preflight(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "schema": "two_person_instance_preflight/v1",
        "generated_at_utc": utc_now(),
        "scene": None,
        "seeds": None,
        "dependencies": {},
        "model_files": find_model_candidates(PROJECT_ROOT),
        "detector": {"requested": args.detector},
        "pose": {"requested_rtmw": str(args.rtmw) if args.rtmw else None},
        "blockers": [],
        "warnings": [],
        "commands": {},
    }
    for module, dist in (
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("rtmlib", "rtmlib"),
        ("onnxruntime", "onnxruntime"),
        ("ultralytics", "ultralytics"),
        ("scipy", "scipy"),
        ("sam2", "sam2"),
        ("torch", "torch"),
        ("smplx", "smplx"),
        ("pytorch3d", "pytorch3d"),
    ):
        report["dependencies"][module] = module_status(module, dist)
    report["dependencies"]["ffprobe"] = {"path": shutil.which("ffprobe"), "present": bool(shutil.which("ffprobe"))}

    if args.scene:
        sp = Path(args.scene).resolve()
        if sp.is_file():
            scene, cameras, warnings = load_scene(sp)
            report["scene"] = {
                "path": str(sp),
                "sha256": sha256_file(sp),
                "cameras": len(cameras),
                "serials": [c.serial for c in cameras],
                "frame_counts": {c.serial: c.frame_count for c in cameras},
                "warnings": warnings,
            }
            report["warnings"].extend(warnings)
        else:
            report["blockers"].append(f"scene not found: {sp}")
    else:
        report["warnings"].append("no --scene supplied; scene checks skipped")

    if args.seeds:
        sp = Path(args.seeds).resolve()
        if not sp.is_file():
            report["blockers"].append(f"seeds not found: {sp}")
        elif report["scene"] and report["scene"].get("cameras"):
            try:
                _, cameras, _ = load_scene(Path(args.scene).resolve())
                ss = load_seeds(sp, cameras)
                report["seeds"] = ss.facts
            except Exception as exc:
                report["blockers"].append(f"seed validation failed: {exc}")
        else:
            report["seeds"] = {"path": str(sp), "sha256": sha256_file(sp), "validated": False}
    else:
        report["warnings"].append("no --seeds supplied; identity-specific tracking cannot run")

    if args.detector == "yolo":
        if not args.weights:
            report["blockers"].append("--detector yolo requires local --weights")
        else:
            wp = Path(args.weights)
            report["detector"].update(
                {
                    "weights": str(wp),
                    "weights_exists": wp.is_file(),
                    "weights_size": int(wp.stat().st_size) if wp.is_file() else None,
                    "ultralytics_present": report["dependencies"]["ultralytics"]["present"],
                }
            )
            if not wp.is_file():
                report["blockers"].append(f"YOLO weights not found: {wp}")
            if not report["dependencies"]["ultralytics"]["present"]:
                report["blockers"].append("ultralytics module is not installed")
    elif args.detector == "external":
        if not args.detections:
            report["blockers"].append("--detector external requires --detections")
        else:
            dp = Path(args.detections)
            report["detector"]["detections"] = str(dp)
            if not dp.is_file():
                report["blockers"].append(f"detections file not found: {dp}")

    rtmw = Path(args.rtmw) if args.rtmw else PROJECT_ROOT / "runtime" / "wholebody" / "rtmw.onnx"
    report["pose"].update(
        {
            "path": str(rtmw),
            "exists": rtmw.is_file(),
            "size": int(rtmw.stat().st_size) if rtmw.is_file() else None,
            "rtmlib_present": report["dependencies"]["rtmlib"]["present"],
            "note": "RTMW is top-down; it is only called with per-instance boxes, never full-frame.",
        }
    )
    if args.rtmw and not rtmw.is_file():
        report["blockers"].append(f"RTMW model not found: {rtmw}")
    if not rtmw.is_file():
        report["warnings"].append(
            "RTMW wholebody model not found at the checked path; tracking can run, but "
            "per-instance keypoints/geometry diagnostics will be unavailable"
        )
    if rtmw.is_file() and not report["dependencies"]["rtmlib"]["present"]:
        report["blockers"].append("rtmlib module is not installed; keypoints cannot be extracted")

    if args.sam2_weights:
        sam2 = Path(args.sam2_weights)
        report["sam2"] = {
            "weights": str(sam2),
            "exists": sam2.is_file(),
            "sam2_module_present": report["dependencies"]["sam2"]["present"],
            "note": "SAM2 is optional mask refinement for later; this runner does not invoke it.",
        }
        if not sam2.is_file():
            report["warnings"].append(f"SAM2 weights not found: {sam2}")

    py = sys.executable
    script = Path(__file__).resolve()
    base = (
        f'"{py}" "{script}" prepare --scene "{Path(args.scene).resolve() if args.scene else "<scene.json>"}" '
        f'--seeds "<seeds.json>" --detector {args.detector} --frames 0:60:1 --out data/instance_inputs/run'
    )
    if args.detector == "yolo":
        base += f' --weights "{args.weights or "<local-yolo.pt>"}"'
    if rtmw.is_file():
        base += f' --rtmw "{rtmw}"'
    report["commands"]["prepare"] = base
    report["next_required_inputs"] = [
        "Codex-labelled seeds.json with exactly two global identities",
        "local person-detector weights (or an external per-frame boxes JSON)",
        "RTMW wholebody ONNX for per-instance 133 keypoints (optional for tracking)",
        "verified frame interval (0:END exclusive) around the real interaction",
    ]
    atomic_write_json(out_dir / "preflight.json", report)
    print(f"[preflight] wrote {out_dir / 'preflight.json'}")
    for b in report["blockers"]:
        print("[preflight] BLOCKER:", b)
    for w in report["warnings"]:
        print("[preflight] warning:", w)
    return 0 if not report["blockers"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def command_observe_hands(args: argparse.Namespace) -> int:
    """Per-view HaMeR hand rotations for one person's full/source-frame subset."""
    import importlib.util

    def _resolve(base: Path, value: Any) -> Path:
        p = Path(str(value))
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    person_path = Path(args.person).resolve()
    if person_path.is_dir():
        person_path = person_path / "person.json"
    person_info = json.loads(person_path.read_text(encoding="utf-8"))
    identity = str(person_info.get("identity") or "")
    if not identity:
        raise ValueError("person.json lacks identity")
    person_frames = [int(x) for x in person_info["frames"]]
    if not person_frames:
        raise ValueError("person.frames is empty")
    if args.frames is None:
        selected_frames = list(person_frames)
        frames_source = "person.frames"
    else:
        requested = {int(x) for x in args.frames}
        missing = sorted(requested - set(person_frames))
        if missing:
            raise ValueError(f"--frames contains frames absent from person.frames: {missing}")
        selected_frames = [f for f in person_frames if f in requested]
        frames_source = "explicit --frames subset of person.frames"
    if not selected_frames:
        raise ValueError("no selected person frames")

    scene_index_path = _resolve(THIS_DIR, args.scene_index)
    if scene_index_path.is_dir():
        scene_index_path = scene_index_path / "index.json"
    scene_index = json.loads(scene_index_path.read_text(encoding="utf-8"))
    scene_map: Dict[str, Dict[int, Path]] = {}
    scene_order = []
    for view in scene_index.get("views") or scene_index.get("cameras") or []:
        serial = str(view.get("serial") or view.get("camera") or "")
        if not serial:
            raise ValueError("scene index view lacks serial")
        scene_order.append(serial)
        frames_map: Dict[int, Path] = {}
        for item in view.get("frames") or []:
            frame = int(item["source_frame"])
            jpeg = _resolve(scene_index_path.parent, item["jpeg"])
            if frame in frames_map:
                raise ValueError(f"duplicate scene frame {serial}/{frame}")
            frames_map[frame] = jpeg
        scene_map[serial] = frames_map

    manifest_path = _resolve(person_path.parent, person_info["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest.get("views") or []
    if not views:
        raise ValueError("person manifest has no views")
    serials = [str(v["serial"]) for v in views]
    if len(set(serials)) != len(serials):
        raise ValueError("manifest view serials are not unique")
    for serial in serials:
        if serial not in scene_map:
            raise ValueError(f"scene index lacks manifest camera {serial}")

    keypoint_path = _resolve(person_path.parent, person_info["keypoints"])
    kp_data = {}
    for view in views:
        serial = str(view["serial"])
        kp_path = keypoint_path / f"{serial}.npz"
        with np.load(kp_path, allow_pickle=False) as z:
            xy = np.asarray(z["xy"], np.float32)
            score = np.asarray(z["score"], np.float32)
            in_image = np.asarray(z["in_image"], bool)
            if xy.ndim != 3 or xy.shape[1:] != (133, 2):
                raise ValueError(f"{kp_path}: xy must be T,133,2, got {xy.shape}")
            if score.shape != xy.shape[:2]:
                raise ValueError(f"{kp_path}: score shape {score.shape} vs xy {xy.shape}")
            if "frames" in z.files:
                kp_frames = [int(x) for x in np.asarray(z["frames"]).tolist()]
            else:
                if xy.shape[0] != len(person_frames):
                    raise ValueError(f"{kp_path}: no frames and T != person.frames")
                kp_frames = list(person_frames)
            kp_index = {f: i for i, f in enumerate(kp_frames)}
            missing_kp = [f for f in selected_frames if f not in kp_index]
            if missing_kp:
                raise ValueError(f"{kp_path}: missing selected frames {missing_kp[:5]}")
            take = [kp_index[f] for f in selected_frames]
            kp_data[serial] = {
                "path": kp_path,
                "sha256": sha256_file(kp_path),
                "xy": xy[take],
                "score": score[take],
                "in_image": in_image[take],
            }

    jpegs: Dict[str, Dict[int, Path]] = {}
    for serial in serials:
        jpegs[serial] = {}
        for frame in selected_frames:
            path = scene_map[serial].get(int(frame))
            if path is None:
                raise ValueError(f"scene index lacks JPEG for {serial}/frame {frame}")
            jpegs[serial][frame] = path

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    parent_dir = THIS_DIR.parent.resolve()
    parent_file = parent_dir / "observe_hand_surface.py"
    if not parent_file.is_file():
        raise FileNotFoundError(parent_file)
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    spec = importlib.util.spec_from_file_location("_mjj_observe_hand_surface", parent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load observe_hand_surface.py")
    hand_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hand_module
    spec.loader.exec_module(hand_module)

    model, cfg, root = hand_module.load()
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    import torch

    model.cuda()
    model.eval().requires_grad_(False)
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device="cuda"))

    V, T = len(serials), len(selected_frames)
    rotations = np.full((V, T, 2, 15, 3, 3), np.nan, np.float32)
    valid = np.zeros((V, T, 2), bool)
    crop_boxes = np.full((V, T, 2, 4), np.nan, np.float32)
    completed_views = np.zeros(V, bool)
    cache_source = {
        "identity": identity,
        "person_sha256": sha256_file(person_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scene_index_sha256": sha256_file(scene_index_path),
        "keypoints": {s: item["sha256"] for s, item in kp_data.items()},
        "model_source_sha256": sha256_file(root / "source.json"),
        "rescale_factor": 2.5,
        "prediction_contract": "official_HaMeR_BGR_left_SRS_local15/v1",
    }
    cache_signature = hashlib.sha256(
        json.dumps(cache_source, sort_keys=True).encode("utf-8")).hexdigest()
    cached = None
    cache_path = out / "observations.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            old = {k: z[k] for k in z.files}
        compatible = str(old.get("source_signature", "")) == cache_signature
        # Migrate the original complete observation file only when all model
        # and input hashes match.  Each individual crop must also match below.
        old_prov_path = out / "provenance.json"
        if not compatible and "source_signature" not in old and old_prov_path.is_file():
            old_prov = json.loads(old_prov_path.read_text(encoding="utf-8"))
            compatible = all(old_prov.get(k) == cache_source[k] for k in (
                "identity", "person_sha256", "manifest_sha256",
                "scene_index_sha256", "model_source_sha256"))
            compatible = compatible and old_prov.get("preprocessing", {}).get("rescale_factor") == 2.5
            compatible = compatible and all(
                old_prov.get("keypoints", {}).get(s, {}).get("sha256") == item["sha256"]
                for s, item in kp_data.items())
        if compatible:
            cached = old
    cached_views = {} if cached is None else {
        str(s): i for i, s in enumerate(cached["serials"])}
    cached_frames = {} if cached is None else {
        int(f): i for i, f in enumerate(cached["frames"])}

    def save_observations():
        atomic_savez(
            cache_path, rotations=rotations,
            frames=np.asarray(selected_frames, np.int64),
            serials=np.asarray(serials, dtype="U32"),
            side=np.asarray([0, 1], np.int32), valid=valid,
            crop_boxes=crop_boxes, completed_views=completed_views,
            source_signature=np.asarray(cache_signature))

    started = time.time()
    prediction_count = 0
    reused_count = 0
    with torch.inference_mode():
        for vi, serial in enumerate(serials):
            kp_xy = kp_data[serial]["xy"]
            kp_score = kp_data[serial]["score"]
            kp_in_image = kp_data[serial]["in_image"]
            for ti, frame in enumerate(selected_frames):
                boxes, sides = [], []
                for side in range(2):
                    ids = np.arange(91 + 21 * side, 112 + 21 * side)
                    good = (kp_score[ti, ids] > 0.5) & np.isfinite(kp_xy[ti, ids]).all(-1)
                    good &= kp_in_image[ti, ids]
                    if int(good.sum()) <= 3:
                        continue
                    xy = kp_xy[ti, ids][good]
                    box = np.r_[xy.min(0), xy.max(0)].astype(np.float32)
                    oi, ot = cached_views.get(serial), cached_frames.get(int(frame))
                    if (oi is not None and ot is not None and cached["valid"][oi, ot, side]
                            and np.array_equal(cached["crop_boxes"][oi, ot, side], box)):
                        rotations[vi, ti, side] = cached["rotations"][oi, ot, side]
                        crop_boxes[vi, ti, side] = box
                        valid[vi, ti, side] = True
                        reused_count += 1
                        continue
                    boxes.append(box)
                    sides.append(side)
                if not boxes:
                    continue
                raw = cv2.imdecode(np.fromfile(jpegs[serial][frame], np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    raise FileNotFoundError(jpegs[serial][frame])
                dataset = ViTDetDataset(
                    cfg,
                    raw,
                    np.asarray(boxes, dtype=np.float32),
                    np.asarray(sides, dtype=np.int64),
                    rescale_factor=2.5,
                )
                for i, side in enumerate(sides):
                    item = dataset[i]
                    x = torch.as_tensor(item["img"], device="cuda", dtype=torch.float32)[None]
                    params, _weak_camera, _ = model(x)
                    r = params["hand_pose"].reshape(15, 3, 3)
                    mapped = mirror @ r @ mirror if side == 0 else r
                    ortho = float((mapped.transpose(-1, -2) @ mapped - torch.eye(3, device="cuda")).abs().max())
                    if not bool(torch.isfinite(mapped).all()) or ortho >= 1e-5:
                        raise RuntimeError(
                            f"non-finite/non-orthogonal hand prediction {serial}/{frame}/side{side}: {ortho}"
                        )
                    rotations[vi, ti, side] = mapped.detach().cpu().numpy().astype(np.float32)
                    crop_boxes[vi, ti, side] = np.asarray(boxes[i], np.float32)
                    valid[vi, ti, side] = True
                    prediction_count += 1
                    print(
                        f"[observe-hands] {identity} {serial} frame={frame} side={side} "
                        f"valid={prediction_count}",
                        flush=True,
                    )
            completed_views[vi] = True
            save_observations()
            print(f"[observe-hands] saved camera {serial}; reused={reused_count}", flush=True)

    save_observations()
    load_report = root / "load_report.json"
    model_source = root / "source.json"
    provenance = {
        "schema": "two_person_hand_surface_observations/v1",
        "identity": identity,
        "person": str(person_path),
        "person_sha256": sha256_file(person_path),
        "scene_index": str(scene_index_path),
        "scene_index_sha256": sha256_file(scene_index_path),
        "scene_camera_order": scene_order,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "view_order_source": "person manifest views",
        "serials": serials,
        "frames_source": frames_source,
        "frames": selected_frames,
        "keypoints": {
            serial: {"path": str(item["path"]), "sha256": item["sha256"]}
            for serial, item in kp_data.items()
        },
        "model_load_report": str(load_report),
        "model_load_report_sha256": sha256_file(load_report),
        "model_source": str(model_source),
        "model_source_sha256": sha256_file(model_source),
        "preprocessing": {
            "input": "scene-index undistorted JPEG, original BGR",
            "dataset": "official ViTDetDataset",
            "rescale_factor": 2.5,
            "bbox_rule": "21 points per side, existing in_image and score>0.5 and finite count>3, min/max pixel bbox",
            "boxes_are_from_rtmw_points_not_fitted_joints": True,
            "other_person_removed": False,
            "views_filtered": False,
        },
        "side_mapping": {"0": "left", "1": "right"},
        "left_mirror": "diag(-1,1,1) @ R @ diag(-1,1,1)",
        "predicted_terms": ["MANO local hand_pose 15 rotations only"],
        "excluded_terms": ["MANO global root", "beta", "depth", "camera", "mean/PCA", "temporal fusion"],
        "rotations_array": "rotations",
        "rotations_shape": list(rotations.shape),
        "rotations_order": "V,T,side,15,3,3",
        "missing_policy": "NaN rotations/crop_boxes, valid=false",
        "valid_count": int(valid.sum()),
        "prediction_count": prediction_count,
        "reused_prediction_count": reused_count,
        "source_signature": cache_signature,
        "cache_policy": "same source/model and identical per-hand pixel crop only; save one observation file after each camera",
        "seconds": round(time.time() - started, 3),
        "no_consensus": True,
        "no_media_or_crop_copies": True,
        "scope": "per-view raw image hand observations; not adopted motion or quality acceptance",
    }
    atomic_write_json(out / "provenance.json", provenance)
    print(f"[observe-hands] wrote {out / 'observations.npz'} and provenance.json", flush=True)
    return 0


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--scene", required=True, help="scene_inventory.py scene.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="inspect dependencies, models, scene, seeds")
    p.add_argument("--scene")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seeds")
    p.add_argument("--detector", choices=("none", "yolo", "external"), default="yolo")
    p.add_argument("--weights")
    p.add_argument("--detections")
    p.add_argument("--rtmw")
    p.add_argument("--sam2-weights")
    p.set_defaults(func=command_preflight)

    p = sub.add_parser("candidates", help="export numbered person candidates + blank seed template")
    add_common(p)
    p.add_argument("--frames", default="0:1:1")
    p.add_argument("--detector", choices=("yolo", "external"), default="yolo")
    p.add_argument("--weights")
    p.add_argument("--detections")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.set_defaults(func=command_candidates)

    p = sub.add_parser("prepare", help="track two seeded identities and write instance inputs")
    add_common(p)
    p.add_argument("--seeds", required=True)
    p.add_argument("--frames", default="0:-1:1", help="START:END[:STEP], END exclusive")
    p.add_argument("--detector", choices=("none", "yolo", "external"), default="yolo")
    p.add_argument("--weights")
    p.add_argument("--detections")
    p.add_argument(
        "--coordinate-space",
        choices=("auto", "raw", "undistorted"),
        default="auto",
        help="space of detector boxes/keypoints; auto reads external JSON, else raw",
    )
    p.add_argument(
        "--seed-coordinate-space",
        choices=("raw", "undistorted"),
        default="raw",
        help="space of seeds.json boxes",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--rtmw", help="RTMW wholebody ONNX; omitted means no keypoints")
    p.add_argument("--overlay-every", type=int, default=10)
    p.add_argument("--iou-gate", type=float, default=0.10)
    p.add_argument("--center-gate", type=float, default=1.5)
    p.add_argument("--max-gap", type=int, default=45)
    p.add_argument("--reproj-threshold", type=float, default=30.0)
    p.add_argument("--kp-score-threshold", type=float, default=0.3)
    p.add_argument("--hand-contact-px", type=float, default=80.0)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True, func=command_prepare)

    p = sub.add_parser("observe-hands", help="per-view HaMeR hand rotations for one person")
    p.add_argument("--person", required=True)
    p.add_argument("--scene-index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=None)
    p.set_defaults(func=command_observe_hands)

    p = sub.add_parser("validate", help="re-check an existing run")
    p.add_argument("--scene", required=True)
    p.add_argument("--run", required=True)
    p.set_defaults(func=command_validate)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    try:
        code = int(args.func(args))
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(f"[done] {args.command} in {time.time() - started:.1f}s exit={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
