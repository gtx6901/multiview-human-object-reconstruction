"""Shared robust multiview triangulation for body, face and hand landmarks."""

import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np

SCORE_MIN = 0.5
MAX_ERROR_PX = 10.0
MIN_VIEWS = 3
MIN_RAY_ANGLE_DEG = 1.5
GROUPS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}


def dlt(Ps, points):
    rows = []
    for P, (x, y) in zip(Ps, points):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.stack(rows))
    h = vt[-1]
    if abs(h[3]) < 1e-12:
        return np.full(3, np.nan)
    return h[:3] / h[3]


def project(Ps, X):
    q = np.einsum("nij,j->ni", Ps, np.r_[X, 1.0])
    return q[:, :2] / q[:, 2:3]


def triangulate_keypoints(keypoints: Path, manifest: Path):
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    views = meta["views"]
    cams = [v["serial"] for v in views]
    Ps = np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in views])
    centers = np.array([-np.array(v["W"])[:3, :3].T @ np.array(v["W"])[:3, 3] for v in views])

    xy = np.stack([np.load(keypoints / f"{c}.npz")["xy"] for c in cams]).astype(np.float64)
    score = np.stack([np.load(keypoints / f"{c}.npz")["score"] for c in cams])
    in_image = np.stack([np.load(keypoints / f"{c}.npz")["in_image"] for c in cams])
    frames, njoints = xy.shape[1], xy.shape[2]

    cand = in_image & (score >= SCORE_MIN)
    xyz = np.full((frames, njoints, 3), np.nan, np.float32)
    valid = np.zeros((frames, njoints), bool)
    support = np.zeros((frames, njoints), np.int16)
    med_err = np.full((frames, njoints), np.inf, np.float32)
    inlier = np.zeros((len(cams), frames, njoints), bool)

    started = time.time()
    for f in range(frames):
        for j in range(njoints):
            ids = np.flatnonzero(cand[:, f, j])
            n = len(ids)
            if n < MIN_VIEWS:
                continue
            pts = xy[ids, f, j]
            best = None
            for a, b in combinations(range(n), 2):
                pair = ids[[a, b]]
                X = dlt(Ps[pair], xy[pair, f, j])
                if not np.isfinite(X).all():
                    continue
                err = np.linalg.norm(project(Ps[ids], X) - pts, axis=1)
                inl = err <= MAX_ERROR_PX
                s = int(inl.sum())
                if s < MIN_VIEWS:
                    continue
                rank = (s, -float(np.median(err[inl])))
                if best is None or rank > best[0]:
                    best = (rank, inl)
            if best is None:
                continue
            keep = ids[best[1]]
            X = dlt(Ps[keep], xy[keep, f, j])
            if not np.isfinite(X).all():
                continue
            err = np.linalg.norm(project(Ps[keep], X) - xy[keep, f, j], axis=1)
            mask = err <= MAX_ERROR_PX
            keep, err = keep[mask], err[mask]
            if len(keep) < MIN_VIEWS:
                continue
            rays = centers[keep] - X
            rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
            if np.degrees(np.arccos(np.clip(np.min(rays @ rays.T), -1.0, 1.0))) < MIN_RAY_ANGLE_DEG:
                continue
            X = dlt(Ps[keep], xy[keep, f, j])
            err = np.linalg.norm(project(Ps[keep], X) - xy[keep, f, j], axis=1)
            xyz[f, j] = X
            valid[f, j] = True
            support[f, j] = len(keep)
            med_err[f, j] = float(np.median(err))
            inlier[keep, f, j] = True

    with np.load(keypoints / f"{cams[0]}.npz", allow_pickle=False) as first_view:
        frame_ids = first_view["frames"] if "frames" in first_view else np.arange(frames)
    return {
        "xyz": xyz,
        "valid": valid,
        "support": support,
        "median_error_px": med_err,
        "inlier": inlier,
        "cameras": np.asarray(cams),
        "frames": frame_ids,
        "score_min": SCORE_MIN,
        "max_error_px": MAX_ERROR_PX,
        "min_views": MIN_VIEWS,
    }
