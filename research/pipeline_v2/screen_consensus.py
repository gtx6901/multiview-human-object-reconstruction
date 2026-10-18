"""Consensus triangulation screening.

For every joint/frame: triangulate all camera pairs, take the coordinate-wise median
of those 3D points as the consensus, reproject it into each candidate view, and keep
only views within a residual gate. This replaces the previous "all-view DLT + 10 px
inlier" definition, whose 4.7 px inlier error was circular (inliers were selected by
that same gate).

Outputs assets/observations_consensus_v1/{observations.npz, report.json} including a
leave-one-camera-out evaluation that never uses the held-out view.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
KEY = PROJECT / "assets" / "keypoints_all300_v1"
MANIFEST = PROJECT / "assets" / "helmet_native000_299_v1" / "manifest.json"
OUT = PROJECT / "assets" / "observations_consensus_v1"

SCORE_MIN = 1.0
GATE_PX = 10.0
MIN_VIEWS = 3
GROUPS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91),
          "left_hand": (91, 112), "right_hand": (112, 133)}


def dlt(Ps, pts):
    rows = []
    for P, (x, y) in zip(Ps, pts):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.stack(rows))
    h = vt[-1]
    return h[:3] / h[3] if abs(h[3]) > 1e-12 else None


def project(P, X):
    q = P @ np.r_[X, 1.0]
    return (q[:2] / q[2]) if abs(q[2]) > 1e-12 else None


def consensus(Ps, pts):
    pts3 = []
    for a, b in combinations(range(len(Ps)), 2):
        X = dlt(Ps[[a, b]], pts[[a, b]])
        if X is not None and np.isfinite(X).all():
            pts3.append(X)
    if len(pts3) < 1:
        return None
    return np.median(np.stack(pts3), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    Ps = np.stack([np.array(v["K_undistorted_native"]) @ np.array(v["W"])[:3] for v in meta["views"]])
    xy = np.stack([np.load(KEY / f"{c}.npz")["xy"] for c in cams]).astype(np.float64)
    score = np.stack([np.load(KEY / f"{c}.npz")["score"] for c in cams])
    in_image = np.stack([np.load(KEY / f"{c}.npz")["in_image"] for c in cams])
    T, J = xy.shape[1], xy.shape[2]
    cand = in_image & (score >= SCORE_MIN)

    xyz = np.full((T, J, 3), np.nan, np.float32)
    valid = np.zeros((T, J), bool)
    support = np.zeros((T, J), np.int16)
    resid = np.full((len(cams), T, J), np.inf, np.float32)
    inlier = np.zeros((len(cams), T, J), bool)

    started = time.time()
    for f in range(T):
        for j in range(J):
            ids = np.flatnonzero(cand[:, f, j])
            if len(ids) < MIN_VIEWS:
                continue
            pts = xy[ids, f, j]
            X = consensus(Ps[ids], pts)
            if X is None:
                continue
            err = np.array([np.linalg.norm(project(Ps[i], X) - xy[i, f, j]) if project(Ps[i], X) is not None else np.inf
                            for i in ids])
            keep = err <= GATE_PX
            if keep.sum() < MIN_VIEWS:
                continue
            xyz[f, j] = consensus(Ps[ids[keep]], pts[keep])
            valid[f, j] = True
            support[f, j] = int(keep.sum())
            resid[ids, f, j] = err
            inlier[ids[keep], f, j] = True

    np.savez_compressed(OUT / "observations.npz", xyz=xyz, valid=valid, support=support,
                        residual_px=resid, inlier=inlier, cameras=np.asarray(cams),
                        frames=np.arange(T), score_min=SCORE_MIN, gate_px=GATE_PX)
    report = {"success": True, "cameras": cams, "frames": int(T), "score_min": SCORE_MIN,
              "gate_px": GATE_PX, "min_views": MIN_VIEWS,
              "estimator": "coordinate-wise median of pairwise triangulations",
              "groups": {}}

    # leave-one-camera-out, never using the held-out view
    loo = []
    for f in range(0, T, 5):
        for j in range(J):
            ids = np.flatnonzero(inlier[:, f, j])
            if len(ids) < MIN_VIEWS + 1:
                continue
            for held in ids:
                rest = ids[ids != held]
                X = consensus(Ps[rest], xy[rest, f, j])
                if X is None:
                    continue
                uv = project(Ps[held], X)
                if uv is None:
                    continue
                loo.append(np.linalg.norm(uv - xy[held, f, j]))
    loo = np.asarray(loo)
    report["loo"] = {"n": int(len(loo)), "median_px": float(np.median(loo)) if len(loo) else None,
                     "p90_px": float(np.percentile(loo, 90)) if len(loo) else None}
    for name, (a, b) in GROUPS.items():
        m = valid[:, a:b]
        report["groups"][name] = {"valid_rate": float(m.mean()),
                                  "median_support": float(np.median(support[:, a:b][m])) if m.any() else 0}
    report["seconds"] = round(time.time() - started, 1)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", report["seconds"], "s; LOO", report["loo"])


if __name__ == "__main__":
    main()
