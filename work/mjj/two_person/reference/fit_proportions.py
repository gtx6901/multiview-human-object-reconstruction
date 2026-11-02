"""One-time body proportions from the static reference capture.

Pipeline
  1. Triangulate the static RTMW keypoints (3 frames x 8 views, pair-RANSAC).
  2. Measure model-free lengths: shoulder width, upper arm, forearm, thigh, shin,
     shoulder-mid to ear.
  3. Fit SMPL-X neutral shape (10 betas) + global scale + one neck offset to those
     lengths, with a strong shape prior. The fit is frozen for every downstream
     motion fit; nothing here touches the 300-frame action.

Output: assets/proportions_v1/{proportions.pt, report.json}
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

PROJECT = Path(__file__).resolve().parent.parent
STATIC = PROJECT / "assets" / "static_reference_v1"
SMPLX = PROJECT / "lhmpp" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
OUT = PROJECT / "assets" / "proportions_v1"

SCORE_MIN = 0.5
MAX_ERROR_PX = 10.0
MIN_VIEWS = 3
# COCO wholebody indices used by the measurement
JOINTS = {"nose": 0, "Lear": 3, "Rear": 4, "Lsho": 5, "Rsho": 6, "Lelb": 7, "Relb": 8,
          "Lwri": 9, "Rwri": 10, "Lhip": 11, "Rhip": 12, "Lkne": 13, "Rkne": 14,
          "Lank": 15, "Rank": 16}
# SMPL-X joint indices
SX = {"Lhip": 1, "Rhip": 2, "Lkne": 4, "Rkne": 5, "Lank": 7, "Rank": 8,
      "Lsho": 16, "Rsho": 17, "Lelb": 18, "Relb": 19, "Lwri": 20, "Rwri": 21, "head": 15}
MEASURE_NAMES = ["shoulder_width", "L_upper_arm", "R_upper_arm", "L_forearm", "R_forearm",
                 "L_thigh", "R_thigh", "L_shin", "R_shin", "shoulder_mid_to_ear"]


def dlt(Ps, points):
    rows = []
    for P, (x, y) in zip(Ps, points):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.stack(rows))
    h = vt[-1]
    return h[:3] / h[3] if abs(h[3]) > 1e-12 else np.full(3, np.nan)


def project(Ps, X):
    q = np.einsum("nij,j->ni", Ps, np.r_[X, 1.0])
    return q[:, :2] / q[:, 2:3]


def triangulate_frame(xy, score, in_image, Ps, centers, frame):
    xyz = np.full((133, 3), np.nan)
    valid = np.zeros(133, bool)
    for j in range(133):
        ids = np.flatnonzero(in_image[:, frame, j] & (score[:, frame, j] >= SCORE_MIN))
        if len(ids) < MIN_VIEWS:
            continue
        pts = xy[ids, frame, j]
        best = None
        for a, b in combinations(range(len(ids)), 2):
            X = dlt(Ps[ids[[a, b]]], xy[ids[[a, b]], frame, j])
            if not np.isfinite(X).all():
                continue
            err = np.linalg.norm(project(Ps[ids], X) - pts, axis=1)
            inl = err <= MAX_ERROR_PX
            if inl.sum() < MIN_VIEWS:
                continue
            rank = (int(inl.sum()), -float(np.median(err[inl])))
            if best is None or rank > best[0]:
                best = (rank, inl)
        if best is None:
            continue
        keep = ids[best[1]]
        X = dlt(Ps[keep], xy[keep, frame, j])
        rays = centers[keep] - X
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        if np.degrees(np.arccos(np.clip(np.min(rays @ rays.T), -1, 1))) < 1.5:
            continue
        xyz[j], valid[j] = X, True
    return xyz, valid


def measure(P):
    def L(a, b):
        return float(np.linalg.norm(P[JOINTS[a]] - P[JOINTS[b]]))

    mid = 0.5 * (P[JOINTS["Lsho"]] + P[JOINTS["Rsho"]])
    return np.array([
        L("Lsho", "Rsho"), L("Lsho", "Lelb"), L("Rsho", "Relb"),
        L("Lelb", "Lwri"), L("Relb", "Rwri"),
        L("Lhip", "Lkne"), L("Rhip", "Rkne"), L("Lkne", "Lank"), L("Rkne", "Rank"),
        0.5 * (np.linalg.norm(mid - P[JOINTS["Lear"]]) + np.linalg.norm(mid - P[JOINTS["Rear"]])),
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((STATIC / "manifest.json").read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    Ps = np.stack([np.array(v["K"]) @ np.array(v["W"])[:3] for v in meta["views"]])
    centers = np.array([-np.array(v["W"])[:3, :3].T @ np.array(v["W"])[:3, 3] for v in meta["views"]])
    kd = np.load(STATIC / "keypoints.npz")
    xy, score, in_image = kd["xy"], kd["score"], kd["in_image"]
    frames = [int(f) for f in kd["frames"]]

    per_frame, valid_frames = [], []
    for fi, f in enumerate(frames):
        xyz, valid = triangulate_frame(xy, score, in_image, Ps, centers, fi)
        need = [JOINTS[k] for k in JOINTS]
        if valid[need].all():
            per_frame.append(measure(xyz))
            valid_frames.append(f)
    assert len(per_frame) >= 1, "no static frame with all measurement joints"
    M = np.stack(per_frame)
    measured = np.median(M, axis=0)
    spread = (np.percentile(M, 90, axis=0) - np.percentile(M, 10, axis=0)) / measured

    # SMPL-X neutral
    d = np.load(SMPLX, allow_pickle=True)
    v_template = np.asarray(d["v_template"], np.float64)
    shapedirs = np.asarray(d["shapedirs"], np.float64)[:, :, :10]
    J_reg = d["J_regressor"]
    J_reg = J_reg.toarray() if hasattr(J_reg, "toarray") else np.asarray(J_reg, np.float64)
    rest = J_reg @ v_template

    head = rest[SX["head"]]
    cand = np.flatnonzero(np.linalg.norm(v_template - head, axis=1) < 0.12)
    earL = int(cand[np.argmax(v_template[cand, 0])])
    earR = int(cand[np.argmin(v_template[cand, 0])])

    def model_lengths(beta):
        v = v_template + shapedirs @ beta
        J = J_reg @ v
        mid = 0.5 * (J[SX["Lsho"]] + J[SX["Rsho"]])
        return np.array([
            np.linalg.norm(J[SX["Lsho"]] - J[SX["Rsho"]]),
            np.linalg.norm(J[SX["Lsho"]] - J[SX["Lelb"]]),
            np.linalg.norm(J[SX["Rsho"]] - J[SX["Relb"]]),
            np.linalg.norm(J[SX["Lelb"]] - J[SX["Lwri"]]),
            np.linalg.norm(J[SX["Relb"]] - J[SX["Rwri"]]),
            np.linalg.norm(J[SX["Lhip"]] - J[SX["Lkne"]]),
            np.linalg.norm(J[SX["Rhip"]] - J[SX["Rkne"]]),
            np.linalg.norm(J[SX["Lkne"]] - J[SX["Lank"]]),
            np.linalg.norm(J[SX["Rkne"]] - J[SX["Rank"]]),
            0.5 * (np.linalg.norm(mid - v[earL]) + np.linalg.norm(mid - v[earR])),
        ])

    base = model_lengths(np.zeros(10))
    scale0 = float(np.median(measured / base))

    def residual(p):
        beta, log_s, neck, delta, d_sh, d_ank = p[:10], p[10], p[11], p[12], p[13], p[14]
        pred = model_lengths(beta) * np.exp(log_s)
        pred[-1] += neck
        # COCO elbow landmarks sit distal to the SMPL-X elbow joint: move the split
        # by one shared offset so beta is not forced to absorb a landmark definition.
        m = measured.copy()
        m[0] -= 2*d_sh
        m[1] -= d_sh; m[3] += delta
        m[2] -= d_sh; m[4] += delta
        m[7] += d_ank; m[8] += d_ank
        rel = (pred - m) / m
        return np.r_[rel, 0.05 * beta, 0.5 * neck / measured[-1], 0.5 * delta / measured[0], 0.5 * d_sh / measured[0], 0.5 * d_ank / measured[0]]

    p0 = np.r_[np.zeros(10), np.log(scale0), 0.0, 0.0, 0.0, 0.0]
    sol = least_squares(residual, p0, bounds=(np.r_[np.full(10, -3.0), np.log(0.2), -0.5, -0.05, -0.05, -0.05],
                                              np.r_[np.full(10, 3.0), np.log(5.0), 0.5, 0.05, 0.05, 0.05]),
                        loss="soft_l1", f_scale=0.05, max_nfev=500)
    beta, log_s, neck, delta, d_sh, d_ank = sol.x[:10], sol.x[10], sol.x[11], sol.x[12], sol.x[13], sol.x[14]
    pred = model_lengths(beta) * np.exp(log_s)
    pred[-1] += neck
    corrected = measured.copy()
    corrected[0] -= 2 * d_sh
    corrected[1] -= d_sh; corrected[3] += delta
    corrected[2] -= d_sh; corrected[4] += delta
    corrected[7] += d_ank; corrected[8] += d_ank
    rel = (pred - corrected) / corrected

    torch.save({"beta": torch.tensor(beta, dtype=torch.float32),
                "scale": float(np.exp(log_s)),
                "neck_offset": float(neck), "elbow_offset": float(delta), "shoulder_offset": float(d_sh), "ankle_offset": float(d_ank),
                "ear_vertices": [earL, earR],
                "source": "assets/static_reference_v1", "frames": valid_frames},
               OUT / "proportions.pt")
    report = {
        "success": True,
        "frames_used": valid_frames,
        "measured_median": dict(zip(MEASURE_NAMES, measured.tolist())),
        "per_frame": [dict(zip(MEASURE_NAMES, row.tolist())) for row in M],
        "cross_frame_relative_spread": dict(zip(MEASURE_NAMES, spread.tolist())),
        "beta": beta.tolist(), "beta_norm": float(np.linalg.norm(beta)),
        "scale": float(np.exp(log_s)), "neck_offset": float(neck), "elbow_offset": float(delta), "shoulder_offset": float(d_sh), "ankle_offset": float(d_ank),
        "model_predicted": dict(zip(MEASURE_NAMES, pred.tolist())),
        "measured_corrected": dict(zip(MEASURE_NAMES, corrected.tolist())),
        "relative_residual": dict(zip(MEASURE_NAMES, rel.tolist())),
        "ear_vertices": [earL, earR],
        "smplx": str(SMPLX),
        "scope": "Static reference only. Lengths are ratios; global scale is fitted separately. "
                 "Neck offset is a rig offset because SMPL-X has no neck-length parameter.",
        "fit": {"cost": float(sol.cost), "optimality": float(sol.optimality), "nfev": int(sol.nfev)},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, m, p, r in zip(MEASURE_NAMES, measured, pred, rel):
        print(f"{name:22s} measured={m:.4f} model={p:.4f} rel={r:+.3%}")
    print("beta", np.round(beta, 3), "scale", round(float(np.exp(log_s)), 4), "neck", round(float(neck), 4), "elbow_offset", round(float(delta), 4), "shoulder", round(float(d_sh), 4), "ankle", round(float(d_ank), 4))


if __name__ == "__main__":
    main()
