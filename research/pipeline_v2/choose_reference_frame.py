"""Rank candidate standing frames for proportion estimation.

Criteria: straight legs and hips, upright torso, arms clear of the torso, ears and
face visible, and enough multi-view support with the helmet not yet worn.
"""
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OBS = PROJECT / "assets" / "observations_all300_v1" / "observations.npz"
KEY = PROJECT / "assets" / "keypoints_all300_v1"
OUT = PROJECT / "assets" / "reference_frame_v1"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def angle_deg(a, b, c):
    v1, v2 = unit(a - b), unit(c - b)
    return float(np.degrees(np.arccos(np.clip(v1 @ v2, -1.0, 1.0))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(OBS)
    xyz, valid = d["xyz"], d["valid"]
    cams = [str(c) for c in d["cameras"]]
    scores = np.stack([np.load(KEY / f"{c}.npz")["score"] for c in cams])

    rows = []
    for f in range(180):  # before the helmet is worn
        need = [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
        if not valid[f, need].all():
            continue
        P = xyz[f]
        knee = 0.5 * (angle_deg(P[11], P[13], P[15]) + angle_deg(P[12], P[14], P[16]))
        hip = 0.5 * (angle_deg(P[5], P[11], P[13]) + angle_deg(P[6], P[12], P[14]))
        torso = P[5:7].mean(0) - P[11:13].mean(0)
        head = P[0] - P[11:13].mean(0)
        up = unit(head) if np.linalg.norm(head) > 1e-6 else torso
        upright = float(np.degrees(np.arccos(np.clip(unit(torso) @ up, -1.0, 1.0))))
        scale = np.linalg.norm(P[0] - P[11:13].mean(0)) + 1e-9
        axis = P[11:13].mean(0) + np.outer(np.linspace(0, 1, 20), P[5:7].mean(0) - P[11:13].mean(0))
        dist = min(np.min(np.linalg.norm(axis - P[9], axis=1)), np.min(np.linalg.norm(axis - P[10], axis=1))) / scale
        vis = float(valid[f, :91].mean())
        ear_vis = int(valid[f, [3, 4]].sum())
        score = knee + hip - 3.0 * abs(180 - upright) * 0.0 + 60.0 * dist + 40.0 * vis + 10.0 * ear_vis
        rows.append({"frame": f, "knee_deg": round(knee, 1), "hip_deg": round(hip, 1),
                     "upright_dev_deg": round(upright, 1), "arm_clearance": round(dist, 3),
                     "valid_body_face": round(vis, 3), "ears": ear_vis, "score": round(score, 1)})
    rows.sort(key=lambda r: -r["score"])
    (OUT / "ranking.json").write_text(json.dumps(rows[:40], indent=2), encoding="utf-8")
    for r in rows[:15]:
        print(r)
    print("best:", rows[0]["frame"] if rows else None)


if __name__ == "__main__":
    main()
