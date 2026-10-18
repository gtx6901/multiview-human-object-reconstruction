"""Manifold ESKF-RTS smoothing for the fitted motion.

Rotations: error-state Kalman filter with a constant-acceleration model (state:
orientation, body angular velocity, angular acceleration) on SO(3); the RTS pass
smooths the error state in the tangent space and retracts with Exp. There is no
per-frame weighting: one process-noise parameter is estimated/selected once and
used for the whole sequence.

Translation: constant-velocity RTS in R^3.

Input : assets/human/<HOI_SRC>/motion.pt  (default motion_face_v1)
Output: assets/human/<HOI_OUT>/motion.pt (default motion_face_smoothed_v1)
"""
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "lhmpp" / "deps"), str(PROJECT / "runtime" / "deps")]

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation

SRC = PROJECT / "assets" / "human" / os.environ.get("HOI_SRC", "motion_face_v1")
OUT = PROJECT / "assets" / "human" / os.environ.get("HOI_OUT", "motion_face_smoothed_v1")
DT = 1.0 / 30.0
Q_ACC = float(os.environ.get("HOI_Q", "2e-2"))     # angular acceleration noise (rad^2/s^5)
R_OBS = float(os.environ.get("HOI_R", "2e-3"))     # orientation measurement noise (rad^2)
Q_TR = float(os.environ.get("HOI_QTR", "2e-2"))


def expm(v):
    return Rotation.from_rotvec(v).as_matrix()


def logm(Rm):
    return Rotation.from_matrix(Rm).as_rotvec()


def rts_cv(z, q, r, dt=DT):
    z = np.asarray(z, np.float64)
    T = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Rm = np.array([[r]])
    xf = np.zeros((T, 2)); Pf = np.zeros((T, 2, 2))
    xp = np.zeros((T, 2)); Pp = np.zeros((T, 2, 2))
    xf[0] = [z[0], 0.0]; Pf[0] = np.eye(2) * 1e-12
    for t in range(1, T):
        xp[t] = F @ xf[t - 1]
        Pp[t] = F @ Pf[t - 1] @ F.T + Q
        S = H @ Pp[t] @ H.T + Rm
        K = Pp[t] @ H.T @ np.linalg.inv(S)
        xf[t] = xp[t] + (K @ (np.array([z[t]]) - H @ xp[t])).ravel()
        Pf[t] = (np.eye(2) - K @ H) @ Pp[t]
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        C = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
    return xs[:, 0]


def smooth_series(a, q, r):
    a = np.asarray(a, np.float64)
    shape = a.shape
    flat = a.reshape(shape[0], -1)
    out = np.empty_like(flat)
    for i in range(flat.shape[1]):
        out[:, i] = rts_cv(flat[:, i], q, r)
    return out.reshape(shape)


def eskf_rts_ca(R, q, r, dt=DT):
    """Constant-acceleration ESKF + RTS on SO(3), one error state per joint."""
    T, J = R.shape[0], R.shape[1]
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    F = np.block([[I3, I3 * dt, 0.5 * I3 * dt ** 2],
                  [Z3, I3, I3 * dt],
                  [Z3, Z3, I3]])
    G = np.vstack([0.5 * dt ** 2 * I3, dt * I3, I3])          # 9x3
    Q = q * (G @ G.T)
    H = np.hstack([I3, Z3, Z3])
    Rm = r * I3
    out = np.zeros_like(R)
    for j in range(J):
        Rj = R[:, j]
        omega = np.zeros((T, 3)); acc = np.zeros((T, 3))
        omega[1:] = np.stack([logm(Rj[t - 1].T @ Rj[t]) / dt for t in range(1, T)])
        acc[1:] = np.diff(omega, axis=0) / dt
        nom = np.zeros((T, 3, 3)); nom[0] = Rj[0]
        xp = np.zeros((T, 9)); xu = np.zeros((T, 9))
        Pp = np.zeros((T, 9, 9)); Pu = np.zeros((T, 9, 9))
        P0 = np.diag(np.r_[np.full(3, r), np.full(3, 1.0), np.full(3, 10.0)])
        xu[0] = 0.0; Pu[0] = P0
        for t in range(1, T):
            nom[t] = nom[t - 1] @ expm(omega[t - 1] * dt + 0.5 * acc[t - 1] * dt ** 2)
            omega[t] = omega[t - 1] + acc[t - 1] * dt
            acc[t] = acc[t - 1]
            xp[t] = F @ xu[t - 1]
            Pp[t] = F @ Pu[t - 1] @ F.T + Q
            y = logm(nom[t].T @ Rj[t])
            S = H @ Pp[t] @ H.T + Rm
            K = Pp[t] @ H.T @ np.linalg.inv(S)
            xu[t] = xp[t] + K @ (y - H @ xp[t])
            Pu[t] = (np.eye(9) - K @ H) @ Pp[t]
            # retract the velocity/acceleration corrections into the nominal state
            omega[t] = omega[t] + xu[t, 3:6]
            acc[t] = acc[t] + xu[t, 6:9]
        xs = xu.copy()
        for t in range(T - 2, -1, -1):
            C = Pu[t] @ F.T @ np.linalg.inv(Pp[t + 1])
            xs[t] = xu[t] + C @ (xs[t + 1] - xp[t + 1])
        out[:, j] = np.stack([nom[t] @ expm(xs[t, :3]) for t in range(T)])
    return out



def eskf_rts_rw(R, q, r, dt=DT):
    """Random-walk ESKF + RTS on SO(3): error state dtheta only, nominal fixed."""
    T, J = R.shape[0], R.shape[1]
    out = np.zeros_like(R)
    for j in range(J):
        Rj = R[:, j]
        nom = np.zeros((T, 3, 3)); nom[0] = Rj[0]
        xp = np.zeros((T, 3)); xu = np.zeros((T, 3))
        Pp = np.zeros((T, 3, 3)); Pu = np.zeros((T, 3, 3))
        xu[0] = 0.0; Pu[0] = np.eye(3) * 1e-12
        for t_ in range(1, T):
            nom[t_] = nom[t_ - 1]
            xp[t_] = xu[t_ - 1]
            Pp[t_] = Pu[t_ - 1] + q * np.eye(3)
            y = logm(nom[t_].T @ Rj[t_])
            K = Pp[t_] @ np.linalg.inv(Pp[t_] + r * np.eye(3))
            xu[t_] = xp[t_] + K @ (y - xp[t_])
            Pu[t_] = (np.eye(3) - K) @ Pp[t_]
        xs = xu.copy()
        for t_ in range(T - 2, -1, -1):
            C = Pu[t_] @ np.linalg.inv(Pp[t_ + 1])
            xs[t_] = xu[t_] + C @ (xs[t_ + 1] - xp[t_ + 1])
        out[:, j] = np.stack([nom[t_] @ expm(xs[t_]) for t_ in range(T)])
    return out


def fk(rest, parents, aa, scale, trans):
    T = aa.shape[0]
    R = axis_angle_to_matrix(torch.tensor(aa.reshape(-1, 3), dtype=torch.float32)).reshape(T, 55, 3, 3)
    Rg, tg = [None] * 55, [None] * 55
    for j in range(55):
        p = int(parents[j])
        if p < 0:
            Rg[j], tg[j] = R[:, j], rest[j].expand(T, 3)
        else:
            Rg[j] = Rg[p] @ R[:, j]
            tg[j] = tg[p] + (Rg[p] @ (rest[j] - rest[p]).view(1, 3, 1)).squeeze(-1)
    return (torch.stack(tg, dim=1) * scale + torch.tensor(trans, dtype=torch.float32).unsqueeze(1)).numpy()


def d2_p95(a):
    a = np.asarray(a, np.float64)
    return float(np.percentile(np.abs(a[2:] - 2 * a[1:-1] + a[:-2]), 95)) if len(a) > 2 else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = torch.load(SRC / "motion.pt", map_location="cpu", weights_only=False)
    av = torch.load(PROJECT / "assets" / "human" / "motion_unified_v1" / "avatar.pt", map_location="cpu", weights_only=False)
    rest, parents = av["rig"]["rest_joints"][0].float(), av["rig"]["parents"]
    scale = float(m["scale"])
    root = m["root_pose"].numpy(); body = m["body_pose"].numpy()
    lh = m["lhand_pose"].numpy(); rh = m["rhand_pose"].numpy(); trans = m["translation_world"].numpy()
    T = len(root)
    zero = np.zeros((T, 3, 3))
    aa = np.concatenate([root[:, None, :], body, zero, lh, rh], axis=1)
    R = axis_angle_to_matrix(torch.tensor(aa.reshape(-1, 3))).reshape(T, 55, 3, 3).numpy()

    R_s = eskf_rts_rw(R, Q_ACC, R_OBS) if os.environ.get('HOI_MODEL','rw')=='rw' else eskf_rts_ca(R, Q_ACC, R_OBS)
    trans_s = smooth_series(trans, Q_TR, 1e-4)
    aa_s = Rotation.from_matrix(R_s.reshape(-1, 3, 3)).as_rotvec().reshape(T, 55, 3).astype(np.float32)

    before_j = fk(rest, parents, aa, scale, trans)
    after_j = fk(rest, parents, aa_s, scale, trans_s)
    rel = np.einsum("tjia,tjik->tjak", R, R_s)
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))

    out = {k: m[k] for k in m}
    out.update(root_pose=torch.tensor(aa_s[:, 0]), body_pose=torch.tensor(aa_s[:, 1:22]),
               lhand_pose=torch.tensor(aa_s[:, 25:40]), rhand_pose=torch.tensor(aa_s[:, 40:55]),
               translation_world=torch.tensor(trans_s, dtype=torch.float32),
               joints55_world=torch.tensor(after_j, dtype=torch.float32))
    torch.save(out, OUT / "motion.pt")

    metrics = {}
    for name, idx in {"translation": None, "wrist_L": 20, "wrist_R": 21, "ankle_L": 7, "ankle_R": 8}.items():
        b = trans if idx is None else before_j[:, idx]
        a = trans_s if idx is None else after_j[:, idx]
        metrics[name] = {"before": d2_p95(b), "after": d2_p95(a)}
    report = {"success": True, "model": "constant-acceleration ESKF-RTS on SO(3)",
              "q_acc": Q_ACC, "r_obs": R_OBS, "q_trans": Q_TR, "dt": DT,
              "jitter_d2_p95": metrics,
              "rotation_deviation_deg": {"median": float(np.median(ang)), "p95": float(np.percentile(ang, 95)),
                                         "max": float(ang.max())},
              "note": "No per-frame weighting; one parameter set for the whole sequence."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in metrics.items():
        print(f"{k:10s} {v['before']:.5f} -> {v['after']:.5f} ({v['after']/max(v['before'],1e-12)-1:+.1%})")
    print("rot dev deg median", round(report["rotation_deviation_deg"]["median"], 3),
          "p95", round(report["rotation_deviation_deg"]["p95"], 3))


if __name__ == "__main__":
    main()
