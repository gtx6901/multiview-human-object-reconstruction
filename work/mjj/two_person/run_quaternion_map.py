#!/usr/bin/env python
"""q-native counterpart of run_full_smooth.py (58 body/feet/hands + optional N face).

Internal state: normalized wxyz local quaternions [T,55,4] plus translation.
No 6D or AA state is used inside the objective.  Legacy AA exists only at the
input/output boundary.

The data term reuses FullObservations58's actual base table/loss for the 58
body/feet/hands landmarks.  Optional face_constraints are attached generically
with N = local.shape[0] (the current stage has N=5), exact face-stage valid mask,
and no second observations: person xy/score remain the source of truth.

The final objective is:
    5.0 * (FullObservations58.loss + input_body_translation_trust
           + 0.5 * full_motion_temporal(q55,t) + image_hand_loss(q55))
The trust is always centered on the actual input q_input/translation_world.
The optional image hand term comes only from --hand-observations; it never
reinitializes the already fitted hand q. The required initialization is still
loaded for FullObservations58 and its source/legacy/frame-alignment validation;
it is not the trust center. Scales are fixed per-frame scales consistent with
this 30fps body-stage data, not physical per-second scales.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
LHM = PROJECT_ROOT / "lhmpp"
WORK_ROOT = PROJECT_ROOT / "work" / "mjj"
SMPLX_DEFAULT = LHM / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def configure_runtime_paths() -> None:
    for p in (LHM, LHM / "vendor", LHM / "deps", PROJECT_ROOT / "runtime" / "deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from runtime_imports import configure

        configure()
    except Exception as exc:
        raise RuntimeError(f"runtime_imports.configure failed: {exc}") from exc


def fk_q(q: torch.Tensor, rest: torch.Tensor, parents: torch.Tensor, scale: float, trans: torch.Tensor):
    from pytorch3d.transforms import quaternion_to_matrix

    R = quaternion_to_matrix(q)
    T = q.shape[0]
    Rg, tg = [None] * 55, [None] * 55
    for j in range(55):
        p = int(parents[j])
        if p < 0:
            Rg[j], tg[j] = R[:, j], rest[j].expand(T, 3)
        else:
            Rg[j] = Rg[p] @ R[:, j]
            tg[j] = tg[p] + (Rg[p] @ (rest[j] - rest[p]).view(1, 3, 1)).squeeze(-1)
    return torch.stack(tg, 1) * scale + trans[:, None], torch.stack(Rg, 1)


def attach_face(full, face_path: Path, frames: np.ndarray, cameras: List[str]) -> Dict[str, Any]:
    """Extend FullObservations58 generically to N face anchors (not hardcoded 3)."""
    fcz = np.load(Path(face_path), allow_pickle=False)
    if "local" not in fcz.files or "valid" not in fcz.files:
        raise ValueError("face_constraints must contain local and valid")
    local = np.asarray(fcz["local"], np.float64)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError(f"face local must be N,3, got {local.shape}")
    N = int(local.shape[0])
    if N == 0:
        return {"N": 0}
    if "coco" not in fcz.files:
        raise ValueError("face_constraints must contain coco")
    coco = np.asarray(fcz["coco"], np.int64).reshape(-1)
    if coco.shape[0] != N:
        raise ValueError("face coco/local N mismatch")
    valid = np.asarray(fcz["valid"], bool)
    if valid.shape != (len(cameras), frames.shape[0], N):
        raise ValueError(
            f"face valid must be V,T,N=({len(cameras)},{frames.shape[0]},{N}), got {valid.shape}"
        )
    if "frames" in fcz.files and not np.array_equal(np.asarray(fcz["frames"], np.int64), frames):
        raise ValueError("face_constraints frames do not match person frame order")
    if "cameras" in fcz.files:
        fcams = [str(x) for x in np.asarray(fcz["cameras"]).tolist()]
        if fcams != cameras:
            raise ValueError(f"face camera order mismatch: {fcams} vs {cameras}")
    # Use person's actual xy/score; only verify the face-stage arrays if present.
    coco_t = torch.as_tensor(coco, dtype=torch.long)
    actual_xy = full.xy[:, :, coco_t]  # T,V,N,2
    actual_score = full.score[:, :, coco_t]  # T,V,N
    if "xy" in fcz.files:
        fxy = torch.tensor(np.asarray(fcz["xy"], np.float64))
        if tuple(fxy.shape) != tuple(actual_xy.shape):
            raise ValueError(f"face xy shape {tuple(fxy.shape)} vs person {tuple(actual_xy.shape)}")
        if not torch.equal(torch.isfinite(fxy), torch.isfinite(actual_xy)):
            raise ValueError('face xy missingness differs from person actual xy')
        finite_xy = torch.isfinite(actual_xy)
        err = float((fxy[finite_xy] - actual_xy[finite_xy]).abs().max()) if bool(finite_xy.any()) else 0.0
        if err > 1e-5:
            raise ValueError(f"face xy does not match person actual xy; max error {err}")
    else:
        err = None
    if "score" in fcz.files:
        fsc = torch.tensor(np.asarray(fcz["score"], np.float64))
        if tuple(fsc.shape) != tuple(actual_score.shape):
            raise ValueError(f"face score shape {tuple(fsc.shape)} vs person {tuple(actual_score.shape)}")
        if not torch.equal(torch.isfinite(fsc), torch.isfinite(actual_score)):
            raise ValueError('face score missingness differs from person actual score')
        finite_sc = torch.isfinite(actual_score)
        serr = float((fsc[finite_sc] - actual_score[finite_sc]).abs().max()) if bool(finite_sc.any()) else 0.0
        if serr > 1e-5:
            raise ValueError(f"face score does not match person actual score; max error {serr}")
    else:
        serr = None
    derived_valid = (actual_score >= 0.5) & torch.isfinite(actual_xy).all(-1)
    if not torch.equal(torch.tensor(valid), derived_valid.permute(1, 0, 2)):
        raise ValueError(
            "face_constraints valid is not exactly score>=0.5 & finite(xy); "
            "no extra front/inlier gate is permitted"
        )

    full.base_count = 58
    full.face_count = N
    full.face_path = str(Path(face_path).resolve())
    full.face_local = torch.tensor(local, dtype=torch.float64)
    full.face_valid = torch.tensor(valid, dtype=torch.bool)  # V,T,N exact stage mask
    for k in range(N):
        full.table.append((int(coco[k]), 15, full.face_local[k].clone(), 1.0))
    full.coco = [x[0] for x in full.table]
    full.joints = torch.tensor([x[1] for x in full.table], dtype=torch.long)
    full.offset = torch.tensor(
        np.stack([np.zeros(3, np.float64) if off is None else off.numpy() for _, _, off, _ in full.table]),
        dtype=torch.float64,
    )
    full.weights = torch.tensor([x[3] for x in full.table], dtype=torch.float64)
    full.base_w_sum = full.weights[: full.base_count].sum()
    full.w_sum = full.weights.sum()
    masks = []
    for k, c in enumerate(full.coco):
        if k < full.base_count:
            m = full.inlier[:, :, c].permute(1, 0) & (full.score[:, :, c] >= 0.5)
        else:
            # EXACT face-stage valid mask; no extra score/front/inlier gate.
            m = full.face_valid[:, :, k - full.base_count].permute(1, 0)
        masks.append(m)
    full.mask = torch.stack(masks, dim=-1)
    return {
        "N": N,
        "coco": coco.tolist(),
        "local": local.tolist(),
        "valid_shape": list(valid.shape),
        "valid_true": int(valid.sum()),
        "xy_match_max": err,
        "score_match_max": serr,
        "selection": "exact face_constraints valid mask; no second observations",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--initialization", required=True)
    ap.add_argument("--face-constraints", default=None)
    ap.add_argument("--hand-observations", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--smplx", default=str(SMPLX_DEFAULT))
    args = ap.parse_args()

    torch.set_num_threads(4)
    configure_runtime_paths()
    from smooth_full_observations import FullObservations58
    from rotation_contract import (
        full_motion_temporal,
        load_motion_quaternions,
        normalize_quaternions,
        relative_log,
        export_motion_with_quaternions,
    )

    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    motion_path = Path(args.motion).resolve()
    if motion_path.is_dir():
        motion_path = motion_path / "motion.pt"
    initialization = Path(args.initialization).resolve()
    if initialization.is_dir():
        initialization = initialization / "motion.pt"
    face_path = Path(args.face_constraints).resolve() if args.face_constraints else None
    out = Path(args.out).resolve()
    if not out.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("output must stay below work/mjj")
    out.mkdir(parents=True, exist_ok=True)

    m = torch.load(motion_path, map_location="cpu", weights_only=False)
    q_input, q_info = load_motion_quaternions(m, verify_legacy=True)
    q_input = normalize_quaternions(q_input).double()
    frames = np.asarray(m.get("frames", np.arange(q_input.shape[0])), np.int64)
    T = q_input.shape[0]
    if len(frames) != T:
        raise ValueError("motion frames/q T mismatch")
    hand_observations = Path(args.hand_observations).resolve() if args.hand_observations else None
    hand_condition = None
    if hand_observations is not None:
        if not hand_observations.is_file():
            raise FileNotFoundError(hand_observations)
        from hand_condition import HandCondition

        hand_condition = HandCondition(hand_observations, frames.tolist())
    init_motion = torch.load(initialization, map_location="cpu", weights_only=False)
    q_init_all, init_info = load_motion_quaternions(init_motion, verify_legacy=True)
    q_init_all = normalize_quaternions(q_init_all).double()
    init_frames = np.asarray(
        init_motion.get("frames", np.arange(q_init_all.shape[0])), np.int64
    )
    init_index = {int(f): i for i, f in enumerate(init_frames)}
    if any(int(f) not in init_index for f in frames):
        raise ValueError("initialization lacks a requested motion frame")
    init_take = torch.tensor([init_index[int(f)] for f in frames], dtype=torch.long)
    q_init = q_init_all[init_take]
    if q_init.shape[0] != T:
        raise ValueError("initialization q T mismatch")

    av = torch.load(avatar, map_location="cpu", weights_only=False)
    rest = av["rig"]["rest_joints"][0].double()
    parents = av["rig"]["parents"].long()
    scale = float(m["scale"].reshape(-1)[0]) if torch.is_tensor(m["scale"]) else float(m["scale"])
    t_input = torch.as_tensor(m["translation_world"], dtype=torch.float64).reshape(T, 3)

    def input_trust(q_cur, t_cur):
        body = 0.5 * relative_log(q_cur[:, 1:22], q_input[:, 1:22]).square().mean() / 1e-3
        trans = 0.5 * (t_cur - t_input).square().mean() / 1e-4
        return body + trans

    def image_hand_loss(q_cur):
        if hand_condition is None:
            return q_cur.new_zeros(())
        return hand_condition.loss(q_cur)

    full = FullObservations58(
        person,
        avatar,
        initialization,
        Path(args.smplx).resolve(),
        expected_t=T,
        face_constraints=None,
    )
    face_meta = {"N": 0}
    if face_path is not None:
        face_meta = attach_face(full, face_path, frames, [str(v["serial"]) for v in full.views])

    with torch.no_grad():
        world_input, Rg_input = fk_q(q_input, rest, parents, scale, t_input)
        data0 = full.loss(world_input, Rg_input)
        trust0 = input_trust(q_input, t_input)
        full_temp0 = full_motion_temporal(q_input, t_input)
        image0 = image_hand_loss(q_input)
        total0 = 5.0 * (data0 + trust0 + 0.5 * full_temp0 + image0)

    q = q_input.clone().requires_grad_(True)
    t = t_input.clone().requires_grad_(True)
    opt = torch.optim.Adam([q, t], lr=3e-3)
    started = time.time()
    history = []
    all_steps_finite = True
    for it in range(args.iters):
        opt.zero_grad()
        world, Rg = fk_q(q, rest, parents, scale, t)
        data = full.loss(world, Rg)
        trust = input_trust(q, t)
        full_temp = full_motion_temporal(q, t)
        image = image_hand_loss(q)
        total = 5.0 * (data + trust + 0.5 * full_temp + image)
        total.backward()
        grad_finite = (
            q.grad is not None and torch.isfinite(q.grad).all()
            and t.grad is not None and torch.isfinite(t.grad).all()
        )
        opt.step()
        with torch.no_grad():
            q.copy_(normalize_quaternions(q))
        all_steps_finite = all_steps_finite and bool(
            torch.isfinite(total) and torch.isfinite(data) and torch.isfinite(trust)
            and torch.isfinite(full_temp) and torch.isfinite(image)
            and grad_finite and torch.isfinite(q).all() and torch.isfinite(t).all()
        )
        if it % 100 == 0 or it == args.iters - 1:
            with torch.no_grad():
                h = {
                    "iter": it,
                    "data": float(data),
                    "trust": float(trust),
                    "full_motion_temporal": float(full_temp),
                    "image_hand_loss": float(image),
                    "total": float(total),
                }
                history.append(h)
                print(h, flush=True)

    with torch.no_grad():
        world_f, Rg_f = fk_q(q, rest, parents, scale, t)
        data_f = full.loss(world_f, Rg_f)
        trust_f = input_trust(q, t)
        full_temp_f = full_motion_temporal(q, t)
        image_f = image_hand_loss(q)
        total_f = 5.0 * (data_f + trust_f + 0.5 * full_temp_f + image_f)
        q_final = normalize_quaternions(q).double()
        t_final = t.detach().double()
        world_final, _ = fk_q(q_final, rest, parents, scale, t_final)
        joints55 = world_final.detach()

    base_motion = dict(m)
    base_motion["translation_world"] = t_final.detach().cpu()
    exported = export_motion_with_quaternions(base_motion, q_final, hemisphere=True)
    exported["joints55_world"] = joints55.cpu()
    exported["frames"] = torch.as_tensor(frames, dtype=torch.long)
    exported["fps"] = m.get("fps", 30.0)
    exported["fps_rational"] = m.get("fps_rational", "30/1")
    torch.save(exported, out / "motion.pt")
    q_exported, q_export_info = load_motion_quaternions(exported, verify_legacy=True)
    export_checks = {
        "frames_preserved": bool(torch.equal(exported["frames"], torch.as_tensor(frames, dtype=torch.long))),
        "fps_preserved": bool(float(exported["fps"]) == float(m.get("fps", 30.0))),
        "betas_preserved": bool(torch.allclose(torch.as_tensor(exported["betas"]), torch.as_tensor(m["betas"]))),
        "scale_preserved": bool(torch.allclose(torch.as_tensor(exported["scale"]), torch.as_tensor(m["scale"]))),
        "joints55_actual_fk": bool(exported["joints55_world"].shape == (T, 55, 3)),
        "q_legacy_consistent": True,
    }

    finite_hist = all(
        all(math.isfinite(float(v)) for k, v in h.items() if k != "iter") for h in history
    )
    face_retained = None
    if full.face_count:
        face_retained = {
            "N": full.face_count,
            "coco": [int(full.coco[full.base_count + k]) for k in range(full.face_count)],
            "local": full.face_local.tolist(),
            "valid_shape": list(full.face_valid.shape),
            "valid_true": int(full.face_valid.sum()),
            "weight_each": 1.0,
            "selection": "exact face_constraints valid; no front/inlier/score second gate",
        }
    report = {
        "schema": "two_person_quaternion_map/v1",
        "person": str(person),
        "identity": info.get("identity"),
        "person_sha256": sha256_file(person),
        "avatar": str(avatar),
        "avatar_size_bytes": avatar.stat().st_size,
        "motion": str(motion_path),
        "motion_sha256": sha256_file(motion_path),
        "initialization": str(initialization),
        "initialization_sha256": sha256_file(initialization),
        "face_constraints": str(face_path) if face_path else None,
        "face_constraints_sha256": sha256_file(face_path) if face_path else None,
        "observations": info["observations"],
        "observations_sha256": sha256_file(Path(info["observations"])),
        "manifest": info["manifest"],
        "manifest_sha256": sha256_file(Path(info["manifest"])),
        "frames": frames.tolist(),
        "frames_source": "motion.frames" if "frames" in m else "arange(T)",
        "fps": float(m.get("fps", 30.0)),
        "fps_source": "motion.fps" if "fps" in m else "default 30.0",
        "T": T,
        "q_input_validation": q_info,
        "initialization_q_validation": init_info,
        "q_order": "wxyz",
        "q_layout": "55 local: root,body21,jaw,leye,reye,lhand15,rhand15",
        "precision": "float64 q/translation/world/data; legacy AA at boundary only",
        "face": face_meta,
        "face_retained": face_retained,
        "hand_observations": (
            hand_condition.provenance() if hand_condition is not None
            else {"provided": False}
        ),
        "landmark_contract": {
            "base": "FullObservations58 base constructor face_constraints=None; body12+feet6+tips10+hand30",
            "base_count": full.base_count,
            "face_count": full.face_count,
            "total_count": len(full.table),
            "base_weight_sum": float(full.base_w_sum),
            "total_weight_sum": float(full.w_sum),
            "face_weight_each": 1.0,
            "2d": "all base+face weights normalized by total w_sum; Huber 5px",
            "3d": "base 58 only normalized by base_w_sum; Huber .02",
            "face_3d": "none",
            "face_valid_source": "exact face_constraints valid mask; person xy/score; no second observations",
            "semantic_limitations": "original RTMW/head L and back-face ghost issues are not fixed; no quality threshold",
        },
        "objective": {
            "loss": "5.0 * (FullObservations58.loss(world,Rg) + input_trust(q,t) + 0.5*full_motion_temporal(q55,t) + image_hand_loss(q55))",
            "data": "FullObservations58.loss(world,Rg)",
            "trust": "0.5*mean(relative_log(q_body21,q_input_body21)^2)/1e-3 + 0.5*mean((t-t_input)^2)/1e-4",
            "trust_center": "actual input q_input / translation_world; initialization is not the center",
            "full_motion_temporal": "shared rotation_contract.full_motion_temporal(q55,t)",
            "full_motion_temporal_terms": (
                "body_motion_temporal(q[:,:1],q[:,1:22],t) "
                "[second_diff(t)/5e-4 + angular_accel(q_body21)/5e-3 + angular_accel(q_root)/5e-3] + "
                "mean(abs(temporal_angular_accel(q[:,25:55])))/5e-3 + "
                "mean(abs(temporal_angular_accel(q[:,22:25])))/5e-3"
            ),
            "full_motion_temporal_weight_in_total": 0.5,
            "image_hand_loss": "0.5*mean(r^T precision r)/45, r=relative_log(q_hand,q_image_mean)",
            "image_hand_loss_position": "inside the existing 5.0 bracket",
            "fixed_scales": "fixed per-frame scales consistent with this 30fps body-stage data; not physical per-second scales",
            "no_normalizer": True,
            "initialization_role": "required by FullObservations58; source/legacy/frame-alignment validation retained; not trust center",
            "iters": args.iters,
            "lr": 3e-3,
            "all55_free": True,
            "no_new_weights": True,
            "no_filters": True,
            "no_joint_locks": True,
            "no_6d_or_aa_state": True,
        },
        "initial": {
            "data": float(data0),
            "trust": float(trust0),
            "full_motion_temporal": float(full_temp0),
            "image_hand_loss": float(image0),
            "total": float(total0),
        },
        "final": {
            "data": float(data_f),
            "trust": float(trust_f),
            "full_motion_temporal": float(full_temp_f),
            "image_hand_loss": float(image_f),
            "total": float(total_f),
        },
        "history": history,
        "history_all_finite": finite_hist,
        "all_steps_finite": bool(all_steps_finite),
        "export_checks": export_checks,
        "export": {
            "local_quaternions_wxyz": True,
            "consistent_legacy": True,
            "actual_fk55": True,
            "q_export_consistency": q_export_info,
            "q_shape": list(q_exported.shape),
        },
        "output_motion": str(out / "motion.pt"),
        "scope": "single current q-native 55-joint MAP; initialization retained for observations/validation; no quality claim.",
    }
    atomic_write_json(out / "quaternion_map_provenance.json", report)
    print(f"[q-map] total {float(total0):.5f} -> {float(total_f):.5f}; finite={finite_hist}")
    print(f"[q-map] wrote {out / 'motion.pt'}")
    print(f"[q-map] provenance {out / 'quaternion_map_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
