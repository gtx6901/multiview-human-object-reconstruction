#!/usr/bin/env python
"""Direct q-native motion fit (wxyz internal, legacy AA boundary only).

This replaces the former temporary ``fit_motion`` string-patching wrapper.  It
uses the same current-person observation package, own avatar rig/rest/scale/beta,
initialization frame selection, world-up foot contact, missingness masks,
landmark weights, 400/200/200 stages, Adam/lr and hand-only stage as
``reference/fit_motion.py``, but keeps root/body/hands as normalized quaternions.

The q time and body trust terms are SO(3) relative-log terms.  That is a semantic
change from raw AA-coordinate temporal/prior terms, not a claim of exact
equivalence.  An optional image-hand tangent likelihood is added only at stage2
when --hand-observations is supplied; it adds no new loss weight or joint lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "mjj"
LHM = PROJECT_ROOT / "lhmpp"
SMPLX = LHM / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
REF_FIT = THIS_DIR / "reference" / "fit_motion.py"

BODY = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
FOOT_VERTEX = {
    17: ("LBigToe", 10),
    18: ("LSmallToe", 10),
    19: ("LHeel", 10),
    20: ("RBigToe", 11),
    21: ("RSmallToe", 11),
    22: ("RHeel", 11),
}
HAND_OFFSET = 91
HAND_BASE = [37, 25, 28, 34, 31]
TIP_VID = [5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022]
STEPS = [400, 200, 200]
TEMPORAL_WEIGHT = 0.5


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


def canonical_offsets(smplx_path: Path = SMPLX):
    """Current fit_motion canonical toe/heel/fingertip offsets, patched hand order."""
    d = np.load(Path(smplx_path), allow_pickle=True)
    v = np.asarray(d["v_template"], np.float64)
    jr = d["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else np.asarray(jr, np.float64)
    J = jr @ v
    from smplx.vertex_ids import vertex_ids

    ids = vertex_ids["smplx"]
    import torch

    feet = []
    for coco, (name, joint) in FOOT_VERTEX.items():
        feet.append((int(coco), int(joint), torch.tensor(v[ids[name]] - J[joint], dtype=torch.float32)))
    tips = []
    for side in range(2):
        for f in range(5):
            distal = HAND_BASE[f] + side * 15 + 2
            tips.append(
                (
                    HAND_OFFSET + side * 21 + 4 * f + 4,
                    int(distal),
                    torch.tensor(v[TIP_VID[side * 5 + f]] - J[distal], dtype=torch.float32),
                )
            )
    return feet, tips


def select_initialization(avatar: Path, initialization: Path, frames: Sequence[int]):
    import torch
    from rotation_contract import load_motion_quaternions, normalize_quaternions

    av = torch.load(avatar, map_location="cpu", weights_only=False)
    seed = torch.load(initialization, map_location="cpu", weights_only=False)
    q_all, q_info = load_motion_quaternions(seed, verify_legacy=True)
    q_all = normalize_quaternions(q_all).float()
    seed_frames = [int(x) for x in (seed["frames"].tolist() if hasattr(seed["frames"], "tolist") else seed["frames"])]
    idx = {f: i for i, f in enumerate(seed_frames)}
    missing = [f for f in frames if f not in idx]
    if missing:
        raise ValueError(f"initialization lacks person frame(s): {missing[:5]}")
    take = torch.tensor([idx[f] for f in frames], dtype=torch.long)
    selected: Dict[str, Any] = {}
    for key, value in seed.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == len(seed_frames):
            selected[key] = value[take]
        else:
            selected[key] = value
    selected["frames"] = torch.tensor(list(frames), dtype=torch.long)
    q_seed = q_all[take]
    avp = av["poses"]
    for key in ("scale", "betas"):
        if key in selected and key in avp:
            a = torch.as_tensor(selected[key], dtype=torch.float32)
            b = torch.as_tensor(avp[key], dtype=torch.float32)
            if a.shape != b.shape or not torch.allclose(a, b, rtol=1e-4, atol=1e-5):
                raise ValueError(f"initialization {key} does not match avatar {key}; refusing inconsistent rig/scale")
    return av, selected, q_seed, q_info


def fk_q(q: Any, rest: Any, parents: Any):
    import torch
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
    return torch.stack(tg, 1), torch.stack(Rg, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--initialization", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-up", type=float, nargs=3, required=True)
    ap.add_argument("--world-up-source", required=True)
    ap.add_argument("--hand-observations", default=None)
    ap.add_argument(
        "--reference",
        default=str(REF_FIT),
        help="provenance-only SHA/reference; never loaded or string-patched by this production fit",
    )
    args = ap.parse_args()

    configure_runtime_paths()
    import torch
    from rotation_contract import (
        body_motion_temporal,
        export_motion_with_quaternions,
        hemisphere_continuous,
        load_motion_quaternions,
        normalize_quaternions,
        relative_log,
    )

    torch.set_num_threads(4)
    person = Path(args.person).resolve()
    if person.is_dir():
        person = person / "person.json"
    info = json.loads(person.read_text(encoding="utf-8"))
    avatar = Path(args.avatar).resolve()
    if avatar.is_dir():
        avatar = avatar / "avatar.pt"
    initialization = Path(args.initialization).resolve()
    if initialization.is_dir():
        initialization = initialization / "motion.pt"
    reference = Path(args.reference).resolve() if args.reference else None
    out = Path(args.out).resolve()
    if not out.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("output must stay below work/mjj")
    if not person.is_file() or not avatar.is_file() or not initialization.is_file():
        raise FileNotFoundError("person/avatar/initialization missing")
    out.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in info["frames"]]
    if not frames:
        raise ValueError("person frames empty")

    av, poses, q_seed, q_info = select_initialization(avatar, initialization, frames)
    rig = av["rig"]
    rest = rig["rest_joints"][0].float()
    parents = rig["parents"].long()
    scale = float(poses["scale"].reshape(-1)[0]) if torch.is_tensor(poses["scale"]) else float(poses["scale"])
    T = len(frames)
    q_seed = normalize_quaternions(q_seed).float()
    if q_seed.shape[0] != T:
        raise ValueError("initialization q/frames mismatch")
    hand_observations = Path(args.hand_observations).resolve() if args.hand_observations else None
    hand_condition = None
    if hand_observations is not None:
        if not hand_observations.is_file():
            raise FileNotFoundError(hand_observations)
        from hand_condition import HandCondition

        hand_condition = HandCondition(hand_observations, frames)
        q_seed = normalize_quaternions(hand_condition.initialize(q_seed)).float()
    trans = torch.as_tensor(poses["translation_world"], dtype=torch.float32).reshape(T, 3).clone().requires_grad_(True)
    betas = torch.as_tensor(poses["betas"], dtype=torch.float32).reshape(-1)

    meta = json.loads(Path(info["manifest"]).read_text(encoding="utf-8"))
    cams = [str(v["serial"]) for v in meta["views"]]
    P = torch.tensor(
        np.stack(
            [
                np.asarray(v["K_undistorted_native"], np.float64)
                @ np.asarray(v["W"], np.float64)[:3]
                for v in meta["views"]
            ]
        ),
        dtype=torch.float32,
    )
    obs = np.load(Path(info["observations"]), allow_pickle=False)
    xyz = torch.tensor(np.asarray(obs["xyz"], np.float32), dtype=torch.float32)
    valid3d = torch.tensor(np.asarray(obs["valid"], bool)) & torch.isfinite(xyz).all(-1)
    inlier = torch.tensor(np.asarray(obs["inlier"], bool))
    key_dir = Path(info["keypoints"])
    xy = torch.stack(
        [torch.tensor(np.load(key_dir / f"{c}.npz", allow_pickle=False)["xy"], dtype=torch.float32) for c in cams]
    ).permute(1, 0, 2, 3)
    score = torch.stack(
        [torch.tensor(np.load(key_dir / f"{c}.npz", allow_pickle=False)["score"], dtype=torch.float32) for c in cams]
    ).permute(1, 0, 2)
    mask2d = inlier.permute(1, 0, 2) & (score >= 0.5)

    feet_off, tip_off = canonical_offsets()
    L: List[Tuple[int, int, Any, float]] = []
    for cj, sj in BODY.items():
        L.append((int(cj), int(sj), None, 1.0))
    for cj, sj, off in feet_off:
        L.append((int(cj), int(sj), off, 0.10))
    for cj, sj, off in tip_off:
        L.append((int(cj), int(sj), off, 0.10))
    for side in range(2):
        for f in range(5):
            for k in range(3):
                coco = HAND_OFFSET + side * 21 + 1 + 4 * f + k
                joint = HAND_BASE[f] + side * 15 + k
                L.append((int(coco), int(joint), None, 0.20))
    if len(L) != 58:
        raise RuntimeError(f"expected 58 landmarks, got {len(L)}")

    q_root = q_seed[:, 0:1].clone().requires_grad_(True)
    q_body = q_seed[:, 1:22].clone().requires_grad_(True)
    q_jaw = q_seed[:, 22:23].clone()
    q_leye = q_seed[:, 23:24].clone()
    q_reye = q_seed[:, 24:25].clone()
    q_lh = q_seed[:, 25:40].clone()
    q_rh = q_seed[:, 40:55].clone()
    q_root0 = q_root.detach().clone()
    q_body0 = q_body.detach().clone()
    trans0 = trans.detach().clone()

    def assemble():
        ident = torch.cat([q_jaw, q_leye, q_reye], 1)
        return torch.cat([q_root, q_body, ident, q_lh, q_rh], 1)

    wu = np.asarray(args.world_up, np.float64)
    wn = float(np.linalg.norm(wu))
    if wn <= 1e-12:
        raise ValueError("world-up vector is zero")
    wu_t = torch.tensor(wu / wn, dtype=torch.float32)
    foot_ids = [k for k, x in enumerate(L) if x[0] in (17, 18, 19, 20, 21, 22)]

    def points_of(tg, Rg):
        pts = []
        for _cj, sj, off, _w in L:
            base = tg[:, sj]
            if off is not None:
                base = base + torch.einsum("tij,j->ti", Rg[:, sj], off)
            pts.append(base)
        return torch.stack(pts, 1)

    def reprojection(P_, pts):
        homo = torch.cat([pts, torch.ones(pts.shape[0], pts.shape[1], 1)], dim=-1)
        q = torch.einsum("vij,tnj->tvni", P_, homo)
        return q[..., :2] / q[..., 2:3].clamp_min(1e-6)

    def evaluate(tg, Rg):
        world = points_of(tg, Rg) * scale + trans.unsqueeze(1)
        uv = reprojection(P, world)
        idx = torch.tensor([x[0] for x in L])
        errs = []
        for k in range(len(L)):
            m = mask2d[:, :, int(idx[k])]
            if bool(m.any()):
                errs.append((uv[:, :, k][m] - xy[:, :, int(idx[k])][m]).norm(dim=-1).mean())
        return float(torch.stack(errs).mean())

    def loss_fn(stage):
        tg, Rg = fk_q(assemble(), rest, parents)
        world = points_of(tg, Rg) * scale + trans.unsqueeze(1)
        uv = reprojection(P, world)
        idx = torch.tensor([x[0] for x in L])
        wgt = torch.tensor(
            [
                x[3] if (stage >= 2 or (x[0] < 91 and (stage >= 1 or x[0] < 17))) else 0.0
                for x in L
            ],
            dtype=torch.float32,
        )
        data2d = torch.zeros(())
        for k in range(len(L)):
            cj = int(idx[k])
            m = mask2d[:, :, cj]
            if not bool(m.any()):
                continue
            # Mask BEFORE subtraction/norm: absent xy may be NaN.
            err = (uv[:, :, k][m] - xy[:, :, cj][m]).norm(dim=-1) / 5.0
            data2d = data2d + wgt[k] * torch.nn.functional.huber_loss(
                err, torch.zeros_like(err), reduction="none", delta=1.0
            ).mean()
        data2d = data2d / wgt.sum()

        data3d = torch.zeros(())
        for k in range(len(L)):
            cj = int(idx[k])
            m = valid3d[:, cj]
            if not bool(m.any()):
                continue
            d = (world[m, k] - xyz[m, cj]).norm(dim=-1) / 0.02
            data3d = data3d + wgt[k] * torch.nn.functional.huber_loss(
                d, torch.zeros_like(d), reduction="mean", delta=1.0
            )
        data3d = data3d / wgt.sum()

        tempo = body_motion_temporal(q_root, q_body, trans)
        prior = 0.5 * (relative_log(q_body, q_body0).square().mean()) / 1e-3
        prior = prior + 0.5 * ((trans - trans0).square().mean()) / 1e-4
        if foot_ids:
            fy = torch.einsum("tnj,j->tn", world[:, foot_ids], wu_t)
            low = fy.min(dim=1).values
            contact = (low - low.median()).abs().mean() / 1e-3
            tempo = tempo + 0.2 * contact
        image_hand_loss = torch.zeros((), device=trans.device, dtype=trans.dtype)
        if stage >= 2 and hand_condition is not None:
            image_hand_loss = hand_condition.loss(assemble())
        return data2d + data3d + TEMPORAL_WEIGHT * tempo + prior + image_hand_loss

    with torch.no_grad():
        tg, Rg = fk_q(assemble(), rest, parents)
        before = evaluate(tg, Rg)
    print("baseline reproj", round(before, 2), "px", flush=True)
    started = time.time()
    history: List[Dict[str, Any]] = []
    all_finite = True

    for stage, steps in enumerate(STEPS):
        if stage == 2:
            q_lh.requires_grad_(True)
            q_rh.requires_grad_(True)
        params = [q_root, q_body, trans] + ([q_lh, q_rh] if stage >= 2 else [])
        opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=4e-4 if stage == 0 else 2e-4)
        for it in range(steps):
            opt.zero_grad()
            loss = loss_fn(stage)
            loss.backward()
            grad_ok = all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in params if p.requires_grad
            )
            opt.step()
            with torch.no_grad():
                q_root.copy_(normalize_quaternions(q_root))
                q_body.copy_(normalize_quaternions(q_body))
                q_lh.copy_(normalize_quaternions(q_lh))
                q_rh.copy_(normalize_quaternions(q_rh))
            if it % max(1, steps // 4) == 0 or it == steps - 1:
                with torch.no_grad():
                    tg, Rg = fk_q(assemble(), rest, parents)
                    h = {
                        "stage": stage,
                        "iter": it,
                        "loss": round(float(loss), 4),
                        "reproj_px": round(evaluate(tg, Rg), 2),
                        "image_hand_loss": (
                            round(float(hand_condition.loss(assemble())), 6)
                            if stage >= 2 and hand_condition is not None else 0.0
                        ),
                    }
                    history.append(h)
                    print(h, flush=True)
            all_finite = all_finite and bool(
                torch.isfinite(loss)
                and grad_ok
                and torch.isfinite(q_root).all()
                and torch.isfinite(q_body).all()
                and torch.isfinite(q_lh).all()
                and torch.isfinite(q_rh).all()
                and torch.isfinite(trans).all()
            )

    with torch.no_grad():
        tg, Rg = fk_q(assemble(), rest, parents)
        after = evaluate(tg, Rg)
        joints = tg * scale + trans.unsqueeze(1)
        q_final = normalize_quaternions(assemble()).float()
        q_store = hemisphere_continuous(q_final)
        exported = export_motion_with_quaternions(
            {
                **poses,
                "translation_world": trans.detach().cpu(),
                "scale": torch.tensor(scale),
                "betas": betas,
                "frames": torch.tensor(frames, dtype=torch.long),
            },
            q_store,
        )
        exported["joints55_world"] = joints.cpu()
        exported["frames"] = torch.tensor(frames, dtype=torch.long)
        torch.save(exported, out / "motion.pt")
        q_exported, q_export_info = load_motion_quaternions(exported, verify_legacy=True)

    history_path = out / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "iter", "loss", "reproj_px", "image_hand_loss"])
        w.writeheader()
        w.writerows(history)

    reference_sha = sha256_file(reference) if reference is not None and reference.is_file() else None
    report = {
        "schema": "two_person_quaternion_fit/v2",
        "success": True,
        "person": str(person),
        "identity": info.get("identity"),
        "avatar": str(avatar),
        "initialization": str(initialization),
        "initialization_q": q_info,
        "frames": frames,
        "T": T,
        "landmarks": len(L),
        "feet": len(feet_off),
        "fingertips": len(tip_off),
        "stage_steps": STEPS,
        "temporal_weight": TEMPORAL_WEIGHT,
        "optimizer": "Adam",
        "learning_rate": "4e-4 stage0, 2e-4 stages1/2",
        "mean_reproj_px_before": before,
        "mean_reproj_px_after": after,
        "history": history,
        "history_csv": str(history_path),
        "history_finite": bool(all_finite),
        "seconds": round(time.time() - started, 1),
        "q_order": "wxyz",
        "q_layout": "root,body21,jaw,leye,reye,lhand15,rhand15",
        "q_representation": "normalized local q; q/-q same rotation; no AA roundtrip in objective",
        "legacy_boundary_only": True,
        "jaw_eyes_preserved_from_input": bool(
            "jaw_pose" in poses or "leye_pose" in poses or "reye_pose" in poses
            or bool(q_seed[:, 22:25, 1:].abs().max() > 1e-6)
        ),
        "objective_changes_from_original": {
            "root_time": "raw AA second difference -> SO(3) relative-log angular acceleration /5e-3",
            "body_time": "raw AA second difference -> SO(3) relative-log angular acceleration /5e-3",
            "body_trust": "raw AA difference -> relative_log(q_body, q_body0) 0.5 square /1e-3",
            "hands": "original staged observation loss plus optional image-hand tangent likelihood at stage2; no hand temporal prior",
            "mathematically_identical_to_original": False,
        },
        "hand_observations": (
            hand_condition.provenance() if hand_condition is not None
            else {"provided": False}
        ),
        "data_contract": {
            "source": "direct expansion of reference/fit_motion.py, not string-patched",
            "body": "COCO 12 body joints",
            "feet": "6 canonical toe/heel offsets",
            "fingertips": "10 canonical fingertip offsets",
            "hand_joints": "30 patched COCO->SMPLX hand joints",
            "missingness": "inlier.permute(1,0,2)&score>=.5; mask before arithmetic",
            "world_up": (wu / wn).tolist(),
            "world_up_source": args.world_up_source,
            "stage_data_isolation": "body -> +feet -> +hands (zeros weights outside stage)",
            "feet_contact": "world[:,foot_ids] dot world-up; unchanged factor 0.2/1e-3",
        },
        "reference": {
            "path": str(reference) if reference else None,
            "sha256": reference_sha,
            "executed": False,
            "note": "--reference is provenance only; this production fit does not load or patch it",
        },
        "export": {
            "local_quaternions_wxyz": True,
            "consistent_legacy": True,
            "legacy_aliases": [
                "root_pose", "body_pose", "lhand_pose", "rhand_pose",
                "left_hand_pose", "right_hand_pose",
            ],
            "q_export_consistency": q_export_info,
            "q_shape": list(q_exported.shape),
            "actual_fk55_world": True,
        },
        "no_stage_directory": True,
        "no_thin_avatar_or_seed_copy": True,
        "no_reference_string_patch": True,
        "no_quality_claim": True,
        "scope": "Direct q-native fit preserving the existing 58-landmark data/stage contract; SO(3) time/trust semantics differ from old AA targets.",
    }
    atomic_write_json(out / "report.json", report)
    atomic_write_json(out / "quaternion_provenance.json", report)
    atomic_write_json(out / "reference_provenance.json", report["reference"])
    print(f"[quaternion-fit] wrote {out / 'motion.pt'}")
    print(f"[quaternion-fit] report {out / 'report.json'} history {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
