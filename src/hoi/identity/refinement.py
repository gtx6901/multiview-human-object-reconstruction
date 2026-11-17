"""Fit a shared facial identity to calibrated images, without changing motion.

Normal displacements are shared by Gaussians attached to each SMPL-X vertex.
Features and covariance scales stay in canonical space and move with the rig.
Camera RGB responses are nuisance parameters with a zero-mean log gauge;
they are not independently measured exposure values or a relighting model.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from hoi.io import write_json
from hoi.runtime import PROJECT_ROOT, configure_runtime, load_rasterization


def load_refinement_targets(segment_index, actor_ids, frames, width, height):
    """Use exact source/local mask rows; an unavailable actor is never background."""
    index_path = Path(segment_index).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else index_path.parent / path

    samples, unavailable = [], []
    for camera in index["cameras"]:
        serial = str(camera["serial"])
        records = {int(row["source_frame"]): row for row in camera["frames"]}
        if len(records) != len(camera["frames"]):
            raise ValueError(f"{serial}: duplicate source frames")
        with np.load(resolve(camera["mask_npz"]), allow_pickle=False) as data:
            packed = data["packed"]
            native_h, native_w = map(int, data["mask_shape"])
            actors = data["actors"].tolist()
            local_frames, present = data["local_frames"], data["present"]
        rows = {int(local): row for row, local in enumerate(local_frames)}
        if len(rows) != len(local_frames):
            raise ValueError(f"{serial}: duplicate mask local frames")
        actor_indices = [actors.index(actor) for actor in actor_ids]
        expected = (len(rows), len(actors), native_h, (native_w + 7) // 8)
        if packed.dtype != np.uint8 or packed.shape != expected:
            raise ValueError(f"{serial}: invalid packed mask layout")
        for source_frame in frames:
            base = {"serial": serial, "source_frame": int(source_frame)}
            record = records.get(source_frame)
            row = None if record is None else rows.get(int(record["local_frame"]))
            if row is None or not present[row, actor_indices].all():
                unavailable.append(base)
                continue
            masks = np.unpackbits(
                packed[row, actor_indices], axis=-1, count=native_w, bitorder="big"
            )
            mask = masks.any(axis=0).astype(np.float32)
            with Image.open(resolve(record["jpeg"])) as image:
                if image.size != (native_w, native_h):
                    raise ValueError(f"{serial}/{source_frame}: image/mask size mismatch")
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            # Composite at native resolution before area resampling the boundary.
            rgb = rgb * mask[..., None] + 1 - mask[..., None]
            samples.append(
                dict(
                    base,
                    target_rgb=torch.from_numpy(
                        cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                    ),
                    target_mask=torch.from_numpy(
                        cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
                    ),
                )
            )
    return samples, unavailable


def linear_rgb(value):
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def encoded_rgb(value):
    # The caller applies sensor saturation after the transfer function.
    value = value.clamp_min(0)
    return torch.where(
        value <= 0.0031308, value * 12.92, 1.055 * value.clamp_min(1e-8).pow(1 / 2.4) - 0.055
    )


class FacialIdentity(torch.nn.Module):
    def __init__(self, bundle, bindings):
        super().__init__()
        self.rig = bundle["rig"]
        face = bindings["is_face"].bool().cuda()
        self.indices = face.nonzero().flatten()
        ids = bindings["query_indx"][face.cpu()].long().cuda()
        self.vertex_ids, self.inverse = ids.unique(return_inverse=True)
        # SMPL-X v_template/shapedirs are already in the zero-pose frame.
        # Recompute normals after beta deformation; neutral_to_zero would
        # incorrectly transform these a second time.
        vertices = bindings["v_template"].float().cuda() + torch.einsum(
            "vck,k->vc",
            bindings["shapedirs"].float().cuda(),
            bundle["poses"]["betas"].flatten()[:10],
        )
        faces = bindings["faces_tensor"].long().cuda()
        triangles = vertices[faces]
        normals = torch.zeros_like(vertices)
        area_normal = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        for corner in range(3):
            normals.index_add_(0, faces[:, corner], area_normal)
        self.normals = torch.nn.functional.normalize(normals[ids], dim=-1)
        self.features = bundle["features"].float()
        self.feature_std = self.features[face].std(0).clamp_min(1e-4)
        self.displacement = torch.nn.Parameter(torch.zeros(len(self.vertex_ids), device="cuda"))
        self.feature_delta = torch.nn.Parameter(
            torch.zeros((len(ids), self.features.shape[1]), device="cuda")
        )
        self.log_scale = torch.nn.Parameter(torch.zeros((len(ids), 3), device="cuda"))
        lookup = torch.full((int(faces.max()) + 1,), -1, device="cuda", dtype=torch.long)
        lookup[self.vertex_ids] = torch.arange(len(self.vertex_ids), device="cuda")
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = lookup[edges]
        self.edges = edges[(edges >= 0).all(-1)].sort(-1).values.unique(dim=0)

    def offset(self):
        return self.normals * self.displacement[self.inverse, None]

    def appearance(self):
        return self.features.index_add(0, self.indices, self.feature_delta)

    def gaussians(self, baseline, transform, scale):
        result = dict(baseline)
        delta = torch.einsum("nij,nj->ni", transform, self.offset()) * scale
        result["xyz"] = baseline["xyz"].index_add(0, self.indices, delta)
        result["scaling"] = baseline["scaling"].index_copy(
            0, self.indices, baseline["scaling"][self.indices] * self.log_scale.exp()
        )
        return result

    def regularization(self, parameters):
        sigma = parameters["displacement_prior_m"]
        d = self.displacement
        smooth = (d[self.edges[:, 0]] - d[self.edges[:, 1]]).square().mean()
        return (
            parameters["geometry_prior"] * (d / sigma).square().mean()
            + parameters["surface_smoothness"] * smooth / sigma**2
            + parameters["feature_prior"] * (self.feature_delta / self.feature_std).square().mean()
            + parameters["scale_prior"] * self.log_scale.square().mean()
        )


def refine_scene(config, frames=None, updates=None):
    """One shared objective for every actor and dataset; save one rolling state."""
    configure_runtime(config.runtime)
    from safetensors import safe_open
    from smplx.lbs import batch_rigid_transform
    from pytorch3d.transforms import axis_angle_to_matrix
    from hoi.rendering.rig import animate, pose_axis_angles
    from hoi.rendering.scene import (
        build_decoder,
        camera_from_scene,
        compare_decoders,
        frame_list,
        load_bundle,
        load_motion,
        load_scene_config,
        rasterize_scene,
        sha256_file,
        to_device,
    )

    parameters = json.loads(
        (PROJECT_ROOT / "configs/reconstruction.json").read_text(encoding="utf-8")
    )["identity_refinement"]
    parameters.update(config.identity_refinement)
    frames = list(frames if frames is not None else parameters["frames"])
    updates = int(updates if updates is not None else parameters["updates"])
    if not frames or len(set(frames)) != len(frames) or updates <= 0:
        raise ValueError("Refinement needs unique source frames and a positive update target")
    width, height = int(parameters["width"]), int(parameters["height"])
    segment = Path(config.preparation["segment_index"])
    if not segment.is_absolute():
        segment = PROJECT_ROOT / segment
    samples, unavailable = load_refinement_targets(
        segment, [a.id for a in config.actors], frames, width, height
    )
    if not samples:
        raise ValueError("No image has masks for all requested actors")
    output = config.output / "identity_refinement"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        a.id: {"avatar": sha256_file(a.avatar), "motion": sha256_file(a.motion)}
        for a in config.actors
    }
    signature = {
        "sources": sources,
        "segment_index": sha256_file(segment),
        "parameters": parameters,
        "frames": frames,
        "implementation": {
            name: sha256_file(PROJECT_ROOT / name)
            for name in (
                "src/hoi/identity/refinement.py",
                "src/hoi/rendering/scene.py",
                "src/hoi/rendering/rig.py",
            )
        },
    }
    write_json(output / "inputs.json", dict(signature, unavailable=unavailable))
    scene, center, world_height, _ = load_scene_config(asdict(config), PROJECT_ROOT)
    raster = load_rasterization(config.runtime)
    checkpoint_path = Path(config.runtime["lhm_root"]) / "weights/LHMPP-700M/model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu", backend="pread") as weights:
        bindings = {
            key: weights.get_tensor("renderer.smplx_model." + key)
            for key in ("is_face", "query_indx")
        }
        bindings["faces_tensor"] = weights.get_tensor(
            "renderer.smplx_model.smplx_layer.faces_tensor"
        )
        bindings["v_template"] = weights.get_tensor("renderer.smplx_model.smplx_layer.v_template")
        bindings["shapedirs"] = weights.get_slice("renderer.smplx_model.smplx_layer.shapedirs")[
            :, :, :10
        ]
    identities, bundles, motions, base_states = [], [], [], []
    for actor in config.actors:
        bundle = load_bundle(actor.avatar)
        if bundles:
            compare_decoders(bundles[0], bundle)
        motion = load_motion(actor.motion, bundle["poses"], bundle["rig"]["rest_joints"].dtype)
        indices = {frame: i for i, frame in enumerate(frame_list(motion))}
        if not set(frames) <= indices.keys():
            raise ValueError(f"{actor.id}: requested source frames missing from motion")
        bundle = to_device(bundle, "cuda")
        motion = to_device(motion, "cuda")
        identity = FacialIdentity(bundle, bindings)
        states = {}
        with torch.no_grad():
            for frame in frames:
                t = indices[frame]
                _, transforms = batch_rigid_transform(
                    axis_angle_to_matrix(pose_axis_angles(motion, t, "cuda")),
                    bundle["rig"]["rest_joints"],
                    bundle["rig"]["parents"],
                )
                transforms = (
                    bundle["rig"]["weights"][identity.indices] @ transforms[0].reshape(-1, 16)
                ).reshape(-1, 4, 4)
                states[frame] = (animate(bundle["rig"], motion, t), transforms[:, :3, :3])
        identities.append(identity)
        bundles.append(bundle)
        motions.append(motion)
        base_states.append(states)
    decoder = build_decoder(bundles[0], "cuda").requires_grad_(False)
    background = bundles[0]["background_embedding"][2].float()
    groups = []
    for identity in identities:
        groups.extend(
            [
                {"params": [identity.displacement], "lr": parameters["geometry_lr"]},
                {"params": [identity.feature_delta], "lr": parameters["feature_lr"]},
                {"params": [identity.log_scale], "lr": parameters["scale_lr"]},
            ]
        )
    serials = sorted({s["serial"] for s in samples})
    log_gain = torch.nn.Parameter(torch.zeros((len(serials), 3), device="cuda"))
    groups.append({"params": [log_gain], "lr": parameters["response_lr"]})
    optimizer = torch.optim.Adam(groups)
    random = np.random.default_rng(parameters["seed"])
    start, queue = 0, []
    path = output / "training.pt"
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Refinement checkpoint inputs/options changed; do not mix experiments")
        for identity, saved in zip(identities, state["identities"]):
            identity.load_state_dict(saved)
        log_gain.data.copy_(state["log_gain"].cuda())
        optimizer.load_state_dict(state["optimizer"])
        random.bit_generator.state = state["rng"]
        start, queue = state["step"], state["queue"]

    flags = []
    for identity, bundle in zip(identities, bundles):
        flag = bundle["features"].new_zeros((len(bundle["features"]), 1))
        flag[identity.indices] = 1
        flags.append(flag)
    # Weights come from the initial joint visibility, never the trainable mask.
    # This prevents lowering opacity/coverage to escape image supervision.
    with torch.no_grad():
        for sample in samples:
            K, W = camera_from_scene(
                scene, sample["serial"], center, world_height, width, height, config.camera_space
            )
            baseline = [states[sample["source_frame"]][0] for states in base_states]
            fields, _ = rasterize_scene(
                raster,
                baseline,
                [b["features"].float() for b in bundles],
                K,
                W,
                width,
                height,
                background,
                flags,
            )
            sample["weight"] = (
                fields[0, ..., -2].float().clamp(0, 1).cpu()
                + parameters["body_weight"] * sample["target_mask"]
            )
            sample["K"], sample["W"] = K, W
    began = time.perf_counter()
    for step in range(start, updates):
        if not queue:
            queue = random.permutation(len(samples)).tolist()
        sample = samples[queue.pop()]
        frame = sample["source_frame"]
        gs = [
            identity.gaussians(*states[frame], motion["scale"])
            for identity, states, motion in zip(identities, base_states, motions)
        ]
        fields, alpha = rasterize_scene(
            raster,
            gs,
            [i.appearance() for i in identities],
            sample["K"],
            sample["W"],
            width,
            height,
            background,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rgb, mask = decoder.predict_rgbs(
                None, fields[0, ..., 3:-1].permute(2, 0, 1)[None], None, height, width
            )
        coverage = mask[0, 0].float().squeeze() * (alpha[0, ..., 0] > 0.001)
        response = log_gain - log_gain.mean(0, keepdim=True)
        gain = response[serials.index(sample["serial"])].exp()
        rgb = encoded_rgb(linear_rgb(rgb[0, 0].float()) * gain).clamp(0, 1)
        predicted = rgb * coverage[..., None] + 1 - coverage[..., None]
        target = sample["target_rgb"].cuda()
        weight = sample["weight"].cuda()
        residual = predicted - target
        photo = (residual.abs() * weight[..., None]).sum() / (3 * weight.sum())
        gradient = sum(
            (
                (residual.diff(dim=axis)).abs()
                * weight.narrow(axis, 0, weight.shape[axis] - 1)[..., None]
            ).sum()
            / (3 * weight.sum())
            for axis in (0, 1)
        )
        silhouette = ((coverage - sample["target_mask"].cuda()).abs() * weight).sum() / weight.sum()
        prior = sum(identity.regularization(parameters) for identity in identities)
        loss = (
            photo
            + parameters["gradient_weight"] * gradient
            + parameters["silhouette_weight"] * silhouette
            + prior
            + parameters["response_prior"] * response.square().mean()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite objective at update {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for identity in identities:
            for value in identity.parameters():
                if value.grad is None or not torch.isfinite(value.grad).all():
                    raise FloatingPointError(
                        f"Missing/non-finite identity gradient at update {step + 1}"
                    )
        optimizer.step()
        if (step + 1) % 10 == 0 or step == start:
            torch.cuda.synchronize()
            print(
                f"[identity refine] {step + 1}/{updates} photo={photo.item():.6f} grad={gradient.item():.6f} prior={prior.item():.6f} sec/update={(time.perf_counter() - began)/(step + 1 - start):.3f} allocated_GB={torch.cuda.max_memory_allocated()/1e9:.3f}",
                flush=True,
            )
        if (step + 1) % parameters["checkpoint_interval"] == 0 or step + 1 == updates:
            from hoi.identity.avatar import move

            state = {
                "step": step + 1,
                "signature": signature,
                "identities": [move(i.state_dict(), "cpu") for i in identities],
                "log_gain": log_gain.detach().cpu(),
                "optimizer": move(optimizer.state_dict(), "cpu"),
                "rng": random.bit_generator.state,
                "queue": queue,
            }
            temporary = output / "training.tmp"
            torch.save(state, temporary)
            temporary.replace(path)
    from hoi.identity.avatar import move

    for actor, bundle, identity in zip(config.actors, bundles, identities):
        bundle["rig"]["zero_coords"] = bundle["rig"]["zero_coords"].index_add(
            0, identity.indices, identity.offset()
        )
        bundle["rig"]["scaling"] = bundle["rig"]["scaling"].index_copy(
            0,
            identity.indices,
            bundle["rig"]["scaling"][identity.indices] * identity.log_scale.exp(),
        )
        bundle["features"] = identity.appearance()
        bundle["identity_refinement"] = {
            "updates": updates,
            "inputs": signature,
            "scope": "canonical face geometry, features and covariance; fixed supplied motion, frozen shared decoder",
        }
        destination = output / actor.id
        destination.mkdir(exist_ok=True)
        torch.save(move(bundle, "cpu"), destination / "avatar.pt")
    write_json(
        output / "response.json",
        {
            "serials": serials,
            "linear_rgb_gain": (log_gain - log_gain.mean(0)).exp().detach().cpu().tolist(),
            "scope": "joint nuisance estimate, not independent physical exposure calibration",
        },
    )
    write_json(
        output / "report.json",
        {
            "updates": updates,
            "samples": len(samples),
            "unavailable": unavailable,
            "elapsed_seconds": time.perf_counter() - began,
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "accepted": False,
        },
    )
