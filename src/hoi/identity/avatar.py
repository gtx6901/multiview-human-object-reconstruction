"""Export decoded LHM appearance as a replayable tensor avatar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def move(x, device):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [move(v, device) for v in x]
    return x


def export_avatar(
    model: Any,
    poses: dict[str, Any],
    output: Path,
    attributes_path: Path,
    latent_path: Path,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write output/avatar.pt and return the bundle with its tensors on CUDA.

    The saved bundle contains CPU tensors, source paths and the upstream
    geometry comparison. The supplied model renderer is moved to CUDA, as in
    the original export contract. No model loading, encoding or fusion runs.
    """
    from hoi.runtime import runtime_settings, configure_runtime

    settings = configure_runtime(runtime_settings(runtime))
    project_root = Path(settings["project_root"])
    lhm_root = Path(settings["lhm_root"])
    import torch
    from core.outputs.output import GaussianAppOutput
    from hoi.rendering.rig import animate

    target = Path(output).expanduser()
    target = (target if target.is_absolute() else project_root / target).resolve()
    attributes_path = Path(attributes_path).expanduser()
    attributes_path = (
        attributes_path if attributes_path.is_absolute() else project_root / attributes_path
    ).resolve()
    latent_path = Path(latent_path).expanduser()
    latent_path = (
        latent_path if latent_path.is_absolute() else project_root / latent_path
    ).resolve()
    model.renderer.cuda()
    model.renderer.device = torch.device("cuda")
    skinner = model.renderer.smplx_model
    target.mkdir(parents=True, exist_ok=True)
    data = move(torch.load(attributes_path, weights_only=True), "cuda")
    latent = torch.load(latent_path, weights_only=True)["result"]
    poses = move(poses, "cuda")
    with torch.inference_mode():
        query = data["query"]["neutral_coords"][0] + data["attrs"][0]["offset_xyz"]
        N = len(query)
        weights = skinner.query_voxel_skinning_weights(query[None]).reshape(N, -1)
        fixed = (skinner.is_rhand + skinner.is_lhand + skinner.is_face) > 0
        weights[fixed] = skinner.skinning_weight[fixed]
        neutral_to_zero = (
            weights @ data["smplx"]["transform_mat_neutral_pose"][0].reshape(-1, 16)
        ).reshape(N, 4, 4)
        zero = skinner.lbs(query[None], neutral_to_zero[None], torch.zeros(1, 3, device="cuda"))[0]
        if poses["betas"].abs().max() > 0:
            from smplx.lbs import blend_shapes

            zero = zero + blend_shapes(poses["betas"], skinner.shape_dirs)[0]
        rest = skinner.get_zero_pose_human(poses["betas"], torch.device("cuda"), None, None)
        rig = {
            "weights": weights.float(),
            "zero_coords": zero.float(),
            "neutral_to_zero": neutral_to_zero.float(),
            "rest_joints": rest.float(),
            "parents": skinner.smplx_layer.parents.cuda(),
            "neutral_rotation": data["attrs"][0]["rotation"].float(),
            **{k: data["attrs"][0][k].float() for k in ["opacity", "scaling", "shs"]},
        }
        params = {
            k: torch.zeros(1, 3, device="cuda")
            for k in ["jaw_pose", "leye_pose", "reye_pose", "trans"]
        }
        for key in ["jaw_pose", "leye_pose", "reye_pose"]:
            if key in poses:
                params[key] = poses[key][:1].reshape(1, 3)
        params.update(
            root_pose=poses["root_pose"][:1],
            body_pose=poses["body_pose"][:1],
            betas=poses["betas"],
            lhand_pose=poses.get("left_hand_pose", torch.zeros(1, 15, 3, device="cuda"))[:1],
            rhand_pose=poses.get("right_hand_pose", torch.zeros(1, 15, 3, device="cuda"))[:1],
            expr=torch.zeros(1, 100, device="cuda"),
            transform_mat_neutral_pose=data["smplx"]["transform_mat_neutral_pose"][0],
        )
        orig, _ = model.renderer.animate_gs_model(
            GaussianAppOutput(**data["attrs"][0]),
            data["query"]["neutral_coords"][0],
            params,
            mesh_meta=data["query"]["mesh_meta"],
        )
        g = orig[0]
        replay = animate(rig, poses, 0)
        local_error = float(
            (
                (g.xyz * poses["scale"] + poses["translation_world"][0] - replay["xyz"])
                / poses["scale"]
            )
            .abs()
            .max()
        )
        rot_error = float(
            1
            - torch.nn.functional.cosine_similarity(g.rotation.float(), replay["rotation"], dim=-1)
            .abs()
            .min()
        )
        assert local_error < 2e-5 and rot_error < 2e-5, (local_error, rot_error)
        bundle = {
            "rig": move(rig, "cpu"),
            "poses": move(poses, "cpu"),
            "features": latent[0][0].cpu(),
            "background_embedding": model.renderer.background_embedding.detach().cpu(),
            "decoder": move(model.neural_renderer.state_dict(), "cpu"),
            "decoder_config": json.loads((lhm_root / "checkpoint_config.json").read_text())[
                "neural_renderer"
            ],
            "scope": "Independent decoded LHM appearance; supplied shared beta applied to dense Gaussian shape blend and rest rig. Supplied full pose fields.",
        }
        bundle["rig_check"] = {
            "gaussians": N,
            "upstream_max_xyz_local_error": local_error,
            "upstream_max_quaternion_cosine_error": rot_error,
            "shared_betas": poses["betas"].cpu().tolist(),
            "sources": [str(attributes_path), str(latent_path)],
        }
        torch.save(bundle, target / "avatar.pt")
    return move(bundle, "cuda")
