"""Render one continuously rotating novel view over the full sequence.

The stock renderer uses a fixed camera per view, so a rotating shot is produced here:
for frame t the camera sits on a circle around the subject at angle 2*pi*t/T, with the
same intrinsics/radius as the existing orbit cameras.
"""
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LHM = PROJECT / "lhmpp"
sys.path.insert(0, str(LHM))
from runtime_imports import configure
configure()
sys.path.insert(0, str(PROJECT / "deps"))

import numpy as np
import torch
from PIL import Image
import importlib.util
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from mesh_renderer import MeshRenderer
from hybrid_compositor import composite, mesh_visible_opacities, composite_clipped_body

spec = importlib.util.spec_from_file_location("gsplat_cuda", __import__("os").environ["GSPLAT_CUDA"])
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
sys.modules["gsplat.csrc"] = backend
from gsplat import rasterization


def look_at(C, T, up=(0.0, 1.0, 0.0)):
    C = np.asarray(C, np.float64); T = np.asarray(T, np.float64)
    f = T - C; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    W = np.eye(4)
    W[:3, 0], W[:3, 1], W[:3, 2] = r, u, f
    W[:3, 3] = -W[:3, :3] @ C
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="assets/human/avatar_shape_smooth_v2")
    ap.add_argument("--motion", default="assets/helmet/worn_helmet_residual_refit_v1")
    ap.add_argument("--mesh", default="assets/helmet/helmet_texture_consistent_lhm_v1")
    ap.add_argument("--cameras", default="renders/temporal_body_full300_v4/cameras.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=2.26)
    ap.add_argument("--height", type=float, default=-0.09)
    ap.add_argument("--start-deg", type=float, default=-128.0)
    args = ap.parse_args()

    out = PROJECT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads((PROJECT / args.cameras).read_text(encoding="utf-8"))
    width, height = cams["width"], cams["height"]
    K = torch.tensor(np.array(cams["views"][8]["K"]), device="cuda", dtype=torch.float32)

    bundle = torch.load(PROJECT / args.avatar / "avatar.pt", map_location="cpu", weights_only=False)
    with init_empty_weights(include_buffers=False):
        decoder = PatchDPT4DecoderOnly(**bundle["decoder_config"])
    decoder.transformer_block = torch.nn.Module()
    decoder.load_state_dict(bundle["decoder"], strict=True, assign=True)
    decoder.eval().cuda()
    rig = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["rig"].items()}
    poses = {k: v.cuda() if torch.is_tensor(v) else v for k, v in bundle["poses"].items()}
    features = bundle["features"].cuda().float()
    bg = bundle["background_embedding"][2].cuda().float()
    motion = {k: v.cuda() if torch.is_tensor(v) else v for k, v in torch.load(PROJECT / args.motion / "sequence.pt", weights_only=False).items()}
    mesh = {k: v.cuda() if torch.is_tensor(v) else v for k, v in torch.load(PROJECT / args.mesh / "mesh.pt", weights_only=False).items()}
    renderer = MeshRenderer()
    T = len(poses["frames"])
    target = torch.tensor([0.0, 0.0, 0.0])

    with torch.inference_mode():
        for t in range(T):
            frame = int(poses["frames"][t])
            g = animate(rig, poses, t)
            world = mesh["vertices_local"].float() @ motion["R"][t].float().T * motion["logscale"].float().exp() + motion["translation_world"][t].float()
            deg = args.start_deg + 360.0 * t / T
            th = math.radians(deg)
            C = np.array([args.radius * math.cos(th), args.height, args.radius * math.sin(th)])
            W = torch.tensor(look_at(C, target.cpu().numpy()), device="cuda", dtype=torch.float32)
            mr = renderer(world, mesh["faces"], K, W, width, height, mesh["colors"].float(),
                          uv=mesh.get("uv"), texture=mesh.get("texture"))
            color, alpha, _ = rasterization(
                means=g["xyz"], quats=g["rotation"], scales=g["scaling"], opacities=g["opacity"].flatten(),
                colors=torch.cat([g["shs"].squeeze(1), features], 1),
                viewmats=W[None], Ks=K[None], width=width, height=height,
                backgrounds=torch.cat([torch.ones(3, device="cuda"), bg])[None],
                packed=False, render_mode="RGB+ED", near_plane=.01, far_plane=100., eps2d=.3)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb, mask = decoder.predict_rgbs(None, color[0, ..., 3:-1].permute(2, 0, 1)[None], None, height, width)
            ba = mask[0, 0].float().squeeze() * (alpha[0, ..., 0] > .001)
            result = composite(rgb[0, 0].float() * ba[..., None], ba, color[0, ..., -1], mr)
            arr = np.uint8(result["rgb"].cpu().numpy().clip(0, 1) * 255)
            Image.fromarray(arr).save(out / f"{frame:04d}.png")
            if t % 50 == 0:
                print("frame", frame, "deg", round(deg, 1), flush=True)
    print("done", T, flush=True)


if __name__ == "__main__":
    main()
