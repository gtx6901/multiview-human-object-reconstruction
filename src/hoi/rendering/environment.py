"""Compose animated Gaussian actors with a lit glTF environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from hoi.io import write_json
from hoi.results import record_video, video_path
from hoi.runtime import PROJECT_ROOT, configure_runtime, load_rasterization


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, down, forward])
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def linear_rgb(value):
    return torch.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def display_rgb(value):
    value = value.clamp(min=0)
    return torch.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)


def placement_basis(config, initial_gaussians, reference_serial=None):
    """One stable, upright local frame shared by the editor and final render."""
    up = np.asarray(config.world_up, float)
    up /= np.linalg.norm(up)
    scene = json.loads(config.scene_path.read_text(encoding="utf-8-sig"))
    reference_serial = reference_serial or scene["views"][0]["serial"]
    view = next((v for v in scene["views"] if v["serial"] == reference_serial), scene["views"][0])
    matrix = np.asarray(view.get("calibration", view)["W"])
    forward = -matrix[2, :3].copy()
    forward -= up * (forward @ up)
    forward /= np.linalg.norm(forward)
    rotation = np.stack([np.cross(up, forward), up, forward])
    points = torch.cat([g["xyz"] for g in initial_gaussians]).cpu().numpy()
    center = points.mean(0)
    center += up * ((points @ up).min() - center @ up)
    return rotation, center


class EnvironmentRenderer:
    """Opaque glTF base color, diffuse lights and depth-tested Gaussian shadows."""

    def __init__(self, options):
        import trimesh
        import nvdiffrast.torch as dr
        from hoi.rendering.mesh import MeshRenderer

        self.dr = dr
        self.mesh_renderer = MeshRenderer()
        self.options = options
        model = Path(options["model"])
        self.model_path = model if model.is_absolute() else PROJECT_ROOT / model
        scene = trimesh.load(self.model_path, force="scene")
        groups = {}
        scale = float(options.get("model_scale", 1))
        offset = np.asarray(options.get("model_offset", [0, 0, 0]))
        for node in scene.graph.nodes_geometry:
            transform, name = scene.graph[node]
            mesh = scene.geometry[name]
            material = getattr(mesh.visual, "material", None)
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            factor = getattr(material, "baseColorFactor", None)
            factor = np.asarray([1, 1, 1, 1] if factor is None else factor, float)
            if factor.max() > 1:
                factor /= 255
            key = (id(image), tuple(factor), getattr(material, "name", ""))
            vertices = trimesh.transform_points(mesh.vertices, transform) * scale + offset
            normal_matrix = np.linalg.inv(transform[:3, :3]).T
            normals = np.asarray(mesh.vertex_normals) @ normal_matrix.T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            uv = getattr(mesh.visual, "uv", None)
            uv = np.zeros((len(vertices), 2)) if uv is None else np.asarray(uv)
            group = groups.setdefault(key, dict(vertices=[], normals=[], uv=[], faces=[], count=0,
                                                 image=image, factor=factor))
            group["faces"].append(np.asarray(mesh.faces) + group["count"])
            for field, value in [("vertices", vertices), ("normals", normals), ("uv", uv)]:
                group[field].append(value)
            group["count"] += len(vertices)
        self.meshes = []
        cpu_vertices, cpu_faces, count = [], [], 0
        texture_cache = {}
        for group in groups.values():
            vertices = np.concatenate(group["vertices"])
            faces = np.concatenate(group["faces"])
            cpu_vertices.append(vertices)
            cpu_faces.append(faces + count)
            count += len(vertices)
            texture = None
            if group["image"] is not None:
                image_key = id(group["image"])
                if image_key not in texture_cache:
                    pixels = np.asarray(group["image"].convert("RGB"), dtype=np.float32) / 255
                    texture_cache[image_key] = torch.as_tensor(pixels, device="cuda")
                texture = texture_cache[image_key]
            self.meshes.append(dict(
                vertices=torch.as_tensor(vertices, device="cuda", dtype=torch.float32),
                faces=torch.as_tensor(faces, device="cuda", dtype=torch.int32),
                normals=torch.as_tensor(np.concatenate(group["normals"]), device="cuda", dtype=torch.float32),
                uv=torch.as_tensor(np.concatenate(group["uv"]), device="cuda", dtype=torch.float32),
                texture=texture,
                factor=torch.as_tensor(group["factor"][:3], device="cuda", dtype=torch.float32),
            ))
        self.vertices_cpu = np.concatenate(cpu_vertices)
        self.faces_cpu = np.concatenate(cpu_faces)
        self.vertices = torch.as_tensor(self.vertices_cpu, device="cuda", dtype=torch.float32)
        self.faces = torch.as_tensor(self.faces_cpu, device="cuda", dtype=torch.int32)
        self.light = options["lighting"]
        size = int(self.light.get("shadow_resolution", 1024))
        focal = size / (2 * np.tan(np.deg2rad(self.light.get("field_of_view", 85)) / 2))
        self.light_k = torch.tensor([[focal, 0, size/2], [0, focal, size/2], [0, 0, 1]], device="cuda", dtype=torch.float32)
        self.light_w = torch.tensor(look_at(self.light["position"], self.light.get("target", [0, 0, 0])), device="cuda", dtype=torch.float32)
        self.light_size = size
        light_buffer = self.mesh_renderer(self.vertices, self.faces, self.light_k, self.light_w, size, size)
        self.static_shadow = light_buffer["depth"]
        self.static_shadow_alpha = light_buffer["hard_alpha"]
        self.camera_cache = {}

    def floor_height(self, x, z, hint=0):
        triangles = self.vertices_cpu[self.faces_cpu]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        v0, v1 = b[:, [0, 2]] - a[:, [0, 2]], c[:, [0, 2]] - a[:, [0, 2]]
        q = np.array([x, z]) - a[:, [0, 2]]
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        valid = np.abs(det) > np.finfo(float).eps
        u = np.divide(q[:, 0]*v1[:, 1]-q[:, 1]*v1[:, 0], det, out=np.zeros(len(a)), where=valid)
        v = np.divide(v0[:, 0]*q[:, 1]-v0[:, 1]*q[:, 0], det, out=np.zeros(len(a)), where=valid)
        y = a[:, 1] + u*(b[:, 1]-a[:, 1]) + v*(c[:, 1]-a[:, 1])
        hits = y[valid & (u >= 0) & (v >= 0) & (u+v <= 1)]
        if not len(hits):
            raise ValueError("Actor placement has no environment surface below it")
        return float(hits[np.argmin(abs(hits-hint))])

    def camera_buffer(self, K, W, width, height):
        key = (K.cpu().numpy().tobytes(), W.cpu().numpy().tobytes(), width, height)
        if key in self.camera_cache:
            return self.camera_cache[key]
        dr = self.dr
        depth = torch.full((height, width), float("inf"), device="cuda")
        xyz = torch.zeros((height, width, 3), device="cuda")
        normals, albedo = torch.zeros_like(xyz), torch.zeros_like(xyz)
        alpha = torch.zeros_like(depth)
        for mesh in self.meshes:
            pc = mesh["vertices"] @ W[:3, :3].T + W[:3, 3]
            image = pc @ K.T
            near, far = .01, 2000.
            clip = torch.stack([
                2*(image[:, 0]+.5*pc[:, 2])/width-pc[:, 2],
                pc[:, 2]-2*(image[:, 1]+.5*pc[:, 2])/height,
                (far+near)/(far-near)*pc[:, 2]-2*far*near/(far-near), pc[:, 2],
            ], -1)[None].contiguous()
            rast, rast_db = dr.rasterize(self.mesh_renderer.context, clip, mesh["faces"], resolution=[height, width])
            attrs = torch.cat([pc[:, 2:3], mesh["vertices"], mesh["normals"], mesh["uv"]], -1)
            attrs, _ = dr.interpolate(attrs[None].contiguous(), rast, mesh["faces"])
            if mesh["texture"] is None:
                rgb = mesh["factor"].expand(1, height, width, 3)
            else:
                uv, uv_db = dr.interpolate(mesh["uv"][None].contiguous(), rast, mesh["faces"],
                                           rast_db=rast_db, diff_attrs="all")
                uv = torch.stack([uv[..., 0], 1-uv[..., 1]], -1).contiguous()
                uv_db = uv_db * uv_db.new_tensor([1, 1, -1, -1])
                rgb = dr.texture(mesh["texture"][None], uv, uv_da=uv_db,
                                 filter_mode="linear-mipmap-linear", boundary_mode="wrap") * mesh["factor"]
            attrs, rgb, mask = attrs[0].flip(0), rgb[0].flip(0), (rast[0, ..., 3] > 0).flip(0)
            closer = mask & (attrs[..., 0] < depth)
            depth = torch.where(closer, attrs[..., 0], depth)
            xyz = torch.where(closer[..., None], attrs[..., 1:4], xyz)
            normals = torch.where(closer[..., None], attrs[..., 4:7], normals)
            albedo = torch.where(closer[..., None], linear_rgb(rgb), albedo)
            alpha = torch.where(closer, torch.ones_like(alpha), alpha)
        buffer = dict(depth=depth, alpha=alpha, xyz=xyz, normals=F.normalize(normals, dim=-1), albedo=albedo)
        # Fixed cameras share their buffers. Moving cameras keep only one buffer.
        self.camera_cache = {key: buffer}
        return buffer

    def render(self, gaussians, rasterization, K, W, width, height):
        buffer = self.camera_buffer(K, W, width, height)
        means = torch.cat([g["xyz"] for g in gaussians])
        shadow, shadow_alpha, _ = rasterization(
            means=means, quats=torch.cat([g["rotation"] for g in gaussians]),
            scales=torch.cat([g["scaling"] for g in gaussians]),
            opacities=torch.cat([g["opacity"].flatten() for g in gaussians]),
            colors=torch.ones((len(means), 1), device="cuda"),
            viewmats=self.light_w[None], Ks=self.light_k[None],
            width=self.light_size, height=self.light_size, packed=False,
            render_mode="RGB+ED", near_plane=.01, far_plane=2000., eps2d=.3,
        )
        receiver = buffer["xyz"] + buffer["normals"] * self.light.get("shadow_normal_bias", .04)
        pc = receiver @ self.light_w[:3, :3].T + self.light_w[:3, 3]
        pixels = pc @ self.light_k.T
        uv = pixels[..., :2] / pixels[..., 2:].clamp(min=1e-8)
        grid = (2*(uv+.5)/self.light_size-1)[None]
        maps = torch.stack([shadow[0, ..., -1], shadow_alpha[0, ..., 0], self.static_shadow, self.static_shadow_alpha])
        bias = float(self.light.get("shadow_bias", .02))
        valid = (pc[..., 2] > 0) & (grid[0].abs() <= 1).all(-1)
        radius = self.light.get("shadow_filter_radius", 1.)
        visibility = torch.zeros_like(buffer["depth"])
        for dy in [-radius, 0, radius]:
            for dx in [-radius, 0, radius]:
                offset = grid.new_tensor([dx, dy]) * (2/self.light_size)
                samples = F.grid_sample(maps[None], grid+offset, align_corners=False, padding_mode="zeros")[0]
                person_block = (pc[..., 2] > samples[0]+bias) & (samples[0] > 0) & valid
                scene_block = (pc[..., 2] > samples[2]+bias) & (samples[2] > 0) & valid
                visibility += (1-samples[1]*person_block) * (1-samples[3]*scene_block) / 9
        light_position = torch.tensor(self.light["position"], device="cuda")
        direction = F.normalize(light_position-buffer["xyz"], dim=-1)
        lambert = (buffer["normals"]*direction).sum(-1).clamp(min=0)
        ambient = torch.tensor(self.light.get("ambient", [.38, .43, .52]), device="cuda")
        direct = torch.tensor(self.light.get("color", [1, .91, .79]), device="cuda") * self.light.get("intensity", 1.)
        illumination = ambient + direct * (lambert*visibility)[..., None]
        rgb = display_rgb(buffer["albedo"]*illumination).clamp(0, 1)
        return dict(depth=buffer["depth"], alpha=buffer["alpha"], premultiplied_rgb=rgb*buffer["alpha"][..., None])


class SceneShow:
    """One loaded scene and its actors; re-render after placement or camera edits."""

    def __init__(self, config, options, target_fps=None):
        self.config = config
        self.options = options
        self.target_fps = target_fps
        self._pano = None
        self._pano_path = None
        self.runtime = configure_runtime(config.runtime)
        self._load_actors()
        self.rebuild_environment()
        self.apply_placement()

    def _load_actors(self):
        from hoi.rendering.rig import animate
        from hoi.rendering.scene import (
            build_decoder,
            compare_decoders,
            frame_list,
            load_bundle,
            load_motion,
            render_scene,
            to_device,
        )

        self.animate = animate
        self.render_scene = render_scene
        bundles = [load_bundle(actor.avatar) for actor in self.config.actors]
        for bundle in bundles[1:]:
            compare_decoders(bundles[0], bundle)
        motions = [
            load_motion(actor.motion, bundle["poses"], bundle["rig"]["rest_joints"].dtype)
            for actor, bundle in zip(self.config.actors, bundles)
        ]
        frames = sorted(set.intersection(*(set(frame_list(m)) for m in motions)))
        if not frames:
            raise ValueError("actor motions share no frame")
        if self.target_fps is not None and self.target_fps != self.config.fps:
            from hoi.motion.resampling import output_times, resample_motion

            times = (np.asarray(frames) - frames[0]) / self.config.fps
            targets = output_times(times, self.config.fps, self.target_fps)
            motions = [
                resample_motion(m, b["rig"], self.config.fps, self.target_fps, targets)
                for m, b in zip(motions, bundles)
            ]
            frames = frame_list(motions[0])
        self.frames = frames
        self.indices = [{f: i for i, f in enumerate(frame_list(m))} for m in motions]
        self.rigs = [to_device(b["rig"], "cuda") for b in bundles]
        self.poses = [to_device(m, "cuda") for m in motions]
        self.features = [b["features"].cuda().float() for b in bundles]
        self.decoder = build_decoder(bundles[0], "cuda")
        self.rasterization = load_rasterization(self.runtime)
        self.background = bundles[0]["background_embedding"][2].cuda().float()
        self.first = [
            animate(rig, pose, index[self.frames[0]])
            for rig, pose, index in zip(self.rigs, self.poses, self.indices)
        ]

    def rebuild_environment(self):
        """Reload the glTF model and shadow map; placement and camera are reapplied."""
        self.env = EnvironmentRenderer(self.options)
        self.apply_placement()

    def apply_placement(self):
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

        options = self.options
        rotation, center = placement_basis(
            self.config, self.first, options["placement"].get("reference_serial")
        )
        yaw = np.deg2rad(options["placement"].get("yaw_degrees", 0))
        rotation = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]) @ rotation
        position = np.asarray(options["placement"]["position"], float)
        if options["placement"].get("snap_to_floor", True):
            position[1] = self.env.floor_height(position[0], position[2], position[1])
        scale = float(options["placement"].get("meters_per_body_height", 1.75))
        translation = position - scale * (rotation @ center)
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.position = position
        self.q = matrix_to_quaternion(torch.tensor(rotation, device="cuda", dtype=torch.float32))
        self.quaternion_multiply = quaternion_multiply

    def camera(self, frame, width, height):
        """Static camera or one of the authored motion presets, evaluated at a frame."""
        camera = self.options["camera"]
        motion = camera.get("motion") or {"type": "static"}
        kind = motion.get("type", "static")
        frames = self.frames
        index = frames.index(frame) if frame in frames else 0
        span = max(len(frames) - 1, 1)
        t = index / span
        if motion.get("ease", True):
            t = t * t * (3 - 2 * t)
        eye_offset = np.asarray(camera["eye"], float)
        target_offset = np.asarray(camera["target"], float)
        eye = self.position + eye_offset
        target = self.position + target_offset
        if kind == "orbit":
            radius = float(motion.get("radius", np.hypot(eye_offset[0], eye_offset[2])))
            angle0 = np.arctan2(eye_offset[2], eye_offset[0])
            angle = angle0 + np.deg2rad(float(motion.get("start_degrees", 0))) + 2 * np.pi * float(motion.get("turns", 1.0)) * t
            eye = self.position + np.array([
                radius * np.cos(angle),
                eye_offset[1] + float(motion.get("height", 0)),
                radius * np.sin(angle),
            ])
            target = self.position + np.array([0.0, float(motion.get("look_height", target_offset[1])), 0.0])
        elif kind == "push_in":
            distance = np.hypot(eye_offset[0], eye_offset[2])
            end = float(motion.get("end_distance", distance * 0.55))
            distance = distance + (end - distance) * t
            direction = np.array([eye_offset[0], eye_offset[2]])
            direction /= max(np.linalg.norm(direction), 1e-9)
            eye = self.position + np.array([
                distance * direction[0],
                eye_offset[1] + float(motion.get("height", 0)),
                distance * direction[1],
            ])
            target = self.position + np.array([0.0, float(motion.get("look_height", target_offset[1])), 0.0])
        elif kind == "crane":
            eye = self.position + eye_offset + np.array([0.0, float(motion.get("rise", 1.2)) * t, 0.0])
            target = self.position + np.array([0.0, float(motion.get("look_height", target_offset[1])), 0.0])
        W = torch.tensor(look_at(eye, target), device="cuda", dtype=torch.float32)
        focal = height / (2 * np.tan(np.deg2rad(camera.get("vertical_fov", 42)) / 2))
        K = torch.tensor([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], device="cuda", dtype=torch.float32)
        return K, W

    def panorama(self, K, W, width, height):
        """Equirectangular backdrop sampled along each pixel ray; display-referred."""
        path = self.options.get("background_image")
        if not path:
            return None
        from PIL import Image

        target = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
        if self._pano is None or self._pano_path != str(target):
            pixels = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32) / 255.0
            self._pano = torch.as_tensor(pixels, device="cuda").permute(2, 0, 1)[None]
            self._pano_path = str(target)
        camera = self.options["camera"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u = torch.arange(width, device="cuda", dtype=torch.float32) + 0.5
        v = torch.arange(height, device="cuda", dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        direction = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1)
        rotation = W[:3, :3]
        world = direction @ rotation
        yaw = np.deg2rad(float(self.options.get("background_yaw", 0)))
        theta = torch.atan2(world[..., 0], -world[..., 2]) + yaw
        phi = torch.acos(world[..., 1].clamp(-1, 1))
        vertical = phi / np.pi
        if self.options.get("background_clip_horizon", True):
            vertical = vertical.clamp(max=0.5)
        grid = torch.stack([theta / np.pi, vertical * 2 - 1], -1)[None]
        grid[..., 0] = (grid[..., 0] + 1) % 2 - 1
        rgb = F.grid_sample(self._pano, grid, align_corners=False, padding_mode="border")[0]
        return rgb.permute(1, 2, 0)

    def render(self, frame, width, height):
        with torch.inference_mode():
            R = torch.tensor(self.rotation, device="cuda", dtype=torch.float32)
            t = torch.tensor(self.translation, device="cuda", dtype=torch.float32)
            q = self.q
            actors = []
            for rig, pose, index in zip(self.rigs, self.poses, self.indices):
                g = self.animate(rig, pose, index[frame])
                g.update(
                    xyz=g["xyz"] @ R.T * self.scale + t,
                    scaling=g["scaling"] * self.scale,
                    rotation=self.quaternion_multiply(q.expand_as(g["rotation"]), g["rotation"]),
                )
                actors.append(g)
            K, W = self.camera(frame, width, height)
            mesh = self.env.render(actors, self.rasterization, K, W, width, height)
            sky = torch.tensor(self.options.get("background", [.60, .74, .88]), device="cuda")
            backdrop = self.panorama(K, W, width, height)
            rgb, _ = self.render_scene(
                self.decoder, self.rasterization, actors, self.features,
                K, W, width, height, self.background, mesh,
                display_background=sky if backdrop is None else backdrop,
            )
            return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def render_environment(config, args):
    """Reuse selected identity, motion, skinning and neural appearance unchanged."""
    from hoi.rendering.scene import sha256_file

    spec_path = args.environment.resolve()
    options = json.loads(spec_path.read_text(encoding="utf-8"))
    fps = config.fps if args.fps is None else args.fps
    show = SceneShow(config, options, target_fps=fps)
    frames = show.frames
    if args.frames:
        requested = set(args.frames)
        if requested - set(frames):
            raise ValueError("Requested environment frames are absent from actor motion")
        frames = [f for f in frames if f in requested]
    if args.video and any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Video needs a consecutive shared frame range")
    camera = options["camera"]
    width, height = args.width or camera.get("width", 1260), args.height or camera.get("height", 756)
    output = config.output / "environment" / spec_path.stem
    output.mkdir(parents=True, exist_ok=True)
    destination = video_path(output, "camera_main")
    process = None
    if args.video:
        process = subprocess.Popen([show.runtime["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination)], stdin=subprocess.PIPE)
    from PIL import Image
    try:
        for frame in frames:
            # Render 2x and downsample: direct low-resolution gaussian rasterization
            # distorts small splats (merged hands, swollen silhouettes).
            # Decode the neural appearance at 2x, then downsample: the DPT decoder's
            # silhouette depends on the feature-map resolution (bodies swell at low res).
            image = show.render(frame, width * 2, height * 2)
            image = np.asarray(Image.fromarray(image).resize((width, height), Image.LANCZOS))
            if frame in {frames[0], frames[len(frames)//2], frames[-1]} or not args.video:
                Image.fromarray(image).save(output / f"frame_{frame:04d}.png")
            if process is not None:
                process.stdin.write(image.tobytes())
            print(f"[environment] {spec_path.stem} frame {frame}", flush=True)
        if process is not None:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Environment video encoding failed")
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    K, W = show.camera(frames[0], width, height)
    report = dict(environment=str(show.env.model_path), environment_sha256=sha256_file(show.env.model_path),
        specification=str(spec_path), actors=[dict(id=a.id, avatar=str(a.avatar), motion=str(a.motion),
        motion_sha256=sha256_file(a.motion)) for a in config.actors], frames=frames, fps=fps,
        placement=dict(rotation=show.rotation.tolist(), translation=show.translation.tolist(), scale=show.scale,
        floor_height=float(show.position[1]), scale_scope="visual scene placement, not independently measured metric scale"),
        camera=dict(K=K.cpu().tolist(), W=W.cpu().tolist()), camera_motion=options["camera"].get("motion"), lighting=show.env.light,
        appearance="Original neural appearance, including captured lighting; no recovered PBR materials or intrinsic relighting",
        shadows="Gaussian expected-depth light-space shadows plus opaque mesh shadow map; approximate volumetric shadow",
        video=str(destination) if args.video else None, image_size=[width, height])
    write_json(output / "report.json", report)
    if args.video:
        record_video(destination, report=output / "report.json", frames=len(frames), fps=fps,
                     description=f"{config.id} 双人交互置入 {spec_path.stem} 虚拟场景")
