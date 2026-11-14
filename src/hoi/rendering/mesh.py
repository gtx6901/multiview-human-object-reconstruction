"""Calibrated-camera mesh rasterization."""

import torch
import nvdiffrast.torch as dr


class MeshRenderer:
    def __init__(self):
        self.context = dr.RasterizeCudaContext()

    def depth_alpha_batch(
        self,
        vertices,
        faces,
        K,
        W,
        width,
        height,
        near=0.01,
        far=100.0,
        colors=None,
        uv=None,
        texture=None,
        texture_confidence=None,
    ):
        """Instanced opaque depths in the same CV convention as __call__."""
        pc = torch.einsum("bij,bnj->bni", W[:, :3, :3], vertices) + W[:, None, :3, 3]
        image = torch.einsum("bij,bnj->bni", K, pc)
        clip = torch.stack(
            [
                2 * (image[..., 0] + 0.5 * pc[..., 2]) / width - pc[..., 2],
                pc[..., 2] - 2 * (image[..., 1] + 0.5 * pc[..., 2]) / height,
                (far + near) / (far - near) * pc[..., 2] - 2 * far * near / (far - near),
                pc[..., 2],
            ],
            -1,
        ).contiguous()
        faces = faces.to(torch.int32).contiguous()
        rast, _ = dr.rasterize(self.context, clip, faces, resolution=[height, width])
        hard = (rast[..., 3:] > 0).float()
        depth, _ = dr.interpolate(pc[..., 2:3].contiguous(), rast, faces)
        alpha = dr.antialias(hard, rast, clip, faces)
        result = {
            "alpha": alpha.flip(1)[..., 0],
            "depth": depth.flip(1)[..., 0],
            "hard_alpha": hard.flip(1)[..., 0],
            "face_id": rast.flip(1)[..., 3].to(torch.long) - 1,
        }
        if uv is not None and texture is not None:
            coordinates, _ = dr.interpolate(
                uv[None].expand(len(vertices), -1, -1).contiguous(), rast, faces
            )
            coordinates = torch.stack(
                [coordinates[..., 0], 1 - coordinates[..., 1]], -1
            ).contiguous()
            rgb = dr.texture(
                texture[None].contiguous(), coordinates, filter_mode="linear", boundary_mode="clamp"
            )
            result["premultiplied_rgb"] = dr.antialias(rgb * hard, rast, clip, faces).flip(1)
            if texture_confidence is not None:
                confidence = dr.texture(
                    texture_confidence[None, ..., None].contiguous(),
                    coordinates,
                    filter_mode="linear",
                    boundary_mode="clamp",
                )
                result["texture_confidence"] = confidence.flip(1)[..., 0]
        elif colors is not None:
            attrs = colors[None].expand(len(vertices), -1, -1).contiguous()
            rgb, _ = dr.interpolate(attrs, rast, faces)
            result["premultiplied_rgb"] = dr.antialias(rgb * hard, rast, clip, faces).flip(1)
        return result

    def alpha_batch(self, vertices, faces, K, W, width, height, near=0.01, far=100.0):
        """Instanced differentiable silhouettes for explicit B independent cameras."""
        assert vertices.ndim == 3 and K.ndim == 3 and W.ndim == 3
        pc = torch.einsum("bij,bnj->bni", W[:, :3, :3], vertices) + W[:, None, :3, 3]
        image = torch.einsum("bij,bnj->bni", K, pc)
        clip = torch.stack(
            [
                2 * (image[..., 0] + 0.5 * pc[..., 2]) / width - pc[..., 2],
                pc[..., 2] - 2 * (image[..., 1] + 0.5 * pc[..., 2]) / height,
                (far + near) / (far - near) * pc[..., 2] - 2 * far * near / (far - near),
                pc[..., 2],
            ],
            -1,
        ).contiguous()
        faces = faces.to(torch.int32).contiguous()
        rast, _ = dr.rasterize(self.context, clip, faces, resolution=[height, width])
        alpha = (rast[..., 3:] > 0).float()
        return dr.antialias(alpha, rast, clip, faces).flip(1)[..., 0]

    def __call__(
        self,
        vertices,
        faces,
        K,
        W,
        width,
        height,
        colors=None,
        near=0.01,
        far=100.0,
        uv=None,
        texture=None,
    ):
        """W maps common world coordinates to OpenCV x-right/y-down/z-forward."""
        pc = vertices @ W[:3, :3].T + W[:3, 3]
        image = pc @ K.T
        clip = torch.stack(
            [
                2 * (image[:, 0] + 0.5 * pc[:, 2]) / width - pc[:, 2],
                pc[:, 2] - 2 * (image[:, 1] + 0.5 * pc[:, 2]) / height,
                (far + near) / (far - near) * pc[:, 2] - 2 * far * near / (far - near),
                pc[:, 2],
            ],
            -1,
        )[None].contiguous()
        faces = faces.to(torch.int32).contiguous()
        rast, _ = dr.rasterize(self.context, clip, faces, resolution=[height, width])
        alpha = (rast[..., 3:] > 0).float()
        depth, _ = dr.interpolate(pc[None, :, 2:3].contiguous(), rast, faces)
        if uv is not None and texture is not None:
            coordinates, _ = dr.interpolate(uv[None].contiguous(), rast, faces)
            coordinates = torch.stack(
                [coordinates[..., 0], 1 - coordinates[..., 1]], -1
            ).contiguous()
            rgb = dr.texture(
                texture[None].contiguous(), coordinates, filter_mode="linear", boundary_mode="clamp"
            )
        else:
            if colors is None:
                colors = torch.ones_like(vertices)
            rgb, _ = dr.interpolate(colors[None].contiguous(), rast, faces)
        rgba = dr.antialias(torch.cat([rgb * alpha, alpha], -1), rast, clip, faces)
        # Rasterizer is bottom-left, input RGB and K are top-left.
        rgba = rgba[0].flip(0)
        depth = depth[0].flip(0)
        return {
            "premultiplied_rgb": rgba[..., :3],
            "alpha": rgba[..., 3],
            "depth": depth[..., 0],
            "hard_alpha": alpha[0, ..., 0].flip(0),
        }
