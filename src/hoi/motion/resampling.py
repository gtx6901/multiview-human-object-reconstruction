"""Resample fitted 3D motion on a shared clock; never alter source observations."""

from fractions import Fraction
import math

import numpy as np
import torch

from hoi.motion.rotations import export_motion_with_quaternions, load_motion_quaternions


def frame_rate(value):
    if isinstance(value, dict):
        value = Fraction(int(value["numerator"]), int(value["denominator"]))
    elif torch.is_tensor(value):
        value = value.item()
    result = Fraction(str(value))
    if result <= 0:
        raise ValueError("Frame rate must be positive")
    return result


def source_times(motion, fps):
    frames = torch.as_tensor(motion["frames"]).cpu().numpy()
    times = np.asarray(
        motion.get("timestamps", (frames - frames[0]) / float(frame_rate(fps))),
        dtype=np.float64,
    )
    if (
        times.shape != frames.shape
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    return times


def output_times(times, source_fps, target_fps):
    """Keep the final source frame's display interval, without pose extrapolation."""
    source_fps, target_fps = frame_rate(source_fps), frame_rate(target_fps)
    duration = Fraction(str(float(times[-1] - times[0]))) + 1 / source_fps
    count = math.ceil(duration * target_fps - Fraction(1, 10**9))
    return times[0] + np.arange(count) / float(target_fps)


def interpolation_indices(times, targets):
    times, targets = np.asarray(times), np.asarray(targets)
    if not np.isfinite(targets).all() or np.any(np.diff(targets) <= 0):
        raise ValueError("Target timestamps must be finite and strictly increasing")
    right = np.clip(np.searchsorted(times, targets, side="right"), 0, len(times) - 1)
    left = np.clip(np.searchsorted(times, targets, side="right") - 1, 0, len(times) - 1)
    alpha = np.zeros(len(targets))
    inside = left != right
    alpha[inside] = (targets[inside] - times[left[inside]]) / (
        times[right[inside]] - times[left[inside]]
    )
    return left, right, alpha


def slerp(q0, q1, alpha):
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    angle = torch.acos(dot.abs().clamp(0, 1))
    # sinc has the exact same-rotation limit; q and -q take the same short arc.
    denominator = torch.sinc(angle / torch.pi)
    value = (
        (1 - alpha) * torch.sinc((1 - alpha) * angle / torch.pi) * q0
        + alpha * torch.sinc(alpha * angle / torch.pi) * q1
    ) / denominator
    return torch.nn.functional.normalize(value, dim=-1)


def resample_motion(motion, rig, source_fps, target_fps, targets=None):
    times = source_times(motion, source_fps)
    targets = output_times(times, source_fps, target_fps) if targets is None else targets
    left, right, alpha = interpolation_indices(times, targets)
    q, info = load_motion_quaternions(motion, verify_legacy=True)
    q = q.detach().cpu().double()
    if q.shape != (len(times), 55, 4) or not torch.isfinite(q).all():
        raise ValueError("Motion rotations must be finite T,55,4 quaternions")
    weights = torch.as_tensor(alpha, dtype=q.dtype)[:, None, None]
    interpolated = slerp(q[left], q[right], weights)
    translation = torch.as_tensor(motion["translation_world"]).detach().cpu().double()
    if translation.shape != (len(times), 3) or not torch.isfinite(translation).all():
        raise ValueError("Motion translation must be finite T,3")
    translation = translation[left] * (1 - weights[:, 0]) + translation[right] * weights[:, 0]
    result = export_motion_with_quaternions(motion, interpolated, hemisphere=True)
    result["translation_world"] = translation
    result["frames"] = torch.arange(len(targets))
    result["timestamps"] = torch.as_tensor(targets, dtype=torch.float64)
    result["fps"] = float(frame_rate(target_fps))
    result["fps_rational"] = str(frame_rate(target_fps))
    from pytorch3d.transforms import quaternion_to_matrix

    rotations = quaternion_to_matrix(interpolated)
    rest = torch.as_tensor(rig["rest_joints"]).detach().cpu().double().reshape(55, 3)
    parents = torch.as_tensor(rig["parents"]).detach().cpu().long()
    global_rotations, joints = [], []
    for joint in range(55):
        parent = int(parents[joint])
        if parent < 0:
            global_rotations.append(rotations[:, joint])
            joints.append(rest[joint].expand(len(targets), -1))
        else:
            global_rotations.append(global_rotations[parent] @ rotations[:, joint])
            joints.append(
                joints[parent]
                + (global_rotations[parent] @ (rest[joint] - rest[parent])[:, None]).squeeze(-1)
            )
    result["joints55_world"] = torch.stack(joints, 1) * motion["scale"] + translation[:, None]
    result["resampling"] = {
        "method": "shortest-arc local-quaternion SLERP; linear translation; FK joints",
        "source_frames": torch.as_tensor(motion["frames"]).tolist(),
        "source_times": times.tolist(),
        "target_times": np.asarray(targets).tolist(),
        "left_indices": left.tolist(),
        "right_indices": right.tolist(),
        "alpha": alpha.tolist(),
        "source_fps": str(frame_rate(source_fps)),
        "target_fps": str(frame_rate(target_fps)),
        "rotation_contract": info,
        "tail": "hold final sample during its remaining display interval",
    }
    return result
