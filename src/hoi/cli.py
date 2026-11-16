"""Single entry point for reconstruction stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

from hoi.runtime import PROJECT_ROOT, configure_runtime
from hoi.settings import ActorConfig, SceneConfig
from hoi.io import write_json


def run_stage(module: str, arguments: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PROJECT_ROOT / "src"), environment.get("PYTHONPATH")])
    )
    command = [sys.executable, "-B", "-u", "-m", module, *arguments]
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f"{module} exited {returncode}; see {output / 'run.log'}")


def activate_runtime(config: SceneConfig) -> None:
    configure_runtime(config.runtime)
    runtime_path = config.output / "runtime.json"
    write_json(runtime_path, config.runtime)
    os.environ["HOI_RUNTIME_CONFIG"] = str(runtime_path)


def save_config(config: SceneConfig) -> None:
    write_json(config.output / "config.json", json.loads(json.dumps(asdict(config), default=str)))


def person_file(config: SceneConfig, actor: ActorConfig) -> Path:
    import numpy as np

    if any(path is None for path in (actor.manifest, actor.observations, actor.keypoints)):
        raise ValueError(f"{actor.id}: prepare the per-person observations first")
    person = config.output / actor.id / "inputs/person.json"
    with np.load(actor.observations, allow_pickle=False) as observations:
        if "frames" in observations:
            frames = observations["frames"].tolist()
        else:
            with np.load(next(actor.keypoints.glob("*.npz")), allow_pickle=False) as keypoints:
                frames = keypoints["frames"].tolist()
    manifest = json.loads(actor.manifest.read_text(encoding="utf-8-sig"))
    write_json(
        person,
        {
            "identity": actor.id,
            "frames": frames,
            "fps": config.fps,
            "manifest": str(actor.manifest),
            "observations": str(actor.observations),
            "keypoints": str(actor.keypoints),
            "gauge": manifest.get("world_normalization", {}),
        },
    )
    return person


def _actor_has_requested_frames(actor: ActorConfig, end: int) -> bool:
    import numpy as np

    expected = np.arange(end)
    try:
        manifest = json.loads(actor.manifest.read_text(encoding="utf-8-sig"))
        keypoint_frames = None
        for view in manifest["views"]:
            with np.load(
                actor.keypoints / f"{view['serial']}.npz", allow_pickle=False
            ) as keypoints:
                frames = keypoints["frames"]
                if not np.array_equal(frames, expected):
                    return False
                if any(keypoints[key].shape[0] != len(frames) for key in ("xy", "score")):
                    return False
                keypoint_frames = frames
        if keypoint_frames is None:
            return False
        with np.load(actor.observations, allow_pickle=False) as observations:
            frames = observations["frames"] if "frames" in observations else keypoint_frames
            return np.array_equal(frames, expected) and observations["xyz"].shape[0] == len(frames)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def _run_needs_preparation(config: SceneConfig) -> bool:
    if any(
        path is None or not path.exists()
        for actor in config.actors
        for path in (actor.manifest, actor.observations, actor.keypoints)
    ):
        return True
    if "end" in config.preparation and any(
        not _actor_has_requested_frames(actor, int(config.preparation["end"]))
        for actor in config.actors
    ):
        return True
    needs_segmentation = any(
        not actor.avatar.is_file()
        or actor.initialization is None
        or not actor.initialization.is_file()
        or (actor.hand_observations is not None and not actor.hand_observations.is_file())
        for actor in config.actors
    )
    if not needs_segmentation:
        return False
    segment_index = config.preparation.get("segment_index")
    if not segment_index:
        return True
    segment_path = Path(segment_index)
    if not segment_path.is_absolute():
        segment_path = PROJECT_ROOT / segment_path
    if not segment_path.is_file():
        return True
    if "end" in config.preparation:
        end = int(config.preparation["end"])
        try:
            index = json.loads(segment_path.read_text(encoding="utf-8-sig"))
            return index.get("frame_range") != [0, end] or index.get("frame_count") != end
        except (OSError, ValueError, TypeError):
            return True
    return False


def prepare_scene(config: SceneConfig) -> SceneConfig:
    from hoi.capture.scene import prepare_capture

    preparation = config.preparation
    if not preparation.get("seeds"):
        raise ValueError("preparation.seeds must specify the identity annotation file")
    seed_path = Path(preparation["seeds"])
    if not seed_path.is_absolute():
        seed_path = PROJECT_ROOT / seed_path
    camera_path = prepare_capture(config, config.output / "capture")
    camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
    end = int(
        preparation["end"]
        if "end" in preparation
        else min(view["decode"]["actual_frames"] for view in camera_data["views"])
    )
    segment = config.output / "segmentation"
    run_stage(
        "hoi.observations.segmentation",
        [
            "--scene",
            str(camera_path),
            "--seeds",
            str(seed_path),
            "--actors",
            *[actor.id for actor in config.actors],
            "--end",
            str(end),
            "--out",
            str(segment),
            "--sam2-root",
            config.runtime.get("sam2_root", str(PROJECT_ROOT / "runtime/vendor/sam2")),
            "--ckpt",
            config.runtime.get(
                "sam2_checkpoint",
                str(PROJECT_ROOT / "runtime/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
            ),
        ],
        segment,
    )
    instances = config.output / "observations"
    run_stage(
        "hoi.observations.instances",
        [
            "prepare",
            "--scene",
            str(camera_path),
            "--seeds",
            str(segment / "seeds_undistorted.json"),
            "--detector",
            "external",
            "--detections",
            str(segment / "boxes_undistorted.json"),
            "--coordinate-space",
            "undistorted",
            "--seed-coordinate-space",
            "undistorted",
            "--frames",
            f"0:{end}:1",
            "--rtmw",
            config.runtime.get("rtmw_model", str(PROJECT_ROOT / "runtime/wholebody/rtmw.onnx")),
            "--out",
            str(instances),
        ],
        instances,
    )
    actors = []
    for actor in config.actors:
        inputs = config.output / actor.id / "inputs"
        run_stage(
            "hoi.observations.person",
            [
                "--scene",
                str(camera_path),
                "--instances",
                str(instances),
                "--manifest",
                str(config.manifest_path),
                "--identity",
                actor.id,
                "--camera-space",
                config.camera_space,
                "--out",
                str(inputs),
            ],
            inputs,
        )
        actors.append(
            replace(
                actor,
                manifest=inputs / "manifest.json",
                observations=inputs / "observations.npz",
                keypoints=inputs / "keypoints",
                initialization=None,
                hand_observations=None,
            )
        )
    prepared = replace(
        config,
        actors=tuple(actors),
        preparation={**preparation, "segment_index": str(segment / "index.json")},
    )
    save_config(prepared)
    return prepared


def identity_scene(config: SceneConfig, reuse: bool = False) -> SceneConfig:
    actors = []
    for actor in config.actors:
        if reuse and actor.avatar.is_file():
            actors.append(actor)
            continue
        directory = config.output / actor.id
        proportions, identity = directory / "proportions", directory / "identity"
        person = person_file(config, actor)
        segment_index = config.preparation.get("segment_index")
        if not segment_index:
            raise ValueError("Identity reconstruction requires prepare's segmentation index")
        proportions_path = actor.proportions
        if proportions_path is None:
            run_stage(
                "hoi.identity.proportions",
                ["--person", str(person), "--out", str(proportions)],
                proportions,
            )
            proportions_path = proportions / "proportions.pt"
        run_stage(
            "hoi.identity.build",
            [
                "--segment-index",
                str(segment_index),
                "--actor",
                actor.id,
                "--source-frame",
                str(config.preparation.get("identity_frame", 0)),
                "--proportions",
                str(proportions_path),
                "--out",
                str(identity),
            ],
            identity,
        )
        actors.append(replace(actor, avatar=identity / "avatar.pt"))
    result = replace(config, actors=tuple(actors))
    save_config(result)
    return result


def initialize_scene(config: SceneConfig, reuse: bool = False) -> SceneConfig:
    actors = []
    segment_index = config.preparation.get("segment_index")
    for actor in config.actors:
        reuse_initial = (
            reuse and actor.initialization is not None and actor.initialization.is_file()
        )
        reuse_hands = (
            reuse and actor.hand_observations is not None and actor.hand_observations.is_file()
        )
        if reuse_initial and (actor.hand_observations is None or reuse_hands):
            actors.append(actor)
            continue
        if not segment_index:
            raise ValueError("Initialization requires prepare's segmentation index")
        person = person_file(config, actor)
        frames = json.loads(person.read_text(encoding="utf-8"))["frames"]
        directory = config.output / actor.id
        initial, hands = directory / "initialization", directory / "hand_observations"
        arguments = [
            "--person",
            str(person),
            "--avatar",
            str(actor.avatar),
            "--segment-index",
            str(segment_index),
            "--frames",
            *map(str, frames),
            "--fps",
            str(config.fps),
            "--world-up",
            *map(str, config.world_up),
            "--world-up-source",
            config.world_up_source,
            "--out",
            str(initial),
        ]
        if actor.motion.is_file():
            arguments.extend(["--source-motion", str(actor.motion)])
        if config.runtime.get("nlf_model"):
            arguments.extend(["--model", config.runtime["nlf_model"]])
        if not reuse_initial:
            run_stage("hoi.motion.initialize", arguments, initial)
        if not reuse_hands:
            run_stage(
                "hoi.observations.instances",
                [
                    "observe-hands",
                    "--person",
                    str(person),
                    "--scene-index",
                    str(segment_index),
                    "--out",
                    str(hands),
                ],
                hands,
            )
        actors.append(
            replace(
                actor,
                initialization=actor.initialization if reuse_initial else initial / "motion.pt",
                hand_observations=(
                    actor.hand_observations if reuse_hands else hands / "observations.npz"
                ),
            )
        )
    result = replace(config, actors=tuple(actors))
    save_config(result)
    return result


def fit_scene(config: SceneConfig) -> SceneConfig:
    parameters = json.loads(
        (PROJECT_ROOT / "configs/reconstruction.json").read_text(encoding="utf-8")
    )
    actors = []
    for actor in config.actors:
        if actor.initialization is None:
            raise ValueError(
                f"{actor.id}: fitting requires manifest, observations, keypoints and initialization"
            )
        directory = config.output / actor.id
        person = person_file(config, actor)
        hand = (
            ["--hand-observations", str(actor.hand_observations)] if actor.hand_observations else []
        )
        common = ["--person", str(person), "--avatar", str(actor.avatar), "--fps", str(config.fps)]
        body, face, motion = directory / "body", directory / "face", directory / "motion"
        run_stage(
            "hoi.motion.body",
            [
                *common,
                "--initialization",
                str(actor.initialization),
                "--world-up",
                *map(str, config.world_up),
                "--world-up-source",
                config.world_up_source,
                "--steps",
                *map(str, parameters["body_steps"]),
                *hand,
                "--out",
                str(body),
            ],
            body,
        )
        run_stage(
            "hoi.motion.face",
            [
                *common,
                "--motion",
                str(body / "motion.pt"),
                "--iters",
                str(parameters["face_iterations"]),
                "--out",
                str(face),
            ],
            face,
        )
        run_stage(
            "hoi.motion.fit",
            [
                *common,
                "--motion",
                str(face / "motion.pt"),
                "--initialization",
                str(actor.initialization),
                "--face-constraints",
                str(face / "face_constraints.npz"),
                "--iters",
                str(parameters["motion_iterations"]),
                *hand,
                "--out",
                str(motion),
            ],
            motion,
        )
        actors.append(replace(actor, motion=motion / "motion.pt"))
    return replace(config, actors=tuple(actors))


def render_scene(config: SceneConfig, args: argparse.Namespace, fitted: bool) -> None:
    render_config = asdict(config)
    render_config["actors"] = [
        {
            "id": actor.id,
            "avatar": str(actor.avatar),
            "motion": str(
                config.output / actor.id / "motion/motion.pt" if fitted else actor.motion
            ),
        }
        for actor in config.actors
    ]
    render_config["scene_path"] = str(config.scene_path)
    render_config["manifest_path"] = str(config.manifest_path)
    render_config["output"] = str(config.output)
    path = config.output / "render_config.json"
    write_json(path, render_config)
    width = args.width or config.rendering.get("width", 1008)
    height = args.height or config.rendering.get("height", 756)
    flags = [
        "--config",
        str(path),
        "--out",
        str(config.output / "render"),
        "--serial",
        args.serial,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    cameras = args.cameras or config.rendering.get("cameras")
    if cameras:
        camera_path = Path(cameras)
        flags.extend(
            [
                "--cameras",
                str(camera_path if camera_path.is_absolute() else PROJECT_ROOT / camera_path),
            ]
        )
    if args.frames:
        flags.extend(["--frames", *map(str, args.frames)])
    if args.video:
        flags.append("--video")
    if args.keep_frames:
        flags.append("--keep-frames")
    run_stage("hoi.rendering.scene", flags, config.output / "render")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "clothing":
        from hoi.clothing.cli import main as clothing_main

        return clothing_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "identity", "initialize", "fit", "render", "run", "plan")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--serial", default="all")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--cameras", type=Path)
    args = parser.parse_args()
    config = SceneConfig.load(args.config)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "scene": config.id,
                    "actors": [actor.id for actor in config.actors],
                    "fps": config.fps,
                    "output": str(config.output),
                    "stages": [
                        "prepare",
                        "identity",
                        "initialize",
                        "body",
                        "face",
                        "motion",
                        "render",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    activate_runtime(config)
    if args.command == "prepare" or (args.command == "run" and _run_needs_preparation(config)):
        config = prepare_scene(config)
    if args.command in ("identity", "run"):
        config = identity_scene(config, reuse=args.command == "run")
    if args.command in ("initialize", "run"):
        config = initialize_scene(config, reuse=args.command == "run")
    if args.command in ("fit", "run"):
        config = fit_scene(config)
    if args.command in ("render", "run"):
        render_scene(config, args, fitted=args.command == "run")
    save_config(config)
    return 0
