"""Run clothing stages from one scene configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hoi.runtime import PROJECT_ROOT, configure_runtime
from hoi.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "bind",
            "parse",
            "layers",
            "canonicalize",
            "position_maps",
            "training_masks",
            "pretrain",
            "train",
            "render",
        ),
    )
    parser.add_argument("--updates", type=int, help="target total updates, not additional updates")
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))

    def path(name: str) -> str:
        value = config.get(name)
        if not value:
            raise ValueError(f"{args.stage} requires configuration field {name}")
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)

    settings = configure_runtime(config.get("runtime"))
    data, output, model = path("data"), Path(path("output")), path("model_config")
    runtime_path = output / "runtime.json"
    write_json(runtime_path, settings)
    write_json(output / "config.json", config)
    os.environ["HOI_RUNTIME_CONFIG"] = str(runtime_path)
    module = "hoi.clothing.integration"
    arguments = ["--config", model, "--data", data, "--out", str(output)]
    if args.stage == "bind":
        module = "hoi.clothing.prepare_inputs"
        arguments = ["--source", path("binding_source"), "--out", data]
    elif args.stage == "parse":
        module = "hoi.clothing.segmentation"
        arguments = ["--config", model, "--data", data]
        if args.frames:
            arguments.extend(["--frames", ",".join(map(str, args.frames))])
    elif args.stage in {"layers", "canonicalize"}:
        module = "hoi.clothing.layer_geometry"
        arguments = ["--config", model, "--data", data]
        if args.stage == "canonicalize":
            arguments.append("--canonicalize-only")
        else:
            for flag, name in (
                ("--template", "source_template"),
                ("--weight-volume", "source_weight_volume"),
                ("--source-canonical-config", "source_canonical_config"),
            ):
                arguments.extend([flag, path(name)])
        if args.frames:
            arguments.extend(["--frames", ",".join(map(str, args.frames))])
    else:
        flag = {"position_maps": "--gen-pos-maps", "training_masks": "--gen-training-masks"}.get(
            args.stage, "--" + args.stage
        )
        arguments.append(flag)
        if args.updates is not None:
            arguments.extend(["--updates", str(args.updates)])
        if args.frames:
            arguments.extend(["--frames", *map(str, args.frames)])
        if args.video:
            arguments.extend(["--video", "--video-camera", str(config.get("video_camera", 0))])
    from hoi.cli import run_stage

    run_stage(module, arguments, output / args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
