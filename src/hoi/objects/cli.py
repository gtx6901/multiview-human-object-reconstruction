"""Run dynamic-object stages through the shared project entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hoi.cli import run_stage
from hoi.objects.multiview import resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "geometry", "train", "render"), required=True
    )
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--serial", default="all")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--novel", action="store_true")
    args = parser.parse_args(argv)
    path = resolve(args.config)
    config = json.loads(path.read_text(encoding="utf-8"))
    modules = {
        "prepare": "multiview",
        "geometry": "geometry",
        "train": "multi4d",
        "render": "render",
    }
    stage_args = ["--config", str(path)]
    if args.stage == "render":
        stage_args.extend(["--serial", args.serial])
        if args.frames:
            stage_args.extend(["--frames", *map(str, args.frames)])
        for flag in ("video", "novel"):
            if getattr(args, flag):
                stage_args.append(f"--{flag}")
    run_stage(
        f"hoi.objects.{modules[args.stage]}",
        stage_args,
        resolve(config["output"]) / args.stage,
    )
    return 0
