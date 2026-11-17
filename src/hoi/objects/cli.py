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
    parser.add_argument("--stage", choices=("prepare", "geometry", "train"), required=True)
    args = parser.parse_args(argv)
    path = resolve(args.config)
    config = json.loads(path.read_text(encoding="utf-8"))
    modules = {"prepare": "multiview", "geometry": "geometry", "train": "multi4d"}
    run_stage(
        f"hoi.objects.{modules[args.stage]}",
        ["--config", str(path)],
        resolve(config["output"]) / args.stage,
    )
    return 0
