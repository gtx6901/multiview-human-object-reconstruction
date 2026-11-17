"""Store generated videos in one local delivery directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from hoi.io import write_json
from hoi.runtime import PROJECT_ROOT

DELIVERY_ROOT = PROJECT_ROOT / "runs/delivery"
COLLECTIONS = ("selected", "views", "review", "exploration")


def video_path(stage_output: Path, camera: str, *, collection: str = "review") -> Path:
    """Use the run's purpose and camera as a stable, descriptive filename."""
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown video collection: {collection}")
    stage_output = stage_output.resolve()
    try:
        parts = stage_output.relative_to(PROJECT_ROOT / "runs").parts
    except ValueError:
        parts = (stage_output.name,)
    parts = [part for part in parts if part not in {"render", "review"}]
    name = "_".join([*parts, str(camera)])
    name = re.sub(r"[^\w-]+", "_", name).strip("_")
    path = DELIVERY_ROOT / collection / f"{name}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_video(path: Path, *, report: Path | None = None, **metadata) -> None:
    """Register an existing video without making another media copy."""
    path = path.resolve()
    relative = path.relative_to(DELIVERY_ROOT)
    collection = relative.parts[0]
    if collection not in COLLECTIONS:
        raise ValueError(f"Video must be in a delivery collection: {path}")
    index = DELIVERY_ROOT / "manifest.json"
    data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    output = path.relative_to(PROJECT_ROOT).as_posix()
    videos = data.setdefault("videos", [])
    entry = next((item for item in videos if item["output"] == output), None)
    if entry is None:
        entry = {"output": output}
        videos.append(entry)
    entry.update(metadata)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    entry.update(collection=collection, bytes=path.stat().st_size, sha256=digest.hexdigest())
    if report is not None:
        entry["report"] = str(Path(report).resolve())
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["deliverables"] = [item for item in videos if item["collection"] == "selected"]
    write_json(index, data)


def main(arguments: list[str] | None = None) -> int:
    """Move a reviewed video into its named collection, updating the index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--name", required=True, help="snake_case name without an extension")
    parser.add_argument("--collection", choices=COLLECTIONS, default="selected")
    parser.add_argument("--replace", action="store_true", help="replace this named result")
    args = parser.parse_args(arguments)
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", args.name):
        parser.error("--name must use lowercase snake_case")
    source = args.video.resolve(strict=True)
    source.relative_to(PROJECT_ROOT)
    if source.suffix.lower() != ".mp4":
        parser.error("--video must be an MP4 file")
    target = DELIVERY_ROOT / args.collection / f"{args.name}.mp4"
    if source != target:
        if target.exists() and not args.replace:
            parser.error(f"Result already exists; use --replace after review: {target}")
        index = DELIVERY_ROOT / "manifest.json"
        data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
        old = source.relative_to(PROJECT_ROOT).as_posix()
        entries = data.get("videos", [])
        metadata = dict(next((item for item in entries if item["output"] == old), {}))
        for key in ("output", "collection", "bytes", "sha256"):
            metadata.pop(key, None)
        metadata.setdefault("source_video", old)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        data["videos"] = [item for item in entries if item["output"] != old]
        write_json(index, data)
        record_video(target, **metadata)
    else:
        record_video(target)
    print(target)
    return 0
