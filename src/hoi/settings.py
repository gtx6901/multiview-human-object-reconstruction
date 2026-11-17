"""Dataset configuration shared by single-person and multi-person scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from hoi.runtime import PROJECT_ROOT, runtime_settings


@dataclass(frozen=True)
class ActorConfig:
    id: str
    avatar: Path
    motion: Path
    manifest: Path | None = None
    observations: Path | None = None
    keypoints: Path | None = None
    initialization: Path | None = None
    hand_observations: Path | None = None
    proportions: Path | None = None


@dataclass(frozen=True)
class SceneConfig:
    id: str
    fps: float
    actors: tuple[ActorConfig, ...]
    scene_path: Path
    manifest_path: Path
    output: Path
    world_up: tuple[float, float, float]
    world_up_source: str
    runtime: dict[str, Any]
    camera_space: str = "raw"
    preparation: dict[str, Any] = field(default_factory=dict)
    rigid_object: dict[str, str] | None = None
    rendering: dict[str, Any] = field(default_factory=dict)
    identity_refinement: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "SceneConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        root = PROJECT_ROOT

        def resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        actors = tuple(
            ActorConfig(
                **{key: value if key == "id" else resolve(value) for key, value in actor.items()}
            )
            for actor in data["actors"]
        )
        identities = [actor.id for actor in actors]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("A scene needs uniquely named actors")
        if not math.isfinite(float(data["fps"])) or float(data["fps"]) <= 0:
            raise ValueError("fps must be finite and positive")
        return cls(
            id=data["id"],
            fps=float(data["fps"]),
            actors=actors,
            scene_path=resolve(data["scene_path"]),
            manifest_path=resolve(data["manifest_path"]),
            output=resolve(data.get("output", f"runs/{data['id']}")),
            world_up=tuple(data["world_up"]),
            world_up_source=data["world_up_source"],
            runtime=runtime_settings(data.get("runtime")),
            camera_space=data.get("camera_space", "raw"),
            preparation=data.get("preparation", {}),
            rigid_object=data.get("rigid_object"),
            rendering=data.get("rendering", {}),
            identity_refinement=data.get("identity_refinement", {}),
        )
