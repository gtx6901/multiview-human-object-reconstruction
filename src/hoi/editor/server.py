"""Local editor for placing rebuilt actors in a glTF scene.

Serves a single-page UI and a small JSON API over the same renderer used by
`pipeline.py render --environment`. Preview frames are real renders, not a
separate approximation.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np

from hoi.runtime import PROJECT_ROOT
from hoi.settings import SceneConfig


LOCK = threading.Lock()
STATE = {
    "show": None,
    "scene": None,
    "model_key": None,
    "config": None,
    "job": None,
}
STATIC = Path(__file__).resolve().parent / "static"
PREVIEW = PROJECT_ROOT / "runs" / "editor_preview.png"
STILL = PROJECT_ROOT / "runs" / "editor_still.png"
FRAME_RE = re.compile(r"\[environment\]\s+\S+\s+frame\s+(\d+)")


def scene_specs() -> list[dict]:
    entries = []
    for path in sorted((PROJECT_ROOT / "configs/environments").glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model = PROJECT_ROOT / spec["model"] if not Path(spec["model"]).is_absolute() else Path(spec["model"])
        entries.append({"name": path.stem, "model": spec["model"], "exists": model.exists()})
    return entries


def load_spec(name: str) -> dict:
    path = PROJECT_ROOT / "configs/environments" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_configs() -> list[dict]:
    entries = []
    for path in sorted((PROJECT_ROOT / "configs/datasets").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries.append({"name": path.stem, "actors": [actor["id"] for actor in data.get("actors", [])]})
    return entries


def save_spec(name: str, spec: dict) -> Path:
    path = PROJECT_ROOT / "configs/environments" / f"{name}.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def model_key(spec: dict) -> str:
    return json.dumps(
        {
            "model": spec["model"],
            "scale": spec.get("model_scale", 1),
            "offset": spec.get("model_offset", [0, 0, 0]),
            "lighting": spec.get("lighting"),
        },
        sort_keys=True,
    )


def ensure_show(name: str, spec: dict):
    from hoi.rendering.environment import SceneShow

    config = STATE["config"]
    if STATE["show"] is not None and STATE["scene"] == name:
        show = STATE["show"]
        key = model_key(spec)
        if key != STATE["model_key"]:
            show.options = spec
            show.rebuild_environment()
            STATE["model_key"] = key
        else:
            show.options = spec
            show.apply_placement()
        return show
    release_show()
    show = SceneShow(config, spec)
    STATE.update(show=show, scene=name, model_key=model_key(spec))
    return show


def release_show():
    STATE["show"] = None
    STATE["scene"] = None
    STATE["model_key"] = None
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def preview(spec: dict, name: str, frame: int, width: int, height: int) -> dict:
    from PIL import Image

    with LOCK:
        show = ensure_show(name, spec)
        frames = show.frames
        if frame not in frames:
            frame = frames[0]
        image = show.render(frame, width, height)
        picture = Image.fromarray(image)
        PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        picture.save(PREVIEW)
        K, W = show.camera(frame, width, height)
        return {
            "image": f"/preview.png?t={time.time_ns()}",
            "frame": frame,
            "frames": frames,
            "width": width,
            "height": height,
            "K": K.cpu().tolist(),
            "W": W.cpu().tolist(),
            "position": show.position.tolist(),
            "up": (show.rotation[1] / np.linalg.norm(show.rotation[1])).tolist(),
        }


def still(spec: dict, name: str, frame: int, width: int, height: int) -> dict:
    from PIL import Image

    with LOCK:
        show = ensure_show(name, spec)
        if frame not in show.frames:
            frame = show.frames[0]
        image = show.render(frame, width * 2, height * 2)
        picture = Image.fromarray(image).resize((width, height), Image.LANCZOS)
        STILL.parent.mkdir(parents=True, exist_ok=True)
        picture.save(STILL)
        return {"image": f"/still.png?t={time.time_ns()}", "frame": frame}


def render_job(name: str, spec: dict) -> None:
    job = STATE["job"]
    spec_path = save_spec(name, spec)
    frames = job["total"]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline.py"),
        "render",
        "--config",
        str(STATE["config_path"]),
        "--environment",
        str(spec_path),
        "--video",
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    job["status"] = "running"
    for line in process.stdout:
        job["lines"].append(line.rstrip())
        match = FRAME_RE.search(line)
        if match:
            job["done"] = int(match.group(1)) + 1
        job["updated"] = time.time()
    code = process.wait()
    job["status"] = "done" if code == 0 else "failed"
    job["code"] = code
    if code == 0:
        from hoi.results import video_path

        output = STATE["config"].output / "environment" / name
        job["video"] = str(video_path(output, "camera_main").relative_to(PROJECT_ROOT))
    job["ended"] = time.time()


class Handler(BaseHTTPRequestHandler):
    server_version = "SceneEditor/1.0"

    def log_message(self, fmt, *args):
        pass

    def json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def send_json(self, value: dict, code: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, content_type: str, cache: bool = False) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
        elif path == "/preview.png":
            self.send_file(PREVIEW, "image/png")
        elif path == "/still.png":
            self.send_file(STILL, "image/png")
        elif path == "/video.mp4":
            job = STATE["job"] or {}
            target = job.get("video")
            if not target:
                self.send_error(404)
                return
            self.send_file(PROJECT_ROOT / target, "video/mp4")
        elif path == "/api/scenes":
            self.send_json({"scenes": scene_specs(), "casts": dataset_configs()})
        elif path == "/api/scene":
            from urllib.parse import parse_qs, urlparse

            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            try:
                self.send_json({
                    "name": name,
                    "spec": load_spec(name),
                    "frames": None,
                    "up": list(STATE["config"].world_up),
                })
            except FileNotFoundError:
                self.send_json({"error": f"unknown scene {name}"}, 404)
        elif path == "/api/job":
            job = STATE["job"]
            self.send_json(job or {"status": "idle"})
        elif path == "/asset":
            from urllib.parse import parse_qs, urlparse

            relative = parse_qs(urlparse(self.path).query).get("path", [""])[0]
            target = (PROJECT_ROOT / relative).resolve()
            if PROJECT_ROOT not in target.parents:
                self.send_error(403)
                return
            types = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
                     ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
            self.send_file(target, types.get(target.suffix.lower(), "application/octet-stream"), cache=True)
        elif path.startswith("/static/"):
            name = path[len("/static/"):]
            target = (STATIC / name).resolve()
            if STATIC not in target.parents:
                self.send_error(403)
                return
            types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".png": "image/png"}
            self.send_file(target, types.get(target.suffix, "application/octet-stream"))
        else:
            self.send_error(404)

    def do_POST(self):
        from urllib.parse import parse_qs, urlparse

        path = self.path.split("?", 1)[0]
        query = parse_qs(urlparse(self.path).query)
        try:
            body = self.json_body()
        except ValueError:
            self.send_json({"error": "invalid JSON"}, 400)
            return
        if path == "/api/preview":
            job = STATE["job"]
            if job and job.get("status") in ("starting", "running"):
                self.send_json({"error": "a render is running; preview is paused"}, 409)
                return
            name = body.get("scene") or query.get("name", [""])[0]
            try:
                result = preview(
                    body["spec"], name, int(body.get("frame", 0)),
                    int(body.get("width", 504)), int(body.get("height", 378)),
                )
                self.send_json(result)
            except Exception as error:  # keep the editor alive on render errors
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)
        elif path == "/api/cast":
            job = STATE["job"]
            if job and job.get("status") in ("starting", "running"):
                self.send_json({"error": "a render is running; cast is locked"}, 409)
                return
            name = body.get("name") or ""
            target = PROJECT_ROOT / "configs/datasets" / f"{name}.json"
            if not target.is_file():
                self.send_json({"error": f"unknown cast {name}"}, 404)
                return
            try:
                config = SceneConfig.load(target)
                release_show()
                STATE["config"] = config
                STATE["config_path"] = str(target)
                self.send_json({"name": name, "actors": [actor.id for actor in config.actors]})
            except Exception as error:
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)
        elif path == "/api/scene":
            name = query.get("name", [""])[0]
            try:
                save_spec(name, body["spec"])
                self.send_json({"saved": name})
            except Exception as error:
                self.send_json({"error": str(error)}, 500)
        elif path == "/api/render":
            if STATE["job"] and STATE["job"].get("status") == "running":
                self.send_json({"error": "a render is already running"}, 409)
                return
            name = body.get("scene") or query.get("name", [""])[0]
            show = ensure_show(name, body["spec"])
            total = len(show.frames)
            release_show()
            STATE["job"] = {
                "status": "starting", "done": 0, "total": total, "lines": [],
                "video": None, "started": time.time(), "scene": name,
            }
            thread = threading.Thread(target=render_job, args=(name, body["spec"]), daemon=True)
            thread.start()
            self.send_json({"started": name, "total": total})
        else:
            self.send_error(404)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    config = SceneConfig.load(args.config.resolve())
    STATE["config"] = config
    STATE["config_path"] = str(args.config.resolve())
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"scene editor: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
