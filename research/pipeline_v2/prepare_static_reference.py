"""Extract undistorted static-reference frames of the same subject.

Source: <STATIC_SOURCE>/<serial>.mp4 (8 cameras,
2048x1536, 300 frames, encoded 2026-08-28 11:20, same session as the action clip).
Calibration/intrinsics are reused from assets/helmet_native000_299_v1/manifest.json
so the static views live in exactly the same undistorted coordinate system.

Output: assets/static_reference_v1/<serial>/<frame:04d>.png + manifest.json
"""
import hashlib
import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = Path(os.environ.get("HOI_STATIC_SOURCE", "<STATIC_SOURCE>"))
MANIFEST = PROJECT / "assets" / "helmet_native000_299_v1" / "manifest.json"
OUT = PROJECT / "assets" / "static_reference_v1"
FRAMES = [60, 150, 240]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for view in meta["views"]:
        serial = view["serial"]
        path = SOURCE / f"{serial}.mp4"
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert (w, h) == (2048, 1536) and n >= max(FRAMES), (serial, n, w, h)
        K = np.array(view["K_native"])
        dist = np.array(view["distortion_opencv5"])
        Ku = np.array(view["K_undistorted_native"])
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, Ku, (w, h), cv2.CV_32FC1)
        out_dir = OUT / serial
        out_dir.mkdir(exist_ok=True)
        for f in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            assert ok, (serial, f)
            rgb = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR)
            Image.fromarray(rgb[:, :, ::-1]).save(out_dir / f"{f:04d}.png")
        cap.release()
        records.append({"serial": serial, "source": str(path), "source_sha256": sha256(path),
                        "frames": FRAMES, "frames_total": n,
                        "K": Ku.tolist(), "W": view["W"], "width": w, "height": h})
        print("done", serial, flush=True)
    report = {"success": True, "source": str(SOURCE), "frames": FRAMES,
              "note": "Intrinsics/distortion reused from the action manifest (same cameras and session).",
              "views": records}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # quick contact sheet
    sheet = Image.new("RGB", (4 * 512, 2 * 384), "white")
    for i, view in enumerate(records):
        img = Image.open(OUT / view["serial"] / f"{FRAMES[1]:04d}.png").convert("RGB").resize((512, 384))
        sheet.paste(img, ((i % 4) * 512, (i // 4) * 384))
    sheet.save(OUT / "views.jpg", quality=90)
    print("complete")


if __name__ == "__main__":
    main()
