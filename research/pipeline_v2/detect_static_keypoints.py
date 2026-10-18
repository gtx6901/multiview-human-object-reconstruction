"""RTMW keypoints for the static reference frames (same subject, no helmet).

No SAM2 mask exists for the static capture, so the person box is bootstrapped:
run RTMW on a few candidate boxes, take the one with the best body score, then
refine the box from the detected body points and run once more.

Output: assets/static_reference_v1/keypoints.npz
  xy[cam,frame,133,2], score[cam,frame,133], in_image[cam,frame,133], boxes[cam,frame,4]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = __import__("hoi_paths").ROOT
sys.path[:0] = [str(ROOT / "deps"), str(ROOT.parent / "lhmpp_20261016/deps")]

from rtmlib import RTMPose  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
STATIC = PROJECT / "assets" / "static_reference_v1"
MODEL = PROJECT / "runtime" / "wholebody_runtime_v1" / "rtmw.onnx"


def run(model, image, box):
    k, s = model(image, bboxes=[box])
    return k[0], s[0]


def bootstrap_box(model, image):
    h, w = image.shape[:2]
    candidates = [
        [0.0, 0.0, float(w), float(h)],
        [0.2 * w, 0.1 * h, 0.8 * w, 0.95 * h],
        [0.3 * w, 0.15 * h, 0.7 * w, 0.95 * h],
    ]
    best = None
    for box in candidates:
        k, s = run(model, image, box)
        body = float(s[:17].mean())
        if best is None or body > best[0]:
            best = (body, k, s, box)
    return best


def refine_box(k, s, width, height):
    pts = k[:23][s[:23] > 0.5]
    if len(pts) < 6:
        return None
    lo, hi = pts.min(0), pts.max(0)
    cx, cy = (lo + hi) / 2.0
    bw, bh = hi[0] - lo[0], hi[1] - lo[1]
    bw, bh = max(bw * 1.35, 64), max(bh * 1.25, 64)
    x0 = min(max(cx - bw / 2, 0), width - bw)
    y0 = min(max(cy - bh / 2, 0), height - bh)
    return [float(x0), float(y0), float(x0 + bw), float(y0 + bh)]


def main():
    meta = json.loads((STATIC / "manifest.json").read_text(encoding="utf-8"))
    cams = [v["serial"] for v in meta["views"]]
    frames = meta["frames"]
    model = RTMPose(str(MODEL), model_input_size=(288, 384), backend="onnxruntime",
                    device="cpu", to_openpose=False)
    started = time.time()
    XY = np.zeros((len(cams), len(frames), 133, 2), np.float32)
    SC = np.zeros((len(cams), len(frames), 133), np.float32)
    IN = np.zeros((len(cams), len(frames), 133), bool)
    BOX = np.zeros((len(cams), len(frames), 4), np.float32)
    qa = Image.new("RGB", (4 * 384, 2 * 288), "white")
    for ci, cam in enumerate(cams):
        h = w = None
        for fi, f in enumerate(frames):
            image = np.asarray(Image.open(STATIC / cam / f"{f:04d}.png").convert("RGB"))
            h, w = image.shape[:2]
            body, k0, s0, box0 = bootstrap_box(model, image)
            box = refine_box(k0, s0, w, h) or box0
            k, s = run(model, image, box)
            xy, sc = k, s
            XY[ci, fi], SC[ci, fi], BOX[ci, fi] = xy, sc, np.array(box, np.float32)
            px, py = xy[:, 0], xy[:, 1]
            IN[ci, fi] = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if fi == 1:
                pic = Image.fromarray(image)
                d = ImageDraw.Draw(pic)
                d.rectangle(box, outline="red", width=4)
                for j, (x, y) in enumerate(xy):
                    if sc[j] > 0.5:
                        color = "lime" if j < 17 else ("cyan" if j < 23 else ("yellow" if j < 91 else "red"))
                        d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
                qa.paste(pic.resize((384, 288)), ((ci % 4) * 384, (ci // 4) * 288))
        print("done", cam, flush=True)
    np.savez_compressed(STATIC / "keypoints.npz", xy=XY, score=SC, in_image=IN, boxes=BOX,
                        cameras=np.asarray(cams), frames=np.asarray(frames),
                        model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest())
    qa.save(STATIC / "qa_grid.jpg", quality=90)
    (STATIC / "keypoints_report.json").write_text(json.dumps({
        "success": True, "cameras": cams, "frames": frames, "seconds": round(time.time() - started, 1),
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "box_strategy": "multi-scale bootstrap on full frame, then refine from body keypoints x1.35",
        "mean_body_score": float(SC[:, :, :17].mean()),
    }, indent=2), encoding="utf-8")
    print("complete", round(time.time() - started, 1), "s")


if __name__ == "__main__":
    main()
