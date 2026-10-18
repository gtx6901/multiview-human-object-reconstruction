"""RTMW wholebody keypoints for all cameras and frames.

One detector only: RTMW (onnx, 288x384) on the SAM2.1 person-mask bounding box,
expanded and clipped. Frames come from the cached undistorted JPEGs, so every
keypoint is already in the original (undistorted) image coordinate system.

Output: assets/keypoints_all300_v1/<serial>.npz
  xy[300,133,2] float32, score[300,133] float32, in_image[300,133] bool,
  boxes[300,4] float32, frames[300] int32
Layout: 0:17 body, 17:23 feet, 23:91 face (68), 91:112 left hand, 112:133 right hand.
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
SAM2 = PROJECT / "assets" / "sam2_all300_v1"
DATA = PROJECT / "assets" / "helmet_native000_299_v1"
OUT = PROJECT / "assets" / "keypoints_all300_v1"
MODEL = PROJECT / "runtime" / "wholebody_runtime_v1" / "rtmw.onnx"

EXPAND = 1.15
GROUPS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91), "left_hand": (91, 112), "right_hand": (112, 133)}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.asarray(Image.open(path).convert("L"))


def mask_box(mask, width, height, expand=EXPAND, fallback=None):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return fallback
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    half_w, half_h = (box[2] - box[0]) * expand / 2.0, (box[3] - box[1]) * expand / 2.0
    return np.array([max(0.0, cx - half_w), max(0.0, cy - half_h),
                     min(width - 1.0, cx + half_w), min(height - 1.0, cy + half_h)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    views = {v["serial"]: v for v in meta["views"]}
    cams = sorted(p.name for p in SAM2.iterdir() if p.is_dir())
    assert len(cams) == 8, cams

    model = RTMPose(str(MODEL), model_input_size=(288, 384), backend="onnxruntime",
                    device="cpu", to_openpose=False)
    started = time.time()
    report = {"success": True, "model": str(MODEL), "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              "model_input": [288, 384], "box_source": "SAM2.1 person mask bbox x1.15, clipped",
              "keypoint_layout": {"0:17": "body", "17:23": "feet", "23:91": "face",
                                  "91:112": "left_hand", "112:133": "right_hand"},
              "score_semantics": "raw SimCC peak score (~0-5), not a calibrated probability",
              "views": []}

    for cam in cams:
        frames = sorted(int(p.stem) for p in (SAM2 / cam / "video_frames").glob("*.jpg"))
        assert frames == list(range(300)), (cam, len(frames))
        xy = np.zeros((len(frames), 133, 2), np.float32)
        score = np.zeros((len(frames), 133), np.float32)
        in_image = np.zeros((len(frames), 133), bool)
        boxes = np.zeros((len(frames), 4), np.float32)
        last_box = None
        empty_masks = 0
        for i, f in enumerate(frames):
            image = load_rgb(SAM2 / cam / "video_frames" / f"{f:04d}.jpg")
            mask = load_mask(SAM2 / cam / "person" / f"{f:04d}.png")
            h, w = mask.shape
            box = mask_box(mask, w, h, fallback=last_box)
            if box is None:
                box = np.array([0.0, 0.0, w - 1.0, h - 1.0])
                empty_masks += 1
            last_box = box
            boxes[i] = box
            k, s = model(image, bboxes=[box.tolist()])
            assert k.shape == (1, 133, 2) and s.shape == (1, 133)
            xy[i] = k[0]
            score[i] = s[0]
            px, py = k[0][:, 0], k[0][:, 1]
            in_image[i] = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        view = views[cam]
        np.savez_compressed(OUT / f"{cam}.npz", xy=xy, score=score, in_image=in_image,
                            boxes=boxes, frames=np.asarray(frames, np.int32))
        report["views"].append({
            "serial": cam, "frames": len(frames), "empty_masks": empty_masks,
            "K": view.get("K_undistorted_native"), "W": view.get("W"),
            "width": view.get("width"), "height": view.get("height"),
            "score_quantiles": {g: [float(np.quantile(score[:, a:b], q)) for q in (0.1, 0.5, 0.9)]
                                for g, (a, b) in GROUPS.items()},
            "in_image_rate": {g: float(in_image[:, a:b].mean()) for g, (a, b) in GROUPS.items()},
        })
        (OUT / "status.json").write_text(json.dumps({"completed": len(report["views"]),
                                                     "seconds": round(time.time() - started, 1)}), encoding="utf-8")
        print("done", cam, flush=True)

    # QA grid: four frames x four cameras with points drawn
    picks = [0, 42, 180, 299]
    sheet = Image.new("RGB", (4 * 384, 4 * 288), "white")
    for row, cam in enumerate(cams[:4]):
        data = np.load(OUT / f"{cam}.npz")
        meta_view = views[cam]
        for col, f in enumerate(picks):
            image = Image.open(SAM2 / cam / "video_frames" / f"{f:04d}.jpg").convert("RGB")
            draw = ImageDraw.Draw(image)
            for j, (x, y) in enumerate(data["xy"][f]):
                if data["in_image"][f, j] and data["score"][f, j] > 0.5:
                    color = "lime" if j < 17 else ("cyan" if j < 23 else ("yellow" if j < 91 else "red"))
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            thumb = image.resize((384, 288))
            ImageDraw.Draw(thumb).text((4, 4), f"{cam} f{f}", fill="blue")
            sheet.paste(thumb, (col * 384, row * 288))
    sheet.save(OUT / "qa_grid.jpg", quality=88)

    report["seconds"] = round(time.time() - started, 1)
    report["qa_grid"] = "qa_grid.jpg"
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
