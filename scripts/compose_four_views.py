"""Four-panel video: GT | reconstruction from the same camera | fixed novel | rotating novel."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2

PROJECT = Path(__file__).resolve().parent.parent
W, H = 1134, 1512
GT_CAM = "21305717"
GT_DIR = PROJECT / "assets" / "capture" / GT_CAM
MASK_DIR = PROJECT / "assets" / "masks" / GT_CAM / "person"
RECON = PROJECT / "runs" / "same_camera"
FIXED = PROJECT / "runs" / "novel_fixed"
ROT = PROJECT / "runs" / "novel_orbit"
OUT = PROJECT / "runs" / "four_views_30fps.mp4"


def fixed_box():
    x0, y0, x1, y1 = 1e9, 1e9, -1e9, -1e9
    for f in range(0, 300, 5):
        m = np.asarray(Image.open(MASK_DIR / f"{f:04d}.png").convert("L")) > 127
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x0, y0 = min(x0, xs.min()), min(y0, ys.min())
        x1, y1 = max(x1, xs.max()), max(y1, ys.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = (y1 - y0) * 1.38
    w = h * W / H
    if w < (x1 - x0) * 1.05:
        w = (x1 - x0) * 1.05
        h = w * H / W
    img_w, img_h = Image.open(GT_DIR / "0000.png").size
    left = min(max(0, cx - w / 2), img_w - w)
    top = min(max(0, cy - h / 2 - h * 0.06), img_h - h)
    return (int(left), int(top), int(left + w), int(top + h))


def main():
    frames = sorted(int(p.stem) for p in ROT.glob("*.png"))
    box = fixed_box()
    print("frames", len(frames), "crop", box)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W * 4, H + 44))
    for f in frames:
        gt = Image.open(GT_DIR / f"{f:04d}.png").convert("RGB").crop(box).resize((W, H))
        sx, sy = W / 2048.0, H / 1536.0
        sb = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
        rc = Image.open(RECON / f"{f:04d}.png").convert("RGB").crop(sb).resize((W, H))
        fx = Image.open(FIXED / f"{f:04d}_orbit_008.png").convert("RGB").resize((W, H))
        rt = Image.open(ROT / f"{f:04d}.png").convert("RGB").resize((W, H))
        sheet = Image.new("RGB", (W * 4, H + 44), "white")
        for i, p in enumerate([gt, rc, fx, rt]):
            sheet.paste(p, (i * W, 44))
        d = ImageDraw.Draw(sheet)
        for i, label in enumerate([f"GT ({GT_CAM})", "RECON (same camera)", "FIXED NOVEL", "ROTATING NOVEL"]):
            d.text((i * W + 10, 14), label, fill="black")
        writer.write(np.asarray(sheet)[:, :, ::-1])
        if f % 100 == 0:
            print("frame", f, flush=True)
    writer.release()
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
