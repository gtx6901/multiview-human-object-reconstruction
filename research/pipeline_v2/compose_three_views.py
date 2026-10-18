"""Compose the three-panel FCS-helmet video: GT camera | fixed novel | rotating novel."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2

PROJECT = Path(__file__).resolve().parent.parent
W, H = 1134, 1512
OUT = PROJECT / "runs" / "fcs_helmet_three_views_30fps.mp4"
GT_CAM = "21305717"
GT_DIR = PROJECT / "assets" / "helmet_native000_299_v1" / GT_CAM
MASK_DIR = PROJECT / "assets" / "sam2_all300_v1" / GT_CAM / "person"
FIXED = PROJECT / "runs" / "_smooth_v2_full300"
ROT = PROJECT / "runs" / "_orbit_rot_v1"


def gt_panel(frame):
    img = Image.open(GT_DIR / f"{frame:04d}.png").convert("RGB")
    m = np.asarray(Image.open(MASK_DIR / f"{frame:04d}.png").convert("L")) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        return ImageOps.fit(img, (W, H))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    h = max(int((ys.max() - ys.min()) * 1.25), H // 2)
    w = int(h * W / H)
    box = (max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
           min(img.width, int(cx + w / 2)), min(img.height, int(cy + h / 2)))
    return img.crop(box).resize((W, H))


def main():
    frames = sorted(int(p.stem) for p in ROT.glob("*.png"))
    print("frames", len(frames), frames[0], frames[-1])
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W * 3, H + 44))
    for f in frames:
        panels = [gt_panel(f),
                  Image.open(FIXED / f"{f:04d}_orbit_008.png").convert("RGB").resize((W, H)),
                  Image.open(ROT / f"{f:04d}.png").convert("RGB").resize((W, H))]
        sheet = Image.new("RGB", (W * 3, H + 44), "white")
        for i, p in enumerate(panels):
            sheet.paste(p, (i * W, 44))
        d = ImageDraw.Draw(sheet)
        for i, label in enumerate(["GT (camera 21305717)", "FIXED NOVEL (orbit_008)", "ROTATING NOVEL"]):
            d.text((i * W + 10, 14), label, fill="black")
        writer.write(np.asarray(sheet)[:, :, ::-1])
        if f % 50 == 0:
            print("frame", f, flush=True)
    writer.release()
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
