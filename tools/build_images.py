"""Convert codex-generated PNGs (WORK/gen/*.png) into the WebP assets.

Icons and the logo are generated on pure black, so alpha is derived from
brightness and the colour is un-premultiplied; the result is cropped to the
glyph and downscaled.

Usage:  python tools/build_images.py [--src .work/gen] [--out assets/img]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# name -> (output path, longest side px, black threshold)
ICONS = {
    "logo": ("logo.webp", 192, 0.05),
    "users": ("icons/users.webp", 96, 0.03),
    "trophy": ("icons/trophy.webp", 96, 0.03),
    "globe": ("icons/globe.webp", 96, 0.03),
    "star": ("icons/star.webp", 96, 0.03),
    "play": ("icons/play.webp", 96, 0.03),
    "arrow": ("icons/arrow.webp", 96, 0.03),
    "globe-glow": ("icons/globe-glow.webp", 128, 0.02),
}


def key_black(src, dst, size, thr):
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
    lum = rgb.max(2)
    a = np.clip((lum - thr) / (lum.max() * 0.92 - thr), 0, 1)
    fg = np.where(a[..., None] > 0, np.clip(rgb / np.maximum(a[..., None], 1e-3), 0, 1), 0)
    im = Image.fromarray((np.dstack([fg, a]) * 255 + .5).astype(np.uint8), "RGBA")
    bb = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    pad = int(max(bb[2] - bb[0], bb[3] - bb[1]) * 0.02)
    im = im.crop((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))
    r = size / max(im.size)
    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "WEBP", quality=92, method=6, alpha_quality=100)
    print(dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst, im.size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / ".work" / "gen")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "img")
    args = ap.parse_args()
    for name, (out, size, thr) in ICONS.items():
        key_black(args.src / f"{name}.png", args.out / out, size, thr)
    Image.open(args.src / "bg.png").convert("RGB").save(args.out / "bg.webp", "WEBP", quality=88, method=6)
