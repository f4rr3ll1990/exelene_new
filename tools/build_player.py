"""Build the head-tracking player layers from references/video.mp4.

Pipeline (each step writes into WORK, default ./.work):
  frames   decode every video frame to PNG (ffmpeg from imageio-ffmpeg)
  segment  estimate torso shift per frame (phase correlation) + rembg masks
  matte    refined RGBA mattes, stabilised and padded       -> WORK/stab/
  input    green-screen canvas of the neutral frame for codex -> WORK/torso_in.png
           (then run codex with tools/prompts/torso.txt and save WORK/torso.png)
  export   body layer + head frames as WebP                 -> assets/img/player/

Usage:  python tools/build_player.py [frames|segment|matte|input|export|all] [--out DIR]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "references" / "video.mp4"
WORK = ROOT / ".work"

F0, F1 = 34, 74      # head-turn segment of the video (inclusive)
NEUTRAL = 52         # most frontal frame; stabilisation reference and default pose
PAD = 24             # transparent margin around stabilised frames
SCALE = 1.8          # video px -> player canvas px
CANVAS = (1440, 1211)
BODY_Y = 640         # body layer starts here (canvas px)


def ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def frame_path(i):
    return WORK / "all" / f"f{i + 1:03d}.png"


def step_frames():
    import imageio_ffmpeg
    (WORK / "all").mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(VIDEO),
                    "-map", "0:0", str(WORK / "all" / "f%03d.png")], check=True)


def phase_shift(a, b):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-6
    r = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if y > r.shape[0] // 2:
        y -= r.shape[0]
    if x > r.shape[1] // 2:
        x -= r.shape[1]
    return y, x


def step_segment():
    from statistics import median
    from rembg import new_session, remove

    gray = lambda i: np.asarray(Image.open(frame_path(i)).convert("L"), dtype=np.float32)
    ref = gray(NEUTRAL)[290:416]
    win = np.hanning(ref.shape[1])[None, :] * np.hanning(ref.shape[0])[:, None]
    idx = np.arange(F0, F1 + 1)
    est = np.array([phase_shift(ref * win, gray(i)[290:416] * win) for i in idx], float)
    # raw estimates are noisy (shoulders also rotate): median, then a cubic fit
    med = np.array([[median(est[max(0, k - 3):k + 4, c]) for c in range(2)] for k in range(len(idx))])
    sm = np.stack([np.polyval(np.polyfit(idx, med[:, c], 3), idx) for c in range(2)], 1)
    sm -= sm[list(idx).index(NEUTRAL)]
    json.dump({int(i): [float(a), float(b)] for i, (a, b) in zip(idx, sm)},
              open(WORK / "shifts.json", "w"), indent=1)

    (WORK / "masks").mkdir(exist_ok=True)
    session = new_session("isnet-general-use")
    for i in idx:
        remove(Image.open(frame_path(i)).convert("RGB"), session=session,
               only_mask=True).save(WORK / "masks" / f"m{i:03d}.png")


def hull(pts):
    pts = sorted(set(pts))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def to_f(img):
    return np.asarray(img).astype(np.float32) / 255


def step_matte():
    """Background is pure black but so is the jersey, so a luma key alone fails.
    alpha = max(eroded rembg core, convex hull of the torso, luma inside dilated mask)."""
    (WORK / "stab").mkdir(exist_ok=True)
    shifts = json.load(open(WORK / "shifts.json"))
    for k, (dy, dx) in shifts.items():
        i = int(k)
        rgb = to_f(Image.open(frame_path(i)).convert("RGB"))
        m = Image.open(WORK / "masks" / f"m{i:03d}.png").convert("L")
        mf = to_f(m)
        H, W = mf.shape
        yy = np.arange(H)[:, None] / H * np.ones((1, W))
        head = yy < 0.66
        core_head = to_f(m.point(lambda v: 255 if v > 200 else 0)
                         .filter(ImageFilter.MinFilter(19)).filter(ImageFilter.GaussianBlur(5)))
        core_body = to_f(m.point(lambda v: 255 if v > 128 else 0)
                         .filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2)))
        ys, xs = np.nonzero((mf > 0.2) & (yy > 0.70))
        poly = hull(list(zip(xs[::3].tolist(), ys[::3].tolist()))
                    + [(int(xs.min()), H - 1), (int(xs.max()), H - 1)])
        hm = Image.new("L", (W, H), 0)
        ImageDraw.Draw(hm).polygon(poly, fill=255)
        hm = to_f(hm.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2)))
        dil = to_f(m.point(lambda v: 255 if v > 15 else 0)
                   .filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2)))
        lum = ss(rgb.max(2), 0.015, 0.20)
        a = np.clip(np.maximum.reduce([np.where(head, core_head, core_body),
                                       hm * (yy > 0.70), np.minimum(dil, lum)]), 0, 1)
        # un-premultiply against the black background
        fg = np.where(a[..., None] > 0.02, np.clip(rgb / np.maximum(a[..., None], 1e-3), 0, 1), 0)
        out = Image.fromarray((np.dstack([fg, a]) * 255 + .5).astype(np.uint8), "RGBA")
        canvas = Image.new("RGBA", (W + 2 * PAD, H + 2 * PAD), (0, 0, 0, 0))
        canvas.paste(out, (PAD + round(dx), PAD + round(dy)))
        canvas.save(WORK / "stab" / f"s{i:03d}.png")


def step_input():
    f = Image.open(WORK / "stab" / f"s{NEUTRAL:03d}.png")
    big = f.resize((round(f.width * SCALE), round(f.height * SCALE)), Image.LANCZOS)
    cv = Image.new("RGBA", (big.width, round(big.height * 1.45)), (0, 255, 0, 255))
    cv.alpha_composite(big)
    cv.convert("RGB").save(WORK / "torso_in.png")


def step_export(out):
    out.mkdir(parents=True, exist_ok=True)
    CW, CH = CANVAS
    # body: chroma-keyed codex outpaint, only below the collar (head comes from the frames)
    t = to_f(Image.open(WORK / "torso.png").convert("RGB").resize((CW, CH), Image.LANCZOS))
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    a = 1 - ss(g - np.maximum(r, b), 0.12, 0.45)
    rgb = np.dstack([r, np.minimum(g, np.maximum(r, b) * 1.02), b])   # despill
    a = a * ss(np.arange(CH)[:, None] * np.ones((1, CW)), 660, 730)
    body = Image.fromarray((np.dstack([rgb, a]) * 255 + .5).astype(np.uint8), "RGBA")
    body = body.crop((0, BODY_Y, CW, CH))
    bb = body.getchannel("A").getbbox()
    body = body.crop((bb[0], 0, bb[2], CH - BODY_Y))
    body.save(out / "body.webp", "WEBP", quality=82, method=6, alpha_quality=90)
    meta = {"body": {"x": bb[0], "y": BODY_Y, "w": body.width, "h": body.height}}

    ids = list(range(F0, F1 + 1))
    U = None
    for i in ids:
        bb = Image.open(WORK / "stab" / f"s{i:03d}.png").getchannel("A").getbbox()
        U = bb if U is None else (min(U[0], bb[0]), min(U[1], bb[1]), max(U[2], bb[2]), max(U[3], bb[3]))
    x0, y0, x1, y1 = [int(v * SCALE) for v in U]
    x0, y0, x1, y1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
    fb = int((PAD + 416 - 6) * SCALE)   # bottom of the video frame; fade into the body layer
    for k, i in enumerate(ids):
        f = Image.open(WORK / "stab" / f"s{i:03d}.png").resize(
            (round(800 * SCALE), round(464 * SCALE)), Image.LANCZOS)
        arr = np.asarray(f).astype(np.float32)
        arr[..., 3] *= 1 - ss(np.arange(arr.shape[0])[:, None], fb - 70, fb - 8)
        f = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA").crop((x0, y0, x1, min(y1, fb)))
        f.save(out / f"f{k:02d}.webp", "WEBP", quality=80, method=6, alpha_quality=85)
    meta["frames"] = {"count": len(ids), "x": x0, "y": y0, "w": x1 - x0,
                      "h": min(y1, fb) - y0, "neutral": ids.index(NEUTRAL)}
    print("constants for assets/js/main.js:", json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["frames", "segment", "matte", "input", "export", "all"])
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "img" / "player")
    args = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    steps = ["frames", "segment", "matte", "input"] if args.step == "all" else [args.step]
    for s in steps:
        print("==", s)
        step_export(args.out) if s == "export" else globals()[f"step_{s}"]()
