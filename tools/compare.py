"""Screenshot the page at the reference size and compare element boxes.

Serve the project first:  python3 -m http.server 8765
Then:                     python tools/compare.py [--url URL] [--mouse X]

Writes .work/shot.png and .work/blend.png (50/50 overlay with the reference)
and prints, per element, the bbox delta (render - reference) in px.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "references" / "reference.png"
W, H = 1591, 989

# name, x0, y0, x1, y1, threshold (max channel > t; negative = dark text < -t)
BOXES = [
    ("logo icon", 70, 15, 145, 100, 120), ("logo name", 150, 30, 340, 60, 150), ("logo sub", 150, 62, 340, 78, 120),
    ("Home", 530, 35, 600, 62, 140), ("Academy", 620, 35, 730, 62, 140), ("Programs", 735, 35, 850, 62, 140),
    ("Success", 855, 35, 1010, 62, 140), ("Contact", 1015, 35, 1120, 62, 140), ("underline", 500, 64, 640, 80, 120),
    ("hdr btn txt", 1350, 35, 1520, 62, 150), ("tagline", 75, 205, 480, 232, 100),
    ("para1", 80, 490, 500, 518, 150), ("para2", 80, 519, 500, 546, 150), ("para3", 80, 547, 500, 575, 150),
    ("join txt", 100, 625, 300, 660, -90), ("play tri", 360, 625, 400, 660, 180), ("2:14", 420, 645, 500, 668, 120),
    ("st icon1", 100, 740, 190, 780, 150), ("st icon2", 240, 740, 320, 780, 150), ("st icon3", 380, 740, 450, 780, 150),
    ("st icon4", 510, 740, 600, 780, 150), ("st num1", 90, 788, 200, 816, 150), ("st num4", 500, 788, 610, 816, 150),
    ("st lbl1", 90, 822, 200, 842, 150), ("st lbl4", 500, 822, 610, 864, 150), ("card ttl", 1370, 188, 1530, 226, 170),
    ("card txt", 1370, 238, 1535, 276, 150), ("card icon", 1310, 195, 1350, 235, 150),
    ("scroll txt", 105, 928, 275, 948, 120), ("scroll line", 270, 932, 430, 944, 100),
]


def boxes(path):
    L = np.asarray(Image.open(path).convert("RGB")).astype(int).max(2)
    out = {}
    for n, x0, y0, x1, y1, t in BOXES:
        r = L[y0:y1, x0:x1]
        ys, xs = np.nonzero(r > t if t > 0 else r < -t)
        out[n] = None if len(xs) == 0 else (x0 + xs.min(), x0 + xs.max(), y0 + ys.min(), y0 + ys.max())
    return out


def title_lines(path):
    """Cap box of each headline line (EXELENA / FOOTBALL / ACADEMY)."""
    L = np.asarray(Image.open(path).convert("RGB")).astype(int).max(2)
    rows = (L[230:470, 84:140] > 170).any(1)
    lines, s = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            y0, y1 = 230 + s, 230 + i - 1
            xs = np.nonzero((L[y0:y1, 80:600] > 170).any(0))[0]
            run = [xs[0]]
            for x in xs[1:]:
                if x - run[-1] > 25:
                    break
                run.append(x)
            lines.append((80 + run[0], 80 + run[-1], y0, y1))
            s = None
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/index.html")
    ap.add_argument("--mouse", type=float, help="move the cursor to this x before the shot")
    args = ap.parse_args()
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    shot = work / "shot.png"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(args.url)
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(1500)
        if args.mouse is not None:
            pg.mouse.move(args.mouse, H / 2)
            pg.wait_for_timeout(1500)
        pg.screenshot(path=str(shot))
        b.close()
    Image.blend(Image.open(shot).convert("RGB"), Image.open(REF).convert("RGB"), 0.5).save(work / "blend.png")

    ref, our = boxes(REF), boxes(shot)
    for n, *_ in BOXES:
        a, b = ref[n], our[n]
        if not (a and b):
            print(f"{n:12s} missing (ref={a}, ours={b})")
            continue
        d = [b[i] - a[i] for i in range(4)]
        flag = "" if max(map(abs, d)) <= 1 else "  <--"
        print(f"{n:12s} x{d[0]:+d},{d[1]:+d}  y{d[2]:+d},{d[3]:+d}{flag}")
    for name, a, b in zip(("EXELENA", "FOOTBALL", "ACADEMY"), title_lines(REF), title_lines(shot)):
        print(f"{name:12s} x{b[0] - a[0]:+d}  y{b[2] - a[2]:+d},{b[3] - a[3]:+d}  (right edge unreliable: bg lights)")
