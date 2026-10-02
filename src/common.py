"""
Shared code for data loading, image rendering and metrics.

evaluate.py, finetune.py (and later app.py) all import from here so the
SAME image preparation is used for baseline evaluation, fine-tuning and
the demo. If these differ, reported accuracy will not match the demo.
"""

import re
import statistics
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


# ------------------------------------------------------------
# InkML parsing
# ------------------------------------------------------------

def parse_inkml(path):
    """Returns (strokes, latex_label).
    strokes = list of strokes, each a list of (x, y) points."""
    root = ET.parse(path).getroot()

    strokes = []
    label = None
    normalized = None

    for el in root.iter():
        tag = el.tag.split("}")[-1]          # ignore XML namespace

        if tag == "trace" and el.text:
            pts = []
            for chunk in el.text.strip().split(","):
                v = chunk.split()
                if len(v) >= 2:
                    pts.append((float(v[0]), float(v[1])))
            if pts:
                strokes.append(pts)

        elif tag == "annotation":
            kind = el.get("type")
            if kind == "normalizedLabel":
                normalized = (el.text or "").strip()
            elif kind == "label":
                label = (el.text or "").strip()

    return strokes, (normalized or label or "")


# ------------------------------------------------------------
# Rendering strokes to an image
# ------------------------------------------------------------

def render_strokes(strokes, max_side=768, pad_mode="square"):
    """Black strokes on white, cropped to the ink and padded.
    pad_mode="square": pad to a square (no distortion when the model
                       resizes it to 384x384).
    pad_mode="stretch": keep the rectangle (the model will stretch it)."""
    xs = [x for s in strokes for x, _ in s]
    ys = [y for s in strokes for _, y in s]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    scale = max_side / max(w, h)

    # Pen thickness relative to the size of a typical stroke,
    # so it looks similar for short and long expressions.
    sizes = []
    for s in strokes:
        sx = [p[0] for p in s]
        sy = [p[1] for p in s]
        sizes.append(max(max(sx) - min(sx), max(sy) - min(sy)) * scale)
    typical = statistics.median(sizes) if sizes else max_side / 10
    line_w = int(min(12, max(3, 0.10 * typical)))

    margin = line_w * 2 + 10
    W = int(w * scale) + 2 * margin
    H = int(h * scale) + 2 * margin

    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    r = line_w / 2

    for s in strokes:
        pts = [((x - x0) * scale + margin, (y - y0) * scale + margin)
               for x, y in s]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse((x - r, y - r, x + r, y + r), fill=0)
            continue
        d.line(pts, fill=0, width=line_w, joint="curve")
        for (x, y) in (pts[0], pts[-1]):
            d.ellipse((x - r, y - r, x + r, y + r), fill=0)

    pad = 40
    padded = Image.new("L", (W + 2 * pad, H + 2 * pad), 255)
    padded.paste(img, (pad, pad))
    img = padded

    if pad_mode == "square":
        side = max(img.size)
        sq = Image.new("L", (side, side), 255)
        sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = sq

    return img.convert("RGB")


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def normalize_latex(text):
    """Remove all whitespace. Spacing is not meaningful in LaTeX math and
    different label styles use different spacing."""
    return re.sub(r"\s+", "", text or "")


def edit_distance(a, b):
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb)
            ))
        prev = cur
    return prev[-1]


def compute_metrics(preds, refs):
    """Exact Match Rate and Character Error Rate (whitespace ignored)."""
    n = len(refs)
    exact = 0
    total_edits = 0
    total_chars = 0
    per_sample = []

    for p, r in zip(preds, refs):
        p, r = normalize_latex(p), normalize_latex(r)
        e = edit_distance(p, r)
        exact += (p == r)
        total_edits += e
        total_chars += len(r)
        per_sample.append(e / max(len(r), 1))

    return {
        "n": n,
        "exact_match": 100.0 * exact / max(n, 1),
        "cer_corpus": 100.0 * total_edits / max(total_chars, 1),
        "cer_mean": 100.0 * sum(per_sample) / max(n, 1),
    }