import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageGrab, Image, ImageChops, ImageTk, ImageDraw, ImageOps, ImageStat
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from display_math import render_math
from common import render_strokes as common_render   # same image pipeline as training
import os
import math
import random
import threading
import queue
import socket
import time

# ============================================================
# LOAD OFFLINE AI MODEL
# ============================================================

# Baseline model. After fine-tuning, change this to your checkpoint,
# for example: r"models/finetuned/best"
MODEL_PATH = "fhswf/TrOCR_Math_handwritten"

# True  = fast: one greedy pass on a square-padded image.
# False = slower but sometimes more accurate: beam search on 2 image shapes
#         (roughly 5-10x more computation on a CPU).
FAST_MODE = True

print("Loading model:", MODEL_PATH)

processor = TrOCRProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Model loaded successfully!")

# ============================================================
# APP WINDOW
# ============================================================

root = tk.Tk()
root.title("MathWrite AI — Handwritten Mathematics to LaTeX")
root.geometry("1280x850")
root.minsize(1100, 780)
root.configure(bg="#08111F")

# Colors
BG = "#08111F"
CARD = "#0F1B2D"
CARD_2 = "#122238"
BORDER = "#203754"
WHITE = "#F8FAFC"
MUTED = "#8FA3BC"
BLUE = "#38BDF8"
BLUE_DARK = "#0284C7"
GREEN = "#22C55E"
GREEN_DARK = "#166534"
PURPLE = "#A78BFA"
AMBER = "#FBBF24"
PINK = "#F472B6"
PANEL = "#0D1B2E"

# ============================================================
# COLOR / ANIMATION HELPERS
# ============================================================

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def lerp_color(a, b, t):
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return "#%02x%02x%02x" % (
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )

GRADIENT_STOPS = [BLUE, PURPLE, PINK, GREEN, BLUE]

def gradient_color(t):
    t = t % 1.0
    seg = t * (len(GRADIENT_STOPS) - 1)
    i = int(seg)
    return lerp_color(GRADIENT_STOPS[i], GRADIENT_STOPS[i + 1], seg - i)

tick = 0
is_busy = False

# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=BG, height=96)
header.pack(fill="x", padx=28, pady=(18, 4))
header.pack_propagate(False)

brand = tk.Frame(header, bg=BG)
brand.pack(side="left", fill="y")

tk.Label(
    brand,
    text="✦  MathWrite AI",
    font=("Segoe UI", 28, "bold"),
    fg=WHITE,
    bg=BG
).pack(anchor="w")

tk.Label(
    brand,
    text="Handwritten mathematics  •  AI recognition  •  LaTeX output",
    font=("Segoe UI", 10),
    fg=MUTED,
    bg=BG
).pack(anchor="w", pady=(3, 0))

# Live AI visualizer
live_box = tk.Frame(
    header,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
live_box.pack(side="right", fill="y")

live_left = tk.Frame(live_box, bg=PANEL)
live_left.pack(side="left", padx=(14, 8), pady=10)

live_dot = tk.Label(
    live_left,
    text="●",
    font=("Segoe UI", 14, "bold"),
    fg=GREEN,
    bg=PANEL
)
live_dot.pack()

tk.Label(
    live_box,
    text="LOCAL AI",
    font=("Segoe UI", 9, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(side="left", anchor="n", pady=(11, 0))

net_badge = tk.Label(
    live_box,
    text="OFFLINE",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg=PANEL,
    width=8,
    anchor="w"
)
net_badge.pack(side="left", anchor="n", padx=(6, 10), pady=(12, 0))

# Live equalizer (idle = gentle wave, busy = jumping bars)
EQ_H = 26
eq_canvas = tk.Canvas(
    live_box, width=46, height=EQ_H,
    bg=PANEL, highlightthickness=0
)
eq_canvas.pack(side="left", padx=(0, 14), pady=10)
eq_bars = [
    eq_canvas.create_rectangle(2 + i * 9, EQ_H - 4, 8 + i * 9, EQ_H,
                               fill=BLUE, outline="")
    for i in range(5)
]

# Floating math particles (drawn once, moved every frame)
floating_canvas = tk.Canvas(
    header,
    width=380,
    height=70,
    bg=BG,
    highlightthickness=0
)
floating_canvas.place(relx=0.40, rely=0.10)

PARTICLE_SYMBOLS = ["+", "∑", "x²", "∫", "π", "√", "∞", "Δ", "≈", "θ"]
particles = []
for sym in PARTICLE_SYMBOLS:
    px = random.uniform(10, 370)
    py = random.uniform(12, 58)
    item = floating_canvas.create_text(
        px, py, text=sym,
        fill="#1E4E73",
        font=("Segoe UI", random.randint(13, 19), "bold")
    )
    particles.append({
        "id": item, "x": px, "y": py,
        "dx": random.choice([-1, 1]) * random.uniform(0.3, 0.9),
        "dy": random.uniform(-0.12, 0.12),
        "phase": random.uniform(0, 6.28),
    })

# Animated gradient accent line under the header
accent_bar = tk.Canvas(root, height=3, bg=BG, highlightthickness=0)
accent_bar.pack(fill="x", padx=28, pady=(0, 6))
ACCENT_SEGMENTS = 120
accent_rects = []

def build_accent(event=None):
    accent_bar.delete("all")
    accent_rects.clear()
    w = max(accent_bar.winfo_width(), 1)
    seg_w = w / ACCENT_SEGMENTS
    for i in range(ACCENT_SEGMENTS):
        accent_rects.append(
            accent_bar.create_rectangle(
                i * seg_w, 0, (i + 1) * seg_w + 1, 3,
                fill=gradient_color(i / ACCENT_SEGMENTS), outline=""
            )
        )

accent_bar.bind("<Configure>", build_accent)

# ============================================================
# MAIN WORKSPACE — FIXED GRID SO BUTTONS ALWAYS STAY VISIBLE
# ============================================================

workspace = tk.Frame(root, bg=BG)
workspace.pack(fill="both", expand=True, padx=28, pady=6)

workspace.grid_columnconfigure(0, weight=1, uniform="cards")
workspace.grid_columnconfigure(1, weight=1, uniform="cards")
workspace.grid_rowconfigure(0, weight=1)

# ============================================================
# INPUT CARD
# ============================================================

input_card = tk.Frame(
    workspace,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)
input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

# colored top strip
tk.Frame(input_card, bg=BLUE, height=3).pack(fill="x")

input_header = tk.Frame(input_card, bg=CARD, height=48)
input_header.pack(fill="x", padx=18, pady=(8, 0))
input_header.pack_propagate(False)

tk.Label(
    input_header,
    text="✎  YOUR INPUT",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=CARD
).pack(side="left", pady=12)

tk.Label(
    input_header,
    text="Draw your expression",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=CARD
).pack(side="right", pady=13)

# ---- Drawing toolbar (pen / eraser / size) ----
toolbar = tk.Frame(input_card, bg=CARD)
toolbar.pack(fill="x", padx=18, pady=(0, 6))

tool = "pen"
pen_width = 4
tool_buttons = {}
size_buttons = {}

def refresh_toolbar():
    for name, b in tool_buttons.items():
        active = (name == tool)
        b.config(
            bg=BLUE_DARK if active else "#1A2B44",
            fg=WHITE if active else MUTED
        )
    for w, b in size_buttons.items():
        active = (w == pen_width)
        b.config(
            bg=PURPLE if active else "#1A2B44",
            fg="#0B1220" if active else MUTED
        )

def set_tool(name):
    global tool
    tool = name
    canvas.config(cursor="pencil" if name == "pen" else "circle")
    refresh_toolbar()

def set_size(w):
    global pen_width
    pen_width = w
    refresh_toolbar()

def small_tool_button(parent, text, cmd):
    b = tk.Button(
        parent, text=text, command=cmd,
        font=("Segoe UI", 8, "bold"),
        bg="#1A2B44", fg=MUTED,
        activebackground="#334155", activeforeground=WHITE,
        relief="flat", bd=0, cursor="hand2", padx=10, pady=3
    )
    b.pack(side="left", padx=(0, 5))
    return b

tool_buttons["pen"] = small_tool_button(toolbar, "✎ Pen", lambda: set_tool("pen"))
tool_buttons["eraser"] = small_tool_button(toolbar, "◌ Eraser", lambda: set_tool("eraser"))

tk.Label(toolbar, text="│", fg=BORDER, bg=CARD).pack(side="left", padx=4)

size_buttons[2] = small_tool_button(toolbar, "Thin", lambda: set_size(2))
size_buttons[4] = small_tool_button(toolbar, "Medium", lambda: set_size(4))
size_buttons[7] = small_tool_button(toolbar, "Bold", lambda: set_size(7))

# Canvas with a clean white writing surface (animated glow border)
canvas_outer = tk.Frame(
    input_card,
    bg=BLUE,
    padx=2,
    pady=2
)
canvas_outer.pack(padx=18, pady=(2, 10), fill="both", expand=True)

canvas = tk.Canvas(
    canvas_outer,
    bg="white",
    cursor="pencil",
    highlightthickness=0
)
canvas.pack(fill="both", expand=True)

# Small live drawing hint
hint_row = tk.Frame(input_card, bg=CARD, height=28)
hint_row.pack(fill="x", padx=18)
hint_row.pack_propagate(False)

hint_dot = tk.Label(
    hint_row,
    text="●  Canvas ready",
    font=("Segoe UI", 8, "bold"),
    fg="#67E8F9",
    bg=CARD
)
hint_dot.pack(side="left")

stroke_chip = tk.Label(
    hint_row,
    text="STROKES 0",
    font=("Segoe UI", 8, "bold"),
    fg=PURPLE,
    bg=CARD
)
stroke_chip.pack(side="left", padx=14)

tk.Label(
    hint_row,
    text="Write large • keep symbols separated",
    font=("Segoe UI", 8),
    fg=MUTED,
    bg=CARD
).pack(side="right")

# ============================================================
# OUTPUT CARD
# ============================================================

output_card = tk.Frame(
    workspace,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)
output_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

tk.Frame(output_card, bg=PURPLE, height=3).pack(fill="x")

output_header = tk.Frame(output_card, bg=CARD, height=48)
output_header.pack(fill="x", padx=18, pady=(8, 0))
output_header.pack_propagate(False)

tk.Label(
    output_header,
    text="✦  AI RECOGNITION",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=CARD
).pack(side="left", pady=12)

output_state = tk.Label(
    output_header,
    text="READY",
    font=("Segoe UI", 8, "bold"),
    fg=GREEN,
    bg=CARD
)
output_state.pack(side="right", pady=13)

# Raw LaTeX result
result_box = tk.Frame(
    output_card,
    bg=CARD_2,
    highlightbackground="#294664",
    highlightthickness=1
)
result_box.pack(fill="x", padx=18, pady=(2, 9))

result_top = tk.Frame(result_box, bg=CARD_2)
result_top.pack(fill="x", padx=12, pady=(9, 2))

tk.Label(
    result_top,
    text="RECOGNIZED LATEX",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg=CARD_2
).pack(side="left")

length_chip = tk.Label(
    result_top,
    text="0 CHARS",
    font=("Segoe UI", 8, "bold"),
    fg=MUTED,
    bg=CARD_2
)
length_chip.pack(side="right")

result_label = tk.Label(
    result_box,
    text="Result will appear here",
    font=("Consolas", 15, "bold"),
    fg=WHITE,
    bg=CARD_2,
    wraplength=500,
    justify="left",
    anchor="w"
)
result_label.pack(fill="x", padx=12, pady=(0, 11))

# Rendered equation
tk.Label(
    output_card,
    text="RENDERED EQUATION",
    font=("Segoe UI", 8, "bold"),
    fg=MUTED,
    bg=CARD
).pack(anchor="w", padx=18)

math_image_frame = tk.Frame(
    output_card,
    bg="white",
    highlightbackground="#CBD5E1",
    highlightthickness=1
)
math_image_frame.pack(fill="both", expand=True, padx=18, pady=(4, 10))

math_image_label = tk.Label(
    math_image_frame,
    text="Your rendered equation will appear here",
    font=("Segoe UI", 10),
    fg="#64748B",
    bg="white"
)
math_image_label.pack(fill="both", expand=True)

# ============================================================
# ACTION BAR
# ============================================================

action_bar = tk.Frame(root, bg=BG, height=58)
action_bar.pack(fill="x", padx=28, pady=(4, 3))
action_bar.pack_propagate(False)

def make_button(parent, text, command, bg_color, hover_color, width=15):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        height=1,
        font=("Segoe UI", 9, "bold"),
        bg=bg_color,
        fg=WHITE,
        activebackground=hover_color,
        activeforeground=WHITE,
        relief="flat",
        bd=0,
        cursor="hand2",
        padx=10,
        pady=7
    )
    button.pack(side="left", padx=4, pady=7)

    def enter(_):
        button.configure(bg=hover_color)

    def leave(_):
        button.configure(bg=bg_color)

    button.bind("<Enter>", enter)
    button.bind("<Leave>", leave)
    return button

left_actions = tk.Frame(action_bar, bg=BG)
left_actions.pack(side="left")

right_actions = tk.Frame(action_bar, bg=BG)
right_actions.pack(side="right")

# ============================================================
# DRAWING
# ============================================================

last_x = None
last_y = None
strokes = []
stroke_tools = {}
stroke_pts = {}
stroke_w = {}
current_stroke = None
stroke_counter = 0

def update_stroke_chip():
    stroke_chip.config(text=f"STROKES {len(strokes)}")

def start_draw(event):
    global last_x, last_y, current_stroke
    last_x = event.x
    last_y = event.y
    current_stroke = None

def draw(event):
    global last_x, last_y, current_stroke, stroke_counter

    if last_x is not None and last_y is not None:
        if current_stroke is None:
            stroke_counter += 1
            current_stroke = f"stroke{stroke_counter}"
            strokes.append(current_stroke)
            stroke_tools[current_stroke] = tool
            stroke_pts[current_stroke] = [(last_x, last_y)]
            stroke_w[current_stroke] = pen_width
            update_stroke_chip()

        if tool == "pen":
            canvas.create_line(
                last_x,
                last_y,
                event.x,
                event.y,
                width=pen_width,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True,
                tags=(current_stroke,)
            )
        else:
            canvas.create_line(
                last_x,
                last_y,
                event.x,
                event.y,
                width=pen_width * 5,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True,
                tags=(current_stroke,)
            )

    if current_stroke is not None:
        stroke_pts[current_stroke].append((event.x, event.y))

    last_x = event.x
    last_y = event.y

def stop_draw(event):
    global last_x, last_y, current_stroke
    last_x = None
    last_y = None
    current_stroke = None

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

def strokes_have_overline(tags=None):

    if tags is None:
        tags = strokes

    boxes = []

    for tag in tags:

        if (
            stroke_tools.get(tag) != "pen"
            or len(stroke_pts.get(tag, [])) < 2
        ):
            continue

        boxes.append(stroke_box(tag))

    if len(boxes) < 2:
        return False

    x0 = min(b[0] for b in boxes)
    x1 = max(b[2] for b in boxes)
    total_w = max(x1 - x0, 1)

    for i, (bx0, by0, bx1, by1) in enumerate(boxes):

        w = bx1 - bx0
        h = by1 - by0

        if w < 0.7 * total_w or h > max(14, w * 0.15):
            continue

        others = [b for j, b in enumerate(boxes) if j != i]

        if all(o[1] >= by1 - 4 for o in others):
            return True

    return False


def render_strokes(tags):
    """Redraw the chosen pen strokes onto a clean white image.
    Independent of screen scaling, so it also lets us recognise
    parts of an expression (numerator, denominator...) separately."""
    S = 2
    M = 40
    W = max(canvas.winfo_width(), 1)
    H = max(canvas.winfo_height(), 1)
    img = Image.new("RGB", (W * S + 2 * M, H * S + 2 * M), "white")
    d = ImageDraw.Draw(img)
    include = set(tags)

    # Use the SAME renderer as evaluation/fine-tuning (src/common.py).
    # (If the eraser was used we fall back to the canvas renderer below,
    #  because erased parts of a stroke cannot be represented there.)
    if not any(stroke_tools.get(t) == "eraser" for t in strokes):
        pen_strokes = [
            stroke_pts[t] for t in strokes
            if t in include and stroke_tools.get(t) == "pen"
            and len(stroke_pts.get(t, [])) >= 2
        ]
        if pen_strokes:
            return common_render(pen_strokes, pad_mode="square")

    for tag in strokes:
        kind = stroke_tools.get(tag)
        if kind == "pen" and tag not in include:
            continue
        pts = stroke_pts.get(tag, [])
        if len(pts) < 2:
            continue
        if kind == "pen":
            wd = max(2, int(stroke_w[tag] * S))
            color = "black"
        else:
            wd = int(stroke_w[tag] * 5 * S)
            color = "white"
        xy = [(x * S + M, y * S + M) for x, y in pts]
        d.line(xy, fill=color, width=wd, joint="curve")
        r = wd / 2
        for (x, y) in (xy[0], xy[-1]):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return img

def stroke_box(tag):
    pts = stroke_pts.get(tag, [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def find_fraction(tag_list=None):

    source = strokes if tag_list is None else tag_list

    tags = [
        t for t in source
        if (
            stroke_tools.get(t) == "pen"
            and len(stroke_pts.get(t, [])) >= 2
        )
    ]

    if len(tags) < 3:
        return None

    best = None

    for bar in tags:

        x0, y0, x1, y1 = stroke_box(bar)

        w = x1 - x0
        h = y1 - y0

        if w < 40 or h > max(20, w * 0.25):
            continue

        pts = stroke_pts[bar]
        (ax, ay), (bx, by) = pts[0], pts[-1]

        def bar_y(x):

            if abs(bx - ax) < 1:
                return (y0 + y1) / 2

            t = min(1, max(0, (x - ax) / (bx - ax)))

            return ay + (by - ay) * t

        margin = max(15, w * 0.1)

        left, numer, denom, right = [], [], [], []

        for t in tags:

            if t == bar:
                continue

            sx0, sy0, sx1, sy1 = stroke_box(t)

            cx = (sx0 + sx1) / 2
            cy = (sy0 + sy1) / 2

            # A tall stroke that crosses the bar line (a bracket or
            # parenthesis around the fraction) is NOT a numerator or
            # denominator. It belongs to the left or right side.
            height = sy1 - sy0
            ybar = bar_y(cx)

            if (
                height > 0
                and (ybar - sy0) > 0.3 * height
                and (sy1 - ybar) > 0.3 * height
            ):
                if cx < (x0 + x1) / 2:
                    left.append(t)
                else:
                    right.append(t)
                continue

            if cx < x0 - margin:
                left.append(t)
            elif cx > x1 + margin:
                right.append(t)
            elif cy < bar_y(cx):
                numer.append(t)
            else:
                denom.append(t)

        if numer and denom and (best is None or w > best[0]):
            best = (w, left, numer, denom, right)

    if best is None:
        return None

    return best[1:]


# Build a tree of pieces: plain text pieces and fractions (nested allowed)
def build_node(tag_list):

    frac = find_fraction(tag_list)

    if frac is None:

        return (
            "ink",
            preprocess_image(render_strokes(tag_list)),
            strokes_have_overline(tag_list)
        )

    left, numer, denom, right = frac

    children = []

    if left:
        children.append(build_node(left))

    children.append(
        ("frac", build_node(numer), build_node(denom))
    )

    if right:
        children.append(build_node(right))

    return ("seq", children)


def build_plan(processed_image):

    tags = [t for t in strokes if stroke_tools.get(t) == "pen"]

    return build_node(tags)


def recognize_node(node):

    kind = node[0]

    if kind == "ink":

        if node[1] is None:
            return ""

        return recognize_image(node[1], node[2])

    if kind == "frac":

        num = recognize_node(node[1])
        den = recognize_node(node[2])

        return "\\frac{" + num + "}{" + den + "}"

    parts = [recognize_node(c) for c in node[1]]

    return " ".join(p for p in parts if p)


def recognize_plan(plan):

    return clean_latex_result(recognize_node(plan))


def undo_stroke():
    if strokes:
        canvas.delete(strokes.pop())
        update_stroke_chip()
        set_status("●  Last stroke removed", BLUE)

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    if image is None:
        return None

    image = image.convert("L")

    # Light text on a dark background (dark-mode screenshots, white-on-black
    # equation images): flip it so the model sees dark ink on white paper.
    if ImageStat.Stat(image).mean[0] < 110:
        image = ImageOps.invert(image)

    # Large safety border to prevent edge cut-off.
    border = 80
    image = ImageOps.expand(image, border=border, fill=255)

    bbox = ImageOps.invert(image).getbbox()

    if bbox is None:
        return None

    left, top, right, bottom = bbox
    margin = 40

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(image.width, right + margin)
    bottom = min(image.height, bottom + margin)

    image = image.crop((left, top, right, bottom))

    image = image.point(lambda p: 0 if p < 200 else 255)

    return image.convert("RGB")


# ============================================================
# MODEL RECOGNITION
# ============================================================

def has_real_overline(image):
    """
    Check whether the handwriting image contains a long horizontal
    stroke above the main expression.

    This prevents TrOCR from turning an ordinary expression such as
    x + y^2 into \\overline{x+y^2} just because the model confuses
    the writing with an overline.
    """
    gray = image.convert("L")

    # Find the actual handwriting area.
    white = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, white)
    bbox = diff.getbbox()

    if bbox is None:
        return False

    gray = gray.crop(bbox)

    # Binary image: black handwriting, white background.
    bw = gray.point(lambda p: 0 if p < 180 else 255)

    width, height = bw.size

    if width < 20 or height < 10:
        return False

    # Look only near the top of the expression.
    top_height = max(5, int(height * 0.22))

    pixels = bw.load()

    # A genuine overline should contain a long horizontal run.
    longest_run = 0

    for y in range(top_height):
        run = 0

        for x in range(width):
            if pixels[x, y] < 128:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0

    # Require a line covering a substantial part of the expression.
    return longest_run >= max(25, int(width * 0.55))


def clean_false_overline(result, image, real_overline=None):
    """
    Remove a model-generated overline only when the handwriting image
    does not contain evidence of a real overline.
    """
    result = result.strip()

    if not result:
        return result

    if "\\overline{" not in result:
        return result

    if real_overline is None:
        real_overline = has_real_overline(image)

    if real_overline:
        return result

    # Remove the outer overline wrapper while preserving its contents.
    while "\\overline{" in result:
        start = result.find("\\overline{")

        if start == -1:
            break

        content_start = start + len("\\overline{")

        depth = 1
        end = content_start

        while end < len(result) and depth:
            if result[end] == "{":
                depth += 1
            elif result[end] == "}":
                depth -= 1
            end += 1

        if depth != 0:
            break

        inside = result[content_start:end - 1]

        result = (
            result[:start]
            + inside
            + result[end:]
        )

    return result


def clean_false_underline(result):

    if not result:
        return result

    result = result.strip()

    while True:

        start = result.find(r"\underline")

        if start == -1:
            break

        brace_start = start + len(r"\underline")

        while (
            brace_start < len(result)
            and result[brace_start].isspace()
        ):
            brace_start += 1

        if (
            brace_start >= len(result)
            or result[brace_start] != "{"
        ):
            result = result[:start] + result[start + len(r"\underline"):]
            continue

        content_start = brace_start + 1
        depth = 1
        end = content_start

        while end < len(result) and depth > 0:

            if result[end] == "{":
                depth += 1
            elif result[end] == "}":
                depth -= 1

            end += 1

        if depth != 0:
            break

        inside = result[content_start:end - 1]
        result = result[:start] + inside + result[end:]

    return result.strip()


def clean_latex_result(
    result,
    image=None,
    real_overline=None
):

    if not result:
        return result

    result = result.strip()

    if image is not None:

        result = clean_false_overline(
            result,
            image,
            real_overline
        )

    result = clean_false_underline(result)

    return result.strip()


def pad_square(img):
    w, h = img.size
    side = max(w, h)
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(img, ((side - w) // 2, (side - h) // 2))
    return sq

def generate_text(img):
    """Returns (latex, confidence score)."""
    pixel_values = processor(
        images=img,
        return_tensors="pt"
    ).pixel_values

    if FAST_MODE:
        ids = model.generate(pixel_values, max_new_tokens=256)
        return processor.batch_decode(ids, skip_special_tokens=True)[0], 0.0

    try:
        out = model.generate(
            pixel_values,
            num_beams=3,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        text = processor.batch_decode(
            out.sequences, skip_special_tokens=True
        )[0]
        score = float(out.sequences_scores[0])
    except Exception:
        ids = model.generate(pixel_values, max_new_tokens=256)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        score = -99.0

    return text, score

def recognize_image(image, real_overline=None):
    # The model squashes every image to a square. Tall or wide
    # expressions get distorted, so also try a square-padded copy
    # (no distortion) and keep whichever the model is more sure of.
    if FAST_MODE:
        candidates = [pad_square(image)]
    else:
        candidates = [image]
        ratio = image.width / max(image.height, 1)
        if ratio < 0.8 or ratio > 1.25:
            candidates.append(pad_square(image))

    best_text, best_score = "", -1e9
    for cand in candidates:
        text, score = generate_text(cand)
        if score > best_score:
            best_text, best_score = text, score

    result = clean_latex_result(
        best_text,
        image,
        real_overline
    )

    return result.strip()

# ============================================================
# BACKGROUND WORKER (keeps the UI animations alive while AI thinks)
# ============================================================

def run_async(work, on_success, on_fail):
    q = queue.Queue()

    def target():
        try:
            q.put(("ok", work()))
        except Exception as e:
            q.put(("err", e))

    threading.Thread(target=target, daemon=True).start()

    def poll():
        try:
            kind, value = q.get_nowait()
        except queue.Empty:
            root.after(60, poll)
            return
        if kind == "ok":
            on_success(value)
        else:
            on_fail(value)

    poll()

# ============================================================
# RESULT DISPLAY
# ============================================================

def display_rendered_math():
    image_path = "recognized_math.png"

    if not os.path.exists(image_path):
        return

    try:
        image = Image.open(image_path).convert("RGB")

        # Make the rendered result fit the panel.
        image.thumbnail(
            (500, 180),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(image)

        math_image_label.config(
            image=photo,
            text=""
        )
        math_image_label.image = photo

    except Exception as e:
        print("Error displaying rendered equation:", e)

# Typewriter effect for the LaTeX result
typing_job = None

def cancel_typing():
    global typing_job
    if typing_job is not None:
        try:
            root.after_cancel(typing_job)
        except Exception:
            pass
        typing_job = None

def type_result(text):
    global typing_job
    cancel_typing()
    length_chip.config(text=f"{len(text)} CHARS")
    result_label.config(text="")
    state = {"i": 0}

    def step():
        global typing_job
        state["i"] = min(len(text), state["i"] + 2)
        result_label.config(text=text[:state["i"]])
        if state["i"] < len(text):
            typing_job = root.after(18, step)
        else:
            typing_job = None

    step()

# ============================================================
# STATUS HELPERS
# ============================================================

def set_status(text, color=BLUE):
    status_label.config(text=text, fg=color)
    root.update_idletasks()

def set_processing(active):
    global is_busy
    is_busy = active
    if active:
        output_state.config(text="PROCESSING...", fg=AMBER)
        live_dot.config(fg=AMBER)
    else:
        output_state.config(text="READY", fg=GREEN)
        live_dot.config(fg=GREEN)

# ============================================================
# CLEAR
# ============================================================

def clear_canvas():
    cancel_typing()
    canvas.delete("all")
    strokes.clear()
    stroke_tools.clear()
    stroke_pts.clear()
    stroke_w.clear()
    update_stroke_chip()

    result_label.config(
        text="Result will appear here"
    )
    length_chip.config(text="0 CHARS")

    math_image_label.config(
        image="",
        text="Your rendered equation will appear here"
    )
    math_image_label.image = None

    output_state.config(text="READY", fg=GREEN)
    set_status("●  Ready — draw an expression to begin", BLUE)

    print("Canvas cleared.")

# ============================================================
# CAPTURE CANVAS
# ============================================================

def capture_canvas():
    # IMPORTANT:
    # Windows display scaling (125%, 150%, etc.) can make
    # ImageGrab capture a different region than the Tkinter canvas.
    # The original working version compensated for Windows DPI.
    # Keep that behavior so the AI receives the actual handwriting.
    root.update_idletasks()

    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    scale = root.winfo_fpixels("1i") / 72

    x = int(x * scale)
    y = int(y * scale)
    width = int(width * scale)
    height = int(height * scale)

    return ImageGrab.grab(
        bbox=(
            x,
            y,
            x + width,
            y + height
        )
    )

# ============================================================
# RECOGNIZE
# ============================================================

def recognize():
    if is_busy:
        return

    try:
        set_processing(True)
        set_status("●  Capturing handwriting...", AMBER)

        image = render_strokes(
            [t for t in strokes if stroke_tools.get(t) == "pen"]
        )
        image.save("user_expression.png")

        set_status("●  Preprocessing image...", AMBER)

        processed_image = preprocess_image(image)

        if processed_image is None:
            set_processing(False)
            set_status("●  No handwriting detected", "#FB7185")

            messagebox.showwarning(
                "No Expression",
                "Please draw a mathematical expression first."
            )
            return

        processed_image.save(
            "user_expression_processed.png"
        )

        set_status("●  AI is recognizing your expression...", AMBER)

        def on_success(result):
            try:
                if not result:
                    result = "Could not recognize expression"

                type_result(result)

                set_status("●  Rendering mathematical expression...", AMBER)

                try:
                    if not render_math(result):
                        raise ValueError("LaTeX could not be rendered")
                    display_rendered_math()
                except Exception as render_error:
                    print("Rendering error:", render_error)
                    math_image_label.config(
                        image="",
                        text="Could not render this expression"
                    )
                    math_image_label.image = None
                    set_status(
                        "●  Recognized, but rendering failed",
                        AMBER
                    )
                    set_processing(False)
                    return

                set_processing(False)
                set_status(
                    f"●  Recognition completed in {time.time() - t0:.1f} s",
                    GREEN
                )

                print("Recognized expression:", result)
            except Exception as e:
                on_fail(e)

        def on_fail(e):
            print("Recognition error:", e)

            set_processing(False)
            set_status("●  Error occurred", "#FB7185")

            messagebox.showerror(
                "Recognition Error",
                str(e)
            )

        t0 = time.time()
        plan = build_plan(processed_image)

        run_async(
            lambda: recognize_plan(plan),
            on_success,
            on_fail
        )

    except Exception as e:
        print("Recognition error:", e)

        set_processing(False)
        set_status("●  Error occurred", "#FB7185")

        messagebox.showerror(
            "Recognition Error",
            str(e)
        )

# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    if is_busy:
        return

    file_path = filedialog.askopenfilename(
        title="Select Handwritten Math Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not file_path:
        return

    try:

        set_processing(True)

        set_status("●  Loading uploaded image...", AMBER)

        original = Image.open(file_path).convert("RGB")

        # Show the uploaded picture on the canvas.
        canvas.delete("all")
        strokes.clear()
        stroke_tools.clear()
        stroke_pts.clear()
        stroke_w.clear()
        update_stroke_chip()

        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        preview = original.copy()
        preview.thumbnail((cw - 30, ch - 30), Image.Resampling.LANCZOS)

        canvas.photo = ImageTk.PhotoImage(preview)   # keep a reference
        canvas.create_image(
            cw // 2, ch // 2,
            image=canvas.photo,
            anchor="center"
        )

        set_status("●  Preprocessing uploaded image...", AMBER)

        processed_image = preprocess_image(original)

        if processed_image is None:

            set_processing(False)
            set_status("●  No handwriting detected", "#FB7185")

            messagebox.showwarning(
                "No Expression",
                "No mathematical expression was detected in the image."
            )
            return

        processed_image.save("user_expression_processed.png")

        set_status("●  AI is recognizing uploaded image...", AMBER)

        def on_success(result):

            try:

                if not result:
                    result = "Could not recognize expression"

                result = clean_latex_result(result)

                type_result(result)

                if not render_math(result):

                    math_image_label.config(
                        image="",
                        text="Could not render this expression"
                    )
                    math_image_label.image = None

                    set_processing(False)
                    set_status("●  Recognized, but rendering failed", AMBER)
                    return

                display_rendered_math()

                set_processing(False)
                set_status("●  Image recognized successfully", GREEN)

            except Exception as e:
                on_fail(e)

        def on_fail(e):

            print("Upload recognition error:", e)

            set_processing(False)
            set_status("●  Upload recognition failed", "#FB7185")

            messagebox.showerror("Upload Error", str(e))

        run_async(
            lambda: recognize_image(processed_image),
            on_success,
            on_fail
        )

    except Exception as e:

        print("Upload error:", e)

        set_processing(False)
        set_status("●  Upload error", "#FB7185")

        messagebox.showerror("Upload Error", str(e))


# ============================================================
# COPY LATEX
# ============================================================

def copy_latex():
    latex = result_label.cget("text")

    if latex == "Result will appear here" or not latex.strip():
        messagebox.showwarning(
            "No Result",
            "Please recognize an expression first."
        )
        return

    root.clipboard_clear()
    root.clipboard_append(latex)
    root.update()

    set_status("●  LaTeX copied to clipboard", GREEN)

# ============================================================
# SAVE RESULT
# ============================================================

def save_result():
    image_path = "recognized_math.png"

    if not os.path.exists(image_path):
        messagebox.showwarning(
            "No Result",
            "Please recognize an expression first."
        )
        return

    save_path = filedialog.asksaveasfilename(
        title="Save Mathematical Expression",
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png")
        ]
    )

    if save_path:
        image = Image.open(image_path)
        image.save(save_path)

        set_status("●  Mathematical expression saved", GREEN)

        messagebox.showinfo(
            "Saved",
            "Mathematical expression saved successfully."
        )

# ============================================================
# BUTTONS
# ============================================================

make_button(
    left_actions,
    "↺  Clear",
    clear_canvas,
    "#243247",
    "#334155",
    12
)

make_button(
    left_actions,
    "↶  Undo",
    undo_stroke,
    "#243247",
    "#334155",
    10
)

make_button(
    left_actions,
    "✦  Recognize",
    recognize,
    BLUE_DARK,
    "#0EA5E9",
    14
)

make_button(
    left_actions,
    "↑  Upload Image",
    upload_image,
    "#243247",
    "#334155",
    16
)

make_button(
    right_actions,
    "▣  Copy LaTeX",
    copy_latex,
    "#243247",
    "#334155",
    15
)

make_button(
    right_actions,
    "↓  Save Result",
    save_result,
    GREEN_DARK,
    "#16A34A",
    15
)

# ============================================================
# LIVE STATUS BAR
# ============================================================

status_bar = tk.Frame(
    root,
    bg="#0D1828",
    highlightbackground=BORDER,
    highlightthickness=1,
    height=34
)
status_bar.pack(fill="x", padx=28, pady=(0, 12))
status_bar.pack_propagate(False)

status_label = tk.Label(
    status_bar,
    text="●  Ready — draw an expression to begin",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg="#0D1828"
)
status_label.pack(side="left", padx=12, pady=7)

net_status = tk.Label(
    status_bar,
    text="LOCAL MODEL  •  OFFLINE MODE",
    font=("Segoe UI", 8, "bold"),
    fg="#64748B",
    bg="#0D1828"
)
net_status.pack(side="right", padx=12, pady=7)

# Shimmering progress bar (only moves while the AI is working)
PROG_W = 160
progress = tk.Canvas(
    status_bar, width=PROG_W, height=6,
    bg="#16283F", highlightthickness=0
)
progress.pack(side="right", padx=10, pady=13)
progress_seg = progress.create_rectangle(-60, 0, -10, 6, fill=BLUE, outline="")

# ============================================================
# LIVE OBJECTS / ANIMATIONS (single master loop)
# ============================================================

set_tool("pen")
set_size(4)

# ---- Wi-Fi / internet monitor (background thread, checks every 3 s) ----
net_online = False
shown_online = None

def check_internet():
    for host in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            s = socket.create_connection(host, timeout=1.5)
            s.close()
            return True
        except OSError:
            continue
    return False

def net_monitor():
    global net_online
    while True:
        net_online = check_internet()
        time.sleep(3)

threading.Thread(target=net_monitor, daemon=True).start()

def refresh_net_labels():
    global shown_online
    if net_online == shown_online:
        return
    shown_online = net_online
    if net_online:
        net_badge.config(text="WI-FI ON", fg=GREEN)
        net_status.config(text="LOCAL MODEL  •  WI-FI ON (NOT REQUIRED)")
    else:
        net_badge.config(text="OFFLINE", fg=BLUE)
        net_status.config(text="LOCAL MODEL  •  OFFLINE MODE ON")

def animate():
    global tick
    tick += 1

    if tick % 10 == 0:
        refresh_net_labels()

    # 1. Pulsing AI status dot
    if is_busy:
        live_dot.config(fg=AMBER if (tick // 4) % 2 == 0 else "#FDE68A")
    else:
        live_dot.config(fg="#86EFAC" if (tick // 12) % 2 == 0 else GREEN)

    # 2. Glowing canvas border (blue <-> purple, faster when busy)
    speed = 0.16 if is_busy else 0.05
    canvas_outer.config(
        bg=lerp_color(BLUE, PURPLE, (math.sin(tick * speed) + 1) / 2)
    )

    # 3. Flowing gradient accent line
    if accent_rects:
        for i, rect in enumerate(accent_rects):
            accent_bar.itemconfig(
                rect,
                fill=gradient_color(i / ACCENT_SEGMENTS - tick * 0.004)
            )

    # 4. Equalizer bars
    for i, bar in enumerate(eq_bars):
        if is_busy:
            h = random.randint(6, EQ_H - 2)
            color = AMBER
        else:
            h = 5 + 4 * (1 + math.sin(tick * 0.15 + i * 0.9))
            color = BLUE
        x0 = 2 + i * 9
        eq_canvas.coords(bar, x0, EQ_H - h, x0 + 6, EQ_H)
        eq_canvas.itemconfig(bar, fill=color)

    # 5. Floating math particles
    for p in particles:
        p["x"] += p["dx"]
        p["y"] += p["dy"] + math.sin(tick * 0.05 + p["phase"]) * 0.25
        if p["x"] > 375:
            p["x"] = 5
        if p["x"] < 5:
            p["x"] = 375
        p["y"] = min(60, max(12, p["y"]))
        floating_canvas.coords(p["id"], p["x"], p["y"])
        glow = (math.sin(tick * 0.04 + p["phase"]) + 1) / 2
        floating_canvas.itemconfig(
            p["id"], fill=lerp_color("#18405F", "#3B82C4", glow)
        )

    # 6. Progress shimmer + hint dot blink
    if is_busy:
        x = (tick * 7) % (PROG_W + 120) - 60
        progress.coords(progress_seg, x, 0, x + 60, 6)
        progress.itemconfig(progress_seg, fill=gradient_color(tick * 0.01))
    else:
        progress.coords(progress_seg, -80, 0, -20, 6)

    hint_dot.config(fg="#67E8F9" if (tick // 15) % 2 == 0 else "#22D3EE")

    root.after(50, animate)

animate()

# ============================================================
# START
# ============================================================

root.mainloop()