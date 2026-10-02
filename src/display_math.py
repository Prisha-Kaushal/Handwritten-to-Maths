import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

OUTPUT_FILE = "recognized_math.png"


# ============================================================
# CHECK BASIC LATEX BRACE BALANCE
# (escaped braces such as \{ and \} are ignored)
# ============================================================

def latex_is_balanced(text):
    balance = 0
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            i += 2          # skip the escaped character
            continue

        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1
            if balance < 0:
                return False

        i += 1

    return balance == 0


# ============================================================
# RENDER MATHEMATICAL EXPRESSION
# Returns True only if a NEW image was produced.
# ============================================================

def render_math(latex_expression):

    # Remove the previous result first, so a failed render can never
    # leave an old equation on screen next to a new LaTeX string.
    try:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
    except OSError:
        pass

    try:
        latex_expression = latex_expression.strip().replace("$", "")

        if not latex_expression:
            print("Empty expression. Rendering skipped.")
            return False

        if not latex_is_balanced(latex_expression):
            print("Invalid LaTeX expression. Rendering skipped.")
            return False

        # Figure + Agg canvas instead of pyplot: no extra GUI window,
        # no clash with Tkinter, and no global state to clean up.
        fig = Figure(figsize=(10, 3))
        FigureCanvasAgg(fig)

        fig.text(
            0.5,
            0.5,
            f"${latex_expression}$",
            fontsize=30,
            ha="center",
            va="center"
        )

        fig.savefig(
            OUTPUT_FILE,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.2
        )

        print("Mathematical expression rendered successfully!")
        print("Saved as:", OUTPUT_FILE)
        return True

    except Exception as e:
        # Matplotlib's math engine does not support every LaTeX command.
        print("Error rendering mathematical expression:")
        print(e)
        return False