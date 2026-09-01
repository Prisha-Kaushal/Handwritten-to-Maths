import matplotlib.pyplot as plt


# ============================================================
# CHECK BASIC LATEX BRACE BALANCE
# ============================================================

def latex_is_balanced(text):

    balance = 0

    for character in text:

        if character == "{":

            balance += 1

        elif character == "}":

            balance -= 1

            if balance < 0:
                return False

    return balance == 0


# ============================================================
# RENDER MATHEMATICAL EXPRESSION
# ============================================================

def render_math(latex_expression):

    try:

        latex_expression = latex_expression.strip()

        # ----------------------------------------------------
        # Prevent Matplotlib from crashing on malformed LaTeX
        # ----------------------------------------------------

        if not latex_is_balanced(
            latex_expression
        ):

            print(
                "Invalid LaTeX expression."
            )

            print(
                "Rendering skipped."
            )

            return False

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 3)
        )

        # ----------------------------------------------------
        # Display mathematical expression
        # ----------------------------------------------------

        plt.text(
            0.5,
            0.5,
            f"${latex_expression}$",
            fontsize=30,
            ha="center",
            va="center"
        )

        # ----------------------------------------------------
        # Remove axes
        # ----------------------------------------------------

        plt.axis(
            "off"
        )

        # ----------------------------------------------------
        # Adjust layout
        # ----------------------------------------------------

        plt.tight_layout()

        # ----------------------------------------------------
        # Save rendered equation
        # ----------------------------------------------------

        plt.savefig(
            "recognized_math.png",
            bbox_inches="tight",
            pad_inches=0.2
        )

        # ----------------------------------------------------
        # Close figure
        # ----------------------------------------------------

        plt.close()

        print(
            "Mathematical expression rendered successfully!"
        )

        print(
            "Saved as: recognized_math.png"
        )

        return True

    except Exception as e:

        print(
            "Error rendering mathematical expression:"
        )

        print(
            e
        )

        # Make sure the figure is closed even if
        # Matplotlib encounters an error.
        plt.close()

        return False