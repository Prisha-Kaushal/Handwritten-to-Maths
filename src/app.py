import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageGrab, Image, ImageChops, ImageTk
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from display_math import render_math
import os


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

processor = TrOCRProcessor.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

print("Model loaded successfully!")


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Handwritten to Mathematical Expression")

root.geometry("950x750")

root.resizable(False, False)

root.configure(bg="#f2f2f2")


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Handwritten to Mathematical Expression",
    font=("Arial", 24, "bold"),
    bg="#f2f2f2"
)

title_label.pack(pady=(20, 5))


subtitle_label = tk.Label(
    root,
    text="Write a mathematical expression below",
    font=("Arial", 12),
    bg="#f2f2f2"
)

subtitle_label.pack(pady=(0, 15))


# ============================================================
# DRAWING CANVAS FRAME
# ============================================================

canvas_frame = tk.Frame(
    root,
    bg="white",
    bd=2,
    relief=tk.SOLID
)

canvas_frame.pack()


# ============================================================
# DRAWING CANVAS
# ============================================================

canvas = tk.Canvas(
    canvas_frame,
    width=850,
    height=350,
    bg="white",
    cursor="pencil"
)

canvas.pack()


# ============================================================
# DRAWING VARIABLES
# ============================================================

last_x = None
last_y = None


# ============================================================
# START DRAWING
# ============================================================

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y


# ============================================================
# DRAW
# ============================================================

def draw(event):

    global last_x, last_y

    if last_x is not None and last_y is not None:

        canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            width=4,
            fill="black",
            capstyle=tk.ROUND,
            smooth=True
        )

    last_x = event.x
    last_y = event.y


# ============================================================
# STOP DRAWING
# ============================================================

def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


# ============================================================
# MOUSE EVENTS
# ============================================================

canvas.bind(
    "<Button-1>",
    start_draw
)

canvas.bind(
    "<B1-Motion>",
    draw
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_draw
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("L")

    white_background = Image.new(
        "L",
        image.size,
        255
    )

    difference = ImageChops.difference(
        image,
        white_background
    )

    bbox = difference.getbbox()

    if bbox is None:

        return None

    image = image.crop(bbox)

    padding = 50

    padded_image = Image.new(
        "L",
        (
            image.width + 2 * padding,
            image.height + 2 * padding
        ),
        255
    )

    padded_image.paste(
        image,
        (
            padding,
            padding
        )
    )

    image = padded_image.point(
        lambda pixel: 0 if pixel < 200 else 255
    )

    image = image.convert("RGB")

    return image


# ============================================================
# RECOGNIZE IMAGE
# ============================================================

def recognize_image(image):

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    generated_ids = model.generate(
        pixel_values
    )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return result.strip()


# ============================================================
# DISPLAY RENDERED IMAGE
# ============================================================

def display_rendered_math():

    image_path = "recognized_math.png"

    if not os.path.exists(image_path):

        return

    try:

        image = Image.open(
            image_path
        )

        max_width = 750
        max_height = 150

        image.thumbnail(
            (
                max_width,
                max_height
            ),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        math_image_label.config(
            image=photo,
            text=""
        )

        math_image_label.image = photo

    except Exception as e:

        print(
            "Error displaying rendered equation:",
            e
        )


# ============================================================
# CLEAR CANVAS
# ============================================================

def clear_canvas():

    canvas.delete("all")

    result_label.config(
        text="Result will appear here"
    )

    status_label.config(
        text="Ready"
    )

    math_image_label.config(
        image="",
        text="Rendered equation will appear here"
    )

    math_image_label.image = None

    print("Canvas cleared.")


# ============================================================
# RECOGNIZE HANDWRITING
# ============================================================

def recognize():

    try:

        status_label.config(
            text="Recognizing..."
        )

        root.update()

        print("\nStarting recognition...")

        # ----------------------------------------------------
        # Get canvas position
        # ----------------------------------------------------

        x = (
            root.winfo_rootx()
            + canvas.winfo_rootx()
            - root.winfo_rootx()
        )

        y = (
            root.winfo_rooty()
            + canvas.winfo_rooty()
            - root.winfo_rooty()
        )

        width = canvas.winfo_width()

        height = canvas.winfo_height()

        # ----------------------------------------------------
        # Capture canvas
        # ----------------------------------------------------

        image = ImageGrab.grab(
            bbox=(
                x,
                y,
                x + width,
                y + height
            )
        )

        image.save(
            "user_expression.png"
        )

        print("Image captured!")

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_image = preprocess_image(
            image
        )

        if processed_image is None:

            status_label.config(
                text="No handwriting detected"
            )

            messagebox.showwarning(
                "Warning",
                "Please draw a mathematical expression first."
            )

            return

        processed_image.save(
            "user_expression_processed.png"
        )

        print(
            "Image preprocessing completed!"
        )

        # ----------------------------------------------------
        # Recognize
        # ----------------------------------------------------

        result = recognize_image(
            processed_image
        )

        print(
            "Recognized expression:"
        )

        print(result)

        # ----------------------------------------------------
        # Display raw result
        # ----------------------------------------------------

        if not result:

            result = "Could not recognize expression"

        result_label.config(
            text=result
        )

        # ----------------------------------------------------
        # Render equation
        # ----------------------------------------------------

        try:

            render_math(
                result
            )

            display_rendered_math()

            status_label.config(
                text="Recognition completed"
            )

        except Exception as render_error:

            print(
                "Rendering error:",
                render_error
            )

            status_label.config(
                text="Recognized, but rendering failed"
            )

    except Exception as e:

        print(
            "Error:",
            e
        )

        status_label.config(
            text="Error occurred"
        )

        messagebox.showerror(
            "Error",
            str(e)
        )


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
            (
                "PNG Image",
                "*.png"
            )
        ]
    )

    if save_path:

        image = Image.open(
            image_path
        )

        image.save(
            save_path
        )

        messagebox.showinfo(
            "Saved",
            "Mathematical expression saved successfully."
        )


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#f2f2f2"
)

button_frame.pack(
    pady=15
)


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=15,
    height=2,
    font=("Arial", 11, "bold")
)

clear_button.grid(
    row=0,
    column=0,
    padx=8
)


# ============================================================
# RECOGNIZE BUTTON
# ============================================================

recognize_button = tk.Button(
    button_frame,
    text="Recognize",
    command=recognize,
    width=15,
    height=2,
    font=("Arial", 11, "bold")
)

recognize_button.grid(
    row=0,
    column=1,
    padx=8
)


# ============================================================
# SAVE BUTTON
# ============================================================

save_button = tk.Button(
    button_frame,
    text="Save Result",
    command=save_result,
    width=15,
    height=2,
    font=("Arial", 11, "bold")
)

save_button.grid(
    row=0,
    column=2,
    padx=8
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Ready",
    font=("Arial", 11),
    bg="#f2f2f2"
)

status_label.pack(
    pady=(0, 8)
)


# ============================================================
# RESULT TITLE
# ============================================================

result_title = tk.Label(
    root,
    text="Recognized Mathematical Expression",
    font=("Arial", 14, "bold"),
    bg="#f2f2f2"
)

result_title.pack()


# ============================================================
# RAW LATEX RESULT
# ============================================================

result_label = tk.Label(
    root,
    text="Result will appear here",
    font=("Arial", 18),
    bg="#f2f2f2"
)

result_label.pack(
    pady=5
)


# ============================================================
# RENDERED EQUATION TITLE
# ============================================================

rendered_title = tk.Label(
    root,
    text="Rendered Equation",
    font=("Arial", 13, "bold"),
    bg="#f2f2f2"
)

rendered_title.pack(
    pady=(5, 0)
)


# ============================================================
# RENDERED EQUATION
# ============================================================

math_image_label = tk.Label(
    root,
    text="Rendered equation will appear here",
    font=("Arial", 12),
    bg="white",
    width=70,
    height=4,
    relief=tk.SOLID,
    bd=1
)

math_image_label.pack(
    pady=10
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()