import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --------------------------------------------------
# 1. Read the InkML file
# --------------------------------------------------

file_path = r"data\raw\mathwriting-2024\train\00001d1472a8709f.inkml"

tree = ET.parse(file_path)
root = tree.getroot()

traces = []

for element in root.iter():
    if element.tag.endswith("trace"):
        points = []

        if element.text:
            for point in element.text.strip().split(","):
                values = point.strip().split()

                if len(values) >= 2:
                    x = float(values[0])
                    y = float(values[1])
                    points.append((x, y))

        if points:
            traces.append(points)

print("Number of strokes:", len(traces))


# --------------------------------------------------
# 2. Create image from handwriting
# --------------------------------------------------

plt.figure(figsize=(8, 4))

for stroke in traces:
    x_values = [point[0] for point in stroke]
    y_values = [point[1] for point in stroke]

    plt.plot(x_values, y_values)

plt.gca().invert_yaxis()
plt.axis("equal")
plt.axis("off")

plt.savefig(
    "handwritten_expression.png",
    bbox_inches="tight",
    pad_inches=0.1
)

plt.close()

print("Handwriting image created!")


# --------------------------------------------------
# 3. Preprocess image
# --------------------------------------------------

image = Image.open("handwritten_expression.png").convert("RGB")
image = image.resize((128, 128))

image.save("processed_expression.png")

print("Image preprocessing completed!")


# --------------------------------------------------
# 4. Load recognition model
# --------------------------------------------------

print("Loading handwriting recognition model...")

processor = TrOCRProcessor.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

print("Model loaded successfully!")


# --------------------------------------------------
# 5. Recognize mathematical expression
# --------------------------------------------------

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(pixel_values)

recognized_expression = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print()
print("Recognized expression:")
print(recognized_expression)


# --------------------------------------------------
# 6. Display recognized mathematics
# --------------------------------------------------

plt.figure(figsize=(10, 3))

plt.text(
    0.5,
    0.5,
    f"${recognized_expression}$",
    fontsize=30,
    ha="center",
    va="center"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    "recognized_math.png",
    bbox_inches="tight",
    pad_inches=0.2
)

print()
print("Final mathematical expression saved as: recognized_math.png")

plt.show()


