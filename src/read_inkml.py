import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

file_path = r"data\raw\mathwriting-2024\train\00001d1472a8709f.inkml"

tree = ET.parse(file_path)
root = tree.getroot()

# Find all trace elements
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

# Draw the handwriting
plt.figure(figsize=(8, 4))

for stroke in traces:
    x_values = [point[0] for point in stroke]
    y_values = [point[1] for point in stroke]

    plt.plot(x_values, y_values)

plt.gca().invert_yaxis()
plt.axis("equal")
plt.axis("off")

# Save handwriting as an image
output_path = "handwritten_expression.png"
plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)

print("Image saved as:", output_path)

plt.show()