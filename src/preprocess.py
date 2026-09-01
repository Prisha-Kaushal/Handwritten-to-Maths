from PIL import Image
import os

input_path = "handwritten_expression.png"
output_path = "processed_expression.png"

# Open image
image = Image.open(input_path).convert("L")

# Resize image
image = image.resize((128, 128))

# Save processed image
image.save(output_path)

print("Image preprocessing completed!")
print("Saved as:", output_path)