from PIL import Image, ImageOps

input_path = "handwritten_expression.png"
output_path = "processed_expression.png"

# Open image
image = Image.open(input_path).convert("L")

# Remove unnecessary empty space around the expression
bbox = ImageOps.invert(image).getbbox()

if bbox:
    image = image.crop(bbox)

# Keep the original aspect ratio
# Add white padding instead of stretching the image
padding = 30
image = ImageOps.expand(image, border=padding, fill=255)

# Make a square canvas without distorting the expression
width, height = image.size
size = max(width, height)

canvas = Image.new("L", (size, size), 255)

x = (size - width) // 2
y = (size - height) // 2

canvas.paste(image, (x, y))

# Save
canvas.save(output_path)

print("Image preprocessing completed!")
print("Saved as:", output_path)