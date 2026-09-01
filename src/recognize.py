from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

image_path = "processed_expression.png"

print("Loading model...")

processor = TrOCRProcessor.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "fhswf/TrOCR_Math_handwritten"
)

print("Model loaded successfully!")

image = Image.open(image_path).convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(pixel_values)

result = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("Recognized expression:")
print(result)