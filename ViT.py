import matplotlib.pyplot as plt
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_path = "images/search/Garmin Techsperts con 2024-264 - Copy.jpg"
image_path = "images/misc/building.png"

image = Image.open(image_path).convert("RGB")

# Display the image
plt.imshow(image)
plt.axis('off')  # Hide the axis
plt.show()

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
