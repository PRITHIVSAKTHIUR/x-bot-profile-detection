import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/x-bot-profile-detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Define class mapping
id2label = {
    "0": "bot",
    "1": "cyborg",
    "2": "real",
    "3": "verified"
}

def detect_profile_type(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Create Gradio UI
iface = gr.Interface(
    fn=detect_profile_type,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4, label="Predicted Profile Type"),
    title="x-bot-profile-detection",
    description="Upload a social media profile picture to classify it as Bot, Cyborg, Real, or Verified using a SigLIP2 model."
)

if __name__ == "__main__":
    iface.launch()
