![ADS.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/qpUh38BrGJA1PNebSFovB.png)

# **x-bot-profile-detection**

> **x-bot-profile-detection** is a SigLIP2-based classification model designed to detect **profile authenticity types on social media platforms** (such as X/Twitter). It categorizes a profile image into four classes: **bot**, **cyborg**, **real**, or **verified**. Built on `google/siglip2-base-patch16-224`, the model leverages advanced vision-language pretraining for robust image classification.

```py
Classification Report:
              precision    recall  f1-score   support

         bot     0.9912    0.9960    0.9936      2500
      cyborg     0.9940    0.9880    0.9910      2500
        real     0.8634    0.9936    0.9239      2500
    verified     0.9948    0.8460    0.9144      2500

    accuracy                         0.9559     10000
   macro avg     0.9609    0.9559    0.9557     10000
weighted avg     0.9609    0.9559    0.9557     10000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/P0MispJOjkolSsYPbYuhC.png)

---

## **Label Classes**

The model predicts one of the following profile types:

```
0: bot        → Automated accounts  
1: cyborg     → Partially automated or suspiciously mixed behavior  
2: real       → Genuine human users  
3: verified   → Verified accounts or official profiles
```

---

## **Installation**

```bash
pip install transformers torch pillow gradio
```

---

## **Example Inference Code**

```python
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
```

---

## **Use Cases**

* Social media moderation and automation detection
* Anomaly detection in public discourse
* Botnet analysis and influence operation research
* Platform integrity and trust verification
