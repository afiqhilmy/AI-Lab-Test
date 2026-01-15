import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Lab Test Real-time Image Classification (PyTorch)",
    page_icon="",
    layout="centered",
)

st.title("Real-time Webcam Image Classification")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit webcam (`st.camera_input`).")


# -----------------------------
# Load labels (ImageNet)
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    r = requests.get(url)
    labels = r.text.strip().split("\n")
    return labels


# -----------------------------
# Load model (once, cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model


labels = load_imagenet_labels()
model = load_model()

# Preprocess pipeline for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -----------------------------
# Webcam input
# -----------------------------
st.subheader("Capture from Webcam")

st.info("Click **Take photo** to capture new frame. Model will class for each time there is a new photo.")

img_data = st.camera_input("Take a photo")

if img_data is not None:
    # Read image from webcam
    image = Image.open(img_data).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Captured Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Inference
    with torch.no_grad():
        outputs = model(input_batch)
        probs = F.softmax(outputs[0], dim=0)

    # Top-5 predictions
    top5_prob, top5_catid = torch.topk(probs, 5)

    with col2:
        st.markdown("### 🔍 Top-5 Predictions")
        for i in range(top5_prob.size(0)):
            label = labels[top5_catid[i]]
            prob = top5_prob[i].item()
            st.write(f"**{label}** — {prob:.4f}")

    # Optional: show as table
    import pandas as pd

    st.markdown("### 📊 Prediction Table")
    df = pd.DataFrame({
        "Label": [labels[idx] for idx in top5_catid],
        "Probability": [float(p) for p in top5_prob],
    })
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Webcam doesn't capture anything. Click **Take photo**.")