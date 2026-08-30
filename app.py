import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_large, resnet18
# --- Private Hugging Face model storage ---
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Chikku2005/caea"

@st.cache_resource(show_spinner=False)
def get_model_path(filename):
    """Download/cache a private model from Hugging Face."""
    token = st.secrets.get("HF_TOKEN", None)
    if not token:
        raise RuntimeError("HF_TOKEN is missing from Streamlit Secrets.")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="model",
        token=token,
    )

ROOT = Path(__file__).parent
DEVICE = torch.device("cpu")

st.set_page_config(page_title="CAEA Tissue Classifier", page_icon="🔬", layout="wide")

@st.cache_data

def load_metadata():
    with open(ROOT / "classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(ROOT / "model_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return classes, config, info

CLASSES, CONFIG, INFO = load_metadata()

@st.cache_resource(show_spinner="Loading CAEA models…")
def load_models():
    models = {}
    for context in ["C1", "C2", "C3"]:
        model = mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
        state = torch.load(ROOT / CONFIG["models"][context], map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        models[context] = model

    baseline = resnet18(weights=None)
    baseline.fc = nn.Linear(baseline.fc.in_features, len(CLASSES))
    state = torch.load(ROOT / CONFIG["models"]["ResNet18"], map_location=DEVICE, weights_only=True)
    baseline.load_state_dict(state)
    baseline.to(DEVICE).eval()
    models["ResNet18"] = baseline
    return models


def spatial_context(image: Image.Image, context: str) -> Image.Image:
    image = image.convert("RGB")
    if context == "C1":
        w, h = image.size
        image = image.crop((int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h)))
    elif context == "C3":
        w, h = image.size
        canvas = Image.new("RGB", (int(w * 1.2), int(h * 1.2)), (0, 0, 0))
        canvas.paste(image, (int(0.1 * w), int(0.1 * h)))
        image = canvas
    return image.resize((224, 224), Image.Resampling.BILINEAR)


def preprocess(image: Image.Image, context: str) -> torch.Tensor:
    image = spatial_context(image, context)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CONFIG["normalize_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CONFIG["normalize_std"], dtype=torch.float32).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).to(DEVICE)


def predict(model, x):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=min(5, len(CLASSES)))
    return [(CLASSES[i], float(p)) for p, i in zip(top.values.cpu(), top.indices.cpu())]


def gradcam(model, x, class_idx):
    activations = []
    gradients = []
    layer = model.features[-1] if hasattr(model, "features") else model.layer4[-1]

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[:, class_idx].sum().backward()
        a = activations[-1]
        g = gradients[-1]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    import matplotlib.cm as cm
    heat = cm.get_cmap("jet")(cam)[..., :3].astype(np.float32)
    overlay = np.clip(0.5 * base + 0.5 * heat, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))

st.title("🔬 CAEA — Context-Aware Tissue Classification")
st.caption("MobileNetV3-Large with Local (C1), Original (C2), Expanded (C3) context + ResNet-18 baseline")

with st.sidebar:
    st.header("Model")
    selected = st.selectbox("Prediction context", ["C1", "C2", "C3", "ResNet18"])
    explain = st.checkbox("Generate Grad-CAM", value=True)
    st.divider()
    st.write(f"**Classes:** {len(CLASSES)}")
    st.write(f"**Device:** CPU")
    st.write("**Training:** PanNuke")
    st.write("**Validation:** Lizard")
    st.write("**Testing:** CoNSeP")

uploaded = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to run CAEA inference.")
    st.markdown("### Contexts")
    st.markdown("- **C1:** Local context\n- **C2:** Original context\n- **C3:** Expanded context\n- **ResNet18:** baseline using original context")
else:
    image = Image.open(uploaded).convert("RGB")
    models = load_models()

    if selected == "ResNet18":
        x = preprocess(image, "C2")
    else:
        x = preprocess(image, selected)

    top = predict(models[selected], x)
    pred_name, pred_conf = top[0]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Prediction")
        st.metric("Predicted tissue", pred_name)
        st.metric("Confidence", f"{pred_conf * 100:.2f}%")
        st.write("### Top 5")
        for name, prob in top:
            st.write(f"**{name}** — {prob * 100:.2f}%")
            st.progress(prob)

    if selected in ["C1", "C2", "C3"]:
        st.subheader("Cross-context comparison")
        cols = st.columns(3)
        context_results = {}
        for col, context in zip(cols, ["C1", "C2", "C3"]):
            result = predict(models[context], preprocess(image, context))[0]
            context_results[context] = result
            col.metric(context, result[0], f"{result[1] * 100:.1f}%")

        votes = [r[0] for r in context_results.values()]
        consensus = max(set(votes), key=votes.count)
        st.info(f"CAEA context consensus: **{consensus}** ({votes.count(consensus)}/3 contexts)")

    if explain:
        st.subheader("Grad-CAM explanation")
        with st.spinner("Generating explanation…"):
            class_idx = CLASSES.index(pred_name)
            cam = gradcam(models[selected], x, class_idx)
            if selected == "ResNet18":
                display_image = spatial_context(image, "C2")
            else:
                display_image = spatial_context(image, selected)
            overlay = overlay_cam(display_image, cam)
        st.image(overlay, caption=f"Grad-CAM — {pred_name}", use_container_width=True)
        st.caption("Highlighted regions indicate image features that most influenced the selected prediction.")
