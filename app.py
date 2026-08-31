import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.models import mobilenet_v3_large, resnet18
from huggingface_hub import hf_hub_download


# ============================================================
# CAEA — Context-Aware Tissue Classification
# ============================================================

st.set_page_config(
    page_title="CAEA Tissue Classifier",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DEVICE = torch.device("cpu")


# ============================================================
# HUGGING FACE MODEL REPOSITORY
# ============================================================

HF_REPO_ID = "Chikku2005/caea"

# EXACT filenames in your Hugging Face model repository
HF_MODEL_FILES = {
    "C1": "MobileNetV3_C1.pth",
    "C2": "MobileNetV3_C2.pth",
    "C3": "MobileNetV3_C3.pth",
    "ResNet18": "ResNet18.pth",
}


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model_path(filename):
    """
    Download/cache a model from Hugging Face.

    If the Hugging Face repository is private,
    HF_TOKEN must be configured in Streamlit Secrets.
    """

    token = st.secrets.get("HF_TOKEN", None)

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model",
            token=token,
        )

        return model_path

    except Exception as e:
        st.error(
            f"""
            ❌ Could not download model from Hugging Face.

            Repository:
            {HF_REPO_ID}

            File:
            {filename}

            Error:
            {e}
            """
        )

        st.stop()


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    classes_path = ROOT / "classes.json"
    config_path = ROOT / "config.json"
    info_path = ROOT / "model_info.json"

    missing_files = []

    if not classes_path.exists():
        missing_files.append("classes.json")

    if not config_path.exists():
        missing_files.append("config.json")

    if not info_path.exists():
        missing_files.append("model_info.json")

    if missing_files:
        st.error(
            "❌ Required metadata files are missing from the GitHub/Streamlit repository:"
        )

        for file in missing_files:
            st.write(f"- `{file}`")

        st.stop()

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return classes, config, info


# ============================================================
# LOAD METADATA
# ============================================================

CLASSES, CONFIG, INFO = load_metadata()


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading CAEA models…")
def load_models():

    models = {}

    # --------------------------------------------------------
    # MobileNetV3 — C1, C2, C3
    # --------------------------------------------------------

    for context in ["C1", "C2", "C3"]:

        # Create MobileNetV3 architecture
        model = mobilenet_v3_large(weights=None)

        # Replace classifier with number of CAEA classes
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            len(CLASSES)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Download model from Hugging Face
        # ----------------------------------------------------

        model_filename = HF_MODEL_FILES[context]

        model_path = get_model_path(model_filename)

        # Load checkpoint
        state = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )

        # Load weights
        model.load_state_dict(state)

        # CPU inference
        model.to(DEVICE)

        # Evaluation mode
        model.eval()

        models[context] = model


    # --------------------------------------------------------
    # ResNet18 baseline
    # --------------------------------------------------------

    baseline = resnet18(weights=None)

    baseline.fc = nn.Linear(
        baseline.fc.in_features,
        len(CLASSES)
    )

    # Download ResNet18 from Hugging Face
    model_filename = HF_MODEL_FILES["ResNet18"]

    model_path = get_model_path(model_filename)

    # Load checkpoint
    state = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Load weights
    baseline.load_state_dict(state)

    baseline.to(DEVICE)

    baseline.eval()

    models["ResNet18"] = baseline

    return models


# ============================================================
# SPATIAL CONTEXT
# ============================================================

def spatial_context(image: Image.Image, context: str) -> Image.Image:

    image = image.convert("RGB")

    # --------------------------------------------------------
    # C1 — Local context
    # --------------------------------------------------------

    if context == "C1":

        w, h = image.size

        image = image.crop(
            (
                int(0.1 * w),
                int(0.1 * h),
                int(0.9 * w),
                int(0.9 * h)
            )
        )

    # --------------------------------------------------------
    # C3 — Expanded context
    # --------------------------------------------------------

    elif context == "C3":

        w, h = image.size

        canvas = Image.new(
            "RGB",
            (
                int(w * 1.2),
                int(h * 1.2)
            ),
            (0, 0, 0)
        )

        canvas.paste(
            image,
            (
                int(0.1 * w),
                int(0.1 * h)
            )
        )

        image = canvas

    # --------------------------------------------------------
    # C2 — Original context
    # --------------------------------------------------------

    return image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image: Image.Image, context: str) -> torch.Tensor:

    image = spatial_context(
        image,
        context
    )

    arr = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    x = torch.from_numpy(
        arr
    ).permute(2, 0, 1)

    mean = torch.tensor(
        CONFIG["normalize_mean"],
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        CONFIG["normalize_std"],
        dtype=torch.float32
    ).view(3, 1, 1)

    x = (x - mean) / std

    return x.unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

def predict(model, x):

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

    top = torch.topk(
        probs,
        k=min(5, len(CLASSES))
    )

    return [
        (
            CLASSES[i],
            float(p)
        )
        for p, i in zip(
            top.values.cpu(),
            top.indices.cpu()
        )
    ]


# ============================================================
# GRAD-CAM
# ============================================================

def gradcam(model, x, class_idx):

    activations = []
    gradients = []

    # --------------------------------------------------------
    # Select target layer
    # --------------------------------------------------------

    if hasattr(model, "features"):

        layer = model.features[-1]

    else:

        layer = model.layer4[-1]


    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(_, __, output):

        activations.append(output)


    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(_, __, grad_output):

        gradients.append(
            grad_output[0]
        )


    h1 = layer.register_forward_hook(
        forward_hook
    )

    h2 = layer.register_full_backward_hook(
        backward_hook
    )


    try:

        model.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        logits[:, class_idx].sum().backward()

        a = activations[-1]

        g = gradients[-1]

        weights = g.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = F.relu(
            (weights * a).sum(
                dim=1,
                keepdim=True
            )
        )

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ].detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam

    finally:

        h1.remove()

        h2.remove()


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def overlay_cam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.asarray(
        image.convert("RGB").resize((224, 224)),
        dtype=np.float32
    ) / 255.0

    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3].astype(
        np.float32
    )

    overlay = np.clip(
        0.5 * base + 0.5 * heat,
        0,
        1
    )

    return Image.fromarray(
        (overlay * 255).astype(np.uint8)
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🔬 CAEA — Context-Aware Tissue Classification"
)

st.caption(
    "MobileNetV3-Large with Local (C1), Original (C2), "
    "Expanded (C3) context + ResNet-18 baseline"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model")

    selected = st.selectbox(
        "Prediction context",
        [
            "C1",
            "C2",
            "C3",
            "ResNet18"
        ]
    )

    explain = st.checkbox(
        "Generate Grad-CAM",
        value=True
    )

    st.divider()

    st.write(
        f"**Classes:** {len(CLASSES)}"
    )

    st.write(
        "**Device:** CPU"
    )

    st.write(
        "**Training:** PanNuke"
    )

    st.write(
        "**Validation:** Lizard"
    )

    st.write(
        "**Testing:** CoNSeP"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a histopathology image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded is None:

    st.info(
        "Upload an image to run CAEA inference."
    )

    st.markdown(
        "### Contexts"
    )

    st.markdown(
        "- **C1:** Local context\n"
        "- **C2:** Original context\n"
        "- **C3:** Expanded context\n"
        "- **ResNet18:** baseline using original context"
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        uploaded
    ).convert("RGB")


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = load_models()


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    if selected == "ResNet18":

        x = preprocess(
            image,
            "C2"
        )

    else:

        x = preprocess(
            image,
            selected
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    top = predict(
        models[selected],
        x
    )

    pred_name, pred_conf = top[0]


    # ========================================================
    # INPUT + PREDICTION
    # ========================================================

    left, right = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Input"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Prediction"
        )

        st.metric(
            "Predicted tissue",
            pred_name
        )

        st.metric(
            "Confidence",
            f"{pred_conf * 100:.2f}%"
        )

        st.write(
            "### Top 5"
        )

        for name, prob in top:

            st.write(
                f"**{name}** — {prob * 100:.2f}%"
            )

            st.progress(
                prob
            )


    # ========================================================
    # CROSS-CONTEXT COMPARISON
    # ========================================================

    if selected in [
        "C1",
        "C2",
        "C3"
    ]:

        st.subheader(
            "Cross-context comparison"
        )

        cols = st.columns(3)

        context_results = {}


        for col, context in zip(
            cols,
            ["C1", "C2", "C3"]
        ):

            result = predict(
                models[context],
                preprocess(
                    image,
                    context
                )
            )[0]

            context_results[context] = result

            col.metric(
                context,
                result[0],
                f"{result[1] * 100:.1f}%"
            )


        # ----------------------------------------------------
        # Consensus
        # ----------------------------------------------------

        votes = [
            r[0]
            for r in context_results.values()
        ]

        consensus = max(
            set(votes),
            key=votes.count
        )

        st.info(
            f"CAEA context consensus: "
            f"**{consensus}** "
            f"({votes.count(consensus)}/3 contexts)"
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    if explain:

        st.subheader(
            "Grad-CAM explanation"
        )

        with st.spinner(
            "Generating explanation…"
        ):

            class_idx = CLASSES.index(
                pred_name
            )

            cam = gradcam(
                models[selected],
                x,
                class_idx
            )

            if selected == "ResNet18":

                display_image = spatial_context(
                    image,
                    "C2"
                )

            else:

                display_image = spatial_context(
                    image,
                    selected
                )

            overlay = overlay_cam(
                display_image,
                cam
            )


        st.image(
            overlay,
            caption=f"Grad-CAM — {pred_name}",
            use_container_width=True
        )

        st.caption(
            "Highlighted regions indicate image features "
            "that most influenced the selected prediction."
        )
