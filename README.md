# CAEA — Context-Aware Ensemble Architecture for Histopathology Tissue Classification

A Streamlit-based deployment of the **Context-Aware Ensemble Architecture (CAEA)** for histopathology tissue classification.

## Overview

CAEA uses multiple image-context views to improve histopathology tissue classification:

- **C1 — Local Context:** focuses on a local crop/region.
- **C2 — Original Context:** uses the original image.
- **C3 — Expanded Context:** provides a wider surrounding tissue context.
- **ResNet-18:** baseline comparison model.

The application provides predictions, confidence scores, Top-5 predictions, context comparison, and model explainability through Grad-CAM where supported by the application.

## Models

The trained model weights are **not stored in this GitHub repository** because of GitHub's browser file-size restrictions.

They are stored in a **private Hugging Face model repository**:

`Chikku2005/caea`

The Streamlit application downloads the required model weights securely at runtime using a Hugging Face access token stored in Streamlit Secrets.

Expected model files:

```text
MobileNetV3_C1.pth
MobileNetV3_C2.pth
MobileNetV3_C3.pth
ResNet18.pth
```

## Repository Structure

```text
CAEA/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── classes.json
├── config.json
├── context_info.json
└── model_info.json
```

## Features

- Upload histopathology images through a web interface.
- Select the CAEA context model or the ResNet-18 baseline.
- Display predicted tissue class and confidence.
- Display Top-5 predictions.
- Compare predictions across C1, C2, and C3.
- Provide context-consensus information.
- Generate Grad-CAM visual explanations where available.
- Run inference through a Streamlit web application.

## Installation

Clone the repository:

```bash
git clone https://github.com/SreekarBharadwaz/CAEA.git
cd CAEA
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Streamlit Secrets

Because the model repository on Hugging Face is private, the application requires a Hugging Face access token.

In Streamlit Community Cloud, open:

**App → Settings → Secrets**

Add:

```toml
HF_TOKEN = "your_huggingface_access_token"
```

Do **not** place the token directly inside `app.py` or commit it to GitHub.

## Deployment

The application is designed for deployment using **Streamlit Community Cloud**.

Deployment configuration:

```text
Repository: SreekarBharadwaz/CAEA
Branch:     main
Main file:  app.py
```

After deployment, configure `HF_TOKEN` in Streamlit Secrets so the application can access the private model repository.

## Configuration Files

- `classes.json` — class labels used by the classifier.
- `config.json` — application/model configuration.
- `context_info.json` — context-view information.
- `model_info.json` — model metadata.

## Security

The GitHub repository and Hugging Face model repository are intended to remain private.

Sensitive credentials such as Hugging Face access tokens must be stored using Streamlit Secrets and must never be committed to the repository.

## Technology Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Hugging Face Hub
- NumPy
- Pillow
- Matplotlib

## Project

**CAEA — Context-Aware Ensemble Architecture for Histopathology Tissue Classification**

This deployment package is intended to provide an interactive demonstration of the trained CAEA models through a web interface.
