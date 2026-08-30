# CAEA — Context-Aware Tissue Classification

Streamlit deployment for the CAEA histopathology image classification project.

## Features
- MobileNetV3-Large: C1 Local, C2 Original, C3 Expanded context
- ResNet-18 baseline
- Top-5 tissue predictions and confidence
- Cross-context consensus
- Grad-CAM visual explanation
- CPU inference for Streamlit Cloud

## Streamlit deployment
1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Select `app.py` as the main file.
4. Deploy.

The model weights are stored in `models/` and are loaded only when an image is uploaded.
