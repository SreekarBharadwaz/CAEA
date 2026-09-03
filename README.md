# CAEA — Context-Aware Ensemble Architecture for Histopathology Tissue Classification

A deep-learning-based histopathology tissue classification system that uses **multiple spatial context views** to improve tissue recognition and provide interpretable predictions through **Grad-CAM**.

CAEA (**Context-Aware Ensemble Architecture**) processes histopathology tissue regions at different spatial scales:

* **C1 — Local Context:** focuses on a smaller region around the target.
* **C2 — Original Context:** preserves the original field of view.
* **C3 — Expanded Context:** incorporates a wider surrounding tissue region.
* **ResNet-18:** used as a conventional baseline for comparison.

The project includes model training, context-aware preprocessing, evaluation, explainability and a **Streamlit deployment interface**.

---

## 1. Project Overview

Histopathology images contain discriminative information at multiple spatial scales. Fine-grained cellular morphology may be important for classification, while surrounding tissue structures can provide additional contextual information.

A conventional image classifier generally receives a single field of view. CAEA instead constructs multiple representations of the same image using different spatial context ratios and evaluates them using separate MobileNetV3-Large models.

The three context representations are:

| Context | Description                  | Context Ratio |
| ------- | ---------------------------- | ------------: |
| **C1**  | Local / fine-grained context |           0.5 |
| **C2**  | Original field of view       |           1.0 |
| **C3**  | Expanded surrounding context |           1.5 |

Each representation is converted to a common **224 × 224** input resolution before being passed to the CNN backbone.

---

## 2. Motivation

Histopathology tissue classification can be difficult because different tissue types may contain visually similar cellular structures.

A small local crop may contain useful cellular information but miss the surrounding tissue organization. Conversely, a large field of view can provide contextual information but may reduce the relative importance of fine-grained structures.

CAEA addresses this problem by explicitly introducing **multi-scale spatial context**.

The underlying idea is:

> **A tissue region should not always be interpreted independently of its surrounding tissue context.**

---

## 3. Proposed CAEA Architecture

### Context-Aware Processing

For an input image \(I\), CAEA generates three spatial representations:

$$
C_1 = \mathcal{R}(Crop(I, \rho_1))
$$

$$
C_2 = \mathcal{R}(Crop(I, \rho_2))
$$

$$
C_3 = \mathcal{R}(Crop(I, \rho_3))
$$

where:

$$
\rho_1 = 0.5,\qquad
\rho_2 = 1.0,\qquad
\rho_3 = 1.5
$$

and \(\mathcal{R}\) represents resizing to the common **224 × 224** resolution.

### Context Pipeline

```text
                    Input Histopathology Image
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
             C1 Local     C2 Original   C3 Expanded
             ρ = 0.5       ρ = 1.0      ρ = 1.5
                 │            │            │
                 ▼            ▼            ▼
             224 × 224     224 × 224    224 × 224
                 │            │            │
                 ▼            ▼            ▼
          MobileNetV3-L   MobileNetV3-L  MobileNetV3-L
                 │            │            │
                 ▼            ▼            ▼
              Class        Class        Class
            Prediction    Prediction    Prediction
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                     Context Comparison
                     / Consensus Analysis
```

The deployment interface exposes the individual context predictions and their confidence values so that the behavior of the model can be examined across spatial scales.

---

## 4. Key Contribution

The main contribution of this implementation is the explicit use of **spatial context variation** for histopathology tissue classification.

Instead of relying exclusively on one fixed crop, CAEA evaluates:

1. **Local morphology**
2. **Original tissue field of view**
3. **Expanded surrounding tissue context**

This allows the system to study how classification behavior changes as the available spatial context changes.

The implementation also combines classification with explainability through **Grad-CAM**, allowing the user to inspect the regions contributing to the model prediction.

---

## 5. Dataset

The training pipeline uses a histopathology tissue dataset containing RGB images of size:

```text
256 × 256 × 3
```

The notebook loads the image data using NumPy memory mapping and processes the images through a PyTorch dataset pipeline.

The experimental configuration shown in the training notebook uses:

```text
Total samples used:       2,000
Training samples:         1,600
Validation samples:         400
Train/Validation split:    80/20
Random state:                42
Stratification:             Yes
```

The notebook reports the underlying image array as containing 2,656 images, while the experiment limits the training experiment to the first 2,000 samples.

---

## 6. Tissue Classes

The classification experiment uses 17 tissue classes:

```text
1.  Adrenal_gland
2.  Bile-duct
3.  Bladder
4.  Breast
5.  Cervix
6.  Colon
7.  Esophagus
8.  Kidney
9.  Lung
10. Ovarian
11. Pancreatic
12. Prostate
13. Skin
14. Stomach
15. Testis
16. Thyroid
17. Uterus
```

The class-to-label mapping is generated from the dataset labels and used consistently during training and inference.

---

## 7. Preprocessing

The preprocessing pipeline consists of:

### 7.1 Pixel normalization

Input pixel values are converted to the range:

```text
[0, 1]
```

when required.

### 7.2 Spatial context extraction

Each image is processed around its geometric center.

The native image side length is:

```text
256 pixels
```

and the context ratios are:

```text
C1 = 0.5
C2 = 1.0
C3 = 1.5
```

For the expanded context, padding is applied when the crop extends beyond the original image boundaries.

### 7.3 Resizing

All context representations are resized to:

```text
224 × 224 × 3
```

using bilinear interpolation.

### 7.4 ImageNet normalization

The MobileNetV3 models use ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 8. Model Architecture

### MobileNetV3-Large

Each context branch uses a pretrained **MobileNetV3-Large** backbone.

The original classification layer is replaced with a task-specific classifier containing:

```text
Input features : 960
Hidden layer   : 1280
Activation     : Hardswish
Dropout        : 0.2
Output classes : 17
```

The three models are trained independently:

```text
MobileNetV3-Large → C1
MobileNetV3-Large → C2
MobileNetV3-Large → C3
```

---

## 9. Baseline Model

For comparison, the project also evaluates:

### ResNet-18

ResNet-18 is used with the **original C2 context** as a conventional baseline.

This provides a comparison between:

```text
Context-aware MobileNetV3
            vs
Conventional ResNet-18
```

The ResNet-18 model is included as a baseline rather than as one of the three context branches.

---

## 10. Training Configuration

The training pipeline uses:

| Parameter               | Configuration     |
| ----------------------- | ----------------- |
| Backbone                | MobileNetV3-Large |
| Number of classes       | 17                |
| Input size              | 224 × 224         |
| Batch size              | 10                |
| Optimizer               | Adam              |
| Initial learning rate   | 1e-4              |
| Weight decay            | 1e-4              |
| Loss                    | CrossEntropyLoss  |
| LR scheduler            | ReduceLROnPlateau |
| Scheduler factor        | 0.5               |
| Scheduler patience      | 1                 |
| Early stopping patience | 3                 |
| Maximum epochs          | 15                |
| Pretrained weights      | ImageNet          |

Mixed-precision training is enabled when CUDA is available.

The best validation-loss model state is restored after training.

---

## 11. Training Results

The reported validation results for the three context models are:

| Context           | Best Validation Accuracy |
| ----------------- | -----------------------: |
| **C1 — Local**    |               **93.75%** |
| **C2 — Original** |               **95.75%** |
| **C3 — Expanded** |               **93.00%** |

In the reported training run, C2 provides the strongest validation accuracy among the three context representations.

For C2, the best reported validation accuracy is:

```text
95.75%
```

at epoch 15, with validation loss:

```text
0.1910
```

The corresponding final training accuracy is:

```text
99.62%
```

These results indicate that the original field of view provided the strongest performance in this particular experimental configuration.

---

## 12. Evaluation

The evaluation pipeline computes:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* IoU
* Dice score
* Confusion matrix
* ROC curves
* AUC
* Context confidence
* Context prediction stability

The classification metrics are calculated from the model predictions and ground-truth labels.

IoU and Dice are computed from the class-wise confusion matrix and averaged across classes.

---

## 13. Explainability

CAEA includes **Grad-CAM-based visual explanations**.

For each selected context, the system can visualize:

```text
Original image
      +
Grad-CAM heatmap
      +
Prediction overlay
```

This provides an indication of which image regions contribute most strongly to the model's prediction.

Example visualization concept:

```text
┌──────────────────┐
│ Original Image   │
└──────────────────┘
          │
          ▼
┌──────────────────┐
│ Grad-CAM Heatmap │
└──────────────────┘
          │
          ▼
┌──────────────────┐
│ Overlay + Class  │
└──────────────────┘
```

The explainability module is intended to improve the interpretability of the classification results rather than treating the model as a completely opaque predictor.

---

## 14. Context Analysis

One of the important analysis components of CAEA is comparing predictions across C1, C2 and C3.

The notebook reports the following mean-confidence analysis:

| Context       | Mean Confidence | Stability vs C2 |
| ------------- | --------------: | --------------: |
| C1 — Local    |          0.3965 |          0.6387 |
| C2 — Original |          0.8011 |          1.0000 |
| C3 — Expanded |          0.7213 |          0.4958 |

This analysis helps investigate whether predictions remain consistent when the spatial field of view changes.

---

## 15. Experimental Comparison

The evaluation framework includes comparison rows for:

```text
Lightweight CNN – Local Context
Lightweight CNN – Original Context
Lightweight CNN – Expanded Context
Grad-CAM baseline – Original Context
CAEA – IoU only
CAEA – IoU + Dice
CAEA – IoU + Dice + SSIM
Proposed CAEA (ESS)
ResNet-18 – Original Context
```

The final experimental table should be updated with the validated results used in the final research report.

**Important:** values generated synthetically or used only for demonstrating the table structure should not be presented as independently measured experimental results.

---

## 16. Deployment Application

The trained models are exposed through a **Streamlit web application**.

The application supports:

* Histopathology image upload
* Model/context selection
* Tissue-class prediction
* Confidence score
* Top-5 predictions
* C1/C2/C3 context comparison
* Context-consensus information
* Grad-CAM visualization
* ResNet-18 baseline prediction

The deployment is intended as an interactive demonstration of the trained models.

---

## 17. Deployment Architecture

```text
User
 │
 ▼
Streamlit Web Interface
 │
 ▼
Image Upload
 │
 ▼
Preprocessing
 │
 ├───────────────┬───────────────┐
 ▼               ▼               ▼
C1              C2              C3
 │               │               │
 ▼               ▼               ▼
MobileNetV3     MobileNetV3     MobileNetV3
 │               │               │
 └───────────────┼───────────────┘
                 ▼
        Prediction Analysis
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Class Prediction       Grad-CAM
       │                   │
       └─────────┬─────────┘
                 ▼
          Streamlit Output
```

---

## 18. Model Weights

The trained model weights are not stored directly in this GitHub repository.

They are hosted in the private Hugging Face repository:

```text
Chikku2005/caea
```

Expected model files:

```text
MobileNetV3_C1.pth
MobileNetV3_C2.pth
MobileNetV3_C3.pth
ResNet18.pth
```

The Streamlit application retrieves the required weights at runtime.

---

## 19. Repository Structure

```text
CAEA/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── classes.json
├── config.json
├── context_info.json
├── model_info.json
│
└── [optional]
    ├── assets/
    ├── screenshots/
    └── examples/
```

### Configuration files

#### `classes.json`

Contains the tissue-class labels used by the application.

#### `config.json`

Contains application and model configuration.

#### `context_info.json`

Describes the C1, C2 and C3 spatial context configurations.

#### `model_info.json`

Contains model metadata used by the deployment application.

---

## 20. Installation

Clone the repository:

```bash
git clone https://github.com/SreekarBharadwaj/CAEA.git
cd CAEA
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 21. Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will provide a local web interface for model inference.

---

## 22. Hugging Face Authentication

Because the model repository is private, the deployment requires a Hugging Face access token.

For Streamlit Community Cloud, configure the secret through:

```text
App
→ Settings
→ Secrets
```

Add:

```toml
HF_TOKEN = "your_huggingface_access_token"
```

Do **not** place the token directly in:

```text
app.py
```

and never commit it to GitHub.

---

## 23. Streamlit Community Cloud Deployment

Deployment configuration:

```text
Repository: SreekarBharadwaj/CAEA
Branch:     main
Main file:  app.py
```

After deployment:

1. Open the Streamlit application settings.
2. Open **Secrets**.
3. Add the Hugging Face access token.
4. Save the secrets.
5. Restart/redeploy the application.
6. Upload a histopathology image.
7. Select the desired model/context.
8. Inspect the prediction and explainability output.

---

## 24. Example Inference Workflow

```text
Upload Image
     │
     ▼
Validate Input
     │
     ▼
Generate Context Views
     │
     ├── C1 Local
     ├── C2 Original
     └── C3 Expanded
     │
     ▼
Run Model Inference
     │
     ▼
Calculate Probabilities
     │
     ▼
Generate Top-5 Predictions
     │
     ▼
Compare Context Predictions
     │
     ▼
Generate Grad-CAM
     │
     ▼
Display Results
```

---

## 25. Technology Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* NumPy
* scikit-learn
* scikit-image

### Computer Vision

* Pillow
* OpenCV
* Matplotlib

### Deployment

* Streamlit
* Hugging Face Hub

---

## 26. Reproducibility

The training configuration should be recorded whenever models are regenerated.

Important reproducibility parameters include:

```text
Dataset version
Number of samples
Train/validation split
Random seed
Context ratios
Input resolution
Backbone
Pretrained weights
Batch size
Learning rate
Optimizer
Weight decay
Number of epochs
Early-stopping configuration
```

The reported experiment uses a stratified 80/20 split with random state 42.

---

## 27. Limitations

This project is an experimental research and demonstration system.

Important limitations include:

* The reported training experiment uses a limited subset of the available images.
* Performance may vary with different dataset splits.
* Histopathology images can exhibit staining, acquisition and scanner variability.
* Model confidence should not be interpreted as clinical certainty.
* Grad-CAM provides an explanatory visualization but does not establish clinical causality.
* The system has not been established as a clinical diagnostic device.
* External validation is required before considering real-world clinical use.

---

## 28. Medical Disclaimer

**This software is intended for research, educational and demonstration purposes only.**

CAEA is not a medical device and should not be used to make, confirm or exclude a medical diagnosis.

Predictions generated by the system should not replace assessment by qualified pathologists or other healthcare professionals.

---

## 29. Future Work

Potential future extensions include:

* End-to-end multi-context feature fusion
* Learned context weighting
* Attention-based context aggregation
* Larger and more diverse datasets
* External multi-dataset validation
* Calibration of prediction confidence
* Additional explainability techniques
* Interactive whole-slide-image analysis
* Uncertainty estimation
* Model quantization and inference optimization
* Clinical workflow integration after appropriate validation

---

## 30. Citation

If this repository or CAEA architecture is used in academic work, please cite the associated research publication:

```text
[Add the final paper citation here]
```

A BibTeX entry can be added once the publication details are finalized.

---

## 31. Acknowledgements

This project uses open-source deep-learning and computer-vision libraries including PyTorch, Torchvision, NumPy, scikit-learn, scikit-image, OpenCV and Streamlit.

The project also makes use of pretrained ImageNet weights for MobileNetV3-Large.

---

## 32. License

Add the appropriate project license here.

For example:

```text
MIT License
```

if the project is intended to be released under the MIT license.

---

# CAEA at a Glance

```text
                 CAEA
                  │
        ┌─────────┼─────────┐
        │         │         │
       C1        C2        C3
     Local    Original   Expanded
        │         │         │
        └─────────┼─────────┘
                  │
            MobileNetV3
                  │
                  ▼
        Tissue Classification
                  │
        ┌─────────┴─────────┐
        │                   │
   Prediction           Explainability
        │                   │
   Top-5 / Confidence    Grad-CAM
        │                   │
        └─────────┬─────────┘
                  ▼
          Interactive CAEA
             Dashboard
```

**CAEA — Context-Aware Ensemble Architecture for Histopathology Tissue Classification**

*Multi-scale context. Tissue-aware classification. Interpretable predictions.*
