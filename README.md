# 🌿 Plant Disease Classification using CNN + Transfer Learning

A deep learning system for automated plant disease detection from leaf images using EfficientNet-B0 with transfer learning, built with PyTorch and deployed via Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
- [Future Improvements](#future-improvements)
- [References](#references)

## 🎯 Overview

Plant diseases cause significant agricultural losses worldwide. Early and accurate detection is crucial for effective treatment. This project implements a deep learning solution that can identify 38 different plant conditions (diseases and healthy states) across 14 crop species using leaf images.

### Problem Statement

Manual identification of plant diseases requires expertise and is time-consuming. This system automates the process using computer vision and deep learning, enabling farmers and agricultural professionals to quickly diagnose plant health issues.

## ✨ Features

- **High Accuracy:** Achieves ~98% accuracy on the PlantVillage dataset
- **Multi-class Classification:** Supports 38 different plant disease classes
- **Transfer Learning:** Utilizes pre-trained EfficientNet-B0 for efficient training
- **Web Interface:** User-friendly Streamlit application for easy predictions
- **Detailed Diagnostics:** Provides disease information and treatment recommendations
- **Top-5 Predictions:** Shows confidence scores for top predictions

## 📁 Project Structure

```
plant-disease-classification/
│
├── data/
│   └── plantvillage dataset/
│       └── color/                 # Dataset images (38 class folders)
│
├── models/
│   └── efficientnet_b0_best.pth   # Trained model weights
│
├── notebooks/
│   └── exploration.py             # Dataset exploration script
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration and hyperparameters
│   ├── dataset.py                 # Dataset class and data loaders
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Evaluation metrics and visualization
│   └── inference.py               # Single image prediction
│
├── app/
│   └── streamlit_app.py           # Streamlit web application
│
├── results/
│   ├── efficientnet_b0_history.csv
│   ├── efficientnet_b0_metrics.csv
│   ├── efficientnet_b0_confusion_matrix.png
│   ├── efficientnet_b0_training_curves.png
│   ├── efficientnet_b0_per_class_metrics.csv
│   └── sota_comparison.csv
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install matplotlib numpy pandas scikit-learn pillow tqdm streamlit seaborn
   ```

4. **Download the dataset** (see [Dataset](#dataset) section)

## 📊 Dataset

This project uses the **PlantVillage Dataset** from Kaggle.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Images | 54,303 |
| Number of Classes | 38 |
| Image Format | RGB, various sizes |
| Plants Covered | 14 species |

### Supported Plants and Diseases

| Plant | Conditions |
|-------|------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

### Download Instructions

**Option A: Using Kaggle CLI**
```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download and extract to `data/` folder

## 🏗️ Model Architecture

### Base Model: EfficientNet-B0

EfficientNet-B0 was chosen for its excellent balance between accuracy and computational efficiency. It uses compound scaling to uniformly scale network width, depth, and resolution.

### Transfer Learning Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    EfficientNet-B0                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Pre-trained Backbone                      │  │
│  │           (Frozen → Unfrozen for fine-tuning)         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Custom Classifier Head                    │  │
│  │   Dropout(0.3) → Linear(1280, 512) → ReLU             │  │
│  │   → Dropout(0.3) → Linear(512, 38)                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Training Strategy

1. **Phase 1 (Initial Training):** Train only classifier head with frozen backbone (5 epochs, LR=1e-3)
2. **Phase 2 (Fine-tuning):** Unfreeze backbone and train entire model (10 epochs, LR=1e-5)

## 🎓 Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy |
| Learning Rate (Phase 1) | 1e-3 |
| Learning Rate (Phase 2) | 1e-5 |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 5 |

### Data Augmentation

- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.3)
- Random Rotation (±20°)
- Color Jitter (brightness, contrast, saturation)
- Normalization (ImageNet mean/std)

### Run Training

```bash
python -m src.train
```

Training takes approximately 45-60 minutes on Apple M4 (MPS) or NVIDIA GPU.

## 📈 Evaluation

### Run Evaluation

```bash
python -m src.evaluate
```

### Metrics Generated

- Accuracy, Precision, Recall, F1-Score
- Top-3 and Top-5 Accuracy
- Confusion Matrix
- Per-class Performance
- Training History Curves

## 💻 Usage

### Command Line Inference

```python
from src.inference import PlantDiseaseClassifier

# Initialize classifier
classifier = PlantDiseaseClassifier("efficientnet_b0")

# Predict on an image
result = classifier.predict("path/to/leaf/image.jpg")

print(f"Plant: {result['plant']}")
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Web Application

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Results

### Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 98.34% |
| Top-3 Accuracy | 99.5% |
| Top-5 Accuracy | 99.8% |
| F1-Score (Macro) | 98.2% |
| F1-Score (Weighted) | 98.3% |

### Comparison with State-of-the-Art

| Model | Accuracy |
|-------|----------|
| AlexNet (Mohanty et al., 2016) | 85.53% |
| VGG16 (Mohanty et al., 2016) | 90.40% |
| ResNet34 (Mohanty et al., 2016) | 91.20% |
| GoogLeNet (Mohanty et al., 2016) | 97.28% |
| VGG19 (Too et al., 2019) | 99.24% |
| ResNet50 (Too et al., 2019) | 99.35% |
| InceptionV3 (Too et al., 2019) | 99.50% |
| DenseNet121 (Too et al., 2019) | 99.75% |
| **Our EfficientNet-B0** | **98.34%** |

### Training Curves

The model shows stable convergence with validation accuracy closely tracking training accuracy, indicating good generalization without overfitting.

## 🎥 Demo

The Streamlit application provides:

1. **Image Upload:** Drag and drop or browse for leaf images
2. **Real-time Prediction:** Instant disease classification
3. **Confidence Scores:** Visual progress bars for prediction confidence
4. **Disease Information:** Description, symptoms, and treatment recommendations
5. **Top-5 Predictions:** Alternative diagnoses with confidence levels

## 🔮 Future Improvements

- [ ] Add ResNet50 for model comparison
- [ ] Implement Grad-CAM for visual explanations
- [ ] Mobile application deployment
- [ ] Support for more plant species
- [ ] Real-time camera feed processing
- [ ] Multi-language support for disease information
- [ ] Integration with agricultural advisory systems

## 📚 References

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.

2. Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019). A Comparative Study of Fine-Tuning Deep Learning Models for Plant Disease Identification. *Computers and Electronics in Agriculture*, 161, 272-279.

3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *International Conference on Machine Learning*.

4. PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Ali Yusifov - Initial work

## 🙏 Acknowledgments

- PlantVillage project for the dataset
- PyTorch team for the deep learning framework
- Streamlit team for the web application framework