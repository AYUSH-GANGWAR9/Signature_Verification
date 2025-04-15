# 📝 Signature Verification System

An advanced signature verification system using deep learning and traditional machine learning techniques to distinguish between genuine and forged signatures.

## 🔍 Overview

This system implements a multi-model approach for signature verification by combining:
- Convolutional Neural Networks (CNN) for feature extraction
- Histogram of Oriented Gradients (HOG) for traditional feature extraction
- LSTM networks for sequential pattern analysis
- Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) for classification

The hybrid approach achieves robust signature verification that outperforms single-model systems.

## ✨ Features

- **Multi-model verification**: Combines results from multiple classifiers for higher accuracy
- **Interactive testing**: Verify individual signature pairs through an easy-to-use interface
- **Batch evaluation**: Test system performance on standard datasets (UTSig, CEDAR)
- **Comprehensive metrics**: View performance metrics including EER, FAR/FRR, and ROC curves
- **Visual comparison**: Side-by-side comparison of test and reference signatures

## 🚀 Installation

### Prerequisites

- Python 3.7+
- GPU support recommended for faster training (but not required)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/signature-verification.git
   cd signature-verification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv signature_env
   
   # On Windows
   signature_env\Scripts\activate
   
   # On macOS/Linux
   source signature_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p models data/UTSig data/CEDAR src
   ```

## 📂 Dataset Structure

Organize your datasets as follows:

### UTSig Dataset
```
data/
└── UTSig/
    └── <user_id>/
        ├── Genuine/
        │   └── <signature_images>
        └── Skilled/
            └── <signature_images>
```

### CEDAR Dataset
```
data/
└── CEDAR/
    ├── full_org/
    │   └── <signature_images>
    └── full_forg/
        └── <signature_images>
```

## 📋 Usage

### Training the Models

```bash
python train.py
```

This will train the CNN feature extractor, LSTM, SVM, and KNN models using the specified dataset and save them to the `models/` directory.

### Interactive Verification

Run the system in interactive mode:

```bash
python test.py
```

This presents a menu where you can:
1. Verify signature pairs by providing file paths
2. Run batch testing on datasets
3. Exit the program

### Command-line Verification

Verify specific signature files:

```bash
python test.py --mode interactive --test path/to/test_signature.png --reference path/to/reference_signature.png
```

### Batch Testing

Evaluate system performance on a dataset:

```bash
python test.py --mode batch --dataset UTSig
```

## 🛠️ System Components

- **config.py**: Configuration settings for paths and parameters
- **src/preprocessing.py**: Image preprocessing utilities
- **src/feature_extraction.py**: HOG and CNN feature extraction
- **src/feature_integration.py**: Feature fusion techniques
- **src/evaluation.py**: Performance metrics calculation
- **train.py**: Model training script
- **test.py**: Interactive testing script

## 📊 Performance Metrics

The system evaluates performance using:
- Accuracy, Precision, Recall, F1-score
- Equal Error Rate (EER)
- False Acceptance Rate (FAR) and False Rejection Rate (FRR)
- ROC curves and AUC scores

## 📦 Requirements

```
numpy>=1.19.5
tensorflow>=2.5.0
scikit-learn>=0.24.2
scikit-image>=0.18.1
matplotlib>=3.4.2
opencv-python>=4.5.2
tqdm>=4.61.0
```

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors <br>

- Ayush gangwar
<br><hr>

## 👏 Acknowledgments

- UTSig and CEDAR signature datasets
- The signature verification research community

## 🔗 Quick Links

- [Report Bug](https://github.com/yourusername/signature-verification/issues)
- [Request Feature](https://github.com/yourusername/signature-verification/issues)

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Build-Passing-success.svg" alt="Build Status">
</p>