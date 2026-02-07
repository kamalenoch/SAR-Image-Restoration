# SAR Image Denoising & Target Detection Framework

This repository contains the implementation of a hybrid framework for Synthetic Aperture Radar (SAR) image despeckling and target detection. The project integrates Wavelet transforms for preprocessing, a Lightweight Convolutional Neural Network (CNN) for feature learning and denoising, and the CA-CFAR algorithm for robust target detection.

This codebase is designed for high-performance SAR image analysis, likely in the context of research submissions (e.g., IEEE Access).

## 🚀 Features

*   **Hybrid Preprocessing**: Utilizes Wavelet transforms to decompose SAR images and reduce initial speckle noise.
*   **Deep Learning Model**: Implements a custom **LightweightCNN** optimized for efficient image restoration.
*   **Target Detection**: Integrated **Cell-Averaging Constant False Alarm Rate (CA-CFAR)** algorithm for identifying targets in the denoised imagery.
*   **Comprehensive Evaluation metrics**:
    *   **PSNR** (Peak Signal-to-Noise Ratio)
    *   **SSIM** (Structural Similarity Index)
    *   **ENL** (Equivalent Number of Looks)
    *   **Runtime** & **Parameter Count** analysis

## 📂 Project Structure

```
├── data/               # Data loading scripts and raw SAR imagery
├── dsp/                # Digital Signal Processing modules (Wavelet transforms)
├── model/              # PyTorch model definitions (LightweightCNN)
├── cfar/               # Target detection algorithms (CA-CFAR)
├── metrics/            # Evaluation metrics (PSNR, SSIM, ENL, Runtime)
├── train.py            # Training loop for the CNN
├── evaluate.py         # Evaluation pipeline
└── main.py             # Main entry point to run the full pipeline
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sar-denoise-detect.git
    cd sar-denoise-detect
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. You will need `torch`, `numpy`, and likely `scikit-image` or `scipy` depending on the metric implementations.
    ```bash
    pip install torch numpy scipy scikit-image
    ```

## ⚡ Usage

To run the complete pipeline (Load -> Denoise -> Detect -> Evaluate), execute the main script:

```bash
python main.py
```

### pipeline flow:
1.  **Input**: Raw SAR images are loaded from `data/`.
2.  **Preprocessing**: Images undergo **Wavelet Denoising** (`dsp/wavelet.py`).
3.  **Inference**: The **LightweightCNN** processes the wavelet-transformed data.
4.  **Detection**: **CA-CFAR** is applied to the output to detect potential targets.
5.  **Output**: Quality metrics (PSNR, SSIM, ENL) and detection counts are printed.

## 🧠 Methodology

### Lightweight CNN
The core model is a streamlined Convolutional Neural Network designed to balance performance with computational efficiency. It consists of multiple convolutional layers with ReLU activation, progressively refining features to output a clean SAR image.

### CA-CFAR
The Cell-Averaging CFAR detector is used to adaptively set thresholds based on background noise levels, making it effective for varying SAR environments.

## 📊 Metrics

The project evaluates performance using:
*   **Visual Quality**: PSNR & SSIM
*   **Speckle Reduction**: ENL
*   **Efficiency**: Model parameter count and inference runtime

---
*Note: This project is intended for research purposes.*
