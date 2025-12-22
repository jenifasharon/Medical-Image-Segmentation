Reproducibility
---------------

Tagged Release: v1.0
Commit Hash: 2c984a1

Below is a **clean, reviewer-ready README.md** you can directly copy into your GitHub repository. It is concise, technical, reproducible, and aligned with your paper.

---

# VHUCS-Net: Advanced Kidney Mass Segmentation

**Transformer-Enhanced U-Net with Contrast-Optimized Protuberance Detection Network**

## Overview

This repository provides the official implementation of **VHUCS-Net**, a dual-track hybrid deep learning framework for accurate kidney and kidney mass segmentation from medical images. The model integrates a **Transformer-enhanced U-Net** for kidney region extraction with a **contrast-optimized Protuberance Detection Network (PDN)** for precise mass localization, followed by feature fusion to obtain refined segmentation outputs.

The framework is designed for **high boundary accuracy**, **robust mass localization**, and **cross-dataset generalization**, validated on kidney CT images and additional biomedical segmentation datasets.

---

## Key Contributions

* Dual-track architecture combining global context modeling and contrast-driven boundary refinement
* Transformer-enhanced U-Net with ViT attention and HRNet for high-resolution feature preservation
* Contrast-optimized PDN with multiscale pooling and boundary refinement for mass segmentation
* Feature fusion strategy for improved IoU and Dice performance
* Extensive validation on Kidney, HAM10000, Blood Cell, and KiTS23 datasets
* Ablation study demonstrating the contribution of each architectural component

---

## Repository Structure

```
├── main.py                     # Main training and evaluation script
├── Baseline/                   # Baseline segmentation models
├── Ablation study/              # Ablation experiments (ViT, HRNet, PDN)
├── Cross-Dataset Evaluation/    # Validation on HAM10000 & Blood Cell datasets
├── KiTS23/                     # KiTS23 dataset experiments
├── train_split.txt             # Training split (plain-text IDs)
├── val_split.txt               # Validation split (plain-text IDs)
├── test_split.txt              # Test split (plain-text IDs)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Dataset Description

The primary experiments use the **Kidney Segmentation Dataset**, consisting of paired kidney images and corresponding tumor masks.

Additional validation datasets:

* **HAM10000** (Skin lesion segmentation)
* **Blood Cell Segmentation Dataset**
* **KiTS23** (2D slices derived from 3D CT volumes)

**Dataset split**:

* Training: 80%
* Validation: 10%
* Testing: 10%

Splits are provided as **plain-text ID files** to ensure full reproducibility.

---

## Experimental Setup

* Environment: Anaconda3 (64-bit), Jupyter Notebook
* CPU: Intel® Core™ i5-10300H (4 cores / 8 threads, 2.5–4.5 GHz)
* RAM: 16 GB
* Precision: float32
* Input size: 256 × 256 × 1
* Batch size: 32
* Epochs: 35
* Optimizer: Adam
* Loss: Dice Loss

**Inference performance**:

* ~0.00147 s per slice
* ~0.044 s per volume (30 slices)
* CPU-only execution (no GPU required)

---

## Evaluation Metrics

* Dice Similarity Coefficient (DSC)
* Intersection over Union (IoU)
* Hausdorff Distance (HD95)
* Average Symmetric Surface Distance (ASSD)

Metrics are reported as **mean ± standard deviation with 95% confidence intervals**.

---

## Results Summary

On the Kidney Segmentation Dataset, **VHUCS-Net** achieves:

* **IoU**: 0.9441
* **Dice**: 0.9712
* **Loss**: 0.0288

The model consistently outperforms baseline architectures including U-Net, UNet++, MobileNetV2, and DeepLabV3+.

---

## Reproducibility

* Fixed random seeds used across experiments
* Explicit train/validation/test split files included
* All hyperparameters documented
* Script-to-table correspondence maintained between code and reported results

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training and evaluation:

```bash
python main.py
```

Ensure dataset paths are correctly set inside `main.py`.

---


