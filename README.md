Reproducibility
---------------

Tagged Release: v5.0
Commit Hash: 7ebb6b9



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

* **Fixed seeds:** All experiments use a fixed random seed (`SEED = 42`) for data splitting, training, and evaluation.
* **Deterministic splits:** Exact dataset partitions are released as plain-text files: `train_split.txt`, `val_split.txt`, and `test_split.txt`.
* **Script-to-table mapping:** Each table and figure in the paper is directly linked to its generating script (see mapping below).


### Fixed Random Seeds

All experiments were conducted using fixed random seeds to ensure deterministic behavior and reproducibility across runs.

```python
# Fixed seeds used in all experiments
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

The same seed (`SEED = 42`) is used consistently for:

* Dataset splitting
* Weight initialization
* Data shuffling
* Training and evaluation across all datasets

---

### Plain-Text Dataset Split Files

To prevent data leakage and guarantee reproducibility, dataset splits are **explicitly defined and released** as plain-text files in the repository.

**Files included in the root directory:**

```
train_split.txt   # Training image IDs
val_split.txt     # Validation image IDs
test_split.txt    # Test image IDs
```

Each file contains **one image ID per line**, ensuring that all experiments can be reproduced exactly without reliance on random or implicit splitting.

These split files are used consistently across:

* Baseline models
* Ablation studies
* Cross-dataset evaluation
* KiTS23 experiments

---
###Script-to-Table Mapping


| Script                  | Paper Content                                        |
| ----------------------- | ---------------------------------------------------- |
| `main.py`               | Primary results, Tables 5–7, 9, 12–13; Figures 12–16 |
| `data_augmentation.py`  | Figures 9–10; Table 3                                |
| `data_split.py`         | Table 8                                              |
| `ablation_study.py`     | Table 11; Figures 18–20                              |
| `visualize_overlays.py` | Tables 4, 10                                         |
| `evaluation_metrics.py` | Table 7; Figure 17                                   |

All results in the paper can be reproduced using the released code, fixed seeds, and provided split files. 


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




