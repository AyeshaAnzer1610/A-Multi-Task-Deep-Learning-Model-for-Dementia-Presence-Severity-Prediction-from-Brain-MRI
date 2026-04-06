# Multi-Task EfficientNet-B0 for Dementia Presence and Severity Prediction

**Ayesha Anzer, Yassine Aribi, Hamzah Faraj**
*Under review — IEEE Access 2026*

---

## Overview

A unified multi-task CNN (MT-CNN) that **simultaneously** performs:
- **Task 1 — Binary detection**: Non-Demented vs. Demented  
- **Task 2 — Ordinal severity staging**: Non-Demented · Very Mild · Mild/Moderate

Built on EfficientNet-B0 with a shared Squeeze-and-Excitation feature head
and cumulative-link ordinal regression outputs. Evaluated with strict
subject-level, leakage-free stratified splits.

---

## Results

### Binary Dementia Detection (Subject-Level)
| Accuracy | AUC   | Macro-F1 |
|----------|-------|----------|
| 96.3%    | 0.988 | —        |

### Ordinal Severity Staging (Subject-Level)
| Accuracy | Macro-AUC | Adjacent errors |
|----------|-----------|-----------------|
| 93.8%    | 0.981     | 100% (0 non-adjacent) |

---

## Repository Structure
├── MT_CNN_Notebook.ipynb     ← Complete run notebook (open this)
├── requirements.txt
├── configs/
│   └── default.yaml          ← All hyperparameters
├── scripts/
│   ├── train.py              ← CLI training entry point
│   └── evaluate.py           ← CLI evaluation entry point
└── src/
├── dataset.py            ← Data loading + label mapping
├── model.py              ← MT-CNN architecture
├── losses.py             ← Combined binary + ordinal loss
├── train.py              ← Two-phase training loop
├── evaluate.py           ← Full evaluation pipeline
├── gradcam.py            ← Grad-CAM + t-SNE
└── utils.py              ← Plotting helpers

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset (no Kaggle account needed)
```python
from datasets import load_dataset
ds = load_dataset("Falah/Alzheimer_MRI", split="train")
```
See `MT_CNN_Notebook.ipynb` Cell 3 for full download + save instructions.

### 3. Train
```bash
python scripts/train.py --data_root /path/to/AlzheimerDataset
```

### 4. Evaluate with Grad-CAM
```bash
python scripts/evaluate.py \
    --checkpoint results/best_binary_auc.pth \
    --gradcam --tsne
```

Or open **`MT_CNN_Notebook.ipynb`** and run all cells top to bottom.

---

## Architecture

- **Backbone**: EfficientNet-B0 pretrained on ImageNet (≈5.3M params)
- **Shared head**: Linear(1280→256) + ReLU + Dropout(0.35) + SE Block
- **Task heads**: 3 × Linear(256→1) — binary, ordinal≥1, ordinal≥2
- **Total**: ≈5.6M parameters

---

## Dataset

**Falah/Alzheimer_MRI** on HuggingFace (mirror of the original 4-class Kaggle dataset):
- 6,400 axial brain MRI slices
- 4 classes: NonDemented · VeryMildDemented · MildDemented · ModerateDemented
- Mapped to 3-class ordinal (CDR 0 / CDR 0.5 / CDR 1–2) for training

---

## Citation
```bibtex
@article{anzer2026mtcnn,
  title   = {A Multi-Task Deep Learning Framework for Joint Dementia Detection
             and Ordinal Severity Staging from Structural Brain MRI},
  author  = {Anzer, Ayesha and Aribi, Yassine and Faraj, Hamzah},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact
**Ayesha Anzer** — aag833@uregina.ca  
Faculty of Engineering and Applied Science, University of Regina, Canada
