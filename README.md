# Multi-Task EfficientNet-B0 for Dementia Presence and Severity Prediction

**Ayesha Anzer** · Faculty of Engineering and Applied Science · University of Regina, Canada  
Presented at the *ISG 15th World Conference*, Vancouver, March 2026

---

## Overview

This repository contains the full implementation of a multi-task deep learning model that simultaneously performs:

- **Task 1 — Dementia Presence Detection** (binary): Non-Demented vs. Demented
- **Task 2 — Clinical Severity Staging** (3-class ordinal): Non-Demented · Very Mild · Mild/Moderate

A single EfficientNet-B0 backbone, pretrained on ImageNet, is fine-tuned end-to-end with a shared feature head and three task-specific output heads using a combined binary + ordinal loss. Predictions are aggregated at the **subject level** to support clinically realistic, patient-level decision-making.

---

## Architecture

```
T1-weighted Brain MRI (224x224x3)
        |
  EfficientNet-B0 Backbone  (pretrained, ImageNet)
        |
  AdaptiveAvgPool2d  -->  1280-dim feature vector
        |
  Shared Head:
    Linear(1280 -> 256) + ReLU + Dropout(0.35)
    SE Block (Squeeze-and-Excitation, 256-dim)
        |
   _____|_____________________
  |           |               |
Binary Head  Ordinal Head >=1  Ordinal Head >=2
Linear(256->1) Linear(256->1) Linear(256->1)
P(Demented)  P(severity>=VeryMild) P(severity>=Mild/Moderate)
```

**Ordinal Decoding (Inference)**

```
P(>=1) >= t1 ?
  No  --> Class 0: Non-Demented
  Yes --> P(>=2) >= t2 ?
            No  --> Class 1: Very Mild
            Yes --> Class 2: Mild / Moderate
```

Thresholds `t1` and `t2` are tuned on the validation set via two-stage grid search.  
Constraint enforced: `P(>=2) <= P(>=1)` at all times.

---

## Results

### Dementia Presence (Binary)

| Level         | Accuracy | AUC  | Macro-F1 |
|---------------|----------|------|----------|
| Image-level   | 88%      | 0.93 | 0.88     |
| Subject-level | **91%**  | **0.95** | **0.90** |

### Severity Staging (3-Class Ordinal)

| Level         | Accuracy | Macro-AUC | Macro-F1 |
|---------------|----------|-----------|----------|
| Image-level   | 75%      | 0.85      | 0.73     |
| Subject-level | **82%**  | **0.90**  | **0.80** |

*Subject-level AUC for severity = macro-average of one-vs-all ROC curves.*

---

## Dataset

All experiments use the publicly available **OASIS (Open Access Series of Imaging Studies)** dataset of T1-weighted structural brain MRI scans of older adults. No proprietary data is used.

- Source: [https://www.oasis-brains.org](https://www.oasis-brains.org)
- Kaggle mirror: [Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- Classes used: Non Demented · Very mild Dementia · Mild Dementia · Moderate Dementia

---

## Repository Structure

```
dementia-multitask-effnet/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── src/
│   ├── dataset.py            # DementiaOrdinalDataset + transforms
│   ├── model.py              # ProposedEffNetOrdinal architecture
│   ├── losses.py             # Combined binary + ordinal loss
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Metrics, threshold tuning, reports
│   └── utils.py              # Seed, subject split, helpers
├── scripts/
│   ├── train.py              # Entry point: full training run
│   └── evaluate.py           # Entry point: evaluate a checkpoint
└── results/                  # Saved checkpoints and outputs (git-ignored)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/dementia-multitask-effnet.git
cd dementia-multitask-effnet
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. GPU is strongly recommended.

---

## Usage

### 1. Prepare Data

Download the OASIS dataset (Kaggle link above) and place the zip at the project root:

```
dementia-multitask-effnet/
└── archive.zip
```

### 2. Train

```bash
python scripts/train.py
```

Checkpoints are saved to `results/` as training progresses:

| File | Saved when |
|------|------------|
| `best_binary_auc.pth`  | Validation subject-level AUC improves |
| `best_severity_f1.pth` | Validation severity macro-F1 improves |
| `final_model.pth`      | End of training |

### 3. Evaluate

```bash
python scripts/evaluate.py --checkpoint results/best_binary_auc.pth
```

Prints image-level and subject-level metrics for both tasks, with tuned thresholds.

---

## Hyperparameters

All defaults are in `configs/default.yaml`. Key values:

| Parameter | Value |
|-----------|-------|
| Image size | 224 x 224 |
| Batch size | 32 |
| Epochs | 25 |
| Head LR | 3e-4 |
| Backbone LR | 3e-5 |
| Weight decay | 1e-4 |
| Freeze epochs | 5 |
| Early stop patience | 5 |
| Binary loss weight | 0.6 |
| Ordinal loss weight | 0.4 |
| SE block reduction | 8 |
| Dropout | 0.35 |

---

## Training Strategy

1. **Warm-up phase (epochs 1–5):** Backbone frozen; only the shared head and task heads are trained.
2. **Fine-tuning phase (epochs 6+):** Full network unfrozen with backbone at 10x lower LR (`3e-5`) than the head (`3e-4`).
3. **Combined early stopping:** Monitors `0.6 × Binary AUC + 0.4 × Severity F1` on the validation set.
4. **Dual checkpointing:** Best binary-AUC model and best severity-F1 model saved independently.
5. **Threshold tuning:** After training, binary and ordinal thresholds are tuned on the validation set (2-stage grid search) before final test evaluation.

---

## Reproducibility

All random seeds are fixed at `42` (Python `random`, NumPy, PyTorch). Data splits use `StratifiedGroupKFold(n_splits=3)` with subject-level grouping to ensure no subject appears in both train and test sets.

```python
set_seed(42)
```

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{anzer2026multitask,
  title     = {A Multi-Task Deep Learning Model for Dementia Presence and Severity Prediction from Brain MRI},
  author    = {Anzer, Ayesha},
  booktitle = {ISG 15th World Conference},
  address   = {Vancouver, Canada},
  month     = {March},
  year      = {2026},
  note      = {Oral Presentation}
}
```

---

## Contact

**Ayesha Anzer**  
Faculty of Engineering and Applied Science  
University of Regina, Canada  
aag833@uregina.ca

---

## License

This project is released for academic and research use. The OASIS dataset is subject to its own data use agreement available at [oasis-brains.org](https://www.oasis-brains.org).
