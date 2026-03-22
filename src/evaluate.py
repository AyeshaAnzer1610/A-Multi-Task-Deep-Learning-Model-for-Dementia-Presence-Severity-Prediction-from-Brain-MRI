"""evaluate.py — Inference, threshold tuning, and metric reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, loader, device: torch.device) -> pd.DataFrame:
    """
    Run *model* over *loader* and collect raw probabilities.

    Returns a DataFrame with columns:
        subject_id, y_bin, y_sev, p_bin, p_ge1, p_ge2
    """
    model.eval()
    rows = []

    for x, y_bin, ge1, ge2, y_sev, sid in loader:
        x = x.to(device, non_blocking=True)
        logit_bin, logit_ge1, logit_ge2 = model(x)

        p_bin = torch.sigmoid(logit_bin).cpu().numpy()
        p_ge1 = torch.sigmoid(logit_ge1).cpu().numpy()
        p_ge2 = torch.sigmoid(logit_ge2).cpu().numpy()

        rows.extend(zip(
            sid,
            y_bin.numpy().astype(int),
            y_sev.numpy().astype(int),
            p_bin, p_ge1, p_ge2,
        ))

    return pd.DataFrame(rows, columns=["subject_id", "y_bin", "y_sev", "p_bin", "p_ge1", "p_ge2"])


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Average image-level probabilities to produce one row per subject."""
    return df.groupby("subject_id").agg({"y_bin": "max", "p_bin": "mean"}).reset_index()


# ---------------------------------------------------------------------------
# Ordinal decoding
# ---------------------------------------------------------------------------

def decode_severity(
    p_ge1: np.ndarray,
    p_ge2: np.ndarray,
    t1: float = 0.5,
    t2: float = 0.5,
) -> np.ndarray:
    """
    Convert ordinal probabilities to a 3-class label using dual thresholds.

    Ordinal constraint enforced: P(>=2) is clipped to P(>=1) so that the
    probability of a higher severity never exceeds that of a lower one.

    Decision rule:
        P(>=1) < t1 --> 0  (Non-Demented)
        P(>=1) >= t1 and P(>=2) < t2 --> 1  (Very Mild)
        P(>=1) >= t1 and P(>=2) >= t2 --> 2  (Mild / Moderate)
    """
    p_ge1 = np.asarray(p_ge1)
    p_ge2 = np.minimum(np.asarray(p_ge2), p_ge1)   # enforce monotonicity

    y = np.zeros_like(p_ge1, dtype=int)
    y[p_ge1 >= t1] = 1
    y[(p_ge1 >= t1) & (p_ge2 >= t2)] = 2
    return y


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------

def report_binary(
    df: pd.DataFrame,
    threshold: float,
    name: str = "SET",
    subject_level: bool = False,
) -> tuple[float, float, float]:
    """Print binary classification metrics and return (acc, balanced_acc, auc)."""
    if subject_level:
        g = aggregate_by_subject(df)
        y_true, y_prob = g["y_bin"].to_numpy(), g["p_bin"].to_numpy()
        level = "SUBJECT"
    else:
        y_true, y_prob = df["y_bin"].to_numpy(), df["p_bin"].to_numpy()
        level = "IMAGE"

    y_hat = (y_prob >= threshold).astype(int)
    acc  = accuracy_score(y_true, y_hat)
    bal  = balanced_accuracy_score(y_true, y_hat)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    print(f"\n--- {name} Binary ({level}) @ threshold={threshold:.3f} ---")
    print(f"  Accuracy: {acc:.4f}  |  Balanced Acc: {bal:.4f}  |  AUC: {auc:.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_true, y_hat))
    return acc, bal, auc


def report_severity(
    df: pd.DataFrame,
    t1: float = 0.5,
    t2: float = 0.5,
    name: str = "SET",
) -> tuple[float, float]:
    """Print severity classification report and return (acc, macro_f1)."""
    y_true = df["y_sev"].to_numpy()
    y_hat  = decode_severity(df["p_ge1"].to_numpy(), df["p_ge2"].to_numpy(), t1=t1, t2=t2)
    acc      = accuracy_score(y_true, y_hat)
    macro_f1 = f1_score(y_true, y_hat, average="macro")

    print(f"\n--- {name} Severity (3-class) @ (t1={t1:.3f}, t2={t2:.3f}) ---")
    print(f"  Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_true, y_hat))
    print(classification_report(y_true, y_hat, digits=4, zero_division=0))
    return acc, macro_f1


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_binary_threshold(df: pd.DataFrame) -> tuple[float, float]:
    """
    Grid-search the binary decision threshold at the subject level,
    maximising balanced accuracy on the provided split.

    Returns: (best_threshold, best_balanced_accuracy)
    """
    g = aggregate_by_subject(df)
    y_true, y_prob = g["y_bin"].to_numpy(), g["p_bin"].to_numpy()

    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        bal = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)

    print(f"Tuned binary threshold: t={best_t:.3f}  |  balanced acc={best_bal:.4f}")
    return best_t, best_bal


def tune_ordinal_thresholds(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Two-stage grid search over (t1, t2) maximising severity macro-F1.

    Stage 1: coarse grid over [0.30, 0.70] in 0.05 steps.
    Stage 2: fine grid ±0.06 around the stage-1 best in 0.01 steps.

    Returns: (best_t1, best_t2, best_macro_f1)
    """
    y_true = df["y_sev"].to_numpy()
    p1, p2 = df["p_ge1"].to_numpy(), df["p_ge2"].to_numpy()

    coarse = np.arange(0.30, 0.701, 0.05)
    best_t1, best_t2, best_f1 = 0.5, 0.5, -1.0

    for t1 in coarse:
        for t2 in coarse:
            mf1 = f1_score(y_true, decode_severity(p1, p2, t1, t2), average="macro")
            if mf1 > best_f1:
                best_f1, best_t1, best_t2 = mf1, t1, t2

    # Refine
    fine1 = np.clip(np.arange(best_t1 - 0.06, best_t1 + 0.061, 0.01), 0.30, 0.70)
    fine2 = np.clip(np.arange(best_t2 - 0.06, best_t2 + 0.061, 0.01), 0.30, 0.70)
    for t1 in np.unique(fine1):
        for t2 in np.unique(fine2):
            mf1 = f1_score(y_true, decode_severity(p1, p2, t1, t2), average="macro")
            if mf1 > best_f1:
                best_f1, best_t1, best_t2 = mf1, t1, t2

    print(f"Tuned ordinal thresholds: t1={best_t1:.3f}, t2={best_t2:.3f}  |  macro-F1={best_f1:.4f}")
    return float(best_t1), float(best_t2), float(best_f1)
