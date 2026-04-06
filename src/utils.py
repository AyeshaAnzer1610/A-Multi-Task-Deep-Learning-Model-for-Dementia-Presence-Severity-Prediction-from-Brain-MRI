"""src/utils.py  —  Seed, device, all plotting helpers"""
import os, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.calibration import calibration_curve


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    if torch.cuda.is_available():
        d    = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name}  ({mem:.1f} GB VRAM)")
    else:
        d = torch.device("cpu")
        print("  CPU mode (training will be slow on CPU)")
    return d


def plot_training_history(history, save_path="results/training_history.png"):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], color="#1976D2", lw=2)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_bin_auc"], label="Binary AUC",  color="#388E3C", lw=2)
    axes[1].plot(history["val_sev_f1"],  label="Severity F1", color="#F57C00", lw=2)
    axes[1].set_title("Validation Metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(history["composite"], color="#7B1FA2", lw=2)
    axes[2].set_title("Composite (0.6·AUC + 0.4·F1)")
    axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="Actual", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(results, save_path="results/roc_curves.png"):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    sp = results["subject_predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fpr, tpr, _ = roc_curve(sp["y_binary"], sp["p_binary"])
    a = sk_auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC={a:.3f}")
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC — Binary Detection")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    p0 = np.clip(1-sp["p_ge1"], 0, 1)
    p1 = np.clip(sp["p_ge1"]-sp["p_ge2"], 0, 1)
    p2 = np.clip(sp["p_ge2"], 0, 1)
    probs = np.stack([p0, p1, p2], axis=1)
    for c, (col, nm) in enumerate(zip(
            ["#1976D2","#F57C00","#C62828"],
            ["Non-Dem.","Very Mild","Mild/Mod."])):
        try:
            fpr, tpr, _ = roc_curve((sp["y_ordinal"]==c).astype(int), probs[:,c])
            a = sk_auc(fpr, tpr)
            axes[1].plot(fpr, tpr, color=col, lw=2, label=f"{nm} ({a:.3f})")
        except Exception:
            pass
    axes[1].plot([0,1],[0,1],"k--", lw=1)
    axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC — Severity (One-vs-All)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_calibration(results, save_path="results/calibration.png", n_bins=10):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    sp = results["subject_predictions"]
    fp, mp = calibration_curve(sp["y_binary"], sp["p_binary"],
                               n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0,1],[0,1],"k--", lw=1, label="Perfect calibration")
    ax.plot(mp, fp, "s-", color="#1976D2", lw=2, label="MT-CNN")
    ax.set(xlabel="Mean predicted prob", ylabel="Fraction of positives",
           title="Calibration — Binary Detection")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")
