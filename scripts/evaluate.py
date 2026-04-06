"""
scripts/evaluate.py  —  Command-line evaluation entry point

Usage:
    python scripts/evaluate.py --checkpoint results/best_binary_auc.pth
    python scripts/evaluate.py --checkpoint results/best_binary_auc.pth --gradcam --tsne
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, yaml, torch
from sklearn.metrics import confusion_matrix

from src.dataset  import build_dataloaders
from src.model    import MTCNNDementia
from src.train    import load_checkpoint
from src.evaluate import full_evaluation_report
from src.gradcam  import GradCAM, tsne_feature_plot
from src.utils    import (set_seed, get_device, plot_roc_curves,
                          plot_calibration, plot_confusion_matrix)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--data_root",   default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--gradcam",     action="store_true")
    p.add_argument("--tsne",        action="store_true")
    p.add_argument("--n_boot",      type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root

    set_seed(cfg["seed"])
    device = get_device()

    _, val_loader, test_loader, val_samp, test_samp = \
        build_dataloaders(
            data_root   = cfg["data_root"],
            image_size  = cfg["image_size"],
            batch_size  = cfg["batch_size"],
            num_workers = cfg.get("num_workers", 2),
            val_fold    = cfg.get("val_fold", 1),
            seed        = cfg["seed"],
        )

    model = MTCNNDementia(
        dropout      = cfg["dropout"],
        se_reduction = cfg.get("se_reduction", 8),
        pretrained   = False,
    ).to(device)

    thr  = load_checkpoint(model, args.checkpoint, device)
    t1, t2, tbin = thr["t1"], thr["t2"], thr["tbin"]

    rd = args.results_dir
    os.makedirs(rd, exist_ok=True)

    results = full_evaluation_report(
        model, test_loader, test_samp, device,
        t1=t1, t2=t2, tbin=tbin, n_boot=args.n_boot,
    )

    sp = results["subject_predictions"]
    plot_confusion_matrix(
        confusion_matrix(sp["y_binary"],  sp["yhat_bin"]),
        ["Non-Dem.", "Demented"],
        "Binary Detection (Subject-Level)",
        os.path.join(rd, "cm_binary.png"),
    )
    plot_confusion_matrix(
        confusion_matrix(sp["y_ordinal"], sp["yhat_ord"]),
        ["Non-Demented", "Very Mild", "Mild/Moderate"],
        "Severity Staging (Subject-Level)",
        os.path.join(rd, "cm_severity.png"),
    )
    plot_roc_curves(results,  os.path.join(rd, "roc_curves.png"))
    plot_calibration(results, os.path.join(rd, "calibration.png"))

    if args.gradcam:
        gc = GradCAM(model)
        gc.visualize_batch(
            test_loader, device, n=8,
            save_dir=os.path.join(rd, "gradcam"),
            tasks=["binary", "ordinal1", "ordinal2"],
        )
        gc.remove_hooks()

    if args.tsne:
        tsne_feature_plot(
            model, test_loader, device,
            save_path=os.path.join(rd, "tsne_features.png"),
        )

    print(f"\n✓ All outputs saved to: {rd}/")


if __name__ == "__main__":
    main()
