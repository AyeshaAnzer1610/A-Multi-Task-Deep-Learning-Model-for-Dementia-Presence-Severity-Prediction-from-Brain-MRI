"""
scripts/train.py  —  Command-line training entry point

Usage:
    python scripts/train.py --data_root /path/to/AlzheimerDataset
    python scripts/train.py --data_root /path/to/data --max_epochs 30
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, yaml, torch

from src.dataset import build_dataloaders
from src.model   import MTCNNDementia
from src.train   import train
from src.utils   import set_seed, get_device, plot_training_history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--data_root",   default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--max_epochs",  type=int, default=None)
    p.add_argument("--seed",        type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:  cfg["data_root"]  = args.data_root
    if args.max_epochs: cfg["max_epochs"] = args.max_epochs
    if args.seed:       cfg["seed"]       = args.seed

    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nData root  : {cfg['data_root']}")
    print(f"Results dir: {args.results_dir}")
    print(f"Max epochs : {cfg['max_epochs']}")

    train_loader, val_loader, test_loader, val_samp, test_samp = \
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
        pretrained   = True,
    ).to(device)
    model.parameter_breakdown()

    train_samples = train_loader.dataset.samples
    history = train(cfg, model, train_loader, val_loader,
                    train_samples, val_samp, device, args.results_dir)

    plot_training_history(
        history,
        save_path=os.path.join(args.results_dir, "training_history.png")
    )
    print("\n✓ Done. Run evaluate.py next.")


if __name__ == "__main__":
    main()
