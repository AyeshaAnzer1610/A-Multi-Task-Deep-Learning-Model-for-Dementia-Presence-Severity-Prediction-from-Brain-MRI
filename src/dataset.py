"""
src/dataset.py
Works with both naming styles:
  Kaggle      : NonDemented, VeryMildDemented, MildDemented, ModerateDemented
  HuggingFace : Non_Demented, Very_Mild_Demented, Mild_Demented, Moderate_Demented
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold

FOLDER_TO_ORDINAL = {
    # Kaggle style
    "NonDemented": 0, "VeryMildDemented": 1,
    "MildDemented": 2, "ModerateDemented": 2,
    # HuggingFace style
    "Non_Demented": 0, "Very_Mild_Demented": 1,
    "Mild_Demented": 2, "Moderate_Demented": 2,
    # lowercase fallback
    "nondemented": 0, "verymilddemented": 1,
    "milddemented": 2, "moderatedemented": 2,
}
ORDINAL_TO_BINARY = {0: 0, 1: 1, 2: 1}
ORDINAL_NAMES     = ["Non-Demented", "Very Mild", "Mild/Moderate"]
BINARY_NAMES      = ["Non-Demented", "Demented"]
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
IMG_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def build_transforms(image_size=224, augment=True):
    norm = [
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if augment:
        aug = [
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(12),
            T.RandomAffine(0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.08, contrast=0.08),
        ]
        return T.Compose(aug + norm)
    return T.Compose(norm)


def collect_samples(data_root, split="train"):
    GROUP_SIZE = 4
    samples    = []
    subj_ctr   = 0
    split_dir  = Path(data_root) / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {split_dir}\n"
            f"Check DATA_ROOT = '{data_root}'"
        )

    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        ord_label = FOLDER_TO_ORDINAL.get(folder.name) \
                 or FOLDER_TO_ORDINAL.get(folder.name.lower())
        if ord_label is None:
            print(f"  [SKIP] Unknown class folder: '{folder.name}'")
            continue

        images = sorted([
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])
        for i, img_path in enumerate(images):
            samples.append({
                "path":       img_path,
                "ordinal":    ord_label,
                "binary":     ORDINAL_TO_BINARY[ord_label],
                "subject_id": subj_ctr + (i // GROUP_SIZE),
            })
        subj_ctr += (len(images) + GROUP_SIZE - 1) // GROUP_SIZE

    if not samples:
        available = [d.name for d in split_dir.iterdir() if d.is_dir()]
        raise RuntimeError(
            f"No images found under {split_dir}\n"
            f"Sub-folders found: {available}"
        )

    print(f"  {split}: {len(samples)} images loaded")
    return samples


class DementiaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image":      img,
            "binary":     torch.tensor(s["binary"],     dtype=torch.float32),
            "ordinal":    torch.tensor(s["ordinal"],    dtype=torch.long),
            "subject_id": torch.tensor(s["subject_id"], dtype=torch.long),
            "path":       s["path"],
        }


def build_dataloaders(data_root, image_size=224, batch_size=32,
                      num_workers=2, val_fold=1, seed=42):
    """
    Returns: train_loader, val_loader, test_loader, val_samples, test_samples
    """
    print("\nLoading data from:", data_root)
    all_train = collect_samples(data_root, "train")
    test_samp = collect_samples(data_root, "test")

    # Stratified group split — no subject leakage
    ord_arr  = np.array([s["ordinal"]    for s in all_train])
    subj_arr = np.array([s["subject_id"] for s in all_train])
    sgkf     = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits   = list(sgkf.split(np.zeros(len(all_train)), ord_arr, subj_arr))
    tr_idx, va_idx = splits[val_fold % 3]

    train_samp = [all_train[i] for i in tr_idx]
    val_samp   = [all_train[i] for i in va_idx]

    # WeightedRandomSampler for class imbalance
    train_labels  = [s["ordinal"] for s in train_samp]
    class_counts  = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_w      = [float(class_weights[l]) for l in train_labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_tf = build_transforms(image_size, augment=True)
    eval_tf  = build_transforms(image_size, augment=False)
    pin      = torch.cuda.is_available()
    kw       = dict(num_workers=num_workers, pin_memory=pin)

    train_loader = DataLoader(DementiaDataset(train_samp, train_tf),
                              batch_size=batch_size, sampler=sampler, **kw)
    val_loader   = DataLoader(DementiaDataset(val_samp,   eval_tf),
                              batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(DementiaDataset(test_samp,  eval_tf),
                              batch_size=batch_size, shuffle=False, **kw)

    dist = np.bincount([s["ordinal"] for s in train_samp], minlength=3)
    print(f"  Train {len(train_samp)} | Val {len(val_samp)} | Test {len(test_samp)}")
    print(f"  Ordinal dist → ND:{dist[0]}  VM:{dist[1]}  M/M:{dist[2]}")
    return train_loader, val_loader, test_loader, val_samp, test_samp
