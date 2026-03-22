"""dataset.py — PyTorch Dataset and data transforms for the OASIS dementia MRI data."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """Augmented transform for the training split."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation and test splits."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DementiaOrdinalDataset(Dataset):
    """
    Dataset for multi-task dementia classification from T1-weighted MRI images.

    Each sample returns:
        img     (C, H, W) tensor
        y_bin   binary label (0 = Non-Demented, 1 = Demented)
        sev_ge1 ordinal threshold: P(severity >= 1), i.e. any dementia
        sev_ge2 ordinal threshold: P(severity >= 2), i.e. Mild or Moderate
        y_sev   integer severity class (0 / 1 / 2)
        subj_id subject identifier string
    """

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.df = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y_sev = int(row["y_sev"])

        return (
            img,
            torch.tensor(row["y_bin"],            dtype=torch.float32),
            torch.tensor(1.0 if y_sev >= 1 else 0.0, dtype=torch.float32),
            torch.tensor(1.0 if y_sev >= 2 else 0.0, dtype=torch.float32),
            torch.tensor(y_sev,                   dtype=torch.long),
            row["subject_id"],
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(
    train_df: "pd.DataFrame",
    val_df:   "pd.DataFrame",
    test_df:  "pd.DataFrame",
    batch_size: int = 32,
    img_size:   int = 224,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> tuple:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = DementiaOrdinalDataset(train_df, get_train_transform(img_size))
    val_ds   = DementiaOrdinalDataset(val_df,   get_eval_transform(img_size))
    test_ds  = DementiaOrdinalDataset(test_df,  get_eval_transform(img_size))

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )
