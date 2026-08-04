"""Training loop for soil pH model."""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partially-downloaded photos
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

from model import SoilpHModel

log = logging.getLogger(__name__)


# ── per-image normalisation ───────────────────────────────────────────────────

class PerImageNorm:
    """
    Normalise each RGB channel to zero mean, unit std per image.

    This removes per-photo brightness/exposure variation (the main confound
    learned by the old model) while keeping relative colour information intact
    and staying compatible with the EfficientNet pretrained feature space.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)   # (3,1,1)
        std = tensor.std(dim=[1, 2], keepdim=True)     # (3,1,1)
        return (tensor - mean) / (std + 1e-6)


# ── augmentation ─────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),   # RGB [0,1]
    PerImageNorm(),          # remove per-photo exposure bias
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    PerImageNorm(),
])


# ── dataset ───────────────────────────────────────────────────────────────────

class SoilDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, farm_ids: list | None = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        # Build consistent farm → index mapping across train and val
        if farm_ids is None:
            farm_ids = sorted(df["farm"].unique())
        self.farm_idx = {f: i for i, f in enumerate(farm_ids)}
        self.n_farms = len(farm_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        farm_ohe = torch.zeros(self.n_farms)
        farm_ohe[self.farm_idx.get(row["farm"], 0)] = 1.0

        return img, farm_ohe, torch.tensor(row["ph"], dtype=torch.float32)


def train(
    df: pd.DataFrame,
    output_dir: str | Path = "checkpoints",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_fraction: float = 0.2,
    freeze_backbone: bool = False,
    use_farm_features: bool = True,
    device: str = "auto",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    # Farm-grouped train/val split — avoids same-farm leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["farm"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    log.info("Train: %d images | Val: %d images", len(train_df), len(val_df))
    log.info("Val farms: %s", sorted(val_df["farm"].unique()))

    # Consistent one-hot encoding across both splits
    all_farm_ids = sorted(df["farm"].unique())
    n_farms = len(all_farm_ids) if use_farm_features else 0

    train_ds = SoilDataset(train_df, TRAIN_TRANSFORMS, farm_ids=all_farm_ids)
    val_ds = SoilDataset(val_df, VAL_TRANSFORMS, farm_ids=all_farm_ids)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SoilpHModel(freeze_backbone=freeze_backbone, n_extra=n_farms).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.5)

    best_val_mae = float("inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, farm_ohe, labels in train_loader:
            imgs = imgs.to(device)
            farm_ohe = farm_ohe.to(device) if use_farm_features else None
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs, farm_ohe)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, farm_ohe, labels in val_loader:
                imgs = imgs.to(device)
                farm_ohe = farm_ohe.to(device) if use_farm_features else None
                val_preds.extend(model(imgs, farm_ohe).cpu().numpy())
                val_labels.extend(labels.numpy())
        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_labels))))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"state_dict": model.state_dict(), "farm_ids": all_farm_ids, "n_extra": n_farms}, best_path)

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train_loss=%.4f  val_MAE=%.4f  best=%.4f",
                     epoch, epochs, train_loss, val_mae, best_val_mae)

    log.info("Training complete. Best val MAE=%.4f  checkpoint=%s", best_val_mae, best_path)
    return best_path
