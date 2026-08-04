"""Evaluate a trained model: MAE, RMSE, R², baseline comparison, scatter plot."""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from model import SoilpHModel
from train import SoilDataset, VAL_TRANSFORMS

log = logging.getLogger(__name__)


def evaluate(
    df,
    checkpoint: str | Path,
    device: str = "auto",
    plot_path: str | Path | None = "eval_scatter.png",
) -> dict:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    farm_ids = ckpt.get("farm_ids")
    n_extra = ckpt.get("n_extra", len(farm_ids) if farm_ids else 0)
    model = SoilpHModel(n_extra=n_extra)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    ds = SoilDataset(df, VAL_TRANSFORMS, farm_ids=farm_ids)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, farm_ohe, labels in loader:
            extra = farm_ohe.to(device) if n_extra > 0 else None
            all_preds.extend(model(imgs.to(device), extra).cpu().numpy())
            all_labels.extend(labels.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    mae = mean_absolute_error(labels, preds)
    rmse = mean_squared_error(labels, preds) ** 0.5
    r2 = r2_score(labels, preds)
    baseline_mae = mean_absolute_error(labels, np.full_like(labels, labels.mean()))

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "baseline_MAE": baseline_mae, "n": len(labels)}

    log.info("─" * 40)
    log.info("Eval on %d samples", len(labels))
    log.info("  MAE          : %.4f pH units", mae)
    log.info("  RMSE         : %.4f pH units", rmse)
    log.info("  R²           : %.4f", r2)
    log.info("  Baseline MAE : %.4f (predict mean)", baseline_mae)
    log.info("  Improvement  : %.1f%%", 100 * (1 - mae / baseline_mae))
    log.info("─" * 40)

    if plot_path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(labels, preds, alpha=0.7, edgecolors="k", linewidths=0.4)
        lims = [min(labels.min(), preds.min()) - 0.2, max(labels.max(), preds.max()) + 0.2]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Measured pH")
        ax.set_ylabel("Predicted pH")
        ax.set_title(f"Soil pH prediction  (MAE={mae:.3f}, R²={r2:.3f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        log.info("Scatter plot saved to %s", plot_path)

    return metrics
