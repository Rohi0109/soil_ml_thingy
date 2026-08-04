"""
Lab* colour feature extraction + classical ML models for soil pH prediction.
Runs leave-one-farm-out cross-validation and prints MAE/RMSE/R².
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import build_dataset

PHOTO_DIR = "downloaded_photos"
CSV_PATH = "raw_data_need_to_map_sample.csv"


def extract_lab_features(img_path: str) -> np.ndarray:
    """17 Lab* features from the top 70% of the image (excludes label card region)."""
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img)
    h = arr.shape[0]
    # crop bottom 30% to remove label card
    arr = arr[: int(h * 0.7)]

    lab = Image.fromarray(arr).convert("LAB")
    lab_arr = np.array(lab).astype(float)
    L, a, b = lab_arr[:, :, 0], lab_arr[:, :, 1], lab_arr[:, :, 2]

    feats = []
    for ch in (L, a, b):
        flat = ch.flatten()
        feats += [
            flat.mean(),
            flat.std(),
            np.percentile(flat, 10),
            np.percentile(flat, 50),
            np.percentile(flat, 90),
        ]
    # derived features
    l_mean = L.mean() + 1e-6
    feats.append(a.mean() / l_mean)   # redness proxy
    feats.append(1.0 - L.mean() / 255)  # darkness
    return np.array(feats, dtype=float)


def build_feature_matrix(df: pd.DataFrame):
    """Average per-image features to one row per unique sample."""
    records = []
    for img_path, grp in df.groupby("image_path"):
        try:
            feat = extract_lab_features(img_path)
        except Exception:
            continue
        row = grp.iloc[0]
        records.append({
            "sample_id": row["sample_id"],
            "farm": row["farm"],
            "ph": row["ph"],
            **{f"f{i}": v for i, v in enumerate(feat)},
        })
    return pd.DataFrame(records)


def evaluate_models(feat_df: pd.DataFrame):
    feature_cols = [c for c in feat_df.columns if c.startswith("f") and c != "farm"]
    X = feat_df[feature_cols].values
    y = feat_df["ph"].values
    groups = feat_df["farm"].values

    logo = LeaveOneGroupOut()

    models = {
        "Ridge":    Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Lasso":    Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01))]),
        "RF":       RandomForestRegressor(n_estimators=100, random_state=42),
        "GBM":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    print(f"\n{'Model':<10} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("-" * 34)

    # baseline: predict farm mean
    baseline_preds = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        baseline_preds[test_idx] = y[train_idx].mean()
    b_mae = mean_absolute_error(y, baseline_preds)
    print(f"{'Baseline':<10} {b_mae:>7.4f} {mean_squared_error(y, baseline_preds)**0.5:>7.4f} {r2_score(y, baseline_preds):>7.4f}")

    best_name, best_mae, best_preds = None, float("inf"), None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models.items():
            preds = np.zeros_like(y)
            for train_idx, test_idx in logo.split(X, y, groups):
                model.fit(X[train_idx], y[train_idx])
                preds[test_idx] = model.predict(X[test_idx])
            mae = mean_absolute_error(y, preds)
            rmse = mean_squared_error(y, preds) ** 0.5
            r2 = r2_score(y, preds)
            print(f"{name:<10} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}")
            if mae < best_mae:
                best_mae, best_name, best_preds = mae, name, preds

    # scatter plot for best model
    plt.figure(figsize=(5, 5))
    plt.scatter(y, best_preds, alpha=0.6)
    lo, hi = min(y.min(), best_preds.min()), max(y.max(), best_preds.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual pH")
    plt.ylabel("Predicted pH")
    plt.title(f"{best_name} (LOFO-CV) — MAE={best_mae:.3f}")
    plt.tight_layout()
    plt.savefig("color_model_scatter.png", dpi=150)
    print(f"\nScatter saved to color_model_scatter.png")

    # feature importance via RF
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop-10 feature importances (RF):")
    print(imp.head(10).to_string())

    imp.head(10).plot.bar(figsize=(8, 3))
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("Building dataset …")
    df = build_dataset(PHOTO_DIR, CSV_PATH)
    if df.empty:
        print("No labeled images found.")
        return

    print(f"  {len(df)} images, {df['sample_id'].nunique()} unique samples, {df['farm'].nunique()} farms")
    print("Extracting Lab* features …")
    feat_df = build_feature_matrix(df)
    print(f"  {len(feat_df)} samples with features")

    if feat_df["farm"].nunique() < 2:
        print("Need ≥2 farms for LOFO-CV.")
        return

    evaluate_models(feat_df)


if __name__ == "__main__":
    main()
