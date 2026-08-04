"""
Soil pH prediction from images — entry point.

Usage:
  uv run python main.py --train              # train on all labeled photos
  uv run python main.py --eval               # evaluate best checkpoint on held-out set
  uv run python main.py --train --eval       # train then immediately evaluate
  uv run python main.py --predict img.jpeg   # predict pH for a single image

Expects:
  downloaded_photos/   — soil images (named  f<farm>-<sample>.jpeg  for labeled ones)
  raw_data_need_to_map_sample.csv  — ground truth pH values
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from dataset import build_dataset
from train import train
from evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PHOTO_DIR = Path("downloaded_photos")
CSV_PATH = Path("Gambia_Raw_Data_2024.xlsx")
CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"

# ── train/test split is done inside train.py (farm-grouped) ──────────────────
# Farms 1-9 collected 3/1/25; farms 10-11 collected 4/1/25 (different day).
# Set HOLDOUT_FARMS to the later-day farms for an honest cross-day eval:
HOLDOUT_FARMS: list[int] = [10, 11]  # collected 4/1/25 vs farms 1-9 on 3/1/25


def cmd_train(args):
    df = build_dataset(PHOTO_DIR, CSV_PATH)
    if len(df) == 0:
        sys.exit(
            "No labeled images found. Rename photos as  f<farm>-<sample>.jpeg  "
            "(e.g. f6-1.jpeg) so they can be matched to CSV sample IDs."
        )
    log.info("Total labeled images: %d  |  unique samples: %d  |  pH range: %.2f–%.2f",
             len(df), df["sample_id"].nunique(), df["ph"].min(), df["ph"].max())

    train_df = df[~df["farm"].isin(HOLDOUT_FARMS)] if HOLDOUT_FARMS else df

    train(
        train_df,
        output_dir=CHECKPOINT_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=args.freeze,
    )


def cmd_eval(args):
    if not BEST_CHECKPOINT.exists():
        sys.exit(f"No checkpoint found at {BEST_CHECKPOINT}. Run --train first.")

    df = build_dataset(PHOTO_DIR, CSV_PATH)
    eval_df = df[df["farm"].isin(HOLDOUT_FARMS)] if HOLDOUT_FARMS else df
    evaluate(eval_df, BEST_CHECKPOINT)


def cmd_predict(image_path: str):
    if not BEST_CHECKPOINT.exists():
        sys.exit(f"No checkpoint found at {BEST_CHECKPOINT}. Run --train first.")

    from model import SoilpHModel
    from train import VAL_TRANSFORMS
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    farm_ids = ckpt.get("farm_ids")
    n_extra = len(farm_ids) if farm_ids else 0
    model = SoilpHModel(n_extra=n_extra)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # predict without farm feature (unknown farm at inference)
        ph = model(tensor, None).item()
    print(f"Predicted pH: {ph:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Soil pH prediction")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", metavar="IMAGE", help="Path to a single image")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", action="store_true", help="Freeze EfficientNet backbone")
    args = parser.parse_args()

    if args.predict:
        cmd_predict(args.predict)
    elif args.train or args.eval:
        if args.train:
            cmd_train(args)
        if args.eval:
            cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
