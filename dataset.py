"""
Parse image filenames → Sample ID → pH label, return a ready-to-use DataFrame.

Filename conventions handled:
  f6-1.jpeg        → F6-1
  f6-1(3).jpeg     → F6-1  (duplicate shot, same label)
  f-61(5).jpeg     → F6-1  (typo variant)
"""

import re
import csv
import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Matches  f<farm>-<plot>  with optional (n) suffix, case-insensitive
_SAMPLE_RE = re.compile(r'f(\d+)-(\d+)', re.IGNORECASE)
# Handles the typo variant  f-<farm><plot>  e.g. f-61  → F6-1
_TYPO_RE = re.compile(r'f-(\d)(\d+)', re.IGNORECASE)


def filename_to_sample_id(name: str) -> str | None:
    """Return normalised sample ID (e.g. 'F6-1') or None if not parseable."""
    stem = Path(name).stem
    m = _SAMPLE_RE.search(stem)
    if m:
        return f"F{m.group(1)}-{m.group(2)}"
    m = _TYPO_RE.search(stem)
    if m:
        return f"F{m.group(1)}-{m.group(2)}"
    return None


def load_ph_csv(data_path: str | Path) -> dict[str, float]:
    """Return {sample_id: avg_pH} from the Excel or CSV ground-truth file.

    For Excel files, the Average is recomputed from the raw A/B/C/D sensor columns
    rather than trusting the pre-calculated Average column.  Duplicate sample IDs
    are averaged with a warning.
    """
    data_path = Path(data_path)
    raw: dict[str, list[float]] = {}

    if data_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(data_path, sheet_name="Raw Data")
        df.columns = df.columns.str.strip()
        sensor_cols = [c for c in df.columns if c.startswith("pH")]
        for _, row in df.iterrows():
            sid = str(row.get("Sample ID", "")).strip().upper()
            if not sid or sid == "NAN":
                continue
            vals = pd.to_numeric(pd.Series([row.get(c) for c in sensor_cols]), errors="coerce")
            avg = float(vals.mean(skipna=True))
            if not np.isnan(avg):
                raw.setdefault(sid, []).append(avg)
    else:
        with open(data_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("Sample ID", "").strip().upper()
                avg = row.get("Average", "").strip()
                if sid and avg:
                    try:
                        raw.setdefault(sid, []).append(float(avg))
                    except ValueError:
                        pass

    ph_map = {}
    for sid, vals in raw.items():
        if len(vals) > 1:
            log.warning("Duplicate sample ID %s — averaging %d measurements: %s → %.4f",
                        sid, len(vals), vals, sum(vals) / len(vals))
        ph_map[sid] = sum(vals) / len(vals)

    log.info("Loaded %d pH labels from %s", len(ph_map), data_path)
    return ph_map


def build_dataset(
    photo_dir: str | Path,
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [image_path, sample_id, ph, farm].
    Only images that can be matched to a pH label are included.
    """
    ph_map = load_ph_csv(csv_path)
    photo_dir = Path(photo_dir)

    rows = []
    unmatched = []
    for img in sorted(photo_dir.glob("*.jpeg")):
        sid = filename_to_sample_id(img.name)
        if sid is None:
            unmatched.append(img.name)
            continue
        sid_upper = sid.upper()
        if sid_upper not in ph_map:
            log.debug("Sample ID %s not in CSV — skipping %s", sid_upper, img.name)
            continue
        farm = int(sid_upper.split("-")[0][1:])
        rows.append({"image_path": str(img), "sample_id": sid_upper, "ph": ph_map[sid_upper], "farm": farm})

    df = pd.DataFrame(rows)
    log.info(
        "Dataset: %d labeled images across %d unique samples (%d files unmatched)",
        len(df), df["sample_id"].nunique() if len(df) else 0, len(unmatched),
    )
    if unmatched:
        log.debug("Unmatched files: %s", unmatched[:5])
    return df
