"""
Rename photo_XXXX.jpeg files to their sample ID label.

Reads the label card via OCR (0° and 180°) and renames:
  photo_0076.jpeg  →  f6-2.jpeg
  photo_0077.jpeg  →  f6-2(1).jpeg   (already a f6-2.jpeg)

Already-named f-* files are left untouched but counted to avoid
duplicate numbering.
"""

import re
import sys
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

PHOTO_DIR = Path("downloaded_photos")
SAMPLE_PATTERN = re.compile(r'\bF\d{1,2}-\d{1,2}\b', re.IGNORECASE)


def ocr_label(reader: easyocr.Reader, path: Path) -> str | None:
    img = Image.open(path)
    img.thumbnail((1024, 1024), Image.LANCZOS)  # downscale for speed
    for angle in (0, 180):
        rotated = img.rotate(angle, expand=True)
        texts = reader.readtext(np.array(rotated), detail=0, paragraph=False)
        ids = SAMPLE_PATTERN.findall(" ".join(texts))
        if ids:
            return ids[0].lower()
    return None


def next_name(label: str, existing: set[str]) -> str:
    """Return the next available filename for this label."""
    base = f"{label}.jpeg"
    if base not in existing:
        return base
    n = 1
    while f"{label}({n}).jpeg" in existing:
        n += 1
    return f"{label}({n}).jpeg"


def main() -> None:
    numbered = sorted(PHOTO_DIR.glob("photo_*.jpeg"))
    if not numbered:
        sys.exit("No photo_XXXX.jpeg files found.")

    # Seed existing names from already-renamed f-* files
    existing: set[str] = {f.name for f in PHOTO_DIR.glob("f*.jpeg")}

    print("Loading EasyOCR …", flush=True)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print(f"Ready. Processing {len(numbered)} files …\n", flush=True)

    failed = []
    renamed = 0

    for i, src in enumerate(numbered, 1):
        if not src.exists():
            print(f"  [{i}/{len(numbered)}] {src.name}  →  SKIPPED (gone)", flush=True)
            continue
        label = ocr_label(reader, src)
        if not label:
            failed.append(src.name)
            print(f"  [{i}/{len(numbered)}] {src.name}  →  COULD NOT READ LABEL", flush=True)
            continue

        dest_name = next_name(label, existing)
        dest = PHOTO_DIR / dest_name
        src.rename(dest)
        existing.add(dest_name)
        renamed += 1
        print(f"  [{i}/{len(numbered)}] {src.name}  →  {dest_name}", flush=True)

    print(f"\nRenamed : {renamed}")
    print(f"Failed  : {len(failed)}")
    if failed:
        print("These need manual renaming:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
