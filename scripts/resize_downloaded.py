"""
Re-download all existing photos in downloaded_photos/ at a smaller resolution,
replacing the originals. Uses the same lh3.googleusercontent.com URLs but with
=s512 instead of =d.

Since the original URLs are not stored locally, this uses Pillow to resize
in-place instead — no re-download needed.

Usage:
    uv run python scripts/resize_downloaded.py               # resize to 512px max
    uv run python scripts/resize_downloaded.py --size 1024   # resize to 1024px max
    uv run python scripts/resize_downloaded.py --dry-run     # preview space savings only
"""

import argparse
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


PHOTO_DIR = Path("downloaded_photos")


def resize_photo(path: Path, max_size: int) -> tuple[int, int]:
    """Resize image to max_size on its longest dimension. Returns (before_bytes, after_bytes)."""
    before = path.stat().st_size
    img = Image.open(path)
    w, h = img.size
    if max(w, h) <= max_size:
        return before, before  # already small enough

    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(path, "JPEG", quality=85, optimize=True)
    after = path.stat().st_size
    return before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512,
                        help="Max dimension in pixels (default: 512)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show how many files would be resized without changing anything")
    args = parser.parse_args()

    all_photos = sorted(PHOTO_DIR.glob("*.jpeg")) + sorted(PHOTO_DIR.glob("*.jpg"))
    needs_resize = []
    for p in all_photos:
        try:
            img = Image.open(p)
            if max(img.size) > args.size:
                needs_resize.append(p)
        except Exception:
            pass

    total_before = sum(p.stat().st_size for p in needs_resize)
    print(f"Photos to resize: {len(needs_resize)} / {len(all_photos)}")
    print(f"Current size of those files: {total_before / 1e9:.2f} GB")

    if args.dry_run:
        print("(dry run — no changes made)")
        return

    print(f"Resizing to max {args.size}px …")
    total_saved = 0
    for i, p in enumerate(needs_resize, 1):
        before, after = resize_photo(p, args.size)
        total_saved += before - after
        if i % 500 == 0 or i == len(needs_resize):
            print(f"  [{i}/{len(needs_resize)}]  saved {total_saved / 1e9:.2f} GB so far …")

    print(f"\nDone. Freed {total_saved / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
