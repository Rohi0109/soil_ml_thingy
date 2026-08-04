"""
Read soil sample label cards from photo_*.jpeg files using Claude vision.

Outputs one line per photo:
    photo_0123.jpeg: F9-5
    photo_0124.jpeg: UNREADABLE

Usage:
    uv run python scripts/label_photos.py --files-list batch.txt --output labels.txt
    uv run python scripts/label_photos.py --start 0 --end 2000 --output labels_0.txt

Results are written after every photo so partial runs are resumable.
Re-running with the same --output file skips already-processed photos.
"""

import argparse
import base64
import re
import sys
from pathlib import Path

import anthropic

PHOTO_DIR = Path("downloaded_photos")
SAMPLE_RE = re.compile(r'\bF\s*(\d{1,2})\s*[-–]\s*(\d{1,2})\b', re.IGNORECASE)

PROMPT = (
    "Look at this soil sample photo. Find the white label card showing the sample ID "
    "in the format F<number>-<number> (e.g. F9-5, F12-3, F1-10). "
    "Reply with ONLY the sample ID (e.g. 'F9-5'). "
    "If you cannot read a clear label card, reply with exactly 'UNREADABLE'."
)


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def read_label(client: anthropic.Anthropic, path: Path) -> str:
    try:
        img_b64 = encode_image(path)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        raw = msg.content[0].text.strip()
        m = SAMPLE_RE.search(raw)
        if m:
            return f"F{m.group(1)}-{m.group(2)}"
        return "UNREADABLE"
    except Exception as e:
        print(f"  ERROR on {path.name}: {e}", file=sys.stderr)
        return "UNREADABLE"


def load_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    for line in output_path.read_text().splitlines():
        if ": " in line:
            done.add(line.split(": ")[0].strip())
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-list", help="Text file with one photo filename per line")
    parser.add_argument("--start", type=int, default=None, help="Start index into sorted photo list")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) into sorted photo list")
    parser.add_argument("--output", required=True, help="Output label file path")
    args = parser.parse_args()

    all_photos = sorted(PHOTO_DIR.glob("photo_*.jpeg"))

    if args.files_list:
        names = Path(args.files_list).read_text().splitlines()
        photos = [PHOTO_DIR / n.strip() for n in names if n.strip()]
    elif args.start is not None:
        end = args.end if args.end is not None else len(all_photos)
        photos = all_photos[args.start:end]
    else:
        photos = all_photos

    output_path = Path(args.output)
    done = load_done(output_path)
    photos = [p for p in photos if p.name not in done]

    print(f"Photos to label: {len(photos)}  (skipping {len(done)} already done)", flush=True)

    client = anthropic.Anthropic()

    with open(output_path, "a") as out:
        for i, photo in enumerate(photos, 1):
            label = read_label(client, photo)
            line = f"{photo.name}: {label}"
            out.write(line + "\n")
            out.flush()
            if i % 50 == 0 or i == 1:
                print(f"  [{i}/{len(photos)}] {photo.name} → {label}", flush=True)

    print(f"\nDone. Results in {output_path}", flush=True)


if __name__ == "__main__":
    main()
