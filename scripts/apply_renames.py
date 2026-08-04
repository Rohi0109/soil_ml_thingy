"""
Apply vision-agent label results to photo_XXXX.jpeg files.
Reads the 6 batch label files and renames using f{farm}-{plot}(.jpeg convention.
"""
from pathlib import Path

PHOTO_DIR = Path("downloaded_photos")
SCRATCHPAD = Path("/private/tmp/claude-501/-Users-rnadgir-andys-thingy/1695e0aa-570f-4781-a8ff-c6da061f7923/scratchpad")
BATCH_FILES = [SCRATCHPAD / f"labels_batch{i}.txt" for i in range(1, 7)]

# Labels flagged as suspicious (not found in CSV)
SUSPICIOUS = {"f2-7", "f9-7", "f10-9", "f3-1", "f14-6"}


def next_name(label: str, existing: set) -> str:
    base = f"{label}.jpeg"
    if base not in existing:
        return base
    n = 1
    while f"{label}({n}).jpeg" in existing:
        n += 1
    return f"{label}({n}).jpeg"


def load_mapping() -> dict:
    mapping = {}
    for bf in BATCH_FILES:
        for line in bf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            photo, _, label = line.partition(": ")
            photo = photo.strip()
            label = label.strip()
            mapping[photo] = label
    return mapping


def main():
    mapping = load_mapping()
    print(f"Loaded {len(mapping)} entries from batch files")

    # Seed existing names from already-renamed f-* files
    existing = {f.name for f in PHOTO_DIR.glob("f*.jpeg")}
    print(f"Already-named files: {len(existing)}")

    skipped_unreadable = []
    skipped_suspicious = []
    skipped_missing = []
    renamed = []

    # Process in photo number order
    for photo_name in sorted(mapping.keys()):
        label_raw = mapping[photo_name]
        src = PHOTO_DIR / photo_name

        if not src.exists():
            skipped_missing.append(photo_name)
            continue

        if label_raw == "UNREADABLE":
            skipped_unreadable.append(photo_name)
            print(f"  SKIP  {photo_name}  (UNREADABLE)")
            continue

        label = label_raw.lower().replace(" ", "").replace("f", "f", 1)
        # Normalize: "F9-5" -> "f9-5"
        label = label_raw.lower()

        if label in SUSPICIOUS:
            skipped_suspicious.append((photo_name, label_raw))
            print(f"  FLAG  {photo_name}  →  {label_raw}  (not in CSV — skipping)")
            continue

        dest_name = next_name(label, existing)
        src.rename(PHOTO_DIR / dest_name)
        existing.add(dest_name)
        renamed.append((photo_name, dest_name))
        print(f"  OK    {photo_name}  →  {dest_name}")

    print(f"\n{'='*50}")
    print(f"Renamed:  {len(renamed)}")
    print(f"Missing:  {len(skipped_missing)}")
    print(f"Unreadable: {len(skipped_unreadable)}")
    if skipped_unreadable:
        for f in skipped_unreadable:
            print(f"  {f}")
    print(f"Suspicious labels (skipped): {len(skipped_suspicious)}")
    for photo, label in skipped_suspicious:
        print(f"  {photo}  →  {label}  (review manually)")


if __name__ == "__main__":
    main()
