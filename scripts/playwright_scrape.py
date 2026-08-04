"""
Scroll through a Google Photos shared album with Playwright and download all images.

Uses network request interception so photos that scroll out of the virtual DOM
are still captured. Keeps scrolling until no new requests appear for PATIENCE
consecutive scroll attempts, then downloads everything collected.

Seen URLs are saved to <output_dir>/seen_urls.txt after each run so re-runs
only download photos that are genuinely new to the album.

Usage:
    uv run python scripts/playwright_scrape.py
    uv run python scripts/playwright_scrape.py --headless
    uv run python scripts/playwright_scrape.py --resolution s512   # ~100 KB/photo (default)
    uv run python scripts/playwright_scrape.py --resolution s1024  # ~350 KB/photo
    uv run python scripts/playwright_scrape.py --resolution d      # original (~4 MB/photo)
    uv run python scripts/playwright_scrape.py --output /Volumes/MyDrive/photos
"""

import argparse
import re
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

# ── config ────────────────────────────────────────────────────────────────────

ALBUM_URL = (
    "https://photos.google.com/share/"
    "AF1QipO-k70TUtxzBeuX-6qabWfynCzx8MUr05Tau2epqWnOdnTLO_7XGIht-tcqCsDefQ"
    "?key=WEQ5eDBJTTllT2U5eVNIUDZUZ3ViZGJvM3FwRmZB"
)
DEFAULT_OUTPUT_DIR = Path("downloaded_photos")
PATIENCE = 15      # stop after this many scrolls with no new URLs
SCROLL_WAIT = 3.0  # seconds to wait after each scroll for lazy-load

# Match lh3 /pw/ photo base URLs (strip everything from = onward)
LH3_RE = re.compile(r'(https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_\-]+)')

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
}

SEEN_URLS_FILE = "seen_urls.txt"


# ── download ──────────────────────────────────────────────────────────────────

def load_seen_urls(output_dir: Path) -> set[str]:
    """Load previously downloaded URLs so we don't re-download them."""
    p = output_dir / SEEN_URLS_FILE
    if not p.exists():
        return set()
    urls = {line.strip() for line in p.read_text().splitlines() if line.strip()}
    print(f"Loaded {len(urls)} previously seen URLs from {p}")
    return urls


def save_seen_urls(output_dir: Path, seen: set[str]) -> None:
    p = output_dir / SEEN_URLS_FILE
    p.write_text("\n".join(sorted(seen)))


def download(base_url: str, dest: Path, session: requests.Session, resolution: str = "s512") -> bool:
    # resolution examples: "d" = original, "s512" = max 512px, "s1024" = max 1024px
    url = base_url + "=" + resolution
    try:
        with session.get(url, headers=HEADERS, timeout=60, stream=True) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "image/jpeg")
            ext = ct.split("/")[-1].split(";")[0].strip() or "jpeg"
            actual = dest.with_suffix(f".{ext}")
            if actual.exists():
                return True
            with open(actual, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        print(f"  WARN: {exc}")
        return False


def next_index(output_dir: Path) -> int:
    existing = [
        int(m.group(1))
        for f in output_dir.glob("photo_*.j*")
        if (m := re.search(r'photo_(\d+)\.', f.name))
    ]
    return max(existing, default=0) + 1


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resolution", default="s512",
                        help="Google Photos size suffix: s512, s1024, d (original). Default: s512 (~100KB/photo)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: downloaded_photos). Use e.g. /Volumes/MyDrive/photos for external drive.")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N URLs (sorted alphabetically) — use when a previous run downloaded N photos but seen_urls.txt was not saved.")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load URLs already downloaded in previous runs — skip them this time
    already_seen = load_seen_urls(output_dir)
    seen: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()

        # Intercept every network request — capture /pw/ photo URLs as they load
        def on_request(request):
            m = LH3_RE.match(request.url)
            if m:
                seen.add(m.group(1))

        page.on("request", on_request)

        print("Opening album …")
        page.goto(ALBUM_URL, wait_until="networkidle", timeout=60_000)
        # Give the JS grid time to render before we start scrolling
        time.sleep(5)
        # Click the centre of the page to give it focus for scroll events
        page.mouse.click(720, 450)
        time.sleep(1)

        no_new = 0
        print("Scrolling to collect photo URLs (intercepting network requests) …")
        while no_new < PATIENCE:
            before = len(seen)
            # Simulate real mouse-wheel scroll — Google Photos lazy-load responds to this
            page.mouse.wheel(0, 3000)
            time.sleep(SCROLL_WAIT)
            gained = len(seen) - before

            if gained == 0:
                no_new += 1
                print(f"  {len(seen)} URLs  (no new × {no_new}/{PATIENCE})")
            else:
                no_new = 0
                print(f"  {len(seen)} URLs  (+{gained})")

        browser.close()

    new_urls = sorted(seen - already_seen)
    if args.skip:
        print(f"Skipping first {args.skip} URLs (--skip flag)")
        new_urls = new_urls[args.skip:]
    print(f"\nFound {len(seen)} total URLs  |  {len(new_urls)} to download\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    idx = next_index(output_dir)
    downloaded = failed = 0
    print(f"Resolution: ={args.resolution}  |  Output: {output_dir}\n")

    for i, url in enumerate(new_urls, 1):
        dest = output_dir / f"photo_{idx:04d}"
        ok = download(url, dest, session, resolution=args.resolution)
        if ok:
            downloaded += 1
            if downloaded % 100 == 0 or downloaded == 1:
                print(f"  [{i}/{len(new_urls)}] {downloaded} downloaded …")
        else:
            failed += 1
        idx += 1
        time.sleep(0.05)

    # Persist the full seen set (old + new) for future runs
    save_seen_urls(output_dir, already_seen | seen)

    print(f"\nDone.  {downloaded} downloaded  |  {failed} failed")
    print(f"Total files: {sum(1 for _ in output_dir.glob('photo_*.j*'))}")


if __name__ == "__main__":
    main()
