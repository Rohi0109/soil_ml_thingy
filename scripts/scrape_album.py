"""
Scrape a public Google Photos shared album and download all images.
Usage: uv run python scrape_album.py
"""

import re
import sys
import time
from pathlib import Path
import browser_cookie3
import requests

ALBUM_URL = (
    "https://photos.google.com/share/"
    "AF1QipO-k70TUtxzBeuX-6qabWfynCzx8MUr05Tau2epqWnOdnTLO_7XGIht-tcqCsDefQ"
    "?key=WEQ5eDBJTTllT2U5eVNIUDZUZ3ViZGJvM3FwRmZB"
)
OUTPUT_DIR = Path("downloaded_photos")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# /pw/ URLs are actual photos; strip trailing size params (=w600-h315-... etc.)
LH3_PATTERN = re.compile(r'(https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_\-]+)(?:=[\w\-]+)?')


def fetch_page(url: str, session: requests.Session) -> str:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_image_urls(html: str) -> list[str]:
    """Pull unique /pw/ photo base URLs; these are actual photos, not avatars."""
    raw = LH3_PATTERN.findall(html)
    return list(dict.fromkeys(raw))  # deduplicate, preserve order


def download_image(base_url: str, idx: int, dest_dir: Path, session: requests.Session) -> bool:
    # =d  → original download; =w4096-h4096 would be a large preview fallback
    url = base_url + "=d"
    dest = dest_dir / f"photo_{idx:04d}.jpg"
    if dest.exists():
        return True
    try:
        with session.get(url, headers=HEADERS, timeout=60, stream=True) as r:
            r.raise_for_status()
            # honour the content-type for extension
            ct = r.headers.get("content-type", "image/jpeg")
            ext = ct.split("/")[-1].split(";")[0].strip() or "jpg"
            dest = dest_dir / f"photo_{idx:04d}.{ext}"
            if dest.exists():
                return True
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        print(f"  WARN [{idx}]: {exc}")
        return False


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()

    # Load Google cookies straight from Chrome — make sure Chrome is closed or at least
    # that you're logged into photos.google.com in Chrome.
    print("Loading Google cookies from Chrome …")
    try:
        cookies = browser_cookie3.chrome(domain_name=".google.com")
        session.cookies.update(cookies)
    except Exception as exc:
        print(f"  WARN: could not read Chrome cookies ({exc}). Trying without auth …")

    print("Fetching album page …")
    try:
        html = fetch_page(ALBUM_URL, session)
    except requests.HTTPError as e:
        sys.exit(f"Failed to load album page: {e}")

    urls = extract_image_urls(html)
    if not urls:
        sys.exit(
            "No image URLs found in the page.\n"
            "The album may require a Google login. "
            "Try exporting cookies from Chrome with the 'Cookie-Editor' extension "
            "and saving as cookies.txt, then re-run with --cookies support."
        )

    print(f"Found {len(urls)} image(s). Downloading to ./{OUTPUT_DIR}/ …")
    downloaded = failed = skipped = 0
    for i, url in enumerate(urls, 1):
        dest_check = OUTPUT_DIR / f"photo_{i:04d}.jpg"
        if dest_check.exists():
            skipped += 1
            continue
        ok = download_image(url, i, OUTPUT_DIR, session)
        if ok:
            downloaded += 1
            print(f"  [{i:>4}/{len(urls)}] saved")
        else:
            failed += 1
        time.sleep(0.05)

    print(f"\nDone. {downloaded} downloaded, {skipped} skipped, {failed} failed.")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
