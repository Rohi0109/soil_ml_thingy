"""
Scrape a public Google Photos shared album and download all images.
Usage: uv run python scrape_album.py
"""

import re
import time
from pathlib import Path

import browser_cookie3
import requests
from playwright.sync_api import sync_playwright

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

LH3_PATTERN = re.compile(r'(https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_\-]+)(?:=[\w\-]+)?')


def collect_all_urls() -> list[str]:
    print("Loading Google cookies from Chrome …")
    raw_cookies = browser_cookie3.chrome(domain_name=".google.com")
    pw_cookies = [
        {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path, "secure": bool(c.secure)}
        for c in raw_cookies if c.value
    ]

    seen: dict[str, None] = {}

    def extract_from_text(text: str):
        for m in LH3_PATTERN.finditer(text):
            seen[m.group(1)] = None

    print("Opening album and intercepting API responses …")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.add_cookies(pw_cookies)
        page = context.new_page()

        def on_response(response):
            ct = response.headers.get("content-type", "")
            if "json" in ct or "javascript" in ct or "text" in ct:
                try:
                    extract_from_text(response.text())
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(ALBUM_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(3000)  # let initial photos render
        # Also scan the full initial page HTML
        extract_from_text(page.content())

        prev_count = 0
        stall_rounds = 0
        while stall_rounds < 8:
            page.evaluate("window.scrollBy(0, window.innerHeight * 3)")
            page.wait_for_timeout(2000)
            extract_from_text(page.content())
            if len(seen) == prev_count:
                stall_rounds += 1
            else:
                stall_rounds = 0
            prev_count = len(seen)
            print(f"  {len(seen)} photo URLs captured …", end="\r")

        browser.close()

    return list(seen.keys())


def download_image(base_url: str, idx: int, dest_dir: Path, session: requests.Session) -> bool:
    url = base_url + "=d"
    dest = dest_dir / f"photo_{idx:04d}.jpg"
    if dest.exists():
        return True
    try:
        with session.get(url, headers=HEADERS, timeout=60, stream=True) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "image/jpeg")
            if not ct.startswith("image/"):
                return False  # skip videos and other non-image media
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

    urls = collect_all_urls()
    print(f"\nDone — {len(urls)} unique photo URLs captured.")

    session = requests.Session()
    try:
        cookies = browser_cookie3.chrome(domain_name=".google.com")
        session.cookies.update(cookies)
    except Exception:
        pass

    print(f"Downloading {len(urls)} images to ./{OUTPUT_DIR}/ …")
    for i, url in enumerate(urls, 1):
        ok = download_image(url, i, OUTPUT_DIR, session)
        if ok:
            print(f"  [{i:>5}/{len(urls)}] saved")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
