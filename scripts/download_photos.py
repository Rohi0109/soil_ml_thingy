"""
Download all images (no videos) from a Google Photos album.

Setup:
  1. Go to https://console.cloud.google.com/
  2. Create a project → enable "Photos Library API"
  3. Create OAuth2 Desktop credentials → download as client_secret.json
  4. Run: uv run python download_photos.py
     - A browser window opens once for auth; token is cached in token.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"
OUTPUT_DIR = Path("downloaded_photos")
BASE_API = "https://photoslibrary.googleapis.com/v1"
PAGE_SIZE = 100  # max allowed by the API


def get_credentials() -> Credentials:
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CLIENT_SECRET_FILE).exists():
                sys.exit(
                    f"Missing {CLIENT_SECRET_FILE}. "
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(TOKEN_FILE).write_text(creds.to_json())
    return creds


def get_headers(creds: Credentials) -> dict:
    return {"Authorization": f"Bearer {creds.token}"}


def find_album(creds: Credentials, name_fragment: str) -> str | None:
    """Return the albumId whose title contains name_fragment (case-insensitive)."""
    for endpoint in ("/albums", "/sharedAlbums"):
        page_token = None
        while True:
            params = {"pageSize": 50}
            if page_token:
                params["pageToken"] = page_token
            resp = requests.get(
                BASE_API + endpoint,
                headers=get_headers(creds),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            key = "albums" if endpoint == "/albums" else "sharedAlbums"
            for album in data.get(key, []):
                if name_fragment.lower() in album.get("title", "").lower():
                    print(f"Found album: \"{album['title']}\" (id={album['id']})")
                    return album["id"]
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return None


def iter_photos(creds: Credentials, album_id: str):
    """Yield media items that are photos (not videos)."""
    page_token = None
    while True:
        body: dict = {"albumId": album_id, "pageSize": PAGE_SIZE}
        if page_token:
            body["pageToken"] = page_token
        resp = requests.post(
            BASE_API + "/mediaItems:search",
            headers=get_headers(creds),
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("mediaItems", []):
            mime = item.get("mimeType", "")
            if mime.startswith("image/"):
                yield item
        page_token = data.get("nextPageToken")
        if not page_token:
            break


def safe_filename(filename: str) -> str:
    return re.sub(r'[^\w.\-]', '_', filename)


def download_item(item: dict, dest_dir: Path, session: requests.Session) -> bool:
    """Download a single media item; return True on success."""
    filename = safe_filename(item.get("filename", item["id"] + ".jpg"))
    dest = dest_dir / filename

    # skip if already downloaded (resume support)
    if dest.exists():
        return True

    # =d suffix requests the full original download
    url = item["baseUrl"] + "=d"
    try:
        with session.get(url, timeout=60, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        print(f"  WARN: failed to download {filename}: {exc}")
        return False


def main():
    # --- find album ---
    creds = get_credentials()
    album_name = input("Enter part of the album name to search for [Gambia]: ").strip()
    if not album_name:
        album_name = "Gambia"

    print(f"Searching for album matching '{album_name}' …")
    album_id = find_album(creds, album_name)
    if not album_id:
        sys.exit(f"No album found matching '{album_name}'. Check spelling or sharing settings.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- iterate and download ---
    session = requests.Session()
    total = downloaded = skipped = failed = 0

    print(f"Downloading images to ./{OUTPUT_DIR}/ …")
    for item in iter_photos(creds, album_id):
        total += 1
        filename = safe_filename(item.get("filename", item["id"] + ".jpg"))
        dest = OUTPUT_DIR / filename
        if dest.exists():
            skipped += 1
            continue

        # refresh token if needed before each batch
        if not creds.valid:
            creds.refresh(Request())
            session.headers.update({"Authorization": f"Bearer {creds.token}"})

        ok = download_item(item, OUTPUT_DIR, session)
        if ok:
            downloaded += 1
            print(f"  [{total:>4}] {filename}")
        else:
            failed += 1

        # polite rate-limit: 1 req/sec is well within quota
        time.sleep(0.1)

    print(f"\nDone. {downloaded} downloaded, {skipped} skipped (already exist), {failed} failed.")
    print(f"Images saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
