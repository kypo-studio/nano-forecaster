"""Download ETTh1 from the official Autoformer repository mirror."""
from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def download(destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Dataset already present: {destination}")
        return destination
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"ETTh1 download failed: {exc}\nDownload {URL} manually to {destination}"
        ) from exc
    if not payload.startswith(b"date,"):
        raise SystemExit("Downloaded file is not a valid ETTh1 CSV")
    destination.write_bytes(payload)
    print(f"Downloaded {len(payload)} bytes, sha256={hashlib.sha256(payload).hexdigest()}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.force)
