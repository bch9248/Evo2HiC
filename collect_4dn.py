#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests

from config import hic_data_dir

TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024


def load_4dn_credentials() -> Optional[Tuple[str, str]]:
    candidates = [
        Path("keypairs.json"),
    ]

    keyfile = None
    for p in candidates:
        if p.exists():
            keyfile = p
            break

    if keyfile is None:
        return None

    data = json.loads(keyfile.read_text())
    if "default" in data and isinstance(data["default"], dict):
        key = data["default"].get("key")
        secret = data["default"].get("secret")
    else:
        key = data.get("key")
        secret = data.get("secret")

    if key and secret:
        return key, secret
    return None


def get_4dn_download_info(file_accession: str, session: requests.Session):
    drs_url = f"https://data.4dnucleome.org/ga4gh/drs/v1/objects/{file_accession}"
    resp = session.get(drs_url, timeout=TIMEOUT)
    resp.raise_for_status()
    meta = resp.json()

    access_methods = meta.get("access_methods", [])
    download_url = None
    for method in access_methods:
        access_url = method.get("access_url", {})
        url = access_url.get("url")
        if url and "@@download" in url:
            download_url = url
            break

    if not download_url:
        download_url = f"https://data.4dnucleome.org/{file_accession}/@@download"

    return download_url


def stream_download(
    url: str,
    out_path: Path,
    session: requests.Session,
    auth: Optional[Tuple[str, str]] = None,
) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with session.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True, auth=auth) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    tmp_path.replace(out_path)


def main() -> int:
    target_acc = "4DNFI2TK7L2F"
    out_dir = Path(hic_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    creds = load_4dn_credentials()
    if not creds or not creds[0] or not creds[1]:
        print("[ERROR] 4DN credentials not found in ./keypairs.json", file=sys.stderr)
        return 1

    out_path = out_dir / f"{target_acc}.hic"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {out_path}")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": "collect-hic/1.0"})

    try:
        url = get_4dn_download_info(target_acc, session)
        print(f"[DOWNLOADING] {target_acc}")
        stream_download(url, out_path, session=session, auth=creds)
        print(f"[OK] Saved to {out_path}")
        return 0
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())