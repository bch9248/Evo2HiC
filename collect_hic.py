#!/usr/bin/env python3
"""
collect_hic.py

Download Hi-C / Micro-C files listed in data/hic_index.tsv into hic_data_dir from config.py.

Expected columns in data/hic_index.tsv include:
- Hi-C Accession
- Dataset Accession
- Organism
- Biosource
- Experiment Type
- split
- Date
- Lab

Behavior:
- ENCODE rows: accession like ENCFF...
- 4DN rows: accession like 4DNFI...
- Existing files are skipped
- A download manifest is written to hic_data_dir/download_manifest.tsv

Requirements:
    pip install requests

Optional for 4DN:
    Create ./keypairs.json or ~/keypairs.json with:
    {
      "default": {
        "key": "YOUR_KEY",
        "secret": "YOUR_SECRET",
        "server": "https://data.4dnucleome.org"
      }
    }
"""

from __future__ import annotations

import csv
import json
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from config import hic_data_dir

INDEX_PATH = Path("data/hic_index.tsv")
TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024  # kept for compatibility, not used by external downloader
MANIFEST_NAME = "download_manifest.tsv"


def load_4dn_credentials() -> Optional[Tuple[str, str]]:
    """
    Load 4DN credentials from ./keypairs.json first, then ~/keypairs.json.

    Supported formats:
    1)
    {
      "default": {
        "key": "...",
        "secret": "...",
        "server": "https://data.4dnucleome.org"
      }
    }

    2)
    {
      "key": "...",
      "secret": "..."
    }
    """
    candidate_files = [
        Path("./keypairs.json"),
        Path.home() / "keypairs.json",
    ]

    keyfile = None
    for path in candidate_files:
        if path.exists():
            keyfile = path
            break

    if keyfile is None:
        return None

    try:
        data = json.loads(keyfile.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse {keyfile}: {e}") from e

    if "default" in data and isinstance(data["default"], dict):
        block = data["default"]
        key = block.get("key")
        secret = block.get("secret")
    else:
        key = data.get("key")
        secret = data.get("secret")

    if key and secret:
        return key, secret

    raise RuntimeError(f"Could not find 'key' and 'secret' in {keyfile}")


def detect_source(hic_accession: str) -> str:
    if hic_accession.startswith("ENCFF"):
        return "ENCODE"
    if hic_accession.startswith("4DNFI"):
        return "4DN"
    raise ValueError(f"Unknown Hi-C accession prefix: {hic_accession}")


def build_output_filename(row: Dict[str, str]) -> str:
    hic_acc = row["Hi-C Accession"].strip()
    return f"{hic_acc}.hic"


def get_encode_download_info(file_accession: str, session: requests.Session) -> Tuple[str, str]:
    """
    Query ENCODE JSON metadata first so we get the direct download URL.
    """
    meta_url = f"https://www.encodeproject.org/files/{file_accession}/?format=json"
    headers = {"accept": "application/json"}

    resp = session.get(meta_url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    meta = resp.json()

    href = meta.get("href")
    submitted_name = meta.get("submitted_file_name")
    title = meta.get("title")

    filename = None
    for candidate in [submitted_name, title]:
        if candidate:
            filename = Path(candidate).name
            break

    if not filename:
        filename = f"{file_accession}.hic"

    if href:
        if href.startswith("http://") or href.startswith("https://"):
            download_url = href
        else:
            download_url = f"https://www.encodeproject.org{href}"
    else:
        download_url = f"https://www.encodeproject.org/files/{file_accession}/@@download/{filename}"

    return download_url, filename


def get_4dn_download_info(file_accession: str, session: requests.Session) -> Tuple[str, str]:
    """
    4DN DRS/object endpoints expose the authenticated access URL.
    """
    drs_url = f"https://data.4dnucleome.org/ga4gh/drs/v1/objects/{file_accession}"
    resp = session.get(drs_url, timeout=TIMEOUT)
    resp.raise_for_status()
    meta = resp.json()

    filename = meta.get("name") or meta.get("description") or f"{file_accession}.hic"
    filename = Path(filename).name

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

    if "." not in filename:
        filename = f"{file_accession}.hic"

    return download_url, filename


def stream_download(
    url: str,
    out_path: Path,
    session: requests.Session,
    auth: Optional[Tuple[str, str]] = None,
) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    aria2c = shutil.which("aria2c")
    wget = shutil.which("wget")
    curl = shutil.which("curl")

    if aria2c:
        cmd = [
            aria2c,
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=10M",
            "--retry-wait=5",
            "--max-tries=10",
            "--timeout=60",
            "--allow-overwrite=true",
            "--file-allocation=none",
            "--out", tmp_path.name,
            "--dir", str(tmp_path.parent),
            url,
        ]
        if auth is not None:
            user, password = auth
            cmd.extend(["--http-user", user, "--http-passwd", password])

    elif wget:
        cmd = [
            wget,
            "-c",
            "-O", str(tmp_path),
            "--tries=10",
            "--timeout=60",
            url,
        ]
        if auth is not None:
            user, password = auth
            cmd.extend(["--user", user, "--password", password])

    elif curl:
        cmd = [
            curl,
            "-L",
            "-C", "-",
            "--retry", "10",
            "--retry-delay", "5",
            "--connect-timeout", "60",
            "-o", str(tmp_path),
            url,
        ]
        if auth is not None:
            user, password = auth
            cmd.extend(["-u", f"{user}:{password}"])

    else:
        raise RuntimeError(
            "No external downloader found. Install aria2c, wget, or curl."
        )

    subprocess.run(cmd, check=True)

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced empty file: {tmp_path}")

    tmp_path.replace(out_path)


def load_rows(index_path: Path) -> list[Dict[str, str]]:
    with open(index_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def append_manifest(manifest_path: Path, row: Dict[str, str]) -> None:
    exists = manifest_path.exists()
    fieldnames = [
        "Hi-C Accession",
        "Dataset Accession",
        "source",
        "status",
        "output_path",
        "download_url",
        "message",
    ]
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    out_dir = Path(hic_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not INDEX_PATH.exists():
        print(f"[ERROR] Missing index file: {INDEX_PATH}", file=sys.stderr)
        return 1

    rows = load_rows(INDEX_PATH)
    manifest_path = out_dir / MANIFEST_NAME

    creds_4dn = load_4dn_credentials()

    session = requests.Session()
    session.headers.update({"User-Agent": "collect-hic/1.0"})

    total = len(rows)
    success = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows, start=1):
        hic_acc = row["Hi-C Accession"].strip()
        dataset_acc = row.get("Dataset Accession", "").strip()

        try:
            source = detect_source(hic_acc)
        except Exception as e:
            failed += 1
            msg = str(e)
            print(f"[{i}/{total}] FAIL {hic_acc}: {msg}")
            append_manifest(
                manifest_path,
                {
                    "Hi-C Accession": hic_acc,
                    "Dataset Accession": dataset_acc,
                    "source": "UNKNOWN",
                    "status": "failed",
                    "output_path": "",
                    "download_url": "",
                    "message": msg,
                },
            )
            continue

        try:
            if source == "ENCODE":
                download_url, _ = get_encode_download_info(hic_acc, session)
            else:
                if creds_4dn is None:
                    raise RuntimeError(
                        "4DN credentials not found. Create ./keypairs.json or ~/keypairs.json before downloading 4DN files."
                    )
                download_url, _ = get_4dn_download_info(hic_acc, session)

            out_name = build_output_filename(row)
            out_path = out_dir / out_name

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                print(f"[{i}/{total}] SKIP {hic_acc} -> {out_path}")
                append_manifest(
                    manifest_path,
                    {
                        "Hi-C Accession": hic_acc,
                        "Dataset Accession": dataset_acc,
                        "source": source,
                        "status": "skipped",
                        "output_path": str(out_path),
                        "download_url": download_url,
                        "message": "already exists",
                    },
                )
                continue

            print(f"[{i}/{total}] DOWNLOADING {source} {hic_acc}")
            auth = creds_4dn if source == "4DN" else None
            stream_download(download_url, out_path, session=session, auth=auth)

            success += 1
            print(f"[{i}/{total}] OK {hic_acc} -> {out_path}")
            append_manifest(
                manifest_path,
                {
                    "Hi-C Accession": hic_acc,
                    "Dataset Accession": dataset_acc,
                    "source": source,
                    "status": "success",
                    "output_path": str(out_path),
                    "download_url": download_url,
                    "message": "",
                },
            )

            time.sleep(0.2)

        except Exception as e:
            failed += 1
            print(f"[{i}/{total}] FAIL {hic_acc}: {e}")
            append_manifest(
                manifest_path,
                {
                    "Hi-C Accession": hic_acc,
                    "Dataset Accession": dataset_acc,
                    "source": source,
                    "status": "failed",
                    "output_path": "",
                    "download_url": download_url if "download_url" in locals() else "",
                    "message": str(e),
                },
            )

    print("\nDone.")
    print(f"  total   : {total}")
    print(f"  success : {success}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")
    print(f"  manifest: {manifest_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())