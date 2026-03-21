#!/usr/bin/env python3
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request

HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/latest/hg38.fa.gz"
OUT_DIR = Path("data/dna/human")
OUT_FA = OUT_DIR / "hg38.fa"
TMP_GZ = OUT_DIR / "hg38.fa.gz"
CHUNK_SIZE = 8 * 1024 * 1024


def download_file(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "python-downloader/1.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        downloaded = 0

        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = downloaded * 100 / total
                print(f"\rDownloading hg38.fa.gz: {downloaded/1e9:.2f} / {total/1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloading hg38.fa.gz: {downloaded/1e9:.2f} GB", end="", flush=True)
    print()


def gunzip_file(src_gz: Path, dest_fa: Path) -> None:
    with gzip.open(src_gz, "rb") as fin, open(dest_fa, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=CHUNK_SIZE)


def build_fai(fasta_path: Path) -> bool:
    try:
        subprocess.run(
            ["samtools", "faidx", str(fasta_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print(f"[WARN] samtools faidx failed: {e}")
        return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FA.exists() and OUT_FA.stat().st_size > 0:
        print(f"[SKIP] FASTA already exists: {OUT_FA}")
    else:
        if not TMP_GZ.exists() or TMP_GZ.stat().st_size == 0:
            print(f"[INFO] Downloading from {HG38_URL}")
            download_file(HG38_URL, TMP_GZ)
        else:
            print(f"[INFO] Reusing existing archive: {TMP_GZ}")

        print("[INFO] Decompressing to hg38.fa ...")
        gunzip_file(TMP_GZ, OUT_FA)
        print(f"[OK] Wrote {OUT_FA}")

    fai_path = Path(str(OUT_FA) + ".fai")
    if fai_path.exists() and fai_path.stat().st_size > 0:
        print(f"[OK] FASTA index already exists: {fai_path}")
    else:
        print("[INFO] Building FASTA index with samtools faidx ...")
        if build_fai(OUT_FA):
            print(f"[OK] Wrote {fai_path}")
        else:
            print("[WARN] Could not build .fai automatically.")
            print("[WARN] Install samtools, then run:")
            print(f"       samtools faidx {OUT_FA}")
            print("[WARN] If pysam has write permission, it may also create the index later.")

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())