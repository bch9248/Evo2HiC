#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen, Request

MAP_URL = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Umap.MultiTrackMappability.bw"
OUT_DIR = Path("data/dna/human")
OUT_BW = OUT_DIR / "k100.Umap.MultiTrackMappability.bw"
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
                print(
                    f"\rDownloading k100.Umap.MultiTrackMappability.bw: "
                    f"{downloaded/1e9:.2f} / {total/1e9:.2f} GB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloading k100.Umap.MultiTrackMappability.bw: {downloaded/1e9:.2f} GB",
                    end="",
                    flush=True,
                )
    print()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_BW.exists() and OUT_BW.stat().st_size > 0:
        print(f"[SKIP] Already exists: {OUT_BW}")
        return 0

    print(f"[INFO] Downloading from {MAP_URL}")
    download_file(MAP_URL, OUT_BW)
    print(f"[OK] Saved to {OUT_BW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())