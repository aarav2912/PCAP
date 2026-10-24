#!/usr/bin/env python3
"""
download_reside.py
Downloads and extracts the RESIDE-Standard dataset (ITS + SOTS subsets).

Usage:
    python data/download_reside.py --subset sots      # only test set (~300 MB)
    python data/download_reside.py --subset its       # training set (~1.5 GB)
    python data/download_reside.py --subset all       # both

Dataset: RESIDE — REalistic Single Image DEhazing
Paper:   Li et al., "Benchmarking Single-Image Dehazing and Beyond", TIP 2019
GitHub:  https://github.com/Boyiliee/RESIDE-dataset-link

NOTE: Official download is via Dropbox. If Dropbox is blocked,
use Kaggle mirror or Baidu Yun links listed in README.md.
"""

import os
import sys
import zipfile
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


# --- Download links ----------------------------------------------------------
LINKS = {
    "sots": {
        "url":    "https://bit.ly/2XZH498",
        "file":   "SOTS.zip",
        "dir":    "data/SOTS",
        "desc":   "SOTS test set (500 indoor + 500 outdoor hazy/GT pairs)",
    },
    "its": {
        "url":    "https://bit.ly/3iwHmh0",
        "file":   "ITS.zip",
        "dir":    "data/ITS",
        "desc":   "ITS training set (13,990 hazy / 1,399 GT indoor images)",
    },
}

BACKUP_INSTRUCTIONS = """
If automatic download fails (Dropbox may require browser login):

1. SOTS (Test set, ~300 MB):
   Dropbox: https://bit.ly/2XZH498
   Kaggle:  Search "RESIDE SOTS" on kaggle.com

2. ITS (Training set, ~1.5 GB):
   Dropbox: https://bit.ly/3iwHmh0
   Baidu:   http://tinyurl.com/yaohd3yv  (password: g0s6)

After downloading, place the .zip files in the data/ directory and run:
   python data/download_reside.py --extract-only
"""


class DownloadBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress bar. Returns True on success."""
    print(f"Downloading {dest.name} from {url}")
    try:
        with DownloadBar(unit='B', unit_scale=True, miniters=1,
                         desc=dest.name) as bar:
            urllib.request.urlretrieve(url, dest, reporthook=bar.update_to)
        return True
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        print(BACKUP_INSTRUCTIONS)
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip archive with progress."""
    print(f"Extracting {zip_path.name} ? {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for i, m in enumerate(members):
            zf.extract(m, extract_to)
            if i % 500 == 0:
                print(f"  {i}/{len(members)} files extracted...", end='\r')
    print(f"  Done. {len(members)} files extracted.          ")


def verify_structure(data_dir: str) -> dict:
    """Check if dataset directories exist and count images."""
    stats = {}
    for name, info in LINKS.items():
        d = Path(info["dir"])
        if d.exists():
            count = sum(1 for f in d.rglob("*") if f.suffix.lower() in
                        {'.png', '.jpg', '.jpeg', '.bmp'})
            stats[name] = count
        else:
            stats[name] = 0
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download RESIDE dataset")
    parser.add_argument("--subset", choices=["sots", "its", "all"],
                        default="sots", help="Which subset to download")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip download, only extract existing .zip files")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing dataset structure")
    args = parser.parse_args()

    root = Path(__file__).parent.parent  # project root
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    if args.verify:
        stats = verify_structure(str(data_dir))
        print("\nDataset status:")
        for name, count in stats.items():
            status = "?" if count > 0 else "?"
            print(f"  {status} {name.upper()}: {count} images in {LINKS[name]['dir']}")
        return

    subsets = list(LINKS.keys()) if args.subset == "all" else [args.subset]

    for subset in subsets:
        info = LINKS[subset]
        zip_path = data_dir / info["file"]
        extract_dir = root / info["dir"]

        print(f"\n??? {subset.upper()}: {info['desc']} ???")

        if extract_dir.exists() and any(extract_dir.rglob("*.png")):
            print(f"  Already exists at {extract_dir}. Skipping.")
            continue

        if not args.extract_only:
            if not zip_path.exists():
                success = download_file(info["url"], zip_path)
                if not success: continue
            else:
                print(f"  .zip already downloaded: {zip_path}")
        else:
            if not zip_path.exists():
                print(f"  [Error] {zip_path} not found. Download first.")
                continue

        extract_zip(zip_path, root / "data")

        # Verify
        count = sum(1 for f in extract_dir.rglob("*")
                    if f.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        print(f"  Verified: {count} images in {extract_dir}")

    print("\nDataset structure expected:")
    print("  data/SOTS/")
    print("    indoor/")
    print("      hazy/   ? 500 hazy test images")
    print("      gt/     ? 500 clean ground truth images")
    print("    outdoor/")
    print("      hazy/")
    print("      gt/")
    print("\nRun the benchmark:")
    print("  ./build/benchmark --hazy data/SOTS/indoor/hazy --gt data/SOTS/indoor/gt --mode all")


if __name__ == "__main__":
    main()
