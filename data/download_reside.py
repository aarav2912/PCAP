#!/usr/bin/env python3
"""
download_reside.py
Robust downloader, extractor, and sample generator for the RESIDE dataset.

Supports:
  - SOTS (Synthetic Objective Testing Set)
  - ITS (Indoor Training Set)
  - Synthetic sample paired dataset for immediate zero-download testing
  - Direct URL download with atomic writes, HTTP validation, HTML detection
  - Kaggle dataset mirror support (balraj98/synthetic-objective-testing-set-sots-reside)
  - Safe extraction with Zip-Slip path traversal protection

Usage:
    # 1. Quick test without large downloads (instant sample hazy/GT pairs):
    python data/download_reside.py --subset sample

    # 2. Download SOTS using verified mirror or Kaggle:
    python data/download_reside.py --subset sots --source kaggle
    python data/download_reside.py --subset sots --url <direct_download_url>

    # 3. Extract manually placed SOTS.zip:
    python data/download_reside.py --extract-only

    # 4. Verify existing dataset:
    python data/download_reside.py --verify
"""

import os
import sys
import shutil
import zipfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# --- Dataset definitions -----------------------------------------------------
DATASETS = {
    "sots": {
        "file": "SOTS.zip",
        "dir": "data/SOTS",
        "desc": "SOTS test set (indoor & outdoor hazy/GT pairs)",
        "kaggle_dataset": "balraj98/synthetic-objective-testing-set-sots-reside",
        # Default official Dropbox link (Note: Dropbox may block direct CLI downloads)
        "default_url": "https://www.dropbox.com/s/y6jupfvitv0dx5w/SOTS.zip?dl=1",
    },
    "its": {
        "file": "ITS.zip",
        "dir": "data/ITS",
        "desc": "ITS training set (indoor hazy/GT pairs)",
        "kaggle_dataset": "balraj98/indoor-training-set-its-reside-standard",
        "default_url": "https://www.dropbox.com/s/72a11b7z9y3r8t6/ITS.zip?dl=1",
    },
}

ZIP_MAGIC_HEADERS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


# --- Progress Bar Helper -----------------------------------------------------
class ProgressBar:
    def __init__(self, total_bytes: int, desc: str = "Downloading"):
        self.total = total_bytes
        self.downloaded = 0
        self.desc = desc

    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        if self.total > 0:
            pct = min(100.0, self.downloaded * 100.0 / self.total)
            mb_down = self.downloaded / (1024 * 1024)
            mb_tot = self.total / (1024 * 1024)
            bar_len = 30
            filled = int(bar_len * pct / 100.0)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"\r  {self.desc}: [{bar}] {pct:5.1f}% ({mb_down:6.1f}/{mb_tot:6.1f} MB)", end="", flush=True)
        else:
            mb_down = self.downloaded / (1024 * 1024)
            print(f"\r  {self.desc}: {mb_down:6.1f} MB downloaded...", end="", flush=True)

    def close(self):
        print()


# --- Download with Validation & HTML Rejection -------------------------------
def download_archive(url: str, dest_path: Path) -> bool:
    """
    Downloads an archive from url into dest_path atomically.
    Validates:
      1. HTTP status == 200
      2. Content-Type is not text/html
      3. Magic bytes match ZIP header (b'PK...')
      4. zipfile.is_zipfile(temp_file)
      5. zipfile integrity check
    Cleans up temporary file on failure.
    """
    part_path = dest_path.with_name(dest_path.name + ".part")
    if part_path.exists():
        part_path.unlink()

    print(f"[Download] Requesting: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/zip,application/octet-stream,*/*",
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status if hasattr(resp, "status") else resp.getcode()
            if status != 200:
                print(f"[Error] HTTP request failed with status code {status}.")
                return False

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                print(f"[Error] Server returned HTML content ('{content_type}') instead of a ZIP archive.")
                print("        The download URL likely redirects to a web login or landing page.")
                return False

            content_length = int(resp.headers.get("Content-Length", 0))
            pbar = ProgressBar(content_length, desc=dest_path.name)

            # Read first chunk to inspect magic bytes
            first_chunk = resp.read(8192)
            if not first_chunk:
                print("[Error] Empty response received from server.")
                return False

            if not any(first_chunk.startswith(magic) for magic in ZIP_MAGIC_HEADERS):
                sample_text = first_chunk[:200].decode("latin1", errors="ignore").strip()
                if "<html" in sample_text.lower() or "<!doctype" in sample_text.lower():
                    print(f"[Error] Server returned an HTML web page instead of a binary ZIP file.")
                    print(f"        Response preview: {sample_text[:120]}...")
                else:
                    print(f"[Error] Response does not have a valid ZIP file header.")
                return False

            with open(part_path, "wb") as f_out:
                f_out.write(first_chunk)
                pbar.update(len(first_chunk))
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

    except urllib.error.HTTPError as e:
        print(f"[Error] HTTP error occurred: {e.code} {e.reason}")
        if part_path.exists(): part_path.unlink()
        return False
    except urllib.error.URLError as e:
        print(f"[Error] Network/URL error occurred: {e.reason}")
        if part_path.exists(): part_path.unlink()
        return False
    except Exception as e:
        print(f"[Error] Unexpected error during download: {e}")
        if part_path.exists(): part_path.unlink()
        return False

    # Validate archive integrity
    print("  Validating ZIP archive integrity...")
    if not zipfile.is_zipfile(part_path):
        print("[Error] Downloaded file failed ZIP header validation.")
        part_path.unlink()
        return False

    try:
        with zipfile.ZipFile(part_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file:
                print(f"[Error] Corrupted entry in zip: {bad_file}")
                part_path.unlink()
                return False
    except Exception as e:
        print(f"[Error] Failed to read ZIP archive: {e}")
        part_path.unlink()
        return False

    # Atomic rename upon successful download and validation
    if dest_path.exists():
        dest_path.unlink()
    part_path.rename(dest_path)
    print(f"  Successfully downloaded and validated: {dest_path}")
    return True


# --- Kaggle Download Helper --------------------------------------------------
def download_from_kaggle(dataset_slug: str, dest_dir: Path) -> bool:
    """Attempts to download dataset via Kaggle CLI / API."""
    print(f"[Kaggle] Attempting download for: {dataset_slug}")
    try:
        import subprocess
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir), "--unzip"]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode == 0:
            print("[Kaggle] Download and extraction complete.")
            return True
        else:
            print(f"[Kaggle] Command failed: {ret.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("[Kaggle] Kaggle CLI ('kaggle') not found.")
        print("         To install: pip install kaggle")
        print("         Place your kaggle.json in ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        print(f"[Kaggle] Error running Kaggle CLI: {e}")
        return False


# --- Safe ZIP Extraction (Zip-Slip Protected) --------------------------------
def safe_extract_zip(zip_path: Path, extract_to: Path):
    """Safely extracts all members from zip_path to extract_to."""
    print(f"[Extract] Extracting {zip_path.name} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    target_base = extract_to.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, m in enumerate(members):
            dest_file = (extract_to / m).resolve()
            try:
                dest_file.relative_to(target_base)
            except ValueError:
                raise RuntimeError(f"Zip-Slip path traversal vulnerability detected in archive entry: {m}")

            zf.extract(m, extract_to)
            if i % 250 == 0 or i == len(members) - 1:
                print(f"\r  {i+1}/{len(members)} files extracted...", end="", flush=True)
    print(f"\n  Done. Extracted {len(members)} files.")


# --- Synthetic Sample Dataset Generator -------------------------------------
def generate_sample_dataset(root_dir: Path, num_samples: int = 5):
    """
    Generates synthetic paired hazy and ground-truth images using
    Koschmieder's physical atmospheric scattering model:
        I(x) = J(x) * t(x) + A * (1 - t(x))
        t(x) = exp(-beta * depth(x))
    Creates pairs in data/sample/ and also mirrors into data/SOTS/indoor/.
    """
    print(f"\n[Sample Generator] Generating {num_samples} synthetic paired test images...")
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError:
        print("[Error] Pillow and numpy are required to generate sample images.")
        print("        Install with: pip install pillow numpy")
        return False

    sample_dir = root_dir / "data" / "sample"
    sots_indoor = root_dir / "data" / "SOTS" / "indoor"

    for d in [sample_dir / "hazy", sample_dir / "gt", sots_indoor / "hazy", sots_indoor / "gt"]:
        d.mkdir(parents=True, exist_ok=True)

    w, h = 512, 512

    for i in range(num_samples):
        gt_img = Image.new("RGB", (w, h), color=(130 + i * 10, 190, 230))
        draw = ImageDraw.Draw(gt_img)

        draw.ellipse([380 - i * 30, 40, 460 - i * 30, 120], fill=(255, 230 - i * 15, 60))
        draw.rectangle([0, h // 2, w, h], fill=(30 + i * 5, 120 - i * 5, 40))

        draw.rectangle([50, 150, 150, 420], fill=(110 + i * 10, 100, 95))
        draw.rectangle([180, 110, 310, 440], fill=(80, 90 + i * 10, 120))
        draw.rectangle([340, 170, 470, 410], fill=(130, 75 + i * 5, 65))

        for wx in range(200, 290, 25):
            for wy in range(130, 400, 35):
                draw.rectangle([wx, wy, wx + 12, wy + 18], fill=(250, 245, 180))

        depth = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            depth[y, :] = 1.0 - 0.7 * (y / h)
        depth[150:420, 50:150] = 0.65
        depth[110:440, 180:310] = 0.40
        depth[170:410, 340:470] = 0.55

        gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
        beta = 1.5 + i * 0.4
        trans = np.exp(-beta * depth)[:, :, np.newaxis]
        A = np.array([0.90 + i * 0.01, 0.92, 0.95], dtype=np.float32)

        hazy_arr = gt_arr * trans + A * (1.0 - trans)
        hazy_arr = np.clip(hazy_arr * 255.0, 0, 255).astype(np.uint8)
        hazy_img = Image.fromarray(hazy_arr)

        base_id = 1400 + i
        gt_name = f"{base_id}.png"
        hazy_name = f"{base_id}_1.png"

        gt_img.save(sample_dir / "gt" / gt_name)
        hazy_img.save(sample_dir / "hazy" / hazy_name)

        gt_img.save(sots_indoor / "gt" / gt_name)
        hazy_img.save(sots_indoor / "hazy" / hazy_name)

    print(f"  Successfully generated {num_samples} sample pairs:")
    print(f"    Sample hazy: {sample_dir / 'hazy'}")
    print(f"    Sample GT:   {sample_dir / 'gt'}")
    print(f"    Also mirrored to standard benchmark path: {sots_indoor}")
    return True


# --- Directory Structure Verification ----------------------------------------
def verify_structure(data_dir: Path) -> dict:
    """Checks for existing dataset directories and counts valid images."""
    stats = {}
    for name, info in DATASETS.items():
        d = data_dir / Path(info["dir"]).name
        if d.exists():
            hazy_cnt = sum(1 for f in d.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "hazy" in str(f).lower())
            gt_cnt = sum(1 for f in d.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and ("gt" in str(f).lower() or "clear" in str(f).lower()))
            total_cnt = sum(1 for f in d.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
            stats[name] = {"hazy": hazy_cnt, "gt": gt_cnt, "total": total_cnt, "path": d}
        else:
            stats[name] = {"hazy": 0, "gt": 0, "total": 0, "path": d}

    sample_dir = data_dir / "sample"
    if sample_dir.exists():
        hazy_cnt = sum(1 for f in (sample_dir / "hazy").glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        gt_cnt = sum(1 for f in (sample_dir / "gt").glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        stats["sample"] = {"hazy": hazy_cnt, "gt": gt_cnt, "total": hazy_cnt + gt_cnt, "path": sample_dir}

    return stats


# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RESIDE Dataset Downloader, Validator & Sample Generator"
    )
    parser.add_argument(
        "--subset",
        choices=["sots", "its", "sample", "all"],
        default="sots",
        help="Dataset subset to obtain (default: sots; 'sample' creates instant test set)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "url", "kaggle", "sample"],
        default="auto",
        help="Source method to use for obtaining the dataset",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom direct URL to download the zip archive from",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to an existing local .zip archive to validate and extract",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip downloading; extract existing .zip archives in data/",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check and report existing dataset status",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of synthetic pairs to generate when using sample mode (default: 5)",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        print("\n=== Dataset Status Verification ===")
        stats = verify_structure(data_dir)
        for name, info in stats.items():
            status = "[READY]" if info["total"] > 0 else "[MISSING]"
            print(f"  {status:9s} {name.upper():7s}: {info['total']} images "
                  f"({info['hazy']} hazy, {info['gt']} GT) at {info['path']}")
        return

    if args.subset == "sample" or args.source == "sample":
        generate_sample_dataset(root, num_samples=args.num_samples)
        print("\nReady! You can now run single-image inference or the benchmark:")
        print("  ./build/dehazer --mode cuda --input data/sample/hazy/1400_1.png --output output_cuda.png")
        print("  ./build/benchmark --hazy data/sample/hazy --gt data/sample/gt --mode all")
        return

    subsets = ["sots", "its"] if args.subset == "all" else [args.subset]

    for subset in subsets:
        info = DATASETS[subset]
        dest_zip = Path(args.zip_path) if args.zip_path else (data_dir / info["file"])
        extract_dir = root / info["dir"]

        print(f"\n========================================================")
        print(f" Preparing {subset.upper()}: {info['desc']}")
        print(f"========================================================")

        if extract_dir.exists():
            existing_imgs = list(extract_dir.rglob("*.png")) + list(extract_dir.rglob("*.jpg"))
            if len(existing_imgs) >= 10:
                print(f"  Dataset already exists with {len(existing_imgs)} images at {extract_dir}.")
                print("  Skipping extraction. Use --verify to inspect.")
                continue

        if args.extract_only:
            if not dest_zip.exists():
                print(f"[Error] Archive not found at {dest_zip}.")
                print("        Download the file first or specify --zip-path.")
                continue
            safe_extract_zip(dest_zip, data_dir)
            continue

        if dest_zip.exists():
            print(f"  Found local archive at {dest_zip}. Validating...")
            if zipfile.is_zipfile(dest_zip):
                safe_extract_zip(dest_zip, data_dir)
                continue
            else:
                print(f"  [Warning] Existing file {dest_zip} is not a valid ZIP file. Removing corrupt file.")
                dest_zip.unlink()

        download_success = False

        if args.url:
            download_success = download_archive(args.url, dest_zip)
        elif args.source == "kaggle":
            download_success = download_from_kaggle(info["kaggle_dataset"], data_dir)
        elif args.source == "auto":
            kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_config.exists() and shutil.which("kaggle"):
                print("  [Auto] Found Kaggle credentials; attempting download via Kaggle CLI...")
                download_success = download_from_kaggle(info["kaggle_dataset"], data_dir)

            if not download_success and info["default_url"]:
                print("  [Auto] Attempting direct URL download...")
                download_success = download_archive(info["default_url"], dest_zip)

        if download_success and dest_zip.exists():
            safe_extract_zip(dest_zip, data_dir)
        else:
            print(f"\n[Notice] Automatic download for {subset.upper()} was not completed.")
            print("=====================================================================")
            print("Actionable Alternatives:")
            print(f"  1. Instant Sample Mode (No large download needed for Colab/smoke test):")
            print(f"     python data/download_reside.py --subset sample")
            print()
            print(f"  2. Download via Kaggle CLI (Free, reliable official mirror):")
            print(f"     pip install kaggle")
            print(f"     # Ensure ~/.kaggle/kaggle.json is configured, then:")
            print(f"     kaggle datasets download -d {info['kaggle_dataset']} -p data/ --unzip")
            print()
            print(f"  3. Custom URL / Google Drive / S3 Mirror:")
            print(f"     python data/download_reside.py --subset {subset} --url <DIRECT_ZIP_URL>")
            print()
            print(f"  4. Manual Placement:")
            print(f"     Place {info['file']} into data/ and run:")
            print(f"     python data/download_reside.py --extract-only")
            print("=====================================================================")

    stats = verify_structure(data_dir)
    print("\nCurrent Dataset Status:")
    for name, st in stats.items():
        print(f"  - {name}: {st['total']} images ({st['hazy']} hazy, {st['gt']} GT)")


if __name__ == "__main__":
    main()
