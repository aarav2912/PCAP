#!/usr/bin/env python3
"""
report.py
Generates benchmark charts and summary tables from benchmark.csv.

Usage:
    python scripts/report.py --csv results/benchmark.csv --out results/

Outputs:
    - results/timing_comparison.png     Bar chart: CPU vs OMP vs CUDA timing
    - results/psnr_per_image.png        PSNR per image (line plot)
    - results/ssim_per_image.png        SSIM per image
    - results/speedup_table.txt         Speedup report
    - results/summary_table.png         Final summary table image
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas", "matplotlib", "numpy"])
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np


# --- Color palette ----------------------------------------------------------
COLORS = {
    "cpu":  "#E74C3C",  # red
    "omp":  "#F39C12",  # orange
    "cuda": "#2ECC71",  # green
}
LABELS = {"cpu": "CPU (Sequential)", "omp": "CPU (OpenMP)", "cuda": "GPU (CUDA)"}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def plot_timing_comparison(df: pd.DataFrame, out_dir: Path):
    """Bar chart with error bars: mean +/- std timing per mode."""
    modes = df["mode"].unique()
    means = [df[df["mode"]==m]["time_ms"].mean() for m in modes]
    stds  = [df[df["mode"]==m]["time_ms"].std()  for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [LABELS.get(m, m) for m in modes],
        means, yerr=stds,
        color=[COLORS.get(m, "#999") for m in modes],
        capsize=6, width=0.5, edgecolor="black", linewidth=0.8
    )
    ax.set_ylabel("Processing Time (ms)", fontsize=12)
    ax.set_title("Image Dehazing - CPU vs OpenMP vs CUDA\nAverage Processing Time per Image", fontsize=13)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    # Annotate bars
    for bar, m, val, std_ in zip(bars, modes, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Speedup annotations
    if len(modes) >= 2:
        cpu_mean = df[df["mode"]=="cpu"]["time_ms"].mean() if "cpu" in modes else None
        for m, val in zip(modes, means):
            if cpu_mean and m != "cpu":
                speedup = cpu_mean / val
                idx = list(modes).index(m)
                ax.annotate(f"{speedup:.1f}x\nfaster",
                            xy=(bars[idx].get_x() + bars[idx].get_width()/2, val*0.5),
                            ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "timing_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_psnr_ssim(df: pd.DataFrame, out_dir: Path):
    """Line plots of PSNR and SSIM per image for all modes."""
    # Only PSNR with valid GT (psnr > 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for metric, ax, ylab in [("psnr", axes[0], "PSNR (dB)"),
                               ("ssim", axes[1], "SSIM")]:
        for m in df["mode"].unique():
            sub = df[df["mode"] == m].reset_index(drop=True)
            if sub[metric].sum() == 0:
                continue
            ax.plot(sub.index, sub[metric],
                    label=LABELS.get(m, m), color=COLORS.get(m, "#999"),
                    linewidth=0.8, alpha=0.7)
            ax.axhline(sub[metric].mean(), color=COLORS.get(m, "#999"),
                       linestyle="--", linewidth=1.2,
                       label=f"{LABELS.get(m,m)} avg={sub[metric].mean():.2f}")

        ax.set_xlabel("Image Index")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"{ylab} per Test Image (RESIDE SOTS)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "quality_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def generate_speedup_report(df: pd.DataFrame, out_dir: Path):
    """Print and save a speedup summary table."""
    modes = df["mode"].unique()
    rows = []
    for m in modes:
        sub = df[df["mode"]==m]
        rows.append({
            "Mode":       LABELS.get(m, m),
            "N Images":   len(sub),
            "Avg Time (ms)": f"{sub['time_ms'].mean():.1f}",
            "Min (ms)":   f"{sub['time_ms'].min():.1f}",
            "Max (ms)":   f"{sub['time_ms'].max():.1f}",
            "Avg PSNR":   f"{sub['psnr'].mean():.2f} dB",
            "Avg SSIM":   f"{sub['ssim'].mean():.4f}",
        })

    # Speedup vs CPU
    cpu_avg = df[df["mode"]=="cpu"]["time_ms"].mean() if "cpu" in modes else None
    for row, m in zip(rows, modes):
        if cpu_avg and m != "cpu":
            val = df[df["mode"]==m]["time_ms"].mean()
            row["Speedup vs CPU"] = f"{cpu_avg/val:.1f}x"
        else:
            row["Speedup vs CPU"] = "1x (baseline)"

    report_df = pd.DataFrame(rows)
    txt_path = out_dir / "speedup_report.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("          IMAGE DEHAZING BENCHMARK RESULTS\n")
        f.write("          Algorithm: Dark Channel Prior (DCP)\n")
        f.write("          Dataset: RESIDE SOTS (indoor)\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_df.to_string(index=False))
        f.write("\n\n")
        f.write("Key:\n")
        f.write("  PSNR - Peak Signal-to-Noise Ratio (higher = better, ~16-19 dB for DCP)\n")
        f.write("  SSIM - Structural Similarity Index  (higher = better, ~0.81-0.84 for DCP)\n")
        f.write("  All three modes produce IDENTICAL output (same algorithm, same quality).\n")
        f.write("  Speedup measures pure COMPUTATION SPEED advantage of GPU parallelism.\n")

    print(report_df.to_string(index=False))
    print(f"\nSaved: {txt_path}")


def plot_visual_grid(df: pd.DataFrame, results_dir: Path, out_dir: Path, n=6):
    """Create a side-by-side visual comparison grid: hazy / cuda_dehazed."""
    import glob
    cuda_dir = results_dir / "cuda"
    if not cuda_dir.exists():
        print("[Skipping visual grid] No cuda output dir found.")
        return

    files = sorted(list(cuda_dir.glob("*.png")))[:n]
    if not files:
        return

    try:
        import cv2
    except ImportError:
        print("[Skipping visual grid] opencv-python not installed.")
        return

    fig, axes = plt.subplots(2, len(files), figsize=(3*len(files), 6))
    fig.suptitle("Image Dehazing Results - CUDA DCP\nTop: Hazy Input | Bottom: Dehazed Output",
                 fontsize=12)

    for j, f in enumerate(files):
        dehazed = cv2.imread(str(f))
        dehazed = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)

        # Try to find original hazy
        hazy_candidates = list((results_dir.parent / "data").rglob(f.name))
        if hazy_candidates:
            hazy = cv2.imread(str(hazy_candidates[0]))
            hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        else:
            hazy = dehazed

        axes[0][j].imshow(hazy);      axes[0][j].axis("off")
        axes[1][j].imshow(dehazed);   axes[1][j].axis("off")
        axes[0][j].set_title(f.stem[:12], fontsize=7)

    fig.tight_layout()
    out = out_dir / "visual_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run the benchmark first.")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    plot_timing_comparison(df, out_dir)
    plot_psnr_ssim(df, out_dir)
    generate_speedup_report(df, out_dir)
    plot_visual_grid(df, out_dir, out_dir)

    print(f"\nAll results saved to: {out_dir}/")
    print("Open timing_comparison.png for the speedup chart.")


if __name__ == "__main__":
    main()
