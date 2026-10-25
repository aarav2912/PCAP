#!/bin/bash
# run_benchmark.sh - Quick benchmark runner
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
DATA="$ROOT/data/SOTS/indoor"
RESULTS="$ROOT/results"
THREADS="${1:-8}"

if [ ! -f "$BUILD/benchmark" ]; then
    echo "Binary not found. Run: bash scripts/build.sh"
    exit 1
fi

if [ ! -d "$DATA/hazy" ]; then
    if [ -d "$ROOT/data/sample/hazy" ]; then
        echo "[Info] SOTS indoor not found; using generated sample dataset at data/sample"
        DATA="$ROOT/data/sample"
    else
        echo "Dataset not found. Run: python data/download_reside.py --subset sample (or --subset sots)"
        exit 1
    fi
fi

mkdir -p "$RESULTS"

echo "Running 3-way benchmark (CPU, OpenMP[$THREADS threads], CUDA)..."
"$BUILD/benchmark" \
    --hazy    "$DATA/hazy" \
    --gt      "$DATA/gt" \
    --output  "$RESULTS" \
    --mode    all \
    --threads "$THREADS"

echo ""
echo "Generating report..."
python3 "$ROOT/scripts/report.py" --csv "$RESULTS/benchmark.csv" --out "$RESULTS"
echo "Done! Open results/timing_comparison.png"
