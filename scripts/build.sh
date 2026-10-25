#!/bin/bash
# build.sh - Build the CUDA Dehazing project
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "===================================================="
echo " Building CUDA Image Dehazing Project"
echo " Root: $ROOT_DIR"
echo "===================================================="

# Check prerequisites
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] CUDA toolkit not found. Install from https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "[ERROR] CMake not found. Install CMake >= 3.18"
    exit 1
fi

NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
echo "[Info] CUDA version: $NVCC_VER"
echo "[Info] Detected GPU:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

CUDA_ARCH="${1:-${CUDA_ARCH:-75}}"
echo "[Info] Target CUDA architecture: $CUDA_ARCH"

# Configure
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    2>&1 | tail -20

# Build
echo ""
echo "[Build] Compiling..."
cmake --build "$BUILD_DIR" --parallel $(nproc) 2>&1

echo ""
echo "===================================================="
echo " Build successful!"
echo " Binaries:"
echo "   $BUILD_DIR/dehazer    - Single-image dehazing"
echo "   $BUILD_DIR/benchmark  - 3-way batch benchmark"
echo ""
echo " Next steps:"
echo "   1. Prepare dataset:   python data/download_reside.py --subset sample"
echo "   2. Run smoke test:    ./build/dehazer --mode cuda --input data/sample/hazy/1400_1.png --output dehazed.png"
echo "   3. Run benchmark:     ./build/benchmark --mode all --hazy data/sample/hazy --gt data/sample/gt --output results/"
echo "   4. Generate charts:   python scripts/report.py --csv results/benchmark.csv"
echo "===================================================="
