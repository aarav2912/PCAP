# CUDA Image Dehazing

> **Single-image dehazing** using the Dark Channel Prior algorithm, accelerated with **CUDA GPU kernels** and **OpenMP CPU parallelism**. Benchmarked 3-ways on the RESIDE dataset.

## Algorithm

**Dark Channel Prior (DCP)** — He et al., CVPR 2009 / TPAMI 2011

Physics model: `I(x) = J(x)·t(x) + A·(1 - t(x))`

Recovery: `J(x) = (I(x) - A) / max(t(x), t_min) + A`

### CUDA Kernels (5 stages)

| Kernel | Stage | Optimization |
|--------|-------|-------------|
| `dark_channel.cu` | Dark Channel min-filter | Shared memory + separable 1D passes |
| `atm_light.cu` | Atmospheric Light | `atomicAdd` histogram + warp `__shfl_down_sync` |
| `transmission.cu` | Raw Transmission | `__launch_bounds__` + texture-friendly reads |
| `guided_filter.cu` | Guided Filter (refinement) | Integral images (SAT) via parallel scans |
| `recovery.cu` | Scene Recovery | `__ldg` L1-cache + `__saturatef` + coalesced |

### OpenMP Integration (CPU-side)

| File | Role |
|------|------|
| `image_io.cpp` | Parallel batch image loading across all CPU cores |
| `metrics.cpp` | Parallel PSNR/SSIM evaluation over 500 test images |
| `dcp_openmp.cpp` | Full DCP pipeline on CPU with `#pragma omp parallel for` |

---

## Dataset: RESIDE-Standard

```
data/SOTS/
  indoor/
    hazy/     ? 500 hazy test images
    gt/       ? 500 clean ground truth
  outdoor/
    hazy/
    gt/
data/ITS/     ? 13,990 hazy training images (optional, for robustness testing)
```

### Download

```bash
# Download SOTS test set only (~300 MB, recommended to start)
python data/download_reside.py --subset sots

# Download both ITS + SOTS
python data/download_reside.py --subset all

# Verify dataset structure
python data/download_reside.py --verify
```

**If Dropbox download fails**, download manually:
- SOTS: https://bit.ly/2XZH498
- ITS:  https://bit.ly/3iwHmh0

Place `.zip` in `data/` and run `python data/download_reside.py --extract-only`

---

## Requirements

| Dependency | Min Version | Purpose |
|-----------|-------------|---------|
| CUDA Toolkit | 11.0 | GPU kernels + Thrust |
| NVIDIA GPU | Pascal (sm_60) | Compute target |
| OpenCV | 4.x | Image I/O only |
| CMake | 3.18 | Build system |
| GCC or MSVC | GCC 9+ / VS 2019+ | C++17 + OpenMP |
| Python 3 | 3.8+ | Dataset scripts |

Install Python deps:
```bash
pip install pandas matplotlib numpy tqdm
```

---

## Build

### Linux/WSL
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows (Visual Studio)
```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release -- /m
```

### macOS (CPU-only, no CUDA)
Not supported (requires NVIDIA GPU).

---

## Usage

### Single image
```bash
# CUDA mode (GPU)
./build/dehazer --mode cuda --input path/to/hazy.png --output dehazed.png

# With ground truth evaluation
./build/dehazer --mode cuda --input hazy.png --output dehazed.png --gt gt.png

# CPU sequential
./build/dehazer --mode cpu --input hazy.png --output dehazed.png

# OpenMP (8 threads)
./build/dehazer --mode omp --input hazy.png --output dehazed.png --threads 8
```

### Full 3-way batch benchmark on SOTS
```bash
./build/benchmark \
    --hazy data/SOTS/indoor/hazy \
    --gt   data/SOTS/indoor/gt \
    --output results/ \
    --mode all \
    --threads 8
```

### Generate charts and report
```bash
python scripts/report.py --csv results/benchmark.csv --out results/
```

---

## Expected Results

| Mode | PSNR (SOTS-Indoor) | SSIM | Speed (1080p) |
|------|-------------------|------|--------------|
| CPU Sequential | ~16–19 dB | ~0.81–0.84 | ~2–5 s |
| OpenMP (8 cores) | same | same | ~0.4–1 s |
| **CUDA (GPU)** | same | same | **~15–50 ms** |

All three modes produce **identical output** — same DCP algorithm, same results.
The speedup is purely from **parallel computation**.

> **Note:** DCP PSNR is lower than deep learning methods (FFA-Net: ~36 dB) because it is
> a physics-based prior with no training. The focus is on demonstrating CUDA parallelism,
> not competing with neural networks.

---

## Project Structure

```
cuda-dehazing/
+-- CMakeLists.txt
+-- README.md
+-- data/
¦   +-- download_reside.py
+-- include/
¦   +-- utils.hpp          ? constants, timer, CUDA_CHECK
¦   +-- dehazer.cuh        ? GPU buffer struct + kernel declarations
+-- src/
¦   +-- main.cu            ? CLI entry point
¦   +-- pipeline/          ? 5 CUDA kernels
¦   ¦   +-- dark_channel.cu
¦   ¦   +-- atm_light.cu
¦   ¦   +-- transmission.cu
¦   ¦   +-- guided_filter.cu
¦   ¦   +-- recovery.cu
¦   +-- cpu/               ? CPU reference implementations
¦   ¦   +-- dcp_cpu.cpp    (sequential)
¦   ¦   +-- dcp_openmp.cpp (multi-core)
¦   +-- utils/
¦   ¦   +-- image_io.cpp   ? OpenMP batch loader
¦   ¦   +-- metrics.cpp    ? PSNR/SSIM (OpenMP parallel)
¦   +-- benchmark/
¦       +-- batch_bench.cpp
+-- scripts/
¦   +-- report.py          ? Generates charts from CSV
+-- results/               ? Output images + benchmark.csv
```

---

## Profiling

Use NVIDIA Nsight Systems to view the kernel timeline:

```bash
nsys profile --trace=cuda,openmp ./build/benchmark --mode cuda --limit 10 \
    --hazy data/SOTS/indoor/hazy --gt data/SOTS/indoor/gt --output results/
nsys-ui report1.nsys-rep
```

---

## References

1. He, K., Sun, J., & Tang, X. (2011). **Single Image Haze Removal Using Dark Channel Prior**. *IEEE TPAMI*, 33(12).
2. He, K., Sun, J., & Tang, X. (2013). **Guided Image Filtering**. *IEEE TPAMI*, 35(6).
3. Li, B., et al. (2019). **Benchmarking Single-Image Dehazing and Beyond**. *IEEE TIP*, 28(1). (RESIDE dataset)
