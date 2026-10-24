/*
 * dark_channel.cu
 * CUDA Kernel 1: Dark Channel computation via separable min-filter.
 *
 * Optimization techniques used:
 *   - Separable 2-pass approach (horizontal then vertical) to reduce
 *     complexity from O(r^2) to O(r) per pixel per pass.
 *   - Shared memory tile loading with halo regions for the horizontal pass
 *     to avoid redundant global memory reads.
 *   - Coalesced global memory access pattern (row-major).
 *
 * Input:  d_input [H x W x 3]  RGB float planar image in [0,1]
 * Output: d_dark  [H x W]      dark channel float in [0,1]
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "utils.hpp"

// --- Pass 1: Channel min then horizontal min-filter --------------------------
// Each thread computes the per-channel minimum for one pixel,
// then applies a horizontal sliding minimum over radius r.

#define BLOCK_W 256
#define HALO    PATCH_RADIUS

__global__ void kernelDarkHorizontal(
    const float* __restrict__ input,
    float*       __restrict__ hpass,
    int W, int H, int radius)
{
    // Shared memory: (BLOCK_W + 2*radius) per row
    extern __shared__ float shmem[];  // size = (BLOCK_W + 2*HALO) * sizeof(float)

    int tx   = threadIdx.x;
    int gx   = blockIdx.x * blockDim.x + tx;
    int gy   = blockIdx.y;

    if (gy >= H) return;

    // -- Load channel-min for this row into shared memory (with halos) --
    int sW = blockDim.x + 2 * radius;

    // Global x index for shared memory slot
    auto loadPixMin = [&](int gxi) -> float {
        int clamped = max(0, min(W-1, gxi));
        const float* p = input + (gy * W + clamped) * 3;
        return fminf(p[0], fminf(p[1], p[2]));
    };

    // Left halo
    if (tx < radius) {
        shmem[tx] = loadPixMin(gx - radius);
    }
    // Main body
    shmem[tx + radius] = loadPixMin(gx);
    // Right halo
    if (tx >= blockDim.x - radius) {
        shmem[tx + 2 * radius] = loadPixMin(gx + radius);
    }

    __syncthreads();

    if (gx >= W) return;

    // -- Horizontal sliding min over shared memory --
    float mn = 1.0f;
    int s0 = tx;              // leftmost shared index
    int s1 = tx + 2*radius;   // rightmost shared index
    for (int s = s0; s <= s1; ++s)
        mn = fminf(mn, shmem[s]);

    hpass[gy * W + gx] = mn;
}

// --- Pass 2: Vertical min-filter over hpass -----------------------------------
// No shared memory here (vertical access is non-coalesced anyway);
// we rely on L1 caching. Each thread handles one column slice.

#define BLOCK_V 32

__global__ void kernelDarkVertical(
    const float* __restrict__ hpass,
    float*       __restrict__ dark,
    int W, int H, int radius)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= W || gy >= H) return;

    int y0 = max(0, gy - radius);
    int y1 = min(H-1, gy + radius);

    float mn = 1.0f;
    for (int y = y0; y <= y1; ++y)
        mn = fminf(mn, hpass[y * W + gx]);

    dark[gy * W + gx] = mn;
}

// --- Host launcher ------------------------------------------------------------
void launchDarkChannel(
    const float* d_input, float* d_dark,
    int W, int H, int radius, cudaStream_t stream)
{
    // Intermediate buffer for horizontal pass
    static float* d_hpass = nullptr;
    static size_t hpass_cap = 0;
    size_t need = (size_t)W * H * sizeof(float);
    if (need > hpass_cap) {
        if (d_hpass) cudaFree(d_hpass);
        CUDA_CHECK(cudaMalloc(&d_hpass, need));
        hpass_cap = need;
    }

    // Pass 1: horizontal
    {
        dim3 block(BLOCK_W, 1);
        dim3 grid((W + BLOCK_W - 1) / BLOCK_W, H);
        size_t smem = (BLOCK_W + 2 * radius) * sizeof(float);
        kernelDarkHorizontal<<<grid, block, smem, stream>>>(
            d_input, d_hpass, W, H, radius);
    }

    // Pass 2: vertical
    {
        dim3 block(BLOCK_V, BLOCK_V);
        dim3 grid((W + BLOCK_V - 1) / BLOCK_V, (H + BLOCK_V - 1) / BLOCK_V);
        kernelDarkVertical<<<grid, block, 0, stream>>>(
            d_hpass, d_dark, W, H, radius);
    }
}
