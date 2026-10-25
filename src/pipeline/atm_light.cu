/*
 * atm_light.cu
 * CUDA Kernel 2: Atmospheric Light Estimation via parallel reduction.
 *
 * Algorithm:
 *   Step A - GPU histogram over dark channel to find the top 0.1% threshold
 *             using atomicAdd into 256-bin histogram.
 *   Step B - Parallel reduction over all pixels above the threshold to
 *             find the one with the highest luminance in the hazy image.
 *
 * Optimization techniques used:
 *   - atomicAdd into shared-memory histogram (per block), then global merge
 *   - Warp-level reduction using __shfl_down_sync for the luminance argmax
 *   - Two-phase reduction (block-level then final single-block pass)
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include "utils.hpp"

// --- Step A: Build histogram of dark channel values ---------------------------
__global__ void kernelHistogram(
    const float* __restrict__ dark,
    int* __restrict__ hist,   // 256 bins (global)
    int N)
{
    __shared__ int s_hist[256];
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = min((int)(dark[idx] * 255.0f), 255);
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256)
        atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
}

// --- Host: find threshold bin from histogram ----------------------------------
static int findTopKBin(const int* h_hist, int N, float frac) {
    int topK = max(1, (int)(frac * N));
    int count = 0;
    for (int bin = 255; bin >= 0; --bin) {
        count += h_hist[bin];
        if (count >= topK) return bin;
    }
    return 0;
}

// --- Step B: Find pixel with max luminance among top-K dark channel pixels ----
struct PixelResult { float lum; float r, g, b; };

__device__ __forceinline__ PixelResult warpReduceMax(PixelResult val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_lum = __shfl_down_sync(0xffffffff, val.lum, offset);
        float or_ = __shfl_down_sync(0xffffffff, val.r,   offset);
        float og   = __shfl_down_sync(0xffffffff, val.g,   offset);
        float ob   = __shfl_down_sync(0xffffffff, val.b,   offset);
        if (other_lum > val.lum) { val.lum = other_lum; val.r = or_; val.g = og; val.b = ob; }
    }
    return val;
}

__global__ void kernelFindAtmLight(
    const float* __restrict__ input,   // HxWx3
    const float* __restrict__ dark,    // HxW
    float* __restrict__ out_lum,       // [gridDim.x]  best luminance per block
    float* __restrict__ out_rgb,       // [gridDim.x * 3]
    int N, float threshold)
{
    __shared__ float s_lum[256];
    __shared__ float s_r  [256];
    __shared__ float s_g  [256];
    __shared__ float s_b  [256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    PixelResult local = { -1.f, 0.f, 0.f, 0.f };
    if (idx < N && dark[idx] >= threshold) {
        float r = input[idx*3 + 0];
        float g = input[idx*3 + 1];
        float b = input[idx*3 + 2];
        // Luminance (BGR order: input[0]=B, [1]=G, [2]=R)
        local.lum = 0.0722f*r + 0.7152f*g + 0.2126f*b;
        local.r = r; local.g = g; local.b = b;
    }

    // Warp reduce
    local = warpReduceMax(local);

    // Write warp results to shared
    if (tid % 32 == 0) {
        s_lum[tid/32] = local.lum;
        s_r  [tid/32] = local.r;
        s_g  [tid/32] = local.g;
        s_b  [tid/32] = local.b;
    }
    __syncthreads();

    // Block reduce (first warp)
    if (tid < 8) {
        PixelResult wval = { s_lum[tid], s_r[tid], s_g[tid], s_b[tid] };
        wval = warpReduceMax(wval);
        if (tid == 0) {
            out_lum[blockIdx.x] = wval.lum;
            out_rgb[blockIdx.x*3+0] = wval.r;
            out_rgb[blockIdx.x*3+1] = wval.g;
            out_rgb[blockIdx.x*3+2] = wval.b;
        }
    }
}

// --- Host launcher ------------------------------------------------------------
void launchAtmLight(
    const float* d_input, const float* d_dark,
    int W, int H,
    float& outR, float& outG, float& outB,
    cudaStream_t stream)
{
    int N = W * H;

    // -- Step A: histogram --
    int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    dim3 hblock(256);
    dim3 hgrid((N + 255) / 256);
    kernelHistogram<<<hgrid, hblock, 0, stream>>>(d_dark, d_hist, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_hist[256];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 256*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hist));

    int thresh_bin = findTopKBin(h_hist, N, ATM_TOP_FRAC);
    float threshold = thresh_bin / 255.0f;

    // -- Step B: argmax luminance --
    int nblocks = hgrid.x;
    float* d_lum; float* d_rgb;
    CUDA_CHECK(cudaMalloc(&d_lum, nblocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb, nblocks * 3 * sizeof(float)));

    kernelFindAtmLight<<<hgrid, hblock, 0, stream>>>(
        d_input, d_dark, d_lum, d_rgb, N, threshold);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_lum(nblocks), h_rgb(nblocks*3);
    CUDA_CHECK(cudaMemcpy(h_lum.data(), d_lum, nblocks*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, nblocks*3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lum)); CUDA_CHECK(cudaFree(d_rgb));

    // Final argmax on host (nblocks is small)
    float best_lum = -1.f; int best_b = 0;
    for (int b = 0; b < nblocks; ++b)
        if (h_lum[b] > best_lum) { best_lum = h_lum[b]; best_b = b; }

    outR = max(h_rgb[best_b*3+0], 1e-3f);
    outG = max(h_rgb[best_b*3+1], 1e-3f);
    outB = max(h_rgb[best_b*3+2], 1e-3f);
}
