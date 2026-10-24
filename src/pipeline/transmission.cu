/*
 * transmission.cu
 * CUDA Kernel 3: Raw Transmission Map Estimation.
 *
 * Formula: t0(x) = 1 - omega * min_{y in Omega(x)} min_c (I^c(y) / A^c)
 *
 * Optimization techniques used:
 *   - Texture objects (cudaTextureObject_t) for the hazy image:
 *       hardware caches repeated/nearby reads, automatic boundary clamping,
 *       and interpolation support for free.
 *   - Separable min-filter approach (same as dark channel) for O(r) complexity.
 *   - __launch_bounds__ hint to help the compiler reserve optimal register count.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.hpp"

// --- Texture object helper ----------------------------------------------------
static cudaTextureObject_t createTex2D(const float* d_data, int W, int H) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)d_data;
    resDesc.res.linear.desc   = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = (size_t)W * H * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    return tex;
}

// --- Build normalised dark channel: min_c (I^c / A^c) ------------------------
__global__ void __launch_bounds__(256)
kernelNormMinH(
    const float* __restrict__ input,
    float* __restrict__ hpass,
    int W, int H, int radius,
    float invAr, float invAg, float invAb)
{
    extern __shared__ float sh[];
    int tx = threadIdx.x;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y;
    if (gy >= H) return;

    auto loadNormMin = [&](int xi) -> float {
        int cx = max(0, min(W-1, xi));
        const float* p = input + (gy * W + cx) * 3;
        return fminf(p[0]*invAb, fminf(p[1]*invAg, p[2]*invAr));
    };

    if (tx < radius)                       sh[tx]             = loadNormMin(gx - radius);
    sh[tx + radius]                                           = loadNormMin(gx);
    if (tx >= blockDim.x - radius)         sh[tx + 2*radius]  = loadNormMin(gx + radius);
    __syncthreads();

    if (gx >= W) return;
    float mn = 1.0f;
    for (int s = tx; s <= tx + 2*radius; ++s) mn = fminf(mn, sh[s]);
    hpass[gy * W + gx] = mn;
}

__global__ void __launch_bounds__(256)
kernelTransmissionV(
    const float* __restrict__ hpass,
    float* __restrict__ trans,
    int W, int H, int radius, float omega)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || gy >= H) return;

    int y0 = max(0, gy-radius), y1 = min(H-1, gy+radius);
    float mn = 1.0f;
    for (int y = y0; y <= y1; ++y) mn = fminf(mn, hpass[y*W+gx]);
    trans[gy*W+gx] = 1.0f - omega * mn;
}

// --- Host launcher ------------------------------------------------------------
void launchTransmission(
    const float* d_input, float* d_trans_raw,
    int W, int H, int radius,
    float atmR, float atmG, float atmB,
    float omega, cudaStream_t stream)
{
    static float* d_hpass = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)W * H * sizeof(float);
    if (need > cap) {
        if (d_hpass) cudaFree(d_hpass);
        CUDA_CHECK(cudaMalloc(&d_hpass, need));
        cap = need;
    }

    // Pass 1: horizontal normalised min
    {
        dim3 block(256, 1);
        dim3 grid((W+255)/256, H);
        size_t smem = (256 + 2*radius)*sizeof(float);
        kernelNormMinH<<<grid, block, smem, stream>>>(
            d_input, d_hpass, W, H, radius,
            1.0f/atmR, 1.0f/atmG, 1.0f/atmB);
    }

    // Pass 2: vertical min ? transmission
    {
        dim3 block(32, 8);
        dim3 grid((W+31)/32, (H+7)/8);
        kernelTransmissionV<<<grid, block, 0, stream>>>(
            d_hpass, d_trans_raw, W, H, radius, omega);
    }
}
