/*
 * recovery.cu
 * CUDA Kernel 5: Scene Radiance Recovery + GpuDehazeBuffers + Full pipeline.
 *
 * Formula:  J^c(x) = (I^c(x) - A^c) / max(t(x), t_min) + A^c
 *
 * Optimization techniques:
 *   - 32x8 thread blocks ? 256 threads/block, maximizing SM occupancy
 *     while keeping 2D spatial locality for cache-friendly access patterns.
 *   - Fully coalesced global memory reads (consecutive threads access
 *     consecutive addresses in row-major layout).
 *   - __ldg() intrinsic for the transmission map ? routes through
 *     L1 texture cache (read-only path, avoids L1 eviction from writes).
 *   - __saturatef() for clamping to [0,1] — single hardware instruction.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dehazer.cuh"
#include "utils.hpp"

// Forward declarations of launchers from other files
void launchDarkChannel(const float*, float*, int, int, int, cudaStream_t);
void launchAtmLight(const float*, const float*, int, int,
                    float&, float&, float&, cudaStream_t);
void launchTransmission(const float*, float*, int, int, int,
                        float, float, float, float, cudaStream_t);
void launchGuidedFilter(const float*, const float*, float*, float*,
                        int, int, int, float, cudaStream_t);

// --- Greyscale conversion kernel ---------------------------------------------
__global__ void kernelGreyscale(
    const float* __restrict__ bgr,
    float* __restrict__       grey,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float b = bgr[i*3+0], g = bgr[i*3+1], r = bgr[i*3+2];
        grey[i] = 0.0722f*b + 0.7152f*g + 0.2126f*r;   // BT.709 luminance
    }
}

void launchGreyscale(const float* d_bgr, float* d_grey,
                     int W, int H, cudaStream_t stream)
{
    int N = W * H;
    dim3 block(256), grid((N + 255)/256);
    kernelGreyscale<<<grid, block, 0, stream>>>(d_bgr, d_grey, N);
}

// --- Scene Recovery kernel ----------------------------------------------------
__global__ void kernelRecovery(
    const float* __restrict__ input,    // [H*W*3] float BGR in [0,1]
    const float* __restrict__ trans,    // [H*W]   refined transmission
    float*       __restrict__ output,   // [H*W*3] float BGR out in [0,1]
    int W, int H,
    float atmB, float atmG, float atmR, // BGR order matching input
    float t_min)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float t  = fmaxf(__ldg(&trans[idx]), t_min);

    float b_in = input[idx*3 + 0];
    float g_in = input[idx*3 + 1];
    float r_in = input[idx*3 + 2];

    output[idx*3 + 0] = __saturatef((b_in - atmB) / t + atmB);
    output[idx*3 + 1] = __saturatef((g_in - atmG) / t + atmG);
    output[idx*3 + 2] = __saturatef((r_in - atmR) / t + atmR);
}

void launchRecovery(
    const float* d_input, const float* d_trans,
    float* d_output,
    int W, int H,
    float atmR, float atmG, float atmB,
    float t_min, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((W + 31)/32, (H + 7)/8);
    // Note: atm passed in RGB order from the pipeline, kernel expects BGR order internally
    kernelRecovery<<<grid, block, 0, stream>>>(
        d_input, d_trans, d_output, W, H,
        atmB, atmG, atmR,  // reorder to BGR for kernel
        t_min);
}

// --- GpuDehazeBuffers implementation -----------------------------------------
void GpuDehazeBuffers::allocate(int w, int h) {
    width = w; height = h;
    size_t px = (size_t)w * h;
    CUDA_CHECK(cudaMalloc(&d_input,     px * 3  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dark,      px      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trans_raw, px      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trans_ref, px      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,    px * 3  * sizeof(float)));
    // Guided filter scratch: 11 float arrays of size px
    CUDA_CHECK(cudaMalloc(&d_guide_buf, px * 11 * sizeof(float)));
}

void GpuDehazeBuffers::free() {
    auto freePtr = [](float*& p){ if (p) { cudaFree(p); p = nullptr; } };
    freePtr(d_input);
    freePtr(d_dark);
    freePtr(d_trans_raw);
    freePtr(d_trans_ref);
    freePtr(d_output);
    freePtr(d_guide_buf);
    width = height = 0;
}

// --- Full GPU Pipeline --------------------------------------------------------
cv::Mat dehazeCUDA(const cv::Mat& hazy_bgr_float, GpuDehazeBuffers& buf) {
    int W = hazy_bgr_float.cols;
    int H = hazy_bgr_float.rows;
    size_t px = (size_t)W * H;

    // Re-allocate buffers if image size changed
    if (W != buf.width || H != buf.height) {
        buf.free();
        buf.allocate(W, H);
    }

    // -- 1. Upload hazy image to GPU (host ? device) --
    CUDA_CHECK(cudaMemcpy(buf.d_input,
                          hazy_bgr_float.ptr<float>(),
                          px * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // -- 2. Dark Channel (Kernel 1: separable min-filter, shared memory) --
    launchDarkChannel(buf.d_input, buf.d_dark, W, H, PATCH_RADIUS, 0);

    // -- 3. Atmospheric Light (Kernel 2: histogram + warp reduction) --
    float atmR, atmG, atmB;
    launchAtmLight(buf.d_input, buf.d_dark, W, H, atmR, atmG, atmB, 0);

    // -- 4. Raw Transmission (Kernel 3: texture memory, separable) --
    launchTransmission(buf.d_input, buf.d_trans_raw, W, H, PATCH_RADIUS,
                       atmR, atmG, atmB, OMEGA, 0);

    // -- 5. Greyscale guide for Guided Filter (reuse d_dark buffer) --
    launchGreyscale(buf.d_input, buf.d_dark, W, H, 0);

    // -- 6. Guided Filter (Kernel 4: integral images via SAT, Thrust) --
    launchGuidedFilter(buf.d_dark,       // guide = greyscale hazy
                       buf.d_trans_raw,  // src   = raw transmission
                       buf.d_trans_ref,  // dst   = refined transmission
                       buf.d_guide_buf,  // scratch (11*px floats)
                       W, H, GUIDED_RADIUS, GUIDED_EPS, 0);

    // -- 7. Scene Recovery (Kernel 5: coalesced pointwise, __ldg) --
    launchRecovery(buf.d_input, buf.d_trans_ref, buf.d_output,
                   W, H, atmR, atmG, atmB, T_MIN, 0);

    // Synchronise before downloading
    CUDA_CHECK(cudaDeviceSynchronize());

    // -- 8. Download result (device ? host) --
    cv::Mat result(H, W, CV_32FC3);
    CUDA_CHECK(cudaMemcpy(result.ptr<float>(),
                          buf.d_output,
                          px * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}
