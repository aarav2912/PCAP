#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// --- GPU Buffer holding all intermediate CUDA allocations ---------------------
struct GpuDehazeBuffers {
    float* d_input;        // [H x W x 3]  input hazy image (float, BGR planar)
    float* d_dark;         // [H x W]      dark channel
    float* d_trans_raw;    // [H x W]      raw transmission map
    float* d_trans_ref;    // [H x W]      refined transmission (guided filter)
    float* d_output;       // [H x W x 3]  scene radiance output
    float* d_guide_buf;    // scratch for guided filter (6 x H*W floats)
    int    width, height;

    void allocate(int w, int h);
    void free();
};

// --- Kernel launchers declared here, defined in .cu files ---------------------
void launchDarkChannel(
    const float* d_input, float* d_dark,
    int width, int height, int radius, cudaStream_t stream = 0);

void launchAtmLight(
    const float* d_input, const float* d_dark,
    int width, int height,
    float& outR, float& outG, float& outB, cudaStream_t stream = 0);

void launchTransmission(
    const float* d_input, float* d_trans_raw,
    int width, int height, int radius,
    float atmR, float atmG, float atmB,
    float omega, cudaStream_t stream = 0);

void launchGuidedFilter(
    const float* d_guide,   // the hazy image grey channel as guide
    const float* d_src,     // raw transmission
    float*       d_dst,     // refined transmission
    float*       d_scratch, // [6 * W * H] scratch
    int width, int height, int radius, float eps,
    cudaStream_t stream = 0);

void launchRecovery(
    const float* d_input, const float* d_trans,
    float* d_output,
    int width, int height,
    float atmR, float atmG, float atmB,
    float t_min, cudaStream_t stream = 0);

// --- Full GPU pipeline (single call) -----------------------------------------
cv::Mat dehazeCUDA(const cv::Mat& hazy_bgr_float, GpuDehazeBuffers& buf);
