#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// --- CUDA Error Checking ------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// --- Timer -------------------------------------------------------------------
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start_t;
    void start() { start_t = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_t).count();
    }
};

struct GpuTimer {
    cudaEvent_t start_e, stop_e;
    GpuTimer()  { cudaEventCreate(&start_e); cudaEventCreate(&stop_e); }
    ~GpuTimer() { cudaEventDestroy(start_e); cudaEventDestroy(stop_e); }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_e, s); }
    float elapsedMs(cudaStream_t s = 0) {
        cudaEventRecord(stop_e, s);
        cudaEventSynchronize(stop_e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_e, stop_e);
        return ms;
    }
};

// --- Common constants ---------------------------------------------------------
constexpr int   PATCH_RADIUS   = 7;    // dark channel window = 15x15
constexpr float OMEGA          = 0.95f; // haze retention factor
constexpr float T_MIN          = 0.10f; // minimum transmission floor
constexpr float GUIDED_EPS     = 1e-3f; // guided filter regulariser
constexpr int   GUIDED_RADIUS  = 40;   // guided filter window radius
constexpr float ATM_TOP_FRAC   = 0.001f; // top 0.1% for atmospheric light

// --- Image loading helper -----------------------------------------------------
inline cv::Mat loadBGRFloat(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Cannot load: " + path);
    cv::Mat f; img.convertTo(f, CV_32FC3, 1.0/255.0);
    return f;
}

inline void saveFloat(const std::string& path, const cv::Mat& img) {
    cv::Mat out; img.convertTo(out, CV_8UC3, 255.0);
    cv::imwrite(path, out);
}
