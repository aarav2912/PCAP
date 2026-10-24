/*
 * main.cu
 * Entry point for the CUDA Image Dehazing pipeline.
 *
 * Usage:
 *   ./dehazer --mode [cpu|omp|cuda] --input <path> --output <path>
 *             [--gt <gt_path>]       # for PSNR/SSIM evaluation
 *             [--threads <N>]        # for omp mode (default: all cores)
 *             [--radius <r>]         # patch radius (default: 7)
 *
 * For batch benchmark, use the `benchmark` binary instead.
 */
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <map>
#include "utils.hpp"
#include <omp.h>
#include "dehazer.cuh"

// Forward declarations from CPU files
cv::Mat dehazeCPU(const cv::Mat& hazy);
cv::Mat dehazeOpenMP(const cv::Mat& hazy, int nthreads);
double computePSNR(const cv::Mat& pred, const cv::Mat& gt);
double computeSSIM(const cv::Mat& pred, const cv::Mat& gt);

// --- CLI argument parser ------------------------------------------------------
std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc-1; i += 2)
        args[argv[i]] = argv[i+1];
    return args;
}

// --- Print GPU info ------------------------------------------------------------
void printGpuInfo() {
    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev == 0) { std::cout << "[CUDA] No GPU found.\n"; return; }
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | Compute %d.%d | %.0f MB VRAM | %d SMs | %.1f GHz\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e6,
           prop.multiProcessorCount,
           prop.clockRate / 1e6);
}

// --- Main ---------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: dehazer --mode [cpu|omp|cuda] --input <path> --output <path>\n"
            "               [--gt <path>] [--threads <N>]\n");
        return 1;
    }

    auto args = parseArgs(argc, argv);
    std::string mode    = args.count("--mode")    ? args["--mode"]    : "cuda";
    std::string in_path = args.count("--input")   ? args["--input"]   : "";
    std::string out_path= args.count("--output")  ? args["--output"]  : "output.png";
    std::string gt_path = args.count("--gt")      ? args["--gt"]      : "";
    int nthreads        = args.count("--threads") ? std::stoi(args["--threads"]) : omp_get_max_threads();

    if (in_path.empty()) { fprintf(stderr, "Error: --input required\n"); return 1; }

    // -- Load hazy image --
    cv::Mat hazy;
    try { hazy = loadBGRFloat(in_path); }
    catch (std::exception& e) { fprintf(stderr, "Load error: %s\n", e.what()); return 1; }

    printf("[Info] Image: %dx%d  Mode: %s\n", hazy.cols, hazy.rows, mode.c_str());
    if (mode == "cuda") printGpuInfo();

    // -- Run pipeline --
    cv::Mat result;
    double elapsed_ms = 0.0;

    if (mode == "cpu") {
        CpuTimer t; t.start();
        result = dehazeCPU(hazy);
        elapsed_ms = t.elapsedMs();

    } else if (mode == "omp") {
        printf("[OMP] Using %d threads\n", nthreads);
        CpuTimer t; t.start();
        result = dehazeOpenMP(hazy, nthreads);
        elapsed_ms = t.elapsedMs();

    } else if (mode == "cuda") {
        GpuDehazeBuffers buf = {};
        // Warm-up (first call includes JIT and alloc overhead)
        {
            GpuTimer wt; wt.start();
            result = dehazeCUDA(hazy, buf);
            float wms = wt.elapsedMs();
            printf("[CUDA] Warm-up: %.2f ms (JIT + alloc included)\n", wms);
        }
        // Timed run
        {
            GpuTimer gt_; gt_.start();
            result = dehazeCUDA(hazy, buf);
            elapsed_ms = gt_.elapsedMs();
        }
        buf.free();

    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode.c_str()); return 1;
    }

    printf("[%s] Processing time: %.2f ms\n", mode.c_str(), elapsed_ms);

    // -- Evaluate vs ground truth --
    if (!gt_path.empty()) {
        try {
            cv::Mat gt = loadBGRFloat(gt_path);
            double psnr = computePSNR(result, gt);
            double ssim = computeSSIM(result, gt);
            printf("[Metrics] PSNR: %.2f dB  |  SSIM: %.4f\n", psnr, ssim);
        } catch (std::exception& e) {
            fprintf(stderr, "[Metrics] Warning: %s\n", e.what());
        }
    }

    // -- Save result --
    try {
        saveFloat(out_path, result);
        printf("[Output] Saved to: %s\n", out_path.c_str());
    } catch (std::exception& e) {
        fprintf(stderr, "Save error: %s\n", e.what()); return 1;
    }

    return 0;
}

