/*
 * batch_bench.cpp
 * 3-way benchmarking: Sequential CPU vs OpenMP vs CUDA
 *
 * Runs all three dehazing modes on the RESIDE SOTS test set (or any
 * paired hazy/GT image directory), records:
 *   - Per-image processing time (ms)
 *   - PSNR and SSIM vs ground truth
 *   - Average, min, max statistics
 *
 * Results are saved to results/benchmark.csv for plotting.
 *
 * Usage:
 *   ./benchmark --hazy <dir> --gt <dir> --output <dir>
 *               [--mode cpu|omp|cuda|all] [--threads <N>] [--limit <N>]
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <omp.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "dehazer.cuh"

namespace fs = std::filesystem;

// Forward declarations
cv::Mat dehazeCPU(const cv::Mat& hazy);
cv::Mat dehazeOpenMP(const cv::Mat& hazy, int nthreads);
double computePSNR(const cv::Mat& pred, const cv::Mat& gt);
double computeSSIM(const cv::Mat& pred, const cv::Mat& gt);
std::vector<std::string> collectImagePaths(const std::string& dir,
                                            const std::vector<std::string>& exts);

// --- Benchmark result for one image ------------------------------------------
struct ImageResult {
    std::string filename;
    double time_ms;
    double psnr;
    double ssim;
};

// --- Run benchmark for a given mode ------------------------------------------
std::vector<ImageResult> runBenchmark(
    const std::string& mode,
    const std::vector<std::string>& hazy_paths,
    const std::vector<std::string>& gt_paths,
    const std::string& out_dir,
    int nthreads)
{
    std::vector<ImageResult> results;
    results.reserve(hazy_paths.size());

    fs::create_directories(out_dir + "/" + mode);

    GpuDehazeBuffers cuda_buf = {};

    // Build lookup maps for ground truth images (by full filename and by stem/prefix)
    std::map<std::string, std::string> gt_by_name;
    std::map<std::string, std::string> gt_by_stem;
    for (const auto& g : gt_paths) {
        fs::path gp(g);
        gt_by_name[gp.filename().string()] = g;
        gt_by_stem[gp.stem().string()] = g;
    }

    // CUDA warm-up: process first image to initialise GPU context
    if (mode == "cuda" && !hazy_paths.empty()) {
        cv::Mat warmup = loadBGRFloat(hazy_paths[0]);
        dehazeCUDA(warmup, cuda_buf);  // discard result
        printf("[CUDA] GPU warm-up done.\n");
    }

    for (size_t i = 0; i < hazy_paths.size(); ++i) {
        // Load hazy
        cv::Mat hazy;
        try { hazy = loadBGRFloat(hazy_paths[i]); }
        catch (...) { fprintf(stderr, "Skip (load fail): %s\n", hazy_paths[i].c_str()); continue; }

        // Process
        cv::Mat dehazed;
        double t_ms = 0.0;

        if (mode == "cpu") {
            CpuTimer t; t.start();
            dehazed = dehazeCPU(hazy);
            t_ms = t.elapsedMs();

        } else if (mode == "omp") {
            CpuTimer t; t.start();
            dehazed = dehazeOpenMP(hazy, nthreads);
            t_ms = t.elapsedMs();

        } else if (mode == "cuda") {
            GpuTimer gt; gt.start();
            dehazed = dehazeCUDA(hazy, cuda_buf);
            t_ms = gt.elapsedMs();
        }

        // Metrics: match ground truth by filename, prefix (e.g. 1400_1 -> 1400), or index fallback
        double psnr = 0.0, ssim = 0.0;
        std::string matched_gt = "";
        fs::path p(hazy_paths[i]);
        std::string h_fname = p.filename().string();
        std::string h_stem = p.stem().string();

        if (gt_by_name.count(h_fname)) {
            matched_gt = gt_by_name[h_fname];
        } else if (gt_by_stem.count(h_stem)) {
            matched_gt = gt_by_stem[h_stem];
        } else {
            // Check prefix before '_' (e.g. 1400_1 -> 1400)
            auto underscore_pos = h_stem.find('_');
            if (underscore_pos != std::string::npos) {
                std::string prefix = h_stem.substr(0, underscore_pos);
                if (gt_by_stem.count(prefix)) {
                    matched_gt = gt_by_stem[prefix];
                }
            }
        }
        if (matched_gt.empty() && i < gt_paths.size()) {
            matched_gt = gt_paths[i];
        }

        if (!matched_gt.empty()) {
            try {
                cv::Mat gt = loadBGRFloat(matched_gt);
                psnr = computePSNR(dehazed, gt);
                ssim = computeSSIM(dehazed, gt);
            } catch (...) {}
        }

        // Save output
        std::string out_file = out_dir + "/" + mode + "/" + p.filename().string();
        try { saveFloat(out_file, dehazed); } catch (...) {}

        results.push_back({ p.filename().string(), t_ms, psnr, ssim });

        if (!matched_gt.empty()) {
            printf("[%s] %3zu/%-3zu  %-30s  %7.1f ms  PSNR=%5.2f dB  SSIM=%.4f\n",
                   mode.c_str(), i+1, hazy_paths.size(),
                   p.filename().string().c_str(), t_ms, psnr, ssim);
        } else {
            printf("[%s] %3zu/%-3zu  %-30s  %7.1f ms\n",
                   mode.c_str(), i+1, hazy_paths.size(),
                   p.filename().string().c_str(), t_ms);
        }
    }

    if (mode == "cuda") cuda_buf.free();

    // Print summary
    if (!results.empty()) {
        double avg_t    = std::accumulate(results.begin(), results.end(), 0.0,
                            [](double s, auto& r){ return s + r.time_ms; }) / results.size();
        double avg_psnr = std::accumulate(results.begin(), results.end(), 0.0,
                            [](double s, auto& r){ return s + r.psnr; }) / results.size();
        double avg_ssim = std::accumulate(results.begin(), results.end(), 0.0,
                            [](double s, auto& r){ return s + r.ssim; }) / results.size();
        double min_t    = std::min_element(results.begin(), results.end(),
                            [](auto& a, auto& b){ return a.time_ms < b.time_ms; })->time_ms;
        double max_t    = std::max_element(results.begin(), results.end(),
                            [](auto& a, auto& b){ return a.time_ms < b.time_ms; })->time_ms;

        printf("\n+------------------------------------------------------+\n");
        printf("|  Mode: %-6s | N=%3zu images                          |\n",
               mode.c_str(), results.size());
        printf("|  Time: avg=%-6.1f ms  min=%-6.1f ms  max=%-6.1f ms  |\n",
               avg_t, min_t, max_t);
        printf("|  PSNR: avg=%-5.2f dB  |  SSIM: avg=%-6.4f          |\n",
               avg_psnr, avg_ssim);
        printf("+------------------------------------------------------+\n\n");
    }

    return results;
}

// --- Save CSV -----------------------------------------------------------------
void saveCSV(const std::string& path, const std::string& mode,
             const std::vector<ImageResult>& results) {
    bool write_header = !fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (write_header) f << "mode,filename,time_ms,psnr,ssim\n";
    for (auto& r : results)
        f << mode << "," << r.filename << ","
          << r.time_ms << "," << r.psnr << "," << r.ssim << "\n";
}

// --- Main ---------------------------------------------------------------------
int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc-1; i += 2) args[argv[i]] = argv[i+1];

    std::string hazy_dir = args.count("--hazy")   ? args["--hazy"]   : "data/SOTS/indoor/hazy";
    std::string gt_dir   = args.count("--gt")     ? args["--gt"]     : "data/SOTS/indoor/gt";
    std::string out_dir  = args.count("--output") ? args["--output"] : "results";
    std::string mode     = args.count("--mode")   ? args["--mode"]   : "all";
    int nthreads         = args.count("--threads")? std::stoi(args["--threads"]) : omp_get_max_threads();
    int limit            = args.count("--limit")  ? std::stoi(args["--limit"])   : INT_MAX;

    printf("+----------------------------------------------------------+\n");
    printf("|           CUDA Image Dehazing Benchmark                  |\n");
    printf("+----------------------------------------------------------+\n");
    printf("Hazy dir : %s\n", hazy_dir.c_str());
    printf("GT dir   : %s\n", gt_dir.c_str());
    printf("Mode     : %s\n", mode.c_str());
    printf("Threads  : %d  (for OMP)\n\n", nthreads);

    // Collect image paths (parallel with OpenMP)
    auto hazy_paths = collectImagePaths(hazy_dir, {".png", ".jpg", ".jpeg", ".bmp"});
    auto gt_paths   = collectImagePaths(gt_dir,   {".png", ".jpg", ".jpeg", ".bmp"});

    if (hazy_paths.empty()) {
        fprintf(stderr, "No images found in: %s\n", hazy_dir.c_str());
        return 1;
    }

    // Apply limit
    if ((int)hazy_paths.size() > limit) hazy_paths.resize(limit);
    if (!gt_paths.empty() && (int)gt_paths.size() > limit) gt_paths.resize(limit);

    if (gt_paths.empty()) {
        printf("Found %zu hazy images (no GT images found; PSNR/SSIM evaluation will be skipped)\n\n",
               hazy_paths.size());
    } else {
        printf("Found %zu hazy / %zu GT images\n\n", hazy_paths.size(), gt_paths.size());
    }

    std::string csv_path = out_dir + "/benchmark.csv";
    fs::create_directories(out_dir);
    // Clear existing CSV
    if (fs::exists(csv_path)) fs::remove(csv_path);

    std::vector<std::string> modes_to_run;
    if (mode == "all") modes_to_run = {"cpu", "omp", "cuda"};
    else               modes_to_run = {mode};

    for (auto& m : modes_to_run) {
        printf("=== Running mode: %s ==================================\n\n", m.c_str());
        auto results = runBenchmark(m, hazy_paths, gt_paths, out_dir, nthreads);
        saveCSV(csv_path, m, results);
    }

    printf("Results saved to: %s\n", csv_path.c_str());
    printf("Run `python3 scripts/report.py --csv %s` to generate charts.\n", csv_path.c_str());
    return 0;
}
