/*
 * image_io.cpp
 * OpenMP-parallel batch image loading/saving for the dehazing benchmark.
 */
#include "utils.hpp"
#include <filesystem>
#include <algorithm>
#include <omp.h>

namespace fs = std::filesystem;

// --- Collect all image paths under a directory --------------------------------
std::vector<std::string> collectImagePaths(const std::string& dir,
                                           const std::vector<std::string>& exts)
{
    std::vector<std::string> paths;
    for (auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        for (auto& e : exts) {
            if (ext == e) { paths.push_back(entry.path().string()); break; }
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// --- Parallel batch loader using OpenMP --------------------------------------
//  Returns vector of [H x W x 3] float32 BGR images, same order as paths.
std::vector<cv::Mat> batchLoadFloatParallel(const std::vector<std::string>& paths,
                                             int num_threads)
{
    std::vector<cv::Mat> imgs(paths.size());
    std::vector<std::string> errors(paths.size());

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < (int)paths.size(); ++i) {
        try {
            imgs[i] = loadBGRFloat(paths[i]);
        } catch (std::exception& e) {
            errors[i] = e.what();
        }
    }

    // Report any load failures (outside parallel region)
    for (size_t i = 0; i < errors.size(); ++i) {
        if (!errors[i].empty())
            std::cerr << "[ImageIO] Failed to load " << paths[i] << ": " << errors[i] << "\n";
    }
    return imgs;
}

// --- Parallel batch saver -----------------------------------------------------
void batchSaveParallel(const std::vector<cv::Mat>& imgs,
                       const std::vector<std::string>& out_paths,
                       int num_threads)
{
    if (imgs.size() != out_paths.size()) {
        throw std::runtime_error("batchSave: imgs/paths size mismatch");
    }

    // Create output dirs (sequential, cheap)
    for (auto& p : out_paths) {
        fs::path fp(p);
        if (fp.has_parent_path()) fs::create_directories(fp.parent_path());
    }

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < (int)imgs.size(); ++i) {
        if (!imgs[i].empty()) saveFloat(out_paths[i], imgs[i]);
    }
}
