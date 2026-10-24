/*
 * dcp_openmp.cpp
 * OpenMP-parallelized Dark Channel Prior dehazing (multi-core CPU).
 *
 * Every compute-heavy step is annotated with #pragma omp parallel for.
 * This serves as the CPU performance baseline to compare against CUDA.
 *
 * Thread safety: all operations use private loop variables; shared read-only
 * data (A, radius, etc.) is accessed safely by multiple threads.
 */
#include "utils.hpp"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <atomic>

// --- 1. Dark Channel (OpenMP over rows) --------------------------------------
cv::Mat darkChannelOMP(const cv::Mat& img, int radius, int nthreads) {
    int H = img.rows, W = img.cols;
    cv::Mat dc(H, W, CV_32F);

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int y = 0; y < H; ++y) {
        float* dc_row = dc.ptr<float>(y);
        int y0 = std::max(0, y-radius), y1 = std::min(H-1, y+radius);
        for (int x = 0; x < W; ++x) {
            float mn = 1.0f;
            int x0 = std::max(0, x-radius), x1 = std::min(W-1, x+radius);
            for (int ny = y0; ny <= y1; ++ny) {
                const float* row = img.ptr<float>(ny);
                for (int nx = x0; nx <= x1; ++nx) {
                    const float* p = row + nx*3;
                    mn = std::min(mn, std::min({p[0], p[1], p[2]}));
                }
            }
            dc_row[x] = mn;
        }
    }
    return dc;
}

// --- 2. Atmospheric Light (parallel partial sort via per-thread local top-K) --
cv::Vec3f estimateAtmLightOMP(const cv::Mat& img, const cv::Mat& dc,
                              int nthreads)
{
    int N   = img.rows * img.cols;
    int topK = std::max(1, (int)(ATM_TOP_FRAC * N));

    // Each thread collects its local top-K, then we merge
    int T = std::min(nthreads, omp_get_max_threads());
    std::vector<std::vector<std::pair<float,int>>> local_tops(T);

    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        auto& lt = local_tops[tid];
        lt.reserve(topK + N/T);

        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            int ry = i / img.cols, rx = i % img.cols;
            lt.push_back({ dc.at<float>(ry, rx), i });
        }

        // Keep only top-K locally
        if ((int)lt.size() > topK) {
            std::partial_sort(lt.begin(), lt.begin()+topK, lt.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });
            lt.resize(topK);
        }
    }

    // Merge local top-K results
    std::vector<std::pair<float,int>> merged;
    merged.reserve(T * topK);
    for (auto& lt : local_tops)
        merged.insert(merged.end(), lt.begin(), lt.end());
    std::partial_sort(merged.begin(), merged.begin()+topK, merged.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    cv::Vec3f A(0,0,0); float best_lum = -1.f;
    for (int k = 0; k < topK; ++k) {
        int idx = merged[k].second;
        int ry = idx / img.cols, rx = idx % img.cols;
        const float* p = img.ptr<float>(ry) + rx*3;
        float lum = 0.2126f*p[2] + 0.7152f*p[1] + 0.0722f*p[0];
        if (lum > best_lum) { best_lum = lum; A = {p[0], p[1], p[2]}; }
    }
    for (int c = 0; c < 3; ++c) A[c] = std::max(A[c], 1e-3f);
    return A;
}

// --- 3. Raw Transmission (OpenMP per row) -------------------------------------
cv::Mat rawTransmissionOMP(const cv::Mat& img, const cv::Vec3f& A,
                           int radius, float omega, int nthreads)
{
    int H = img.rows, W = img.cols;

    // Normalise image by A
    cv::Mat norm(H, W, CV_32FC3);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int y = 0; y < H; ++y) {
        const float* src = img.ptr<float>(y);
        float* dst = norm.ptr<float>(y);
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                dst[x*3+c] = src[x*3+c] / A[c];
    }

    cv::Mat dc_norm = darkChannelOMP(norm, radius, nthreads);

    cv::Mat t(H, W, CV_32F);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < H*W; ++i)
        t.at<float>(i/W, i%W) = 1.0f - omega * dc_norm.at<float>(i/W, i%W);
    return t;
}

// --- 4. Guided Filter (OpenCV box-filter, then OpenMP for pointwise ops) ------
static cv::Mat boxFilter32F(const cv::Mat& src, int r) {
    cv::Mat dst;
    cv::boxFilter(src, dst, CV_32F, cv::Size(2*r+1, 2*r+1),
                  cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    return dst;
}

cv::Mat guidedFilterOMP(const cv::Mat& guide, const cv::Mat& src,
                        int radius, float eps, int nthreads)
{
    // OpenCV box-filter is already internally threaded.
    // We add OpenMP to the point-wise arithmetic steps.
    cv::Mat mean_I  = boxFilter32F(guide, radius);
    cv::Mat mean_p  = boxFilter32F(src,   radius);
    cv::Mat mean_Ip = boxFilter32F(guide.mul(src),   radius);
    cv::Mat mean_II = boxFilter32F(guide.mul(guide),  radius);

    int N = guide.rows * guide.cols;
    cv::Mat a(guide.rows, guide.cols, CV_32F);
    cv::Mat b(guide.rows, guide.cols, CV_32F);

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < N; ++i) {
        float cov = mean_Ip.at<float>(i) - mean_I.at<float>(i)*mean_p.at<float>(i);
        float var  = mean_II.at<float>(i) - mean_I.at<float>(i)*mean_I.at<float>(i);
        a.at<float>(i) = cov / (var + eps);
        b.at<float>(i) = mean_p.at<float>(i) - a.at<float>(i)*mean_I.at<float>(i);
    }

    cv::Mat mean_a = boxFilter32F(a, radius);
    cv::Mat mean_b = boxFilter32F(b, radius);

    cv::Mat out(guide.rows, guide.cols, CV_32F);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < N; ++i)
        out.at<float>(i) = mean_a.at<float>(i)*guide.at<float>(i) + mean_b.at<float>(i);
    return out;
}

// --- 5. Scene Recovery (OpenMP over pixels) -----------------------------------
cv::Mat recoverSceneOMP(const cv::Mat& img, const cv::Mat& t_refined,
                        const cv::Vec3f& A, float t_min, int nthreads)
{
    int H = img.rows, W = img.cols;
    cv::Mat out(H, W, CV_32FC3);

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int y = 0; y < H; ++y) {
        const float* p = img.ptr<float>(y);
        float* q = out.ptr<float>(y);
        const float* tp = t_refined.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            float t = std::max(tp[x], t_min);
            for (int c = 0; c < 3; ++c)
                q[x*3+c] = std::clamp((p[x*3+c] - A[c]) / t + A[c], 0.0f, 1.0f);
        }
    }
    return out;
}

// --- Full OpenMP Pipeline -----------------------------------------------------
cv::Mat dehazeOpenMP(const cv::Mat& hazy_bgr_float, int nthreads) {
    cv::Mat guide;
    cv::cvtColor(hazy_bgr_float, guide, cv::COLOR_BGR2GRAY);

    cv::Mat dc    = darkChannelOMP(hazy_bgr_float, PATCH_RADIUS, nthreads);
    cv::Vec3f A   = estimateAtmLightOMP(hazy_bgr_float, dc, nthreads);
    cv::Mat t_raw = rawTransmissionOMP(hazy_bgr_float, A, PATCH_RADIUS, OMEGA, nthreads);
    cv::Mat t_ref = guidedFilterOMP(guide, t_raw, GUIDED_RADIUS, GUIDED_EPS, nthreads);
    return          recoverSceneOMP(hazy_bgr_float, t_ref, A, T_MIN, nthreads);
}
