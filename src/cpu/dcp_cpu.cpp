/*
 * dcp_cpu.cpp
 * Sequential (single-threaded) Dark Channel Prior dehazing.
 * This is the baseline reference implementation - correctness over speed.
 *
 * Algorithm:  He et al. "Single Image Haze Removal Using Dark Channel Prior"
 *             CVPR 2009 / TPAMI 2011
 *
 * Pipeline:
 *   1. Compute dark channel  J^dark(x) = min_{y in O(x)} min_c J^c(y)
 *   2. Estimate atmospheric light A from top 0.1% brightest dark-channel pixels
 *   3. Estimate raw transmission t_0(x) = 1 - ? * dark(I/A)
 *   4. Refine transmission via Guided Filter (using hazy image as guide)
 *   5. Recover scene radiance  J(x) = (I(x)-A) / max(t(x), t_min) + A
 */
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

// --- 1. Dark Channel ----------------------------------------------------------
cv::Mat darkChannel(const cv::Mat& img, int radius) {
    int H = img.rows, W = img.cols;
    cv::Mat dc(H, W, CV_32F);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float mn = 1.0f;
            int y0 = std::max(0, y-radius), y1 = std::min(H-1, y+radius);
            int x0 = std::max(0, x-radius), x1 = std::min(W-1, x+radius);
            for (int ny = y0; ny <= y1; ++ny)
                for (int nx = x0; nx <= x1; ++nx) {
                    const float* p = img.ptr<float>(ny) + nx*3;
                    mn = std::min(mn, std::min({p[0], p[1], p[2]}));
                }
            dc.at<float>(y, x) = mn;
        }
    }
    return dc;
}

// --- 2. Atmospheric Light -----------------------------------------------------
// Returns the atmospheric light A as a 3-element array [B, G, R]
cv::Vec3f estimateAtmLight(const cv::Mat& img, const cv::Mat& dc) {
    int N = img.rows * img.cols;
    int topK = std::max(1, (int)(ATM_TOP_FRAC * N));

    // Flatten dark channel with indices
    std::vector<std::pair<float,int>> vals(N);
    for (int i = 0; i < N; ++i)
        vals[i] = { dc.at<float>(i/img.cols, i%img.cols), i };

    // Partial sort - only need top-K
    std::partial_sort(vals.begin(), vals.begin()+topK, vals.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    // Among those top-K pixels, find the one with highest luminance in hazy img
    cv::Vec3f A(0, 0, 0);
    float best_lum = -1.f;
    for (int k = 0; k < topK; ++k) {
        int idx = vals[k].second;
        int ry = idx / img.cols, rx = idx % img.cols;
        const float* p = img.ptr<float>(ry) + rx*3;
        float lum = 0.2126f*p[2] + 0.7152f*p[1] + 0.0722f*p[0]; // R G B
        if (lum > best_lum) { best_lum = lum; A = {p[0], p[1], p[2]}; }
    }
    // Clamp A to avoid divide-by-zero
    for (int c = 0; c < 3; ++c) A[c] = std::max(A[c], 1e-3f);
    return A;
}

// --- 3. Raw Transmission ------------------------------------------------------
cv::Mat rawTransmission(const cv::Mat& img, const cv::Vec3f& A,
                        int radius, float omega)
{
    int H = img.rows, W = img.cols;
    cv::Mat t(H, W, CV_32F);

    // Normalise image by A, then dark-channel of normalised image
    cv::Mat norm(H, W, CV_32FC3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float* p = img.ptr<float>(y) + x*3;
            float* q = norm.ptr<float>(y) + x*3;
            for (int c = 0; c < 3; ++c) q[c] = p[c] / A[c];
        }

    cv::Mat dc_norm = darkChannel(norm, radius);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            t.at<float>(y,x) = 1.0f - omega * dc_norm.at<float>(y,x);

    return t;
}

// --- 4. Guided Filter ---------------------------------------------------------
// Fast O(N) guided filter via integral images (box-filter approximation)
static cv::Mat boxFilter32F(const cv::Mat& src, int r) {
    cv::Mat dst;
    cv::boxFilter(src, dst, CV_32F, cv::Size(2*r+1, 2*r+1),
                  cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    return dst;
}

cv::Mat guidedFilter(const cv::Mat& guide, const cv::Mat& src,
                     int radius, float eps)
{
    cv::Mat mean_I  = boxFilter32F(guide, radius);
    cv::Mat mean_p  = boxFilter32F(src,   radius);
    cv::Mat mean_Ip = boxFilter32F(guide.mul(src), radius);
    cv::Mat mean_II = boxFilter32F(guide.mul(guide), radius);

    cv::Mat cov_Ip  = mean_Ip - mean_I.mul(mean_p);
    cv::Mat var_I   = mean_II - mean_I.mul(mean_I);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = boxFilter32F(a, radius);
    cv::Mat mean_b = boxFilter32F(b, radius);

    return mean_a.mul(guide) + mean_b;
}

// --- 5. Scene Recovery --------------------------------------------------------
cv::Mat recoverScene(const cv::Mat& img, const cv::Mat& t_refined,
                     const cv::Vec3f& A, float t_min)
{
    int H = img.rows, W = img.cols;
    cv::Mat out(H, W, CV_32FC3);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float* p = img.ptr<float>(y) + x*3;
            float* q = out.ptr<float>(y) + x*3;
            float t = std::max(t_refined.at<float>(y,x), t_min);
            for (int c = 0; c < 3; ++c)
                q[c] = std::clamp((p[c] - A[c]) / t + A[c], 0.0f, 1.0f);
        }
    return out;
}

// --- Full CPU Pipeline --------------------------------------------------------
cv::Mat dehazeCPU(const cv::Mat& hazy_bgr_float) {
    // Use single grey channel as guided filter guide
    cv::Mat guide;
    cv::cvtColor(hazy_bgr_float, guide, cv::COLOR_BGR2GRAY);

    cv::Mat dc      = darkChannel(hazy_bgr_float, PATCH_RADIUS);
    cv::Vec3f A     = estimateAtmLight(hazy_bgr_float, dc);
    cv::Mat t_raw   = rawTransmission(hazy_bgr_float, A, PATCH_RADIUS, OMEGA);
    cv::Mat t_ref   = guidedFilter(guide, t_raw, GUIDED_RADIUS, GUIDED_EPS);
    return            recoverScene(hazy_bgr_float, t_ref, A, T_MIN);
}
