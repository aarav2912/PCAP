/*
 * metrics.cpp
 * PSNR and SSIM computation, with OpenMP parallelism for batch evaluation.
 *
 *  PSNR = 10 * log10(255^2 / MSE)
 *  SSIM  (Wang et al. 2004)  per luminance channel then averaged
 */
#include "utils.hpp"
#include <omp.h>
#include <cmath>
#include <numeric>

static constexpr double C1 = (0.01 * 255) * (0.01 * 255); // (K1*L)^2
static constexpr double C2 = (0.03 * 255) * (0.03 * 255); // (K2*L)^2
static constexpr int    SSIM_WIN = 11; // Gaussian window size

// --- Single-image PSNR (float images, values in [0,1]) -----------------------
double computePSNR(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size() && pred.type() == gt.type());
    cv::Mat diff;
    cv::absdiff(pred, gt, diff);
    diff = diff.mul(diff);                      // squared differences
    cv::Scalar s = cv::sum(diff);
    double mse = (s[0] + s[1] + s[2]) / (3.0 * pred.rows * pred.cols);
    if (mse < 1e-12) return 100.0;
    // Images are in [0,1]; scale MSE to 8-bit equivalent
    return 10.0 * std::log10(1.0 / mse);
}

// --- Single-channel SSIM helper ----------------------------------------------
static double ssimChannel(const cv::Mat& c1, const cv::Mat& c2) {
    cv::Mat mu1, mu2, sigma1, sigma2, sigma12;
    int k = SSIM_WIN;
    cv::GaussianBlur(c1,          mu1,     cv::Size(k,k), 1.5);
    cv::GaussianBlur(c2,          mu2,     cv::Size(k,k), 1.5);
    cv::GaussianBlur(c1.mul(c1),  sigma1,  cv::Size(k,k), 1.5);
    cv::GaussianBlur(c2.mul(c2),  sigma2,  cv::Size(k,k), 1.5);
    cv::GaussianBlur(c1.mul(c2),  sigma12, cv::Size(k,k), 1.5);

    sigma1  -= mu1.mul(mu1);
    sigma2  -= mu2.mul(mu2);
    sigma12 -= mu1.mul(mu2);

    cv::Mat num = (2.0 * mu1.mul(mu2) + C1).mul(2.0 * sigma12 + C2);
    cv::Mat den = (mu1.mul(mu1) + mu2.mul(mu2) + C1).mul(sigma1 + sigma2 + C2);

    cv::Mat ssim_map;
    cv::divide(num, den, ssim_map);
    return cv::mean(ssim_map)[0];
}

// --- Single-image SSIM (float images [0,1]) -----------------------------------
double computeSSIM(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size() && pred.channels() == 3);
    std::vector<cv::Mat> cp, cg;
    cv::split(pred, cp);
    cv::split(gt, cg);
    double s = 0.0;
    for (int c = 0; c < 3; ++c) s += ssimChannel(cp[c], cg[c]);
    return s / 3.0;
}

// --- Batch evaluation with OpenMP ---------------------------------------------
struct BatchMetrics {
    std::vector<double> psnr;
    std::vector<double> ssim;
    double avg_psnr, avg_ssim;
};

BatchMetrics evaluateBatch(const std::vector<cv::Mat>& preds,
                           const std::vector<cv::Mat>& gts,
                           int num_threads)
{
    size_t N = preds.size();
    BatchMetrics m;
    m.psnr.resize(N, 0.0);
    m.ssim.resize(N, 0.0);

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < (int)N; ++i) {
        if (!preds[i].empty() && !gts[i].empty()) {
            m.psnr[i] = computePSNR(preds[i], gts[i]);
            m.ssim[i] = computeSSIM(preds[i], gts[i]);
        }
    }

    m.avg_psnr = std::accumulate(m.psnr.begin(), m.psnr.end(), 0.0) / N;
    m.avg_ssim = std::accumulate(m.ssim.begin(), m.ssim.end(), 0.0) / N;
    return m;
}
