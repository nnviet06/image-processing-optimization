#include "filters.h"
#include "utils.h"
#include <iostream>
#include <cmath>

// Build 1D Gaussian kernel
static std::vector<double> build_kernel(int kernel_size, double sigma) {
    std::vector<double> kernel(kernel_size);
    int half = kernel_size / 2;
    double sum = 0.0;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - half;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

cv::Mat gaussian_naive(const cv::Mat& src, int kernel_size, double sigma) {
    CV_Assert(src.type() == CV_8UC1);  // Grayscale only
    CV_Assert(kernel_size % 2 == 1);   // Must be odd

    int rows = src.rows;
    int cols = src.cols;
    int half = kernel_size / 2;

    std::vector<double> kernel = build_kernel(kernel_size, sigma);

    cv::Mat temp = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat dst  = cv::Mat::zeros(rows, cols, CV_8UC1);

    // Horizontal pass
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            double val = 0.0;
            for (int k = -half; k <= half; k++) {
                int cc = c + k;
                if (cc < 0) cc = 0;
                if (cc >= cols) cc = cols - 1;
                val += kernel[k + half] * src.at<uchar>(r, cc);
            }
            temp.at<double>(r, c) = val;
        }
    }

    // Vertical pass
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            double val = 0.0;
            for (int k = -half; k <= half; k++) {
                int rr = r + k;
                if (rr < 0) rr = 0;
                if (rr >= rows) rr = rows - 1;
                val += kernel[k + half] * temp.at<double>(rr, c);
            }
            dst.at<uchar>(r, c) = static_cast<uchar>(std::round(val));
        }
    }

    return dst;
}

int main(int argc, char* argv[]) {
    std::string image_path = (argc > 1) ? argv[1] : "test_image.png";

    cv::Mat src;
    try {
        src = load_image(image_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << src.cols << "x" << src.rows << std::endl;

    // Warm up
    gaussian_naive(src, 15, 2.0);

    // Benchmark: run 5 times, take average
    const int RUNS = 5;
    double total_ms = 0.0;
    cv::Mat result;
    Timer timer;

    for (int i = 0; i < RUNS; i++) {
        timer.start();
        result = gaussian_naive(src, 15, 2.0);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / RUNS;
    std::cout << "Naive Gaussian (avg " << RUNS << " runs): " 
              << avg_ms << " ms" << std::endl;

    save_image("output_naive.png", result);
    std::cout << "Saved: output_naive.png" << std::endl;

    return 0;
}