// Phase 1.3: gaussian_threaded.cpp
#include "filters.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>

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

// Horizontal pass (single row)
static void horizontal_pass_row(const cv::Mat& src, cv::Mat& temp,
                                const std::vector<double>& kernel, int row) {
    int kernel_size = kernel.size();
    int half = kernel_size / 2;
    int cols = src.cols;

    const uchar* src_row = src.ptr<uchar>(row);
    double* temp_row = temp.ptr<double>(row);

    for (int c = 0; c < cols; c++) {
        double val = 0.0;
        for (int k = -half; k <= half; k++) {
            int cc = c + k;
            if (cc < 0) cc = 0;
            if (cc >= cols) cc = cols - 1;
            val += kernel[k + half] * src_row[cc];
        }
        temp_row[c] = val;
    }
}

// Vertical pass (single row)
static void vertical_pass_row(const cv::Mat& temp, cv::Mat& dst,
                              const std::vector<double>& kernel, int row) {
    int kernel_size = kernel.size();
    int half = kernel_size / 2;
    int rows = temp.rows;
    int cols = temp.cols;

    uchar* dst_row = dst.ptr<uchar>(row);

    for (int c = 0; c < cols; c++) {
        double val = 0.0;
        for (int k = -half; k <= half; k++) {
            int rr = row + k;
            if (rr < 0) rr = 0;
            if (rr >= rows) rr = rows - 1;
            val += kernel[k + half] * temp.at<double>(rr, c);
        }
        dst_row[c] = static_cast<uchar>(std::round(val));
    }
}

cv::Mat gaussian_threaded(const cv::Mat& src, int kernel_size, double sigma) {
    CV_Assert(src.type() == CV_8UC1);  // Grayscale only
    CV_Assert(kernel_size % 2 == 1);   // Must be odd

    int rows = src.rows;
    int cols = src.cols;

    std::vector<double> kernel = build_kernel(kernel_size, sigma);

    cv::Mat temp = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_8UC1);

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback

    // Horizontal pass (multi-threaded)
    {
        std::vector<std::thread> threads;
        int rows_per_thread = (rows + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            int start_row = t * rows_per_thread;
            int end_row = std::min(start_row + rows_per_thread, rows);

            threads.emplace_back([&, start_row, end_row]() {
                for (int r = start_row; r < end_row; r++) {
                    horizontal_pass_row(src, temp, kernel, r);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    // Vertical pass (multi-threaded)
    {
        std::vector<std::thread> threads;
        int rows_per_thread = (rows + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            int start_row = t * rows_per_thread;
            int end_row = std::min(start_row + rows_per_thread, rows);

            threads.emplace_back([&, start_row, end_row]() {
                for (int r = start_row; r < end_row; r++) {
                    vertical_pass_row(temp, dst, kernel, r);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    return dst;
}