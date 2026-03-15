#include "utils.h"
#include <iostream>
#include <stdexcept>

// Setting up timer

Timer::Timer() {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    return duration.count();
}

// Image utilities

cv::Mat load_image(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return img;
}
void save_image(const std::string& path, const cv::Mat& img) {
    if (!cv::imwrite(path, img)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}
void display_image(const std::string& window_name, const cv::Mat& img) {
    cv::imshow(window_name, img);
    cv::waitKey(0);
}