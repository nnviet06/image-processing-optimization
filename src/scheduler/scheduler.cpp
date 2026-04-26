// src/scheduler/scheduler.cpp
#include "scheduler.h"
#include "filters.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <algorithm>

// Forward declarations (we'll call these from CPU and GPU implementations)
extern cv::Mat gaussian_threaded(const cv::Mat& src, int kernel_size, double sigma);

// GPU function declaration (we need to create a wrapper or declare from gaussian_kernel.cu)
extern cv::Mat gaussian_gpu_kernel(const cv::Mat& src, int kernel_size, double sigma);

ScheduleProfile profile_and_schedule(const cv::Mat& src, int kernel_size, double sigma) {
    ScheduleProfile profile;
    
    std::cout << "\n=== SCHEDULER: Profiling CPU vs GPU ===" << std::endl;
    std::cout << "Image: " << src.cols << "x" << src.rows << std::endl;
    
    // Warm up both paths
    std::cout << "Warming up..." << std::endl;
    gaussian_threaded(src, kernel_size, sigma);
    gaussian_gpu_kernel(src, kernel_size, sigma);
    
    // Profile CPU path (best CPU: multi-threaded)
    std::cout << "\nProfiling CPU (multi-threaded)..." << std::endl;
    const int PROFILE_RUNS = 3;
    double cpu_total = 0.0;
    
    for (int i = 0; i < PROFILE_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = gaussian_threaded(src, kernel_size, sigma);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        cpu_total += duration.count();
        std::cout << "  Run " << (i+1) << ": " << duration.count() << " ms" << std::endl;
    }
    profile.cpu_time_ms = cpu_total / PROFILE_RUNS;
    std::cout << "  Average: " << profile.cpu_time_ms << " ms" << std::endl;
    
    // Profile GPU path (best single image GPU: kernel only, no batch overhead)
    std::cout << "\nProfiling GPU (CUDA kernel)..." << std::endl;
    double gpu_total = 0.0;
    
    for (int i = 0; i < PROFILE_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = gaussian_gpu_kernel(src, kernel_size, sigma);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        gpu_total += duration.count();
        std::cout << "  Run " << (i+1) << ": " << duration.count() << " ms" << std::endl;
    }
    profile.gpu_time_ms = gpu_total / PROFILE_RUNS;
    std::cout << "  Average: " << profile.gpu_time_ms << " ms" << std::endl;
    
    // Decision logic: simple comparison
    std::cout << "\n=== DECISION ===" << std::endl;
    std::cout << "CPU time: " << profile.cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU time: " << profile.gpu_time_ms << " ms" << std::endl;
    
    if (profile.gpu_time_ms < profile.cpu_time_ms) {
        profile.chosen_device = "GPU";
        double speedup = profile.cpu_time_ms / profile.gpu_time_ms;
        std::cout << "Choosing GPU (" << speedup << "x faster)" << std::endl;
    } else {
        profile.chosen_device = "CPU";
        double speedup = profile.gpu_time_ms / profile.cpu_time_ms;
        std::cout << "Choosing CPU (" << speedup << "x faster)" << std::endl;
    }
    
    return profile;
}

cv::Mat gaussian_scheduled(const cv::Mat& src, int kernel_size, double sigma) {
    ScheduleProfile profile = profile_and_schedule(src, kernel_size, sigma);
    
    std::cout << "\nExecuting on " << profile.chosen_device << "..." << std::endl;
    
    if (profile.chosen_device == "GPU") {
        return gaussian_gpu_kernel(src, kernel_size, sigma);
    } else {
        return gaussian_threaded(src, kernel_size, sigma);
    }
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

    // Run scheduler and execute
    cv::Mat result = gaussian_scheduled(src, 15, 2.0);

    save_image("output_scheduler.png", result);
    std::cout << "\nSaved: output_scheduler.png" << std::endl;

    return 0;
}