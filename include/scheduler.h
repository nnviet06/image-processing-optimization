// include/scheduler.h
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <opencv2/opencv.hpp>
#include <string>

struct ScheduleProfile {
    double cpu_time_ms;
    double gpu_time_ms;
    std::string chosen_device;  // "CPU" or "GPU"
    
    ScheduleProfile() : cpu_time_ms(0), gpu_time_ms(0), chosen_device("") {}
};

// Profile both CPU and GPU paths, return the faster one
ScheduleProfile profile_and_schedule(const cv::Mat& src, int kernel_size, double sigma);

// Wrapper that applies the chosen implementation
cv::Mat gaussian_scheduled(const cv::Mat& src, int kernel_size, double sigma);

#endif