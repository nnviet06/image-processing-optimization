# IMAGE PROCESSING OPTIMIZATION

High-performance image processing library demonstrating CPU optimization (SIMD, threading, cache-awareness) and GPU acceleration (CUDA) with intelligent scheduler for heterogeneous computing trade-offs.

**Focus:** Understanding when to use CPU vs GPU through layered optimization, benchmarking, and empirical validation.

---

## Project Overview

Three-phase optimization of Gaussian blur demonstrating progression from scalar CPU code to vectorized/threaded optimization to GPU acceleration with intelligent scheduling:

1. **Phase 1: CPU Optimization** — SIMD, threading, cache techniques
2. **Phase 2: GPU Acceleration** — CUDA kernels, batch processing
3. **Phase 3: Smart Scheduler** — Automated CPU/GPU selection via profiling

**Full benchmark results:** See [BENCHMARKS.md](data/benchmarks/BENCHMARKS.md)

---

## Requirements

### Hardware
- **GPU (optional but recommended):** NVIDIA GPU with CUDA compute capability 8.0+ (tested on RTX 3050)
- **CPU:** Multi-core processor (tested on 8-core CPU)

### Software
- **CMake** 3.10+
- **C++ compiler** with C++17 support (g++, clang, MSVC)
- **OpenCV** 4.0+ (for image I/O and display)
- **CUDA Toolkit** 13.0+ (for GPU support, optional)
- **Threads library** (for CPU multi-threading)

### Build Dependencies
```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build libopencv-dev

# CUDA (optional, for GPU support)
# Download from https://developer.nvidia.com/cuda-downloads
```

---

## Build & Run

### 1. Configure
```bash
mkdir build
cd build
cmake -G Ninja ..
```

### 2. Build
```bash
ninja
```

### 3. Run Individual Implementations

**Phase 1: CPU Optimization**
```bash
./gaussian_naive ../data/test1.png
./gaussian_simd ../data/test1.png
./gaussian_threaded ../data/test1.png
./gaussian_cache_aware ../data/test1.png
```

**Phase 2: GPU Acceleration**
```bash
./gaussian_gpu_kernel ../data/test1.png
./gaussian_gpu_optimized ../data/test1.png
./gaussian_gpu_batch ../data/test1.png
```

**Phase 3: Smart Scheduler**
```bash
./gaussian_scheduler ../data/test1.png
```

The scheduler will:
1. Profile both CPU (multi-threaded) and GPU (CUDA kernel) paths
2. Display timing for each implementation
3. Automatically choose the faster device
4. Execute and save result

---

**Decision Logic:**
```cpp
if (gpu_time_ms < cpu_time_ms) {
    return GPU;
} else {
    return CPU;
}
```

Scheduler profiles both paths on real hardware, adapts to different systems automatically.

---

## Known issues and limitations

- [#4: Threading performance regression on smaller images](https://github.com/nnviet06/image-processing-optimization/issues/4)
  - Impact: Threading slower than SIMD on 3K images
  - Priority: Low (larger images favor threading)

- [#5: Cache-aware tiling slower than naive](https://github.com/nnviet06/image-processing-optimization/issues/5)
  - Impact: Phase 1.4 implementation broken
  - Reason: Separable convolution not cache-bound; tiling overhead > benefit
  - Priority: Low (threading already optimal)

---

## Repository Structure
```
image-processing-optimization/
├── src/
│   ├── cpu/
│   │   ├── naive/
│   │   │   └── gaussian.cpp              # Phase 1.1: Naive baseline
│   │   └── optimized/
│   │       ├── gaussian_simd.cpp         # Phase 1.2: AVX2 vectorization
│   │       ├── gaussian_simd_impl.cpp    # (no main, for scheduler)
│   │       ├── gaussian_threaded.cpp     # Phase 1.3: Multi-threading
│   │       ├── gaussian_threaded_impl.cpp # (no main, for scheduler)
│   │       └── gaussian_cache_aware.cpp  # Phase 1.4: Cache-aware (broken)
│   ├── gpu/
│   │   ├── kernels/
│   │   │   ├── gaussian_kernel.cu        # Phase 2.1: Basic CUDA kernel
│   │   │   └── gaussian_kernel_impl.cu   # (no main, for scheduler)
│   │   ├── memory/
│   │   │   └── gaussian_gpu_optimized.cu # Phase 2.2: Pinned memory + streams
│   │   ├── batch/
│   │   │   └── gaussian_batch.cu         # Phase 2.3: Batch pipeline
│   │   └── utils.cu                      # GPU utilities
│   └── scheduler/
│       └── scheduler.cpp                 # Phase 3: Smart scheduler
├── include/
│   ├── filters.h                         # CPU/GPU function declarations
│   ├── utils.h                           # Timing + image I/O utilities
│   └── scheduler.h                       # Scheduler profiling struct
├── data/
│   ├── test1.png                         # 3840×2160 (4K) test image
│   ├── test2.png                         # 3000×2000 (3K) test image
│   └── benchmarks/
│       └── BENCHMARKS.md                 # Detailed benchmark results
├── CMakeLists.txt                        # Build configuration
├── README.md                             
├── benchmark.py                          # Python benchmarking automation
├── LICENSE                               
└── .gitignore                            
```
---

## Key Findings

1. **CPU Optimization:** SIMD and threading provide solid speedups (5-6x), but cache-aware optimizations require profile-guided tuning.

2. **GPU Acceleration:** CUDA kernels dominate compute-bound tasks (100-200x speedup), with PCIe overhead amortized quickly on larger images.

3. **Scheduling:** Heterogeneous workloads benefit from adaptive scheduling. Profiling real hardware > static heuristics.

4. **Separable Convolution:** Already efficient on CPU (good data reuse); GPU shines due to massive parallelism and bandwidth.

---


