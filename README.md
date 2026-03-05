# IMAGE PROCESSING OPTIMIZATION

High-performance image processing library demonstrating CPU optimization (SIMD, threading, cache-awareness) and GPU acceleration (CUDA) with scheduler for computing trade-offs.

**Focus:** Understanding when to use CPU vs GPU through layered optimization and benchmarking.

## Phase 1: CPU Optimization

Gaussian blur optimization from naive to cache-aware:

- **Phase 1.1 (Naive)**: Baseline scalar implementation 
- **Phase 1.2 (SIMD)**: AVX2 vectorization 
- **Phase 1.3 (Threading)**: Multi-threaded with thread pool 
- **Phase 1.4 (Cache-aware)**: Tile-based memory optimization
