# CPU Optimization Benchmarks

## Phase 1: Gaussian Blur 

### test1.png - 3840x2160 4K

| Implementation | Time (ms) | Speedup | Status | Notes |
|---|---|---|---|---|
| Naive (1.1) | 1360 | 1.0x | ✓ | Baseline scalar |
| SIMD AVX2 (1.2) | 260 | 5.2x | ✓ | AVX2 vectorization, FMA |
| Threading (1.3) | 220 | 6.2x | ✓ | 4-thread row-level parallelism |
| Cache-aware (1.4) | 1140 | 1.2x | ✗ | Tiling overhead > benefit (separable convolution not cache-bound) |

### test2.png (3000x2000)

| Implementation | Time (ms) | Speedup vs Naive | Status | Notes |
|---|---|---|---|---|
| Naive (1.1) | 1378 | 1.0x | ✓ | Baseline |
| SIMD AVX2 (1.2) | 297 | 4.6x | ✓ | Vectorization wins |
| Threading (1.3) | 601 | 2.3x | ✓ | Threading overhead > benefit |
| Cache-aware (1.4) | 1284 | 1.07x | ✗ | Tiling overhead still broken |

---

## Phase 2: GPU Gaussian Blur

### test1.png - 3840x2160 4K

| Implementation | Time (ms) | Speedup vs Naive | Status | Notes |
|---|---|---|---|---|
| GPU Kernel (2.1) | 20.6 | 138.7x | ✓ | Basic CUDA kernel |
| GPU Optimized (2.2) | ~40 | 71.4x | ✓ | Pinned memory overhead |
| GPU Batch 10x (2.3) | 29.3/img | 97.4x | ✓ | Amortized PCIe, batch=10 (64MB limit) |

### test2.png (3000x2000)

| Implementation | Time (ms) | Speedup vs Naive CPU | Status | Notes |
|---|---|---|---|---|
| GPU Kernel (2.1) | 12.1 | 168.2x | ✓ | Smaller image, faster |
| GPU Optimized (2.2) | ~27 | 75.4x | ✓ | Pinned memory overhead |
| GPU Batch 10x (2.3) | 24.85/img | 81.8x | ✓ | Faster per-image, higher throughput (40.4 img/sec) |