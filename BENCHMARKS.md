# CPU Optimization Benchmarks

## Phase 1: Gaussian Blur 

### test1.png - 3840x2160 4K

| Implementation | Time (ms) | Speedup | Status | Notes |
|---|---|---|---|---|
| Naive (1.1) | 2856.76 | 1.0x | ✓ | Baseline scalar |
| SIMD AVX2 (1.2) | 497.36 | 5.7x | ✓ | Vectorized |
| Threading (1.3) | 352.24 | 8.1x | ✓ | 4-thread parallelism |
| Cache-aware (1.4) | 2135.79 | 0.75x | ✗ | currently broken |

### test2.png (3000x2000) naive = 2035.74 ms