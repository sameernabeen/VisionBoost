
## Performance Optimization Results

### Benchmark Environment
- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
- Dataset: 500 synthetic images (1280 × 720)
- Image processing: OpenCV
- Parallel loading: ThreadPoolExecutor (16 workers)

### CPU Pipeline Benchmark

| Run | Baseline (s) | Optimized (s) |
|---|---:|---:|
| 1 | 12.95 | 3.0390 |
| 2 | 12.99 | 3.0959 |
| 3 | 12.67 | 3.1152 |
| Median | 12.95 | 3.0959 |

### Performance Summary

- Baseline median: 12.95 seconds
- Optimized median: 3.0959 seconds
- Speedup: 4.18x
- Runtime reduction: 76.09%
- Optimized throughput: 161.50 images/second
- Images processed: 500
- Failed images: 0

### GPU Experiment

A separate PyTorch tensor-resizing experiment measured:
- CPU: 0.3252 seconds
- GPU: 0.0154 seconds
- GPU speedup: 21.07x

Note: This is a synthetic tensor benchmark, not an end-to-end
computer vision pipeline benchmark.



## Complete Pipeline Worker Optimization

Dataset: 500 images (1280x720)

| Workers | Median Runtime (s) | Throughput (img/s) |
|---:|---:|---:|
| 1 | 12.0499 | 41.49 |
| 2 | 6.8568 | 72.92 |
| 4 | 4.1594 | 120.21 |
| 8 | 3.0186 | 165.64 |
| 16 | 2.4882 | 200.95 |

### Key Findings

- Best tested configuration: 16 workers
- Single-worker runtime: 12.0499 seconds
- 16-worker runtime: 2.4882 seconds
- Speedup: 4.84x
- Runtime reduction: 79.35%
- Failed images: 0

The benchmark uses parallel image loading followed by
sequential OpenCV transformations and image saving.

Results are specific to this hardware, dataset and workload.
