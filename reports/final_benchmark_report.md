# VisionBoost: Final Performance Benchmark Report

Generated: 2026-11-09 17:31:53

## 1. Project Overview

VisionBoost is a computer vision pipeline optimization project.

It evaluates image processing performance through profiling,
parallel image loading, batch processing, configurable execution,
memory monitoring, GPU experiments, and robust error handling.

The pipeline performs:

1. Image loading
2. Image resizing
3. Grayscale conversion
4. Gaussian blur
5. Canny edge detection
6. Processed image saving


## 2. Experimental Environment

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5800H |
| CPU Cores | 8 cores / 16 logical processors |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM | 4 GB |
| System RAM | Approximately 24 GB |
| Python | 3.14.4 |
| PyTorch | 2.15.0+cu132 |
| Operating System | Windows |
| IDE | Visual Studio Code |

### Benchmark environment notes

- CPU pipeline benchmarks use OpenCV image-processing operations.
- Parallel image loading uses Python ThreadPoolExecutor.
- GPU tensor benchmarking uses PyTorch and CUDA.
- The GPU tensor benchmark is separate from the complete
  OpenCV image-processing pipeline.
- Performance results depend on hardware, workload,
  storage conditions, and software versions.


## 3. Pipeline Profiling Results
| Operation | Time (seconds) | Time Share |
| --- | --- | --- |
| Image Reading | 16.204 | 91.67% |
| Resizing | 0.1873 | 1.06% |
| Grayscale Conversion | 0.0469 | 0.27% |
| Gaussian Blur | 0.087 | 0.49% |
| Canny Edge Detection | 0.2013 | 1.14% |
| Image Saving | 0.9429 | 5.33% |

**Main bottleneck:** Image Reading accounted for 91.67% of the profiled execution time.

## 4. Parallel Image Loading Benchmark
| Workers | Images | Median Time (seconds) | Throughput (images/sec) |
| --- | --- | --- | --- |
| 1 | 500 | 13.4941 | 37.05 |
| 2 | 500 | 5.4218 | 92.22 |
| 4 | 500 | 2.8346 | 176.39 |
| 8 | 500 | 1.8552 | 269.51 |
| 16 | 500 | 1.1822 | 422.93 |

The highest measured loading throughput was 422.93 images/sec with 16 workers.
This represents approximately 11.41x the single-worker throughput in this loading-only test.

## 5. Configurable Pipeline Benchmark
| Dataset | Workers | Batch Size | Processed | Failed | Runtime (sec) | Throughput | Sampled RAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 16 | 32 | 5000 | 0 | 27.8202 | 179.73 | 144.89 |
| 5000 | 8 | 64 | 5000 | 0 | 30.0786 | 166.23 | 228.1 |
| 5000 | 4 | 16 | 5000 | 0 | 44.157 | 113.23 | 101.5 |
| 2500 | 8 | 32 | 2500 | 0 | 16.4311 | 152.15 | 141.8 |

**Fastest recorded configuration:** 16 workers, batch size 32, 179.73 images/sec.

These are individual recorded runs, not repeated statistical comparisons.

## 6. Automated Benchmark Result
| Dataset | Workers | Batch | Processed | Failed | Runtime (sec) | Throughput | Sampled RAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 16 | 32 | 5000 | 0 | 28.7599 | 173.85 | 143.75 |

## 7. Additional Previously Recorded Experiments

The following results were recorded during earlier project
experiments. They are presented separately because they are
not sourced from the CSV files above.

| Experiment | Recorded Result |
|---|---|
| Baseline vs optimized pipeline | 4.18x speedup |
| Batch memory optimization | 89.87% lower peak sampled RSS |
| Full-pipeline worker scaling | 4.84x speedup |
| GPU tensor resizing | 21.07x speedup |
| Dataset scale testing | Up to 5,000 images |

### Interpretation and limitations

- GPU tensor resizing is a separate synthetic tensor experiment,
  not a full computer vision pipeline GPU speedup.
- Memory measurements are peak sampled RSS, not continuous
  operating-system peak measurements.
- Worker scaling results depend on the tested hardware,
  workload, and implementation.
- Historical measurements are not directly interchangeable
  with the CSV benchmark results.
- The memory reduction applies to the specific all-at-once
  versus batch-processing experiment.


## 8. Error Handling and Reliability

The robust pipeline includes:

1. Image-loading validation.
2. Detection of corrupted or unreadable images.
3. Processing exception handling.
4. CSV error reporting.
5. Error logging.
6. Retry handling with up to two additional attempts.

### Latest error report

No error records were available in the latest pipeline error report.

### Reliability validation

Previously verified experiments:

| Test | Processed | Failed |
|---|---:|---:|
| Corrupted-image test | 10 | 1 |
| Clean 5,000-image test | 5,000 | 0 |

The corrupted image was attempted three times:
one initial attempt and two retries.

The clean 5,000-image test validates normal execution
with retry handling enabled. It does not demonstrate
retries being triggered during that run.


## 9. Reproducibility

### Install dependencies

Activate the project virtual environment and run:

```powershell
pip install -r requirements.txt
```

### Execute the robust pipeline

```powershell
python src/robust_pipeline.py --input dataset/scale_input --output dataset/robust_output --workers 16 --batch-size 32
```

### Generate the final benchmark report

```powershell
python src/final_benchmark_report.py
```

### View the generated report

```powershell
code reports/final_benchmark_report.md
```

The input dataset must be available in the specified
directory before running the pipeline.


## 10. Conclusion

VisionBoost demonstrates practical computer vision pipeline
optimization through:

- Performance profiling
- Parallel image loading
- Configurable worker and batch settings
- Memory-aware processing
- GPU tensor experimentation
- Robust failure handling

The retry-enabled pipeline successfully processed a clean
dataset of 5,000 images with zero failures.

The experiments identified image loading and decoding
as the largest bottleneck in the original profiled workload.

The project demonstrates how profiling, concurrency,
batch processing, and reliability mechanisms can be
combined to improve an image-processing workflow.


## 11. Future Improvements

1. Perform repeated benchmark runs and report medians.
2. Evaluate alternative image-loading strategies.
3. Benchmark different image resolutions and formats.
4. Investigate GPU acceleration of the complete pipeline.
5. Measure memory continuously for more accurate peak RSS.
6. Add automated tests for corrupted and inaccessible files.
7. Evaluate performance on larger and real-world datasets.


---

**Project:** VisionBoost

**Report Type:** Performance and Reliability Evaluation

**Implementation:** Python, OpenCV, PyTorch, CUDA

**Benchmark Dataset:** Synthetic JPEG images
