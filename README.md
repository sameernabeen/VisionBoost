# VisionBoost
### Performance Optimization and Memory-Efficient Computer Vision Pipeline

VisionBoost is a Python-based computer vision engineering project focused on improving image-processing performance, reducing memory consumption, and building reliable, scalable image-processing workflows.

The project investigates how parallel image loading, batch processing, configurable execution, and error-handling strategies affect the performance of an OpenCV image-processing pipeline.

## Project Objectives

- Establish a baseline image-processing pipeline.
- Identify performance bottlenecks through profiling.
- Improve image-loading performance using multithreading.
- Reduce memory consumption through batch processing.
- Evaluate pipeline performance across different dataset sizes.
- Implement retry mechanisms and failure logging.
- Compare runtime, throughput, memory usage, and reliability.
- Build reproducible benchmarking experiments.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Computer vision | OpenCV |
| Numerical computing | NumPy |
| Parallel processing | ThreadPoolExecutor |
| Deep learning experiments | PyTorch |
| System monitoring | psutil |
| Data analysis | Pandas |
| Visualization | Matplotlib |
| Testing | pytest |
| Version control | Git and GitHub |

## Experimental Environment

The experiments were conducted on the following system:

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 7 5800H |
| CPU cores | 8 cores / 16 logical processors |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4 GB |
| System RAM | Approximately 24 GB |
| Operating system | Windows |
| Python | 3.14.4 |
| PyTorch | 2.15.0+cu132 |
| IDE | Visual Studio Code |

**Important:** The main OpenCV image-processing pipeline uses CPU operations. CUDA-enabled PyTorch was installed for a separate GPU experiment; the GPU results should not be interpreted as end-to-end acceleration of the OpenCV pipeline.

## Image Processing Workflow

Each image passes through the following operations:

1. Image loading using OpenCV.
2. Image resizing to 640 × 360.
3. Conversion from BGR to grayscale.
4. Gaussian blur.
5. Canny edge detection.
6. Saving the processed image.

### Pipeline Architecture

```text
             Input Dataset
                   |
                   v
              Image Loading
                   |
                   v
             Image Resizing
                   |
                   v
           Grayscale Conversion
                   |
                   v
              Gaussian Blur
                   |
                   v
             Canny Edge Detection
                   |
                   v
             Output Image Saving
                   |
                   v
             Benchmark Reports
```

## Project Structure

```text
VisionBoost/
│
├── dataset/
│   ├── input/
│   └── output/
│
├── src/
│   ├── generate_data.py
│   ├── baseline.py
│   ├── benchmark.py
│   ├── profile_pipeline.py
│   ├── optimized_pipeline.py
│   ├── robust_pipeline.py
│   ├── memory_benchmark.py
│   ├── memory_profiler.py
│   ├── compare_pipeline_memory.py
│   ├── randomized_benchmark.py
│   ├── scale_benchmark.py
│   ├── configurable_pipeline.py
│   └── final_benchmark_report.py
│
├── tests/
│   └── test_robust_pipeline.py
│
├── docs/
│   └── architecture.md
│
├── reports/
│   ├── randomized_benchmark/
│   ├── memory_comparison/
│   └── final_benchmark_report.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

Generated datasets and intermediate outputs are excluded from version control.

## Pipeline Implementations

### 1. Baseline Pipeline

The baseline implementation performs image loading and processing without the later parallel-loading and batch-processing optimizations.

It establishes a reference point for evaluating subsequent improvements.

### 2. Optimized Pipeline

The optimized pipeline introduces:

- Parallel image loading using ThreadPoolExecutor.
- Configurable worker count.
- Batch-based image loading.
- OpenCV image transformations.
- Throughput measurement.

The current batch configuration uses 16 workers and a batch size of 32.

### 3. Robust Pipeline

The robust implementation introduces:

- Image-loading retry handling.
- Processing and saving failure handling.
- Configurable input and output directories.
- Configurable worker and batch sizes.
- Error logging.
- CSV performance reports.

The pipeline retries failed operations up to three total attempts.

## Performance Benchmark Results

### Randomized Benchmark

A randomized benchmark compared the baseline, optimized, and robust implementations.

**Experimental configuration:**

- Dataset: 500 synthetic images.
- Image resolution: 1280 × 720.
- Repetitions: 5 per pipeline.
- Total executions: 15.
- Random seed: 42.
- Runtime measured externally around Python subprocesses, including interpreter startup.

| Pipeline | Median Runtime | Median Throughput |
|---|---:|---:|
| Baseline | 7.7357 sec | 64.64 img/sec |
| Optimized | 2.1886 sec | 228.46 img/sec |
| Robust | 2.5610 sec | 195.23 img/sec |

**Performance improvements relative to baseline:**

| Metric | Optimized | Robust |
|---|---:|---:|
| Median speedup | 3.53× | 3.02× |
| Runtime reduction | 71.71% | 66.89% |

All 15 executions processed 500 images each, with zero reported failures.

The optimized pipeline achieved the highest median throughput in this experiment.

The optimized pipeline also showed greater run-to-run variation than the robust pipeline. Its standard deviation was 1.1361 seconds, compared with 0.1400 seconds for the robust pipeline.

These results are specific to the tested environment and workload.

### Memory Optimization Experiment

An additional experiment evaluated memory consumption using separate subprocesses.

**Measurement methodology:**

- Dataset: 500 images.
- Repetitions: 3 per pipeline.
- Sampling interval: 50 milliseconds.
- Metric: Median peak sampled aggregate process-tree RSS.

| Pipeline | Median Runtime | Median Sampled RSS |
|---|---:|---:|
| Baseline | 7.2375 sec | 55.04 MB |
| Optimized (batched) | 2.4962 sec | 128.54 MB |
| Robust | 2.5358 sec | 143.52 MB |

The original optimized implementation had previously recorded 1,170.63 MB median peak sampled process-tree RSS.

After introducing batch-based image loading with a batch size of 32, the measured value decreased to 128.54 MB.

This represents an approximately 89.02% reduction between the two experiments.

The batching modification traded approximately 6.94% higher median runtime for substantially lower measured memory consumption.

**Measurement limitation:** Sampled process-tree RSS is not an exact instantaneous peak-memory measurement. Sampling can miss short-lived spikes, and summing process RSS can double-count shared memory pages.

The before-and-after results come from separate experiments and should be interpreted as preliminary measurements.

### Large-Scale Dataset Experiment

The pipeline was also evaluated with larger synthetic datasets.

Configuration: 16 workers and batch size 32.

| Dataset Size | Runtime | Throughput | Sampled RAM |
|---:|---:|---:|---:|
| 1,000 | 7.5832 sec | 131.87 img/sec | 138.70 MB |
| 2,500 | 17.2021 sec | 145.33 img/sec | 141.17 MB |
| 5,000 | 31.3578 sec | 159.45 img/sec | 143.46 MB |

All reported scale-test executions processed their complete datasets without failures.

These are individual recorded scale-test runs, not repeated median measurements.

### Separate GPU Tensor Experiment

A separate PyTorch experiment evaluated tensor resizing on CPU and GPU.

| Execution | Runtime |
|---|---:|
| CPU | 0.3252 sec |
| GPU | 0.0154 sec |

The experiment recorded approximately 21.07× speedup on GPU.

This is a synthetic tensor-resizing experiment. It does not represent the performance of the complete OpenCV image-processing pipeline and excludes the costs of transferring image data between CPU and GPU.

## Reliability and Testing

The robust pipeline includes retry handling and error reporting.

The automated test suite currently contains eight tests:

- Valid image processing.
- Corrupted image handling.
- Missing input directory.
- Empty input directory.
- Invalid worker count.
- Invalid batch size.
- Multiple valid images.
- Image-saving failure handling.

The latest recorded test execution passed all eight tests.

Run the tests using:

```bash
pytest -v
```

## Installation and Setup

### Prerequisites

- Python installed on Windows.
- Git.
- Visual Studio Code (recommended).

### Clone the Repository

```bash
git clone https://github.com/sameernabeen/VisionBoost.git
cd VisionBoost
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Generate the Dataset

```powershell
python src/generate_data.py
```

### Run the Pipelines

Baseline:

```powershell
python src/baseline.py
```

Optimized:

```powershell
python src/optimized_pipeline.py
```

Robust:

```powershell
python src/robust_pipeline.py
```

### Run Benchmarks

Randomized comparison:

```powershell
python src/randomized_benchmark.py
```

Memory comparison:

```powershell
python src/compare_pipeline_memory.py
```

Run automated tests:

```powershell
pytest -v
```

## Key Engineering Findings

1. Parallel image loading can substantially improve image-processing throughput.
2. Increasing worker count does not guarantee proportional performance improvements.
3. Batch processing can reduce the number of simultaneously retained image arrays.
4. Memory consumption and processing speed must be evaluated together.
5. Retry handling and failure logging improve pipeline reliability.
6. Randomized execution order helps reduce systematic execution-order bias.
7. Benchmark results must be interpreted in the context of hardware, dataset size, implementation, and measurement methodology.

## Current Limitations

- Experiments use synthetic image datasets.
- Results are hardware- and workload-dependent.
- Memory measurements use periodic sampling.
- The randomized benchmark uses five repetitions per pipeline.
- GPU acceleration has only been evaluated separately using synthetic tensors.
- Further testing is required on real-world image datasets.
- Batch-size and worker-count configurations need broader repeated evaluation.

## Future Improvements

- Systematic worker-count and batch-size tuning.
- More rigorous memory profiling.
- Evaluation on real-world image datasets.
- GPU acceleration of the complete image-processing workflow.
- Comparison of additional image-loading and processing strategies.
- Benchmarking across different image resolutions.
- Improved experiment automation and visualization.
- Expanded reliability and integration testing.

## Author

**Shaik Sameer**

Computer Science and Data Analytics  
Indian Institute of Technology Patna

GitHub: https://github.com/sameernabeen

---

*VisionBoost is an experimental computer vision engineering project. Benchmark values are measurements from the documented experiments, not universal performance guarantees.*
