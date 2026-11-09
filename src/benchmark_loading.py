
import cv2
import os
import glob
import time
import statistics
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = "dataset/input"

image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)

if not image_paths:
    raise FileNotFoundError("No images found!")

# Read one image before benchmarking
cv2.imread(image_paths[0])

def read_image(path):
    return cv2.imread(path)

def benchmark_loading(workers):

    execution_times = []

    for repetition in range(3):

        start = time.perf_counter()

        if workers == 1:

            images = map(read_image, image_paths)

            count = sum(
                1 for image in images if image is not None
            )

        else:

            with ThreadPoolExecutor(
                max_workers=workers
            ) as executor:

                images = executor.map(
                    read_image,
                    image_paths
                )

                count = sum(
                    1 for image in images
                    if image is not None
                )

        elapsed = time.perf_counter() - start

        execution_times.append(elapsed)

    return {
        "Workers": workers,
        "Images": count,
        "Average Time": statistics.mean(execution_times),
        "Median Time": statistics.median(execution_times),
        "Throughput": count / statistics.median(execution_times)
    }


results = []

print("\n===== PARALLEL IMAGE LOADING BENCHMARK =====")

for workers in [1, 2, 4, 8, 16]:

    result = benchmark_loading(workers)

    results.append(result)

    print(
        f"Workers: {workers:2} | "
        f"Median: {result['Median Time']:.4f}s | "
        f"Throughput: {result['Throughput']:.2f} images/sec"
    )

report = pd.DataFrame(results)

os.makedirs("reports", exist_ok=True)

report.to_csv(
    "reports/loading_benchmark.csv",
    index=False
)

print("\nBenchmark saved to reports/loading_benchmark.csv")
