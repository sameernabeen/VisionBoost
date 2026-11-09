
import cv2
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = Path("dataset/input")
IMAGE_PATHS = list(INPUT_DIR.glob("*.jpg"))

WORKER_COUNTS = [1, 2, 4, 8, 16]
REPEATS = 3

def load_image(path):
    return cv2.imread(str(path))

def benchmark(workers):
    times = []

    for run in range(REPEATS):
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            images = list(executor.map(load_image, IMAGE_PATHS))

        elapsed = time.perf_counter() - start

        successful = sum(img is not None for img in images)

        if successful != len(IMAGE_PATHS):
            print(f"Warning: {len(IMAGE_PATHS) - successful} images failed")

        times.append(elapsed)

    return statistics.median(times)

print("\n===== VISIONBOOST WORKER BENCHMARK =====")
print("Images:", len(IMAGE_PATHS))

results = []

for workers in WORKER_COUNTS:
    median_time = benchmark(workers)
    throughput = len(IMAGE_PATHS) / median_time

    results.append((workers, median_time, throughput))

    print(
        f"Workers: {workers:2d} | "
        f"Median: {median_time:.4f}s | "
        f"Throughput: {throughput:.2f} images/sec"
    )

best = max(results, key=lambda x: x[2])

print("\n===== BEST CONFIGURATION =====")
print("Workers:", best[0])
print("Median time:", round(best[1], 4), "seconds")
print("Throughput:", round(best[2], 2), "images/sec")

