
import cv2
import time
import psutil
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = Path("dataset/input")
OUTPUT_DIR = Path("dataset/memory_benchmark_output")

IMAGE_PATHS = sorted(INPUT_DIR.glob("*.jpg"))

WORKERS = 16
BATCH_SIZE = 32
REPEATS = 3

process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

def load_image(path):
    return cv2.imread(str(path))

def process_image(path, image, output_dir):
    if image is None:
        return False

    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return cv2.imwrite(str(output_dir / path.name), edges)

def run_all_at_once():
    output_dir = OUTPUT_DIR / "all_at_once"
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_memory = memory_mb()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        images = list(executor.map(load_image, IMAGE_PATHS))

    peak_memory = max(peak_memory, memory_mb())

    processed = 0
    failed = 0

    for path, image in zip(IMAGE_PATHS, images):
        if process_image(path, image, output_dir):
            processed += 1
        else:
            failed += 1

        peak_memory = max(peak_memory, memory_mb())

    elapsed = time.perf_counter() - start
    return elapsed, peak_memory, processed, failed

def run_batched():
    output_dir = OUTPUT_DIR / "batched"
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_memory = memory_mb()
    start = time.perf_counter()

    processed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for start_index in range(0, len(IMAGE_PATHS), BATCH_SIZE):
            batch_paths = IMAGE_PATHS[
                start_index:start_index + BATCH_SIZE
            ]

            images = list(executor.map(load_image, batch_paths))

            peak_memory = max(peak_memory, memory_mb())

            for path, image in zip(batch_paths, images):
                if process_image(path, image, output_dir):
                    processed += 1
                else:
                    failed += 1

            del images

            peak_memory = max(peak_memory, memory_mb())

    elapsed = time.perf_counter() - start
    return elapsed, peak_memory, processed, failed

def benchmark(function):
    results = []

    for run in range(REPEATS):
        results.append(function())

    median_time = statistics.median(r[0] for r in results)
    median_peak = statistics.median(r[1] for r in results)

    processed = results[-1][2]
    failed = results[-1][3]

    return median_time, median_peak, processed, failed

print("\n===== VISIONBOOST MEMORY BENCHMARK =====")
print("Images:", len(IMAGE_PATHS))
print("Workers:", WORKERS)
print("Batch size:", BATCH_SIZE)

print("\nRunning all-at-once benchmark...")
all_result = benchmark(run_all_at_once)

print("Running batched benchmark...")
batch_result = benchmark(run_batched)

print("\n===== RESULTS =====")

print("\nALL AT ONCE")
print(f"Median time: {all_result[0]:.4f} seconds")
print(f"Peak sampled RAM: {all_result[1]:.2f} MB")
print(f"Processed: {all_result[2]}")
print(f"Failed: {all_result[3]}")

print("\nBATCH PROCESSING")
print(f"Median time: {batch_result[0]:.4f} seconds")
print(f"Peak sampled RAM: {batch_result[1]:.2f} MB")
print(f"Processed: {batch_result[2]}")
print(f"Failed: {batch_result[3]}")

print("\n===== MEMORY COMPARISON =====")
print(
    f"Sampled RAM difference: "
    f"{all_result[1] - batch_result[1]:.2f} MB"
)
