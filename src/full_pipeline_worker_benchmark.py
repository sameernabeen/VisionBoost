
import cv2
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = Path("dataset/input")
OUTPUT_DIR = Path("dataset/worker_benchmark_output")

IMAGE_PATHS = sorted(INPUT_DIR.glob("*.jpg"))
WORKER_COUNTS = [1, 2, 4, 8, 16]
REPEATS = 3

def load_image(path):
    return cv2.imread(str(path))

def run_pipeline(workers):
    output_dir = OUTPUT_DIR / f"workers_{workers}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    # Load images concurrently
    with ThreadPoolExecutor(max_workers=workers) as executor:
        images = list(executor.map(load_image, IMAGE_PATHS))

    processed = 0
    failed = 0

    # Apply the same OpenCV operations to every image
    for path, image in zip(IMAGE_PATHS, images):
        if image is None:
            failed += 1
            continue

        resized = cv2.resize(image, (640, 360))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        output_path = output_dir / path.name

        if cv2.imwrite(str(output_path), edges):
            processed += 1
        else:
            failed += 1

    elapsed = time.perf_counter() - start

    return elapsed, processed, failed

print("\n===== FULL PIPELINE WORKER BENCHMARK =====")
print("Input images:", len(IMAGE_PATHS))

results = []

for workers in WORKER_COUNTS:
    run_times = []

    for run in range(REPEATS):
        elapsed, processed, failed = run_pipeline(workers)
        run_times.append(elapsed)

    median_time = statistics.median(run_times)
    throughput = processed / median_time

    results.append((workers, median_time, throughput))

    print(
        f"Workers: {workers:2d} | "
        f"Median: {median_time:.4f}s | "
        f"Throughput: {throughput:.2f} images/sec | "
        f"Processed: {processed} | Failed: {failed}"
    )

best = max(results, key=lambda x: x[2])

print("\n===== BEST COMPLETE PIPELINE CONFIGURATION =====")
print("Workers:", best[0])
print("Median time:", round(best[1], 4), "seconds")
print("Throughput:", round(best[2], 2), "images/sec")
