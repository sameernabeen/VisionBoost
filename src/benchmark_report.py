
import csv
import time
import cv2
import psutil
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_DIR = Path("dataset/scale_input")
OUTPUT_DIR = Path("dataset/automated_output")
REPORT_DIR = Path("reports")

WORKERS = 16
BATCH_SIZE = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATHS = sorted(INPUT_DIR.glob("*.jpg"))
process = psutil.Process()

def load_image(path):
    return cv2.imread(str(path))

def process_image(path, image):
    if image is None:
        return False

    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return cv2.imwrite(str(OUTPUT_DIR / path.name), edges)

def run_pipeline():
    peak_ram = process.memory_info().rss
    processed = 0
    failed = 0

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for index in range(0, len(IMAGE_PATHS), BATCH_SIZE):
            paths = IMAGE_PATHS[index:index + BATCH_SIZE]
            images = list(executor.map(load_image, paths))

            peak_ram = max(
                peak_ram,
                process.memory_info().rss
            )

            for path, image in zip(paths, images):
                try:
                    if process_image(path, image):
                        processed += 1
                    else:
                        failed += 1
                except Exception as error:
                    print(f"Error processing {path.name}: {error}")
                    failed += 1

            del images

            peak_ram = max(
                peak_ram,
                process.memory_info().rss
            )

    elapsed = time.perf_counter() - start
    throughput = processed / elapsed if elapsed else 0

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(IMAGE_PATHS),
        "workers": WORKERS,
        "batch_size": BATCH_SIZE,
        "processed": processed,
        "failed": failed,
        "runtime_seconds": round(elapsed, 4),
        "throughput_images_per_second": round(throughput, 2),
        "peak_sampled_ram_mb": round(
            peak_ram / (1024 * 1024), 2
        )
    }

def save_report(result):
    report_file = REPORT_DIR / "benchmark_report.csv"

    file_exists = report_file.exists()

    with report_file.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print("\nReport saved:", report_file)

if __name__ == "__main__":
    if not IMAGE_PATHS:
        raise SystemExit(
            f"No JPG images found in {INPUT_DIR}"
        )

    print("===== VISIONBOOST AUTOMATED PIPELINE =====")
    print("Images:", len(IMAGE_PATHS))
    print("Workers:", WORKERS)
    print("Batch size:", BATCH_SIZE)

    result = run_pipeline()

    print("\n===== BENCHMARK RESULTS =====")

    for key, value in result.items():
        print(f"{key}: {value}")

    save_report(result)
