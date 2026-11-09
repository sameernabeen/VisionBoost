
import cv2
import psutil
import time
import statistics
import subprocess
import sys
import json
from pathlib import Path

INPUT_DIR = Path("dataset/input")
OUTPUT_DIR = Path("dataset/memory_profile_output")

IMAGE_PATHS = sorted(INPUT_DIR.glob("*.jpg"))

WORKERS = 16
BATCH_SIZE = 32
REPEATS = 3

SCRIPT_PATH = Path(__file__).resolve()

def run_child(mode):
    import os
    from concurrent.futures import ThreadPoolExecutor

    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss

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

    output_dir = OUTPUT_DIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    processed = 0
    failed = 0

    def update_peak():
        nonlocal peak_memory
        peak_memory = max(peak_memory, process.memory_info().rss)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        if mode == "all":
            images = list(executor.map(load_image, IMAGE_PATHS))
            update_peak()

            for path, image in zip(IMAGE_PATHS, images):
                if process_image(path, image, output_dir):
                    processed += 1
                else:
                    failed += 1
                update_peak()

            del images

        else:
            for index in range(0, len(IMAGE_PATHS), BATCH_SIZE):
                paths = IMAGE_PATHS[index:index + BATCH_SIZE]
                images = list(executor.map(load_image, paths))
                update_peak()

                for path, image in zip(paths, images):
                    if process_image(path, image, output_dir):
                        processed += 1
                    else:
                        failed += 1

                del images
                update_peak()

    elapsed = time.perf_counter() - start

    print(json.dumps({
        "mode": mode,
        "runtime_seconds": elapsed,
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "processed": processed,
        "failed": failed
    }))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_child(sys.argv[1])
    else:
        results = {"all": [], "batch": []}

        print("\n===== VISIONBOOST MEMORY PROFILER =====")
        print("Images:", len(IMAGE_PATHS))
        print("Workers:", WORKERS)
        print("Batch size:", BATCH_SIZE)

        for mode in ["all", "batch"]:
            print(f"\nTesting {mode.upper()} mode")

            for run in range(REPEATS):
                command = [
                    sys.executable,
                    str(SCRIPT_PATH),
                    mode
                ]

                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )

                result = json.loads(completed.stdout.strip().splitlines()[-1])
                results[mode].append(result)

                print(
                    f"Run {run + 1}: "
                    f"{result['runtime_seconds']:.4f}s | "
                    f"{result['peak_memory_mb']:.2f} MB | "
                    f"Processed: {result['processed']} | "
                    f"Failed: {result['failed']}"
                )

        print("\n===== MEDIAN RESULTS =====")

        for mode in ["all", "batch"]:
            median_time = statistics.median(
                r["runtime_seconds"] for r in results[mode]
            )
            median_memory = statistics.median(
                r["peak_memory_mb"] for r in results[mode]
            )

            print(f"\n{mode.upper()}")
            print(f"Median runtime: {median_time:.4f}s")
            print(f"Median peak RAM: {median_memory:.2f} MB")

            print(
                "Processed:",
                results[mode][-1]["processed"],
                "| Failed:",
                results[mode][-1]["failed"]
            )
