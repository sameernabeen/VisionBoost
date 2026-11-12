
import argparse
import csv
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import psutil
from concurrent.futures import ThreadPoolExecutor


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "dataset" / "input"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "configuration_output"
REPORT_DIR = PROJECT_DIR / "reports" / "configuration_benchmark"

WORKERS_LIST = [4, 8, 12, 16]
BATCH_SIZES = [8, 16, 32, 64]

REPETITIONS = 3
RANDOM_SEED = 42
SAMPLE_INTERVAL = 0.05


def read_image(path):
    return path, cv2.imread(str(path))


def process_image(path, image):
    if image is None:
        return False

    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    output_path = OUTPUT_DIR / path.name
    return cv2.imwrite(str(output_path), edges)


def child_run(workers, batch_size):
    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    processed = 0
    failed = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start:start + batch_size]

            for path, image in executor.map(read_image, batch):
                if process_image(path, image):
                    processed += 1
                else:
                    failed += 1

    runtime = time.perf_counter() - start_time

    print(f"RESULT_PROCESSED={processed}")
    print(f"RESULT_FAILED={failed}")
    print(f"RESULT_RUNTIME={runtime:.6f}")


def sample_process_tree(process):
    total = 0

    try:
        processes = [process] + process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]

    for item in processes:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total / (1024 * 1024)


def run_configuration(workers, batch_size):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitored = psutil.Process(process.pid)
    peak_memory = 0

    start = time.perf_counter()

    while process.poll() is None:
        peak_memory = max(
            peak_memory,
            sample_process_tree(monitored)
        )
        time.sleep(SAMPLE_INTERVAL)

    stdout, stderr = process.communicate()
    runtime = time.perf_counter() - start

    if process.returncode != 0:
        raise RuntimeError(
            f"Configuration failed: workers={workers}, "
            f"batch={batch_size}\n{stderr}\n{stdout}"
        )

    def get_value(name):
        for line in stdout.splitlines():
            if line.startswith(name + "="):
                return float(line.split("=", 1)[1])
        raise ValueError(f"Missing {name} in child output:\n{stdout}")

    processed = int(get_value("RESULT_PROCESSED"))
    failed = int(get_value("RESULT_FAILED"))

    throughput = processed / runtime if runtime > 0 else 0

    return {
        "workers": workers,
        "batch_size": batch_size,
        "runtime_seconds": round(runtime, 4),
        "processed": processed,
        "failed": failed,
        "throughput_images_per_second": round(throughput, 2),
        "peak_sampled_process_tree_rss_mb": round(peak_memory, 2),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_count = len(list(INPUT_DIR.glob("*.jpg")))

    if image_count != 500:
        raise ValueError(
            f"Expected 500 JPG images, found {image_count}"
        )

    configurations = [
        (workers, batch)
        for workers in WORKERS_LIST
        for batch in BATCH_SIZES
    ]

    results_file = REPORT_DIR / "configuration_results.csv"
    summary_file = REPORT_DIR / "configuration_summary.txt"

    randomizer = random.Random(RANDOM_SEED)
    results = []

    fields = [
        "round",
        "workers",
        "batch_size",
        "runtime_seconds",
        "processed",
        "failed",
        "throughput_images_per_second",
        "peak_sampled_process_tree_rss_mb",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for round_number in range(1, REPETITIONS + 1):
            order = configurations.copy()
            randomizer.shuffle(order)

            print(f"\n========== ROUND {round_number} ==========")

            for workers, batch_size in order:
                print(
                    f"Testing workers={workers}, "
                    f"batch_size={batch_size}"
                )

                result = run_configuration(workers, batch_size)
                result["round"] = round_number

                if result["processed"] != image_count or result["failed"] != 0:
                    raise RuntimeError(
                        f"Unexpected image count: {result}"
                    )

                writer.writerow(result)
                file.flush()
                results.append(result)

                print(result)

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("VISIONBOOST CONFIGURATION BENCHMARK\n")
        file.write("=" * 60 + "\n")
        file.write(f"Dataset size: {image_count}\n")
        file.write(f"Repetitions: {REPETITIONS}\n")
        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(f"Worker counts: {WORKERS_LIST}\n")
        file.write(f"Batch sizes: {BATCH_SIZES}\n")
        file.write(
            "Memory metric: median peak sampled aggregate process-tree RSS\n\n"
        )

        summaries = []

        for workers, batch_size in configurations:
            group = [
                row for row in results
                if row["workers"] == workers
                and row["batch_size"] == batch_size
            ]

            median_runtime = statistics.median(
                row["runtime_seconds"] for row in group
            )
            median_throughput = statistics.median(
                row["throughput_images_per_second"] for row in group
            )
            median_memory = statistics.median(
                row["peak_sampled_process_tree_rss_mb"] for row in group
            )

            summaries.append({
                "workers": workers,
                "batch_size": batch_size,
                "runtime": median_runtime,
                "throughput": median_throughput,
                "memory": median_memory,
            })

        file.write(
            "WORKERS | BATCH | MEDIAN TIME | MEDIAN THROUGHPUT | MEDIAN RSS\n"
        )

        for row in summaries:
            file.write(
                f"{row['workers']:7} | "
                f"{row['batch_size']:5} | "
                f"{row['runtime']:11.4f} | "
                f"{row['throughput']:17.2f} | "
                f"{row['memory']:10.2f} MB\n"
            )

        fastest = min(summaries, key=lambda row: row["runtime"])
        lowest_memory = min(summaries, key=lambda row: row["memory"])

        file.write("\nFASTEST CONFIGURATION\n")
        file.write(
            f"Workers: {fastest['workers']}, "
            f"Batch: {fastest['batch_size']}\n"
        )

        file.write("\nLOWEST MEMORY CONFIGURATION\n")
        file.write(
            f"Workers: {lowest_memory['workers']}, "
            f"Batch: {lowest_memory['batch_size']}\n"
        )

    print("\nCONFIGURATION BENCHMARK COMPLETED")
    print(f"CSV: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.child:
        child_run(args.workers, args.batch_size)
    else:
        main()
