import csv
import random
import re
import subprocess
import sys
import time
import statistics
from pathlib import Path

import psutil


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "dataset" / "input"
REPORT_DIR = PROJECT_DIR / "reports" / "memory_comparison"
CSV_FILE = REPORT_DIR / "memory_results.csv"

DATASET_SIZE = len(list(INPUT_DIR.glob("*.jpg")))
REPETITIONS = 3
RANDOM_SEED = 42
SAMPLE_INTERVAL = 0.05

PIPELINES = {
    "baseline": [
        sys.executable,
        str(PROJECT_DIR / "src" / "baseline.py"),
    ],
    "optimized": [
        sys.executable,
        str(PROJECT_DIR / "src" / "optimized_pipeline.py"),
    ],
    "robust": [
        sys.executable,
        str(PROJECT_DIR / "src" / "robust_pipeline.py"),
        "--input", str(INPUT_DIR),
        "--output", str(PROJECT_DIR / "dataset" / "memory_comparison_output"),
        "--workers", "16",
        "--batch-size", "32",
        "--report-dir", str(REPORT_DIR),
    ],
}


def get_process_tree_memory(process):
    """Return aggregate RSS of the process and its current descendants."""
    total_memory = 0

    try:
        processes = [process] + process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]

    for item in processes:
        try:
            total_memory += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_memory / (1024 * 1024)


def extract_count(patterns, output):
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def run_pipeline(name, command):
    print(f"\nRunning: {name}")

    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitored_process = psutil.Process(process.pid)
    peak_sampled_ram = 0.0

    while process.poll() is None:
        current_memory = get_process_tree_memory(monitored_process)
        peak_sampled_ram = max(peak_sampled_ram, current_memory)
        time.sleep(SAMPLE_INTERVAL)

    stdout, stderr = process.communicate()

    # Capture one final memory sample.
    try:
        final_memory = get_process_tree_memory(monitored_process)
        peak_sampled_ram = max(peak_sampled_ram, final_memory)
    except psutil.NoSuchProcess:
        pass

    runtime = time.perf_counter() - start_time

    if process.returncode != 0:
        print(stdout)
        print(stderr)
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")

    processed = extract_count(
        [r"Images processed:\s*(\d+)", r"processed:\s*(\d+)"],
        stdout,
    )

    failed = extract_count(
        [r"Failed:\s*(\d+)", r"failed:\s*(\d+)"],
        stdout,
    )

    if processed is None:
        raise ValueError(
            f"Could not find processed image count in {name} output:\n{stdout}"
        )

    if failed is None:
        failed = DATASET_SIZE - processed

    throughput = processed / runtime if runtime > 0 else 0

    result = {
        "pipeline": name,
        "runtime_seconds": round(runtime, 4),
        "processed": processed,
        "failed": failed,
        "throughput_images_per_second": round(throughput, 2),
        "peak_sampled_process_tree_rss_mb": round(peak_sampled_ram, 2),
        "return_code": process.returncode,
    }

    print(result)
    return result


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_SIZE != 500:
        raise ValueError(
            f"Expected 500 input images, found {DATASET_SIZE}. "
            "Check dataset/input before running."
        )

    results = []
    randomizer = random.Random(RANDOM_SEED)

    fieldnames = [
        "round",
        "order_position",
        "pipeline",
        "runtime_seconds",
        "processed",
        "failed",
        "throughput_images_per_second",
        "peak_sampled_process_tree_rss_mb",
        "return_code",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for round_number in range(1, REPETITIONS + 1):
            order = list(PIPELINES.keys())
            randomizer.shuffle(order)

            print(f"\n========== ROUND {round_number} ==========")
            print("Execution order:", order)

            for position, name in enumerate(order, start=1):
                result = run_pipeline(name, PIPELINES[name])

                result["round"] = round_number
                result["order_position"] = position

                writer.writerow(result)
                file.flush()
                results.append(result)

    summary_file = REPORT_DIR / "memory_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("VISIONBOOST MEMORY COMPARISON\n")
        file.write("=" * 55 + "\n")
        file.write(f"Dataset size: {DATASET_SIZE}\n")
        file.write(f"Repetitions: {REPETITIONS}\n")
        file.write(f"Sampling interval: {SAMPLE_INTERVAL} seconds\n")
        file.write(
            "Memory metric: peak sampled aggregate process-tree RSS\n\n"
        )

        for name in PIPELINES:
            pipeline_results = [
                row for row in results if row["pipeline"] == name
            ]

            median_runtime = statistics.median(
                row["runtime_seconds"] for row in pipeline_results
            )

            median_memory = statistics.median(
                row["peak_sampled_process_tree_rss_mb"]
                for row in pipeline_results
            )

            median_throughput = statistics.median(
                row["throughput_images_per_second"]
                for row in pipeline_results
            )

            total_processed = sum(
                row["processed"] for row in pipeline_results
            )

            total_failed = sum(
                row["failed"] for row in pipeline_results
            )

            file.write(f"{name.upper()}\n")
            file.write(f"Median runtime: {median_runtime:.4f} sec\n")
            file.write(
                f"Median throughput: {median_throughput:.2f} img/sec\n"
            )
            file.write(
                f"Median peak sampled process-tree RSS: "
                f"{median_memory:.2f} MB\n"
            )
            file.write(f"Total processed: {total_processed}\n")
            file.write(f"Total failed: {total_failed}\n\n")

    print("\nMEMORY BENCHMARK COMPLETED")
    print(f"CSV: {CSV_FILE}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()