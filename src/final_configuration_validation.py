
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
OUTPUT_DIR = PROJECT_DIR / "dataset" / "final_validation_output"
REPORT_DIR = PROJECT_DIR / "reports" / "final_configuration_validation"

REPETITIONS = 10
RANDOM_SEED = 42
SAMPLE_INTERVAL = 0.05

CONFIGURATIONS = [
    {"name": "memory_efficient", "workers": 4, "batch_size": 64},
    {"name": "balanced", "workers": 8, "batch_size": 32},
    {"name": "fastest_candidate", "workers": 12, "batch_size": 64},
]


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


def sample_process_tree_memory(process):
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


def run_configuration(config, round_number):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--workers", str(config["workers"]),
        "--batch-size", str(config["batch_size"]),
    ]

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitored_process = psutil.Process(process.pid)
    peak_memory = 0.0

    start = time.perf_counter()

    while process.poll() is None:
        peak_memory = max(
            peak_memory,
            sample_process_tree_memory(monitored_process)
        )
        time.sleep(SAMPLE_INTERVAL)

    stdout, stderr = process.communicate()
    external_runtime = time.perf_counter() - start

    if process.returncode != 0:
        raise RuntimeError(
            f"Configuration failed: {config['name']}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    def extract_value(key):
        for line in stdout.splitlines():
            if line.startswith(key + "="):
                return float(line.split("=", 1)[1])
        raise ValueError(f"Missing {key} in child output:\n{stdout}")

    processed = int(extract_value("RESULT_PROCESSED"))
    failed = int(extract_value("RESULT_FAILED"))
    internal_runtime = extract_value("RESULT_RUNTIME")

    if processed != DATASET_SIZE or failed != 0:
        raise RuntimeError(
            f"Unexpected processing result: {config['name']}, "
            f"processed={processed}, failed={failed}"
        )

    throughput = processed / external_runtime

    return {
        "round": round_number,
        "configuration": config["name"],
        "workers": config["workers"],
        "batch_size": config["batch_size"],
        "external_runtime_seconds": round(external_runtime, 4),
        "pipeline_runtime_seconds": round(internal_runtime, 4),
        "processed": processed,
        "failed": failed,
        "throughput_images_per_second": round(throughput, 2),
        "peak_sampled_process_tree_rss_mb": round(peak_memory, 2),
        "return_code": process.returncode,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    randomizer = random.Random(RANDOM_SEED)
    results = []

    fields = [
        "round",
        "configuration",
        "workers",
        "batch_size",
        "external_runtime_seconds",
        "pipeline_runtime_seconds",
        "processed",
        "failed",
        "throughput_images_per_second",
        "peak_sampled_process_tree_rss_mb",
        "return_code",
    ]

    csv_file = REPORT_DIR / "validation_results.csv"
    summary_file = REPORT_DIR / "validation_summary.txt"

    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for round_number in range(1, REPETITIONS + 1):
            order = CONFIGURATIONS.copy()
            randomizer.shuffle(order)

            print(f"\n========== ROUND {round_number} ==========")

            for config in order:
                print(
                    f"\nTesting {config['name']}: "
                    f"{config['workers']} workers, "
                    f"batch {config['batch_size']}"
                )

                result = run_configuration(config, round_number)

                writer.writerow(result)
                file.flush()
                results.append(result)

                print(result)

    summaries = []

    for config in CONFIGURATIONS:
        group = [
            row for row in results
            if row["configuration"] == config["name"]
        ]

        summaries.append({
            "name": config["name"],
            "workers": config["workers"],
            "batch_size": config["batch_size"],
            "median_runtime": statistics.median(
                row["external_runtime_seconds"] for row in group
            ),
            "median_pipeline_runtime": statistics.median(
                row["pipeline_runtime_seconds"] for row in group
            ),
            "median_throughput": statistics.median(
                row["throughput_images_per_second"] for row in group
            ),
            "median_memory": statistics.median(
                row["peak_sampled_process_tree_rss_mb"] for row in group
            ),
            "runtime_stdev": statistics.stdev(
                row["external_runtime_seconds"] for row in group
            ),
            "total_processed": sum(row["processed"] for row in group),
            "total_failed": sum(row["failed"] for row in group),
        })

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("VISIONBOOST FINAL CONFIGURATION VALIDATION\n")
        file.write("=" * 65 + "\n")
        file.write(f"Dataset size: {DATASET_SIZE}\n")
        file.write(f"Repetitions per configuration: {REPETITIONS}\n")
        file.write(f"Total executions: {len(results)}\n")
        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(
            "Memory: median peak sampled aggregate process-tree RSS\n"
        )
        file.write(
            "External runtime includes Python interpreter startup.\n\n"
        )

        for row in summaries:
            file.write(f"{row['name'].upper()}\n")
            file.write(
                f"Configuration: {row['workers']} workers, "
                f"batch {row['batch_size']}\n"
            )
            file.write(
                f"Median external runtime: "
                f"{row['median_runtime']:.4f} sec\n"
            )
            file.write(
                f"Median pipeline runtime: "
                f"{row['median_pipeline_runtime']:.4f} sec\n"
            )
            file.write(
                f"Median throughput: "
                f"{row['median_throughput']:.2f} img/sec\n"
            )
            file.write(
                f"Median peak sampled RSS: "
                f"{row['median_memory']:.2f} MB\n"
            )
            file.write(
                f"Runtime standard deviation: "
                f"{row['runtime_stdev']:.4f} sec\n"
            )
            file.write(
                f"Total processed: {row['total_processed']}\n"
            )
            file.write(
                f"Total failed: {row['total_failed']}\n\n"
            )

        fastest = min(summaries, key=lambda x: x["median_runtime"])
        lowest_memory = min(summaries, key=lambda x: x["median_memory"])

        file.write("FASTEST CONFIGURATION\n")
        file.write(
            f"{fastest['name']}: {fastest['workers']} workers, "
            f"batch {fastest['batch_size']}\n\n"
        )

        file.write("LOWEST MEMORY CONFIGURATION\n")
        file.write(
            f"{lowest_memory['name']}: {lowest_memory['workers']} workers, "
            f"batch {lowest_memory['batch_size']}\n"
        )

    print("\nFINAL VALIDATION COMPLETED")
    print(f"Results: {csv_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    DATASET_SIZE = len(list(INPUT_DIR.glob("*.jpg")))

    if args.child:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        child_run(args.workers, args.batch_size)
    else:
        if DATASET_SIZE != 500:
            raise ValueError(
                f"Expected 500 input images, found {DATASET_SIZE}"
            )
        main()
