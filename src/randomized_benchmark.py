
import csv
import random
import re
import statistics
import subprocess
import sys
import time

from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "dataset" / "input"

REPORT_DIR = PROJECT_DIR / "reports" / "randomized_benchmark"

CSV_FILE = REPORT_DIR / "benchmark_results.csv"

SUMMARY_FILE = REPORT_DIR / "benchmark_summary.txt"

REPETITIONS = 5

RANDOM_SEED = 42

TIMEOUT_SECONDS = 600

DATASET_SIZE = len(list(INPUT_DIR.glob("*.jpg")))


# ============================================================
# 2. PIPELINE CONFIGURATION
# ============================================================

PIPELINES = {
    "baseline": [
        sys.executable,
        str(PROJECT_DIR / "src" / "baseline.py")
    ],

    "optimized": [
        sys.executable,
        str(PROJECT_DIR / "src" / "optimized_pipeline.py")
    ],

    "robust": [
        sys.executable,
        str(PROJECT_DIR / "src" / "robust_pipeline.py"),
        "--input", str(INPUT_DIR),
        "--output", str(
            PROJECT_DIR / "dataset" / "randomized_robust_output"
        ),
        "--workers", "16",
        "--batch-size", "32",
        "--report-dir", str(REPORT_DIR)
    ]
}


# ============================================================
# 3. EXTRACT RESULTS FROM PIPELINE OUTPUT
# ============================================================

def extract_number(pattern, output):

    match = re.search(
        pattern,
        output,
        re.IGNORECASE | re.MULTILINE
    )

    if match:
        return int(match.group(1))

    return None


def extract_results(name, output):

    if name in ("baseline", "optimized"):

        processed = extract_number(
            r"Images processed:\s*(\d+)",
            output
        )

    else:

        processed = extract_number(
            r"^\s*processed:\s*(\d+)",
            output
        )

    failed = extract_number(
        r"^\s*Failed:\s*(\d+)",
        output
    )

    if failed is None:
        failed = extract_number(
            r"^\s*failed:\s*(\d+)",
            output
        )

    if processed is None:
        raise ValueError(
            f"Could not extract processed count for {name}"
        )

    if failed is None:
        failed = DATASET_SIZE - processed

    return processed, failed


# ============================================================
# 4. RUN ONE PIPELINE
# ============================================================

def run_pipeline(name, command, round_number, position):

    print("\n" + "=" * 55)
    print(f"ROUND {round_number} | POSITION {position}")
    print(f"PIPELINE: {name.upper()}")
    print("=" * 55)

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS
    )

    elapsed = time.perf_counter() - start

    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:

        raise RuntimeError(
            f"{name} pipeline failed with exit code "
            f"{result.returncode}"
        )

    processed, failed = extract_results(
        name,
        result.stdout
    )

    throughput = (
        processed / elapsed
        if elapsed > 0
        else 0
    )

    row = {
        "round": round_number,
        "order_position": position,
        "pipeline": name,
        "runtime_seconds": round(elapsed, 4),
        "processed": processed,
        "failed": failed,
        "throughput_images_per_second": round(
            throughput, 2
        ),
        "return_code": result.returncode,
        "random_seed": RANDOM_SEED
    }

    print(f"External runtime: {elapsed:.4f} seconds")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Throughput: {throughput:.2f} images/sec")

    return row


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_result(row):

    fieldnames = [
        "round",
        "order_position",
        "pipeline",
        "runtime_seconds",
        "processed",
        "failed",
        "throughput_images_per_second",
        "return_code",
        "random_seed"
    ]

    file_exists = CSV_FILE.exists()

    with CSV_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# 6. GENERATE SUMMARY
# ============================================================

def generate_summary(rows):

    lines = []

    lines.append("VISIONBOOST RANDOMIZED BENCHMARK")
    lines.append("=" * 55)

    lines.append(f"Dataset size: {DATASET_SIZE}")
    lines.append(f"Repetitions: {REPETITIONS}")
    lines.append(f"Random seed: {RANDOM_SEED}")

    lines.append(
        "\nRuntime is measured externally around each "
        "Python subprocess. It includes interpreter startup."
    )

    lines.append(
        "\nAll comparisons use the same input dataset."
    )

    lines.append("\n" + "=" * 55)

    summary = {}

    for name in PIPELINES:

        selected = [
            row for row in rows
            if row["pipeline"] == name
        ]

        runtimes = [
            row["runtime_seconds"]
            for row in selected
        ]

        throughputs = [
            row["throughput_images_per_second"]
            for row in selected
        ]

        if not runtimes:
            continue

        median_runtime = statistics.median(runtimes)

        mean_runtime = statistics.mean(runtimes)

        std_runtime = (
            statistics.stdev(runtimes)
            if len(runtimes) > 1
            else 0
        )

        median_throughput = statistics.median(
            throughputs
        )

        total_processed = sum(
            row["processed"] for row in selected
        )

        total_failed = sum(
            row["failed"] for row in selected
        )

        summary[name] = median_runtime

        lines.append(f"\n{name.upper()}")

        lines.append(
            f"Median runtime: {median_runtime:.4f} sec"
        )

        lines.append(
            f"Mean runtime: {mean_runtime:.4f} sec"
        )

        lines.append(
            f"Standard deviation: {std_runtime:.4f} sec"
        )

        lines.append(
            f"Median throughput: {median_throughput:.2f} img/sec"
        )

        lines.append(
            f"Total processed: {total_processed}"
        )

        lines.append(
            f"Total failed: {total_failed}"
        )

    if "baseline" in summary:

        baseline_time = summary["baseline"]

        for name in ("optimized", "robust"):

            if name in summary:

                runtime = summary[name]

                speedup = baseline_time / runtime

                reduction = (
                    (baseline_time - runtime)
                    / baseline_time
                ) * 100

                lines.append(
                    f"\n{name.upper()} VS BASELINE"
                )

                lines.append(
                    f"Median speedup: {speedup:.2f}x"
                )

                lines.append(
                    f"Runtime reduction: {reduction:.2f}%"
                )

    report = "\n".join(lines)

    SUMMARY_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)


# ============================================================
# 7. MAIN BENCHMARK
# ============================================================

def main():

    if not INPUT_DIR.is_dir():
        raise SystemExit(
            f"Input directory not found: {INPUT_DIR}"
        )

    if DATASET_SIZE != 500:
        raise SystemExit(
            f"Expected 500 images, found {DATASET_SIZE}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Start a fresh CSV for this experiment.
    if CSV_FILE.exists():
        CSV_FILE.unlink()

    rng = random.Random(RANDOM_SEED)

    results = []

    print("\nVISIONBOOST RANDOMIZED BENCHMARK")

    print(f"Dataset: {DATASET_SIZE} images")
    print(f"Repetitions: {REPETITIONS}")
    print(f"Random seed: {RANDOM_SEED}")

    for round_number in range(1, REPETITIONS + 1):

        order = list(PIPELINES.keys())

        rng.shuffle(order)

        print("\nRandomized execution order:")
        print(" -> ".join(order))

        for position, name in enumerate(order, start=1):

            row = run_pipeline(
                name,
                PIPELINES[name],
                round_number,
                position
            )

            save_result(row)

            results.append(row)

    # Verify all executions completed.
    if len(results) != REPETITIONS * len(PIPELINES):

        raise RuntimeError(
            "Unexpected number of benchmark results"
        )

    generate_summary(results)

    print("\n" + "=" * 55)
    print("RANDOMIZED BENCHMARK COMPLETED")
    print(f"CSV: {CSV_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
