
import argparse
import csv
import logging
import time
import cv2
import psutil

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    filename="reports/pipeline_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionBoost robust image pipeline"
    )

    parser.add_argument("--input", default="dataset/scale_input")
    parser.add_argument("--output", default="dataset/robust_output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)

    return parser.parse_args()

def safe_load(path):
    try:
        image = cv2.imread(str(path))

        if image is None:
            return path, None, "Image is corrupted or unreadable"

        return path, image, None

    except Exception as error:
        return path, None, str(error)

def process_image(path, image, output_dir):
    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    output_path = output_dir / path.name

    if not cv2.imwrite(str(output_path), edges):
        raise IOError("Failed to save processed image")

def main():
    args = parse_args()

    if args.workers < 1 or args.batch_size < 1:
        raise SystemExit("Workers and batch size must be positive")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    report_dir = Path("reports")

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))

    if not image_paths:
        raise SystemExit("No JPG images found")

    error_rows = []
    processed = 0
    failed = 0

    process = psutil.Process()
    peak_sampled_ram = process.memory_info().rss

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        for index in range(0, len(image_paths), args.batch_size):

            batch_paths = image_paths[
                index:index + args.batch_size
            ]

            loaded_results = list(
                executor.map(safe_load, batch_paths)
            )

            peak_sampled_ram = max(
                peak_sampled_ram,
                process.memory_info().rss
            )

            for path, image, load_error in loaded_results:

                if load_error:
                    failed += 1
                    error_rows.append({
                        "filename": str(path),
                        "stage": "loading",
                        "error": load_error
                    })

                    logging.error(
                        "%s | loading | %s",
                        path,
                        load_error
                    )
                    continue

                try:
                    process_image(path, image, output_dir)
                    processed += 1

                except Exception as error:
                    failed += 1

                    error_rows.append({
                        "filename": str(path),
                        "stage": "processing_or_saving",
                        "error": str(error)
                    })

                    logging.exception(
                        "Failed processing %s",
                        path
                    )

            del loaded_results

            peak_sampled_ram = max(
                peak_sampled_ram,
                process.memory_info().rss
            )

    elapsed = time.perf_counter() - start

    error_file = report_dir / "pipeline_errors.csv"

    with error_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "stage", "error"]
        )

        writer.writeheader()
        writer.writerows(error_rows)

    summary = {
        "dataset_size": len(image_paths),
        "processed": processed,
        "failed": failed,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "runtime_seconds": round(elapsed, 4),
        "throughput": round(
            processed / elapsed if elapsed else 0, 2
        ),
        "peak_sampled_ram_mb": round(
            peak_sampled_ram / (1024 * 1024), 2
        )
    }

    print("\n===== VISIONBOOST ROBUST PIPELINE =====")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nError report:", error_file)
    print("Error log: reports/pipeline_errors.log")

if __name__ == "__main__":
    main()
