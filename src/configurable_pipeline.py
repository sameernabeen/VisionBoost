
import argparse
import cv2
import csv
import time
import psutil

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VisionBoost configurable image pipeline"
    )

    parser.add_argument(
        "--input",
        default="dataset/scale_input",
        help="Input image directory"
    )

    parser.add_argument(
        "--output",
        default="dataset/configurable_output",
        help="Output image directory"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    return parser.parse_args()


def load_image(path):
    return cv2.imread(str(path))


def process_image(path, image, output_dir):
    if image is None:
        return False

    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return cv2.imwrite(
        str(output_dir / path.name),
        edges
    )


def run_pipeline(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    image_paths = sorted(input_dir.glob("*.jpg"))

    if not image_paths:
        raise SystemExit(f"No JPG images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    peak_ram = process.memory_info().rss

    processed = 0
    failed = 0

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        for index in range(0, len(image_paths), args.batch_size):
            paths = image_paths[
                index:index + args.batch_size
            ]

            images = list(executor.map(load_image, paths))

            peak_ram = max(
                peak_ram,
                process.memory_info().rss
            )

            for path, image in zip(paths, images):
                try:
                    if process_image(path, image, output_dir):
                        processed += 1
                    else:
                        failed += 1

                except Exception as error:
                    print(f"Error: {path.name}: {error}")
                    failed += 1

            del images

    elapsed = time.perf_counter() - start

    throughput = processed / elapsed if elapsed else 0

    result = {
        "dataset_size": len(image_paths),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "processed": processed,
        "failed": failed,
        "runtime_seconds": round(elapsed, 4),
        "throughput": round(throughput, 2),
        "peak_sampled_ram_mb": round(
            peak_ram / (1024 * 1024), 2
        )
    }

    return result


def save_report(result):
    report_path = Path("reports/configurable_results.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = report_path.exists()

    with report_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    args = parse_arguments()

    if args.workers < 1 or args.batch_size < 1:
        raise SystemExit("Workers and batch size must be positive.")

    print("\n===== VISIONBOOST CONFIGURABLE PIPELINE =====")
    print("Input:", args.input)
    print("Output:", args.output)
    print("Workers:", args.workers)
    print("Batch size:", args.batch_size)

    result = run_pipeline(args)

    print("\n===== RESULTS =====")

    for key, value in result.items():
        print(f"{key}: {value}")

    save_report(result)
