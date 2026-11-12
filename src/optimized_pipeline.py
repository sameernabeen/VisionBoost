
import argparse
import cv2
import time

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


DEFAULT_WORKERS = 8
DEFAULT_BATCH_SIZE = 32


def read_image(path):
    return path, cv2.imread(str(path))


def process_image(path, image, output_dir):

    if image is None:
        return False

    try:
        resized = cv2.resize(image, (640, 360))

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            100,
            200
        )

        output_path = output_dir / path.name

        return cv2.imwrite(str(output_path), edges)

    except cv2.error:
        return False


def run_pipeline(input_dir, output_dir, workers, batch_size):

    if workers < 1:
        raise ValueError("Workers must be at least 1.")

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(input_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError("No JPG images found in the input directory.")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:

        for start in range(0, len(image_paths), batch_size):

            batch_paths = image_paths[start:start + batch_size]

            loaded_images = executor.map(read_image, batch_paths)

            for path, image in loaded_images:

                if process_image(path, image, output_dir):
                    processed += 1
                else:
                    failed += 1

                del image

    total_time = time.perf_counter() - start_time

    throughput = (
        processed / total_time
        if total_time > 0
        else 0
    )

    print("\n===== VISIONBOOST OPTIMIZED PIPELINE =====")

    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Workers:", workers)
    print("Batch size:", batch_size)

    print("Images found:", len(image_paths))
    print("Images processed:", processed)
    print("Failed:", failed)

    print("Total time:", round(total_time, 4), "seconds")
    print("Throughput:", round(throughput, 2), "images/sec")

    return processed, failed, total_time


def main():

    parser = argparse.ArgumentParser(
        description="VisionBoost configurable image-processing pipeline"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/input"),
        help="Input image directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/output_optimized"),
        help="Output image directory"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel image-loading workers"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of images loaded per batch"
    )

    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output,
        args.workers,
        args.batch_size
    )


if __name__ == "__main__":
    main()
