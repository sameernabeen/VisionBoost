
import cv2
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = "dataset/input"
OUTPUT_DIR = "dataset/output_optimized"

WORKERS = 16
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)


def read_image(path):
    return path, cv2.imread(path)


def process_image(path, image):

    if image is None:
        return False

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

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(path)
    )

    return cv2.imwrite(output_path, edges)


start_time = time.perf_counter()

processed = 0
failed = 0

with ThreadPoolExecutor(max_workers=WORKERS) as executor:

    for start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[start:start + BATCH_SIZE]

        for path, image in executor.map(read_image, batch_paths):

            if process_image(path, image):
                processed += 1
            else:
                failed += 1

            del image

total_time = time.perf_counter() - start_time

print("\n===== BATCH-OPTIMIZED PIPELINE RESULTS =====")

print("Workers:", WORKERS)
print("Batch size:", BATCH_SIZE)
print("Images processed:", processed)
print("Failed:", failed)

print("Total time:", round(total_time, 4), "seconds")

if total_time > 0:
    print(
        "Throughput:",
        round(processed / total_time, 2),
        "images/sec"
    )
