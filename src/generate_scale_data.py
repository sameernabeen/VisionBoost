
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=1000)
args = parser.parse_args()

COUNT = args.count
OUTPUT_DIR = Path("dataset/scale_input")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

existing = list(OUTPUT_DIR.glob("*.jpg"))

if existing:
    raise SystemExit(
        "Dataset already contains images. "
        "Use an empty folder or remove old generated images first."
    )

print(f"Generating {COUNT} synthetic images...")

start = time.perf_counter()

for i in range(COUNT):
    image = np.random.randint(
        0, 256, (720, 1280, 3), dtype=np.uint8
    )

    path = OUTPUT_DIR / f"image_{i:05d}.jpg"

    success = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        raise RuntimeError(f"Failed to generate {path}")

    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{COUNT}")

elapsed = time.perf_counter() - start

print("\n===== DATASET GENERATION COMPLETE =====")
print("Images:", COUNT)
print(f"Time: {elapsed:.2f} seconds")
print(f"Output: {OUTPUT_DIR}")
