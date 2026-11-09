
import cv2
import time
import psutil
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = Path("dataset/scale_input")
OUTPUT_DIR = Path("dataset/scale_output_5000")

WORKERS = 16
BATCH_SIZE = 32

IMAGE_PATHS = sorted(INPUT_DIR.glob("*.jpg"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

process = psutil.Process()
peak_ram = process.memory_info().rss

def update_memory():
    global peak_ram
    peak_ram = max(peak_ram, process.memory_info().rss)

def load_image(path):
    return cv2.imread(str(path))

def process_image(path, image):
    if image is None:
        return False

    resized = cv2.resize(image, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return cv2.imwrite(str(OUTPUT_DIR / path.name), edges)

print("\n===== VISIONBOOST SCALE BENCHMARK =====")
print("Images:", len(IMAGE_PATHS))
print("Workers:", WORKERS)
print("Batch size:", BATCH_SIZE)

if not IMAGE_PATHS:
    raise SystemExit("No images found in dataset/scale_input")

start = time.perf_counter()
processed = 0
failed = 0

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    for index in range(0, len(IMAGE_PATHS), BATCH_SIZE):
        paths = IMAGE_PATHS[index:index + BATCH_SIZE]
        images = list(executor.map(load_image, paths))

        update_memory()

        for path, image in zip(paths, images):
            if process_image(path, image):
                processed += 1
            else:
                failed += 1

        del images
        update_memory()

elapsed = time.perf_counter() - start
throughput = processed / elapsed
peak_ram_mb = peak_ram / (1024 * 1024)

print("\n===== RESULTS =====")
print("Processed:", processed)
print("Failed:", failed)
print(f"Runtime: {elapsed:.4f} seconds")
print(f"Throughput: {throughput:.2f} images/sec")
print(f"Peak sampled RAM: {peak_ram_mb:.2f} MB")
