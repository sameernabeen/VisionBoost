
import cv2
import os
import glob
import time
import psutil
import pandas as pd

INPUT_DIR = "dataset/input"
OUTPUT_DIR = "dataset/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)

if not image_paths:
    raise FileNotFoundError("No JPG images found in dataset/input")

timings = {
    "Image Reading": 0.0,
    "Resizing": 0.0,
    "Grayscale Conversion": 0.0,
    "Gaussian Blur": 0.0,
    "Canny Edge Detection": 0.0,
    "Image Saving": 0.0
}

processed_count = 0
failed_count = 0

process = psutil.Process(os.getpid())

total_start = time.perf_counter()

for image_path in image_paths:

    # 1. Read image
    start = time.perf_counter()
    image = cv2.imread(image_path)
    timings["Image Reading"] += time.perf_counter() - start

    if image is None:
        failed_count += 1
        continue

    # 2. Resize
    start = time.perf_counter()
    resized = cv2.resize(image, (640, 360))
    timings["Resizing"] += time.perf_counter() - start

    # 3. Convert to grayscale
    start = time.perf_counter()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    timings["Grayscale Conversion"] += time.perf_counter() - start

    # 4. Gaussian blur
    start = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    timings["Gaussian Blur"] += time.perf_counter() - start

    # 5. Canny edge detection
    start = time.perf_counter()
    edges = cv2.Canny(blurred, 100, 200)
    timings["Canny Edge Detection"] += time.perf_counter() - start

    # 6. Save processed image
    start = time.perf_counter()

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(image_path)
    )

    success = cv2.imwrite(output_path, edges)

    timings["Image Saving"] += time.perf_counter() - start

    if success:
        processed_count += 1
    else:
        failed_count += 1

total_time = time.perf_counter() - total_start

# Calculate profiling report
rows = []

for operation, duration in timings.items():

    percentage = (
        duration / total_time * 100
        if total_time > 0 else 0
    )

    rows.append({
        "Operation": operation,
        "Time (seconds)": round(duration, 4),
        "Percentage (%)": round(percentage, 2)
    })

report = pd.DataFrame(rows)

print("\n========== OPERATION PROFILING ==========")
print(report.to_string(index=False))

print("\n========== OVERALL PERFORMANCE ==========")
print("Images found:", len(image_paths))
print("Images processed:", processed_count)
print("Failed images:", failed_count)
print("Total runtime:", round(total_time, 4), "seconds")
print(
    "Throughput:",
    round(processed_count / total_time, 2),
    "images/second"
)

print(
    "Python memory:",
    round(process.memory_info().rss / (1024 ** 2), 2),
    "MB"
)

# Save profiling results
os.makedirs("reports", exist_ok=True)

report.to_csv(
    "reports/profiling_results.csv",
    index=False
)

print("\nProfiling report saved to reports/profiling_results.csv")
