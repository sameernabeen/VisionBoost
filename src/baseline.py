
import cv2
import os
import glob
import time

INPUT_DIR = "dataset/input"
OUTPUT_DIR = "dataset/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)

start_time = time.perf_counter()

processed_count = 0

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        continue

    # Resize image
    resized = cv2.resize(
        image,
        (640, 360)
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detect edges
    edges = cv2.Canny(
        blurred,
        100,
        200
    )

    # Save processed image
    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(image_path)
    )

    success = cv2.imwrite(output_path, edges)

    if success:
        processed_count += 1

end_time = time.perf_counter()

total_time = end_time - start_time

print("\n===== BASELINE RESULTS =====")

print("Images processed:", processed_count)

print("Execution time:",
      round(total_time, 2), "seconds")

if total_time > 0:
    print(
        "Throughput:",
        round(processed_count / total_time, 2),
        "images/second"
    )
