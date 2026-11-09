
import cv2
import numpy as np
import os

OUTPUT_DIR = "dataset/input"
TOTAL_IMAGES = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(TOTAL_IMAGES):

    # Generate a random image
    image = np.random.randint(
        0,
        256,
        (720, 1280, 3),
        dtype=np.uint8
    )

    # Draw a rectangle
    cv2.rectangle(
        image,
        (100, 100),
        (500, 400),
        (255, 0, 0),
        5
    )

    # Draw a circle
    cv2.circle(
        image,
        (700, 350),
        100,
        (0, 255, 0),
        5
    )

    filename = f"image_{i:04d}.jpg"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(output_path, image)

print(f"Generated {TOTAL_IMAGES} images successfully!")
