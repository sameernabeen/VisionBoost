
import os
import time
import psutil
import cv2

process = psutil.Process(os.getpid())

print("===== SYSTEM INFORMATION =====")

print("CPU cores:", os.cpu_count())

print("RAM usage:",
      round(process.memory_info().rss / (1024 ** 2), 2),
      "MB")

print("OpenCV version:", cv2.__version__)

print("OpenCV threads:", cv2.getNumThreads())

print("===== BENCHMARK ENVIRONMENT READY =====")
