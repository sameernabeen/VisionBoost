
import time
import torch

print("=" * 45)
print("VISIONBOOST GPU BENCHMARK")
print("=" * 45)

if not torch.cuda.is_available():
    print("CUDA is not available!")
    raise SystemExit

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory:", round(
    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
), "GB")

# Create a large image-like tensor
batch = torch.rand(8, 3, 720, 1280)

# CPU benchmark
start = time.perf_counter()

for _ in range(20):
    result_cpu = torch.nn.functional.interpolate(
        batch,
        size=(360, 640),
        mode="bilinear",
        align_corners=False
    )

cpu_time = time.perf_counter() - start

# GPU benchmark
batch_gpu = batch.to(device)

# Warm-up
for _ in range(5):
    _ = torch.nn.functional.interpolate(
        batch_gpu,
        size=(360, 640),
        mode="bilinear",
        align_corners=False
    )

torch.cuda.synchronize()

start = time.perf_counter()

for _ in range(20):
    result_gpu = torch.nn.functional.interpolate(
        batch_gpu,
        size=(360, 640),
        mode="bilinear",
        align_corners=False
    )

torch.cuda.synchronize()
gpu_time = time.perf_counter() - start

print("\n--- RESULTS ---")
print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")

print("\nBenchmark completed!")
