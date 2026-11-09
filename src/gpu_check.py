
import torch

print("PyTorch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

    print("CUDA version:", torch.version.cuda)

else:

    print("CUDA is not available.")
    print("We will investigate the PyTorch installation and drivers.")
