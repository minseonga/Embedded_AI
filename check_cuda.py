
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Check your PyTorch installation and Docker runtime (needs --runtime nvidia).")
