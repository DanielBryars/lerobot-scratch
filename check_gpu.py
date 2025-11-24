import torch
print(torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Name:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
