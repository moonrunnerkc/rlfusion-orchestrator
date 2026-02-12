# Author: Bradley R. Kinnard
# Quick GPU stack verification for RTX 5070 Blackwell

import torch
print('PyTorch CUDA check:')
print(f"  Available: {torch.cuda.is_available()}")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"  Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}")  # Expect (12, 0) for 5070

import faiss
res = faiss.StandardGpuResources()
print('FAISS GPU check:')
print(f"  Resources loaded: {res}")
print('GPU stack: GREEN LIGHT! 🚀')
