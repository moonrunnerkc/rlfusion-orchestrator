# Author: Bradley R. Kinnard
# Full GPU stack test - embeddings, FAISS, PyTorch training

import time
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("FULL GPU STACK TEST - RTX 5070 BLACKWELL")
print("=" * 70)

# 1. PyTorch GPU Check
print("\n1️⃣  PyTorch GPU Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 2. SentenceTransformers GPU Embeddings
print("\n2️⃣  Testing GPU Embeddings (SentenceTransformers):")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Loading model on: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

test_texts = [
    "GPU acceleration is working perfectly",
    "FAISS vector search with CUDA support",
    "RTX 5070 Blackwell architecture rocks",
    "Reinforcement learning with PPO training",
    "Multi-modal retrieval with CLIP models"
] * 20  # 100 texts total

print(f"   Embedding {len(test_texts)} texts on GPU...")
start = time.time()
embeddings = model.encode(test_texts, convert_to_numpy=True, show_progress_bar=False)
gpu_time = time.time() - start
print(f"   ✅ Embedded in {gpu_time:.3f}s ({len(test_texts)/gpu_time:.1f} texts/sec)")
print(f"   Embedding shape: {embeddings.shape}")
print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 3. FAISS CPU Index (GPU FAISS has kernel compatibility issues with Blackwell)
print("\n3️⃣  Testing FAISS CPU Indexing:")
d = embeddings.shape[1]  # dimension
print(f"   Creating CPU index (dim={d})...")
print(f"   Note: FAISS GPU kernels not yet compiled for compute 12.0 (Blackwell)")

# Create CPU index
index = faiss.IndexFlatL2(d)
print(f"   CPU Index created: ✅")

# Add vectors
start = time.time()
index.add(embeddings)
add_time = time.time() - start
print(f"   Added {len(embeddings)} vectors in {add_time:.3f}s")

# Search
print(f"   Running search (top-5 neighbors)...")
start = time.time()
D, I = index.search(embeddings[:10], 5)
search_time = time.time() - start
print(f"   ✅ Searched in {search_time:.4f}s")
print(f"   Sample distances: {D[0][:3]}")

# 4. PyTorch Tensor Operations
print("\n4️⃣  Testing PyTorch GPU Tensor Ops:")
size = 10000
print(f"   Creating {size}x{size} tensors on GPU...")
a = torch.randn(size, size, device='cuda')
b = torch.randn(size, size, device='cuda')

start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
matmul_time = time.time() - start
print(f"   ✅ Matrix multiplication: {matmul_time:.3f}s")
print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 5. Summary
print("\n" + "=" * 70)
print("🚀 GPU STACK PERFORMANCE SUMMARY")
print("=" * 70)
print(f"✅ GPU Embeddings:    {len(test_texts)/gpu_time:.1f} texts/sec (GPU)")
print(f"✅ FAISS Add:         {len(embeddings)/add_time:.0f} vectors/sec (CPU)")
print(f"✅ FAISS Search:      {search_time*1000:.2f}ms for 10 queries (CPU)")
print(f"✅ PyTorch MatMul:    {matmul_time:.3f}s ({size}x{size}) (GPU)")
print(f"✅ Peak GPU Memory:   {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
print("=" * 70)
print("🎉 GPU STACK: GREEN LIGHT!")
print("⚠️  FAISS GPU: Waiting for Blackwell kernel compilation")
print("=" * 70)
