#!/bin/bash
# ============================================================================
# NVIDIA Blackwell GPU Compatibility Fix
# ============================================================================
# Target: RTX 5070, RTX 5080, RTX 5090 (Blackwell architecture, sm_120)
#
# Problem: PyTorch stable releases may not include Blackwell (sm_120) CUDA
#          kernels, causing cuBLAS errors during tensor operations. This is
#          because Blackwell shipped after the PyTorch stable release cycle.
#
# Solution: Install PyTorch nightly builds which include Blackwell support.
#           Once a stable PyTorch release ships with sm_120 support, this
#           script is no longer needed.
#
# When to use: Only if you see cuBLAS or CUDA errors on RTX 50-series GPUs.
#              Not needed for Ampere (RTX 30xx), Ada Lovelace (RTX 40xx), or
#              CPU-only setups.
#
# Author: Bradley R. Kinnard
# ============================================================================

echo "Installing PyTorch nightly for Blackwell GPU support..."

# Activate virtual environment
source venv/bin/activate

# Install PyTorch nightly with CUDA 12.4 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

echo "Done! Restart the server now."
