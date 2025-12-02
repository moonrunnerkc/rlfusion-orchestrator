#!/bin/bash
# Fix for Blackwell GPU (RTX 5070) cuBLAS issues
# This installs PyTorch nightly with proper Blackwell support

echo "Installing PyTorch nightly for Blackwell GPU support..."

# Activate virtual environment
source venv/bin/activate

# Install PyTorch nightly with CUDA 12.4 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

echo "Done! Restart the server now."
