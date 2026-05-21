#!/bin/bash
# ============================================================================
# NVIDIA Blackwell GPU Compatibility Fix
# ============================================================================
# Target: RTX 5070, RTX 5080, RTX 5090 (Blackwell architecture, sm_120)
#
# What this does: rebuilds llama-cpp-python with the Blackwell CUDA arch
# flags. CLAUDE.md and the Dockerfile use the same flags
# (-DCMAKE_CUDA_ARCHITECTURES=100;120). The PyTorch cu128 wheels in the
# main requirements file already include sm_120 kernels, so no torch
# pinning is needed here.
#
# When to use: Only if `llama_cpp` ImportError mentions a missing Blackwell
# kernel or you see cuBLAS errors on RTX 50-series GPUs at first inference.
#
# Author: Bradley R. Kinnard
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# pick the closest .venv we can find
for candidate in "$PROJECT_ROOT/.venv" "$PROJECT_ROOT/venv" "$PROJECT_ROOT/backend/venv"; do
    if [ -f "$candidate/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$candidate/bin/activate"
        echo "[fix_blackwell] activated $candidate"
        break
    fi
done

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[fix_blackwell] No virtual environment active. Create one first." >&2
    exit 1
fi

echo "[fix_blackwell] Rebuilding llama-cpp-python with Blackwell CUDA arch flags..."
CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100;120" \
    pip install --no-cache-dir --force-reinstall llama-cpp-python

echo "[fix_blackwell] Done. Restart the server."
