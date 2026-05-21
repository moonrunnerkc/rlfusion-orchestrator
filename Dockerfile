# Author: Bradley R. Kinnard
# Dockerfile - RLFusion Orchestrator (2-path CAG+Graph architecture)
# Multi-stage: builder compiles llama-cpp-python with CUDA Blackwell flags,
# production image excludes training scripts, test files, and dead deps.
# GPU profile for docker-compose; CPU fallback also supported.

# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential cmake git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY backend/requirements.txt backend/requirements.txt

# Install torch (CUDA 12.1) and llama-cpp-python with Blackwell flags
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN CUDACXX=/usr/local/cuda/bin/nvcc \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100;120" \
    pip install --no-cache-dir llama-cpp-python

RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Stage 2: Production ─────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Non-root execution: every endpoint mutates files under data/, db/, or
# indexes/, and the container can be made to read those mount points
# in r/w mode without granting it root over anything else on the host.
RUN groupadd --system rlfusion \
    && useradd --system --gid rlfusion --create-home --shell /usr/sbin/nologin rlfusion

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code only. data/ and models/ are bind-mounted at
# runtime so a poisoned image cannot inject documents or model weights
# into a deployment.
COPY backend/ backend/
COPY conftest.py .
COPY scripts/init_db.sh scripts/init_db.sh
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Pre-create writable mount targets owned by the runtime user.
RUN mkdir -p /app/db /app/indexes /app/data /app/models \
    && chown -R rlfusion:rlfusion /app \
    && chmod +x /usr/local/bin/entrypoint.sh

USER rlfusion

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')" || exit 1

# entrypoint.sh idempotently initializes the SQLite database against the
# bind-mounted db/ volume before handing off to uvicorn.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
