# Author: Bradley R. Kinnard
# Dockerfile - RLFusion Orchestrator (2-path CAG+Graph architecture)
# Multi-stage: builder compiles llama-cpp-python with CUDA Blackwell flags,
# production image runs as an unprivileged user, models/data are bind-mounted
# at runtime rather than baked in.

# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential cmake git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY backend/requirements.txt backend/requirements.txt

# llama-cpp-python with Blackwell flags. Torch comes from the lockfile install
# below so the cu128 wheels match the runtime CUDA stack.
RUN CUDACXX=/usr/local/cuda/bin/nvcc \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100;120" \
    pip install --no-cache-dir llama-cpp-python

RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Stage 2: Production ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip sqlite3 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin --uid 1500 rlfusion

WORKDIR /app

# Copy installed packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code only. Data, db and models are mounted by compose at
# runtime so the image stays slim and rebuilds do not invalidate the cache
# on every doc upload.
COPY backend/ backend/
COPY conftest.py .
COPY scripts/init_db.sh scripts/init_db.sh

RUN mkdir -p /app/db /app/indexes /app/data/docs /app/models \
    && chown -R rlfusion:rlfusion /app

USER rlfusion

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')" || exit 1

# tini keeps PID 1 sane and reaps zombies; bash entrypoint initializes the
# DB schema on first boot, then handing off to uvicorn.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-c", "bash scripts/init_db.sh && python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
