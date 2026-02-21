# Author: Bradley R. Kinnard
# Dockerfile - RLFusion Orchestrator backend
# Supports both CPU and GPU via docker-compose profiles.
# Multi-arch: builds for linux/amd64 and linux/arm64 (Jetson, M-series, Snapdragon).

FROM python:3.10-slim

WORKDIR /app

# Detect architecture at build time for conditional deps
ARG TARGETARCH

# System deps for faiss, numpy, sqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

# Install CPU-only torch first (GPU users mount nvidia runtime via compose)
# ARM64 gets torch from the default index (wheels available for aarch64)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio && \
        pip install --no-cache-dir faiss-cpu; \
    else \
        pip install --no-cache-dir \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir faiss-cpu; \
    fi && \
    pip install --no-cache-dir -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-deps torch torchvision torchaudio faiss-gpu-cu12 || true

COPY backend/ backend/
COPY conftest.py .
COPY data/ data/
COPY indexes/ indexes/
COPY models/ models/
COPY scripts/ scripts/

# Initialize the database if it does not exist
RUN mkdir -p db && bash scripts/init_db.sh

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
