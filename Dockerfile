# Author: Bradley R. Kinnard
# Dockerfile - RLFusion Orchestrator backend
# Supports both CPU and GPU via docker-compose profiles.

FROM python:3.10-slim

WORKDIR /app

# System deps for faiss, numpy, sqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

# Install CPU-only torch first (GPU users mount nvidia runtime via compose)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir faiss-cpu && \
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
