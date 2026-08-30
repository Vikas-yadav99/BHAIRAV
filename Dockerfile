# ============================================
# BHAIRAV — Multi-stage Docker Build
# ============================================

# --- Stage 1: Base Python ---
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV + YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Dependencies ---
FROM base AS deps

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ml.txt

# --- Stage 3: Application ---
FROM deps AS app

COPY src/ src/
COPY scripts/ scripts/
COPY dashboard/ dashboard/
COPY models/ models/

# Download YOLO model at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Non-root user
RUN useradd -m -r bhairav && chown -R bhairav:bhairav /app
USER bhairav

EXPOSE 8000

# Default: run the server
CMD ["python", "-m", "bhairav.serve", "--host", "0.0.0.0", "--port", "8000"]
