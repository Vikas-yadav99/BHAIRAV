# BHAIRAV — City Safety Surveillance System
# Multi-stage build for production deployment
FROM python:3.12-slim AS base

# System deps for OpenCV, numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[ml]" 2>/dev/null || \
    pip install --no-cache-dir fastapi uvicorn numpy opencv-python-headless \
    Pillow ultralytics pyyaml

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY dashboard/ dashboard/
COPY output/ output/ 2>/dev/null || true

# Non-root user
RUN useradd -m bhairav && chown -R bhairav:bhairav /app
USER bhairav

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "scripts.serve:app", "--host", "0.0.0.0", "--port", "8000"]
