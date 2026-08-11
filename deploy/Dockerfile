# BHAIRAV - production image
# Build:  docker build -t bhairav -f deploy/Dockerfile .
# Run:    see docker-compose.yml (nginx terminates TLS in front)
FROM python:3.12-slim

# OpenCV needs these at runtime; keep the image small otherwise
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) the app
COPY config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY dashboard ./dashboard
# models/ must exist in the build context (a .gitkeep keeps it present on a
# fresh clone); run scripts/fetch_models.py to populate the real weights
COPY models ./models

# non-root runtime user
RUN useradd --create-home --uid 1000 bhairav \
    && mkdir -p /data/evidence /data/output \
    && chown -R bhairav:bhairav /data /app
USER bhairav

# evidence + state live on a volume; the pipeline reads its source from config
VOLUME ["/data"]
ENV BHAIRAV_EVIDENCE_KEY="" \
    BHAIRAV_SECRET="" \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["python", "scripts/serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--evidence", "/data/evidence"]
