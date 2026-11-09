# Model-free research demo image. Provision trusted checkpoints at runtime.

# Stage 1: Build the React Dashboard
FROM node:22-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci --prefer-offline
COPY web/ ./
RUN npm run build

# Stage 2: Python Backend & Runtime Environment
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (libsndfile, ffmpeg for telecom codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Checkpoints are intentionally not distributed or baked into the image.
# Mount a trusted checkpoint directory read-only at /app/checkpoints.
RUN mkdir -p /app/checkpoints
# Copy source code, configurations, and tracked demo data assets
COPY vaanishield/ /app/vaanishield/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY demo/ /app/demo/

# Copy compiled frontend from Stage 1 into web/dist (matching demo/api/server.py WEB_DIST_DIR)
COPY --from=frontend-builder /app/web/dist /app/web/dist

# This checks process liveness, not model readiness. /api/ready is separate.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Run FastAPI production ASGI server
CMD ["uvicorn", "demo.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
