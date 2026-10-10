# Multi-Stage Production Dockerfile for VaaniShield Enterprise SaaS

# Stage 1: Build the React Dashboard
FROM node:20-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci --prefer-offline || npm install
COPY web/ ./
RUN npm run build

# Stage 2: Python Backend & Runtime Environment
FROM python:3.10-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (libsndfile, ffmpeg for telecom codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, checkpoints, configurations, and data assets
COPY vaanishield/ /app/vaanishield/
COPY configs/ /app/configs/
COPY checkpoints/ /app/checkpoints/
COPY data/ /app/data/
COPY demo/ /app/demo/

# Copy compiled frontend from Stage 1 into demo/web_dist
COPY --from=frontend-builder /app/web/dist /app/demo/web_dist

EXPOSE 8000

# Run FastAPI production ASGI server
CMD ["uvicorn", "demo.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
