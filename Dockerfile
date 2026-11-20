# Multi-stage production container for Room2Rug Multimodal Search Service
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY configs/ configs/
COPY app.py .
COPY pyproject.toml .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    CATALOG_PATH=data/products.json \
    RERANKER_PATH=models/lambdamart_reranker.txt

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
