# AI Research & Knowledge Assistant — container image
FROM python:3.11-slim

# System deps needed by PyMuPDF / TensorFlow / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime data directories (mount volumes here in production)
RUN mkdir -p data/raw_documents data/vector_db data/dataset models

ENV PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Train the classifier at build time if a labelled dataset is present, then serve.
CMD ["sh", "-c", "python -m src.ml.train_classifier || echo 'Skipping classifier training (no dataset found).'; uvicorn main:app --host ${APP_HOST} --port ${APP_PORT}"]
