FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV MODEL_PATH=/app/artifacts/model.joblib \
    META_PATH=/app/artifacts/model_meta.json \
    BACKEND_URL=http://localhost:8000

EXPOSE 8000

CMD ["uvicorn", "src.fraudml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
