FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY cli.py ./
COPY app ./app
COPY prompts ./prompts

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[rag,vlm]"

RUN mkdir -p \
    /app/cache/huggingface \
    /app/data/vector_store

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=10 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]