FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .
COPY src/ src/

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels .[api,all]

FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

COPY pyproject.toml .
COPY README.md .
COPY src/ src/

RUN python -c "from fastembed import TextEmbedding; TextEmbedding()"

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import synaptoroute" || exit 1

CMD ["python", "-m", "uvicorn", "synaptoroute.api:app", "--host", "0.0.0.0", "--port", "8000"]
