FROM python:3.11-slim

WORKDIR /app

# Install any required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python project configuration
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir .[api]

# Cache fastembed weights during build
RUN python -c "from fastembed import TextEmbedding; TextEmbedding()" || true

COPY examples/ examples/

EXPOSE 8000

CMD ["uvicorn", "examples.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
