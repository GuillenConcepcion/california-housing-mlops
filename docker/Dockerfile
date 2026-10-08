# Multi-stage Dockerfile for Ajuste de Hiperparametros Vivienda California
# Lead Architect: Guillén Concepción
FROM python:3.13-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Final Runtime Image
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

COPY --from=builder /root/.local /root/.local

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY dashboards/ ./dashboards/

RUN pip install --no-deps -e .

EXPOSE 8000 8501

CMD ["python", "scripts/serve.py"]
