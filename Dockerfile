# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Python build tools (for any package without a prebuilt wheel, e.g. asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc make libc6-dev ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build all runtime deps into a wheelhouse
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip wheel --no-cache-dir --wheel-dir=/wheels -r /tmp/requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Minimal runtime deps (curl for the HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels
COPY --from=builder /wheels /tmp/wheels

# Install from wheelhouse (fast, no compilation)
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY frontend/ ./frontend/

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create logs directory and set ownership
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

ENTRYPOINT ["python", "app/main.py"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${HEALTH_PORT:-8080}/health || exit 1
