# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# System dependencies (TA-Lib C build + Python build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc make libc6-dev \
    autotools-dev automake autoconf libtool \
    wget curl ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build TA-Lib C library from source
WORKDIR /tmp
RUN git clone https://github.com/TA-Lib/ta-lib.git \
    && cd ta-lib \
    && autoreconf -fiv \
    && ./configure --prefix=/usr/local --enable-shared --enable-static \
    && make -j$(nproc) \
    && make install \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/talib.conf \
    && ldconfig \
    && cd / && rm -rf /tmp/ta-lib

# Install TA-Lib Python wrapper + runtime deps into a wheelhouse
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" \
       LD_LIBRARY_PATH=/usr/local/lib \
       PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
       pip wheel --no-cache-dir --wheel-dir=/wheels TA-Lib \
    && pip wheel --no-cache-dir --wheel-dir=/wheels -r /tmp/requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Minimal runtime deps (only what TA-Lib shared lib needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib shared library from builder
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/talib.conf && ldconfig

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:${HEALTH_PORT:-8080}/health || exit 1
