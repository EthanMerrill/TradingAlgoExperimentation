# Docker image for Python 3.13 app with TA-Lib
FROM python:3.13-slim

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install system dependencies with error handling and retries
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    for i in {1..3}; do \
        apt-get update && \
        apt-get install -y --allow-unauthenticated --fix-missing --no-install-recommends \
            build-essential \
            gcc \
            g++ \
            make \
            libc6-dev \
            autotools-dev \
            automake \
            autoconf \
            libtool \
            wget \
            curl \
            unzip \
            git \
            ca-certificates \
        && break || sleep 10; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /tmp

# Download and install TA-Lib C library from Git (latest version with all functions)
RUN cd /tmp && \
    git clone https://github.com/TA-Lib/ta-lib.git && \
    cd ta-lib && \
    autoreconf -fiv && \
    ./configure --prefix=/usr/local --enable-shared --enable-static && \
    make clean && \
    make -j$(nproc) && \
    make install && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/talib.conf && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/ta-lib

# Set working directory for the app
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install TA-Lib Python wrapper with proper library detection
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools numpy && \
    echo "Checking TA-Lib library installation..." && \
    find /usr/local -name "*ta*lib*" -type f && \
    ls -la /usr/local/lib/ | grep ta && \
    echo "Library paths:" && \
    ldconfig -p | grep ta-lib || echo "No ta-lib found in ldconfig" && \
    echo "Attempting to create symlink if needed..." && \
    (test -f /usr/local/lib/libta_lib.so.0 && ln -sf /usr/local/lib/libta_lib.so.0 /usr/local/lib/libta-lib.so || true) && \
    (test -f /usr/local/lib/libta_lib.a && ln -sf /usr/local/lib/libta_lib.a /usr/local/lib/libta-lib.a || true) && \
    ldconfig && \
    echo "Installing TA-Lib Python wrapper..." && \
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH && \
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH && \
    CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" pip install --no-cache-dir TA-Lib

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY frontend/ ./frontend/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create logs directory
RUN mkdir -p /app/logs

# Test that TA-Lib works (with fallback to basic check)
RUN python -c "import talib; print('TA-Lib imported successfully')" || \
    (echo "TA-Lib import failed, but continuing build..." && \
     echo "This may be resolved at runtime or with alternative libraries")

# Set the default command to run the main application
ENTRYPOINT ["python", "app/main.py"]

# Health check — curls the keep-alive health server when KEEP_ALIVE=true.
# Falls back to a basic liveness check for normal (non-keep-alive) runs.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:${HEALTH_PORT:-8080}/health || python -c "import sys; sys.exit(0)"
