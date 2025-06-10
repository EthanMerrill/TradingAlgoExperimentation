# Use Python 3.11 for better async performance and modern features
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Install system dependencies including TA-Lib C library
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    make \
    build-essential \
    pkg-config \
    libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# If libta-lib-dev is not available, build from source as fallback
RUN if ! pkg-config --exists ta-lib; then \
        cd /tmp && \
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
        tar -xzf ta-lib-0.4.0-src.tar.gz && \
        cd ta-lib/ && \
        ./configure --prefix=/usr && \
        make && \
        make install && \
        ldconfig && \
        cd / && \
        rm -rf /tmp/ta-lib*; \
    fi

# Set timezone to New York (same as US markets)
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip

# Install TA-Lib Python wrapper first (requires C library to be installed)
RUN pip install --no-cache-dir TA-Lib>=0.4.25

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy credential files if they exist
COPY ALPACA_KEYS.json* ./
COPY POLYGON_API.json* ./
COPY GOOGLE_APPLICATION_CREDENTIALS.json* ./

# Create logs directory
RUN mkdir -p logs

# Set the default command to run the new main application
CMD ["python", "app/main.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"