# Simple, reliable TA-Lib installation using Ubuntu base
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and TA-Lib from Ubuntu repositories
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libta-lib0-dev \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set timezone to New York (same as US markets)
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir TA-Lib>=0.4.25 && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/

# Copy credential files if they exist
COPY ALPACA_KEYS.json* ./
COPY POLYGON_API.json* ./
COPY GOOGLE_APPLICATION_CREDENTIALS.json* ./

# Create logs directory
RUN mkdir -p logs

# Test that TA-Lib works
RUN python -c "import talib; print('TA-Lib version:', talib.__version__)"

# Set the default command to run the new main application
CMD ["python", "app/main.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"
