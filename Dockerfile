# Use Python 3.11 for better async performance and modern features
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to New York (same as US markets)
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
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