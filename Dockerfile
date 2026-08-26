# Use official Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install essential system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration files first for efficient Docker layer caching
COPY pyproject.toml MANIFEST.in ./
COPY src/ ./src/

# Install dependencies and project package in editable mode
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

# Copy remaining codebase assets
COPY . .

# Expose port 8000 for FastAPI + Gradio telemetry dashboard
EXPOSE 8000

# Default entrypoint command
CMD ["python", "launch.py"]
