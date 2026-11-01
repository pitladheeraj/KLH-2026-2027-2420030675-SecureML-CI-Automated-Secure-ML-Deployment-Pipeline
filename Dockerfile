# SecureML-CI: Production Hardened Dockerfile
# Base: Minimal Debian-based Python slim image
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="models/model.joblib"

WORKDIR /app

# Install security updates and curl for container healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security Best Practice: Create a non-root system user and group
RUN groupadd -r mlgroup && useradd -r -g mlgroup -s /sbin/nologin -d /app mluser

# Copy application code and model artifacts
COPY src/ /app/src/
COPY models/ /app/models/

# Assign ownership to non-root user
RUN chown -R mluser:mlgroup /app

# Switch execution context to non-root user
USER mluser

# Expose FastAPI listening port
EXPOSE 8000

# Container healthcheck for automated monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start prediction service with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
