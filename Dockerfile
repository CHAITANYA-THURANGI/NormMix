# Production Dockerfile for NormMix AI Universal Studio
# Fully compatible with Hugging Face Spaces (Docker), Render, Railway, and Cloud Run

FROM python:3.11-slim

# Setup non-root user with UID 1000 (Required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch first (fast ~180MB download, avoids downloading 2.5GB CUDA in cloud CPU containers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code and assign ownership to user 1000
COPY --chown=user:user . /home/user/app

# Ensure writable directories for logs and active learning feedback
RUN mkdir -p data/processed experiments/checkpoints experiments/logs && \
    chown -R user:user /home/user/app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Expose both Hugging Face Spaces standard port (7860) and standard HTTP port (8000)
EXPOSE 7860
EXPOSE 8000

# Dynamically bind to the platform's assigned PORT (defaults to 7860 on Hugging Face, 8000 locally/Render)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
