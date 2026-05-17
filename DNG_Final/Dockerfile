# Stage 1: Base Python environment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV, LibRAW need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libraw-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install CPU-only PyTorch (smaller image, sufficient for inference)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY api/          ./api/
COPY model/        ./model/
COPY data/dataset.py ./data/dataset.py
COPY utils/        ./utils/
COPY pipeline/     ./pipeline/
COPY dashboard/    ./dashboard/
COPY checkpoints_v3/ ./checkpoints_v3/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
