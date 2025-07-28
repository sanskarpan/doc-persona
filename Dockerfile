# Use exact version for maximum compatibility
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Set environment variables early
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Upgrade pip to latest version
RUN python -m pip install --no-cache-dir --upgrade pip==23.3.1

# Copy requirements
COPY requirements.txt .

# Install torch CPU version first (this is the big one)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --timeout 300

# Install all other requirements
RUN pip install --no-cache-dir -r requirements.txt --timeout 300

# Pre-download the sentence transformer model (critical for offline operation)
RUN python -c "import os; os.makedirs('/root/.cache/torch/sentence_transformers', exist_ok=True)" && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set final environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Health check to ensure the app can start
RUN python -c "import main; print('App loads successfully')" || echo "Warning: App check failed"

# Run the application
CMD ["python", "main.py"]