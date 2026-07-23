# Use Ubuntu with CUDA support
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Make scripts executable
RUN chmod +x /workspace/src/*.sh

# Run setup script from root directory
RUN cd /workspace && bash src/setup.sh

# Create output directories
RUN mkdir -p /workspace/finetuning_output /workspace/hf_export /workspace/logs

# Set entrypoint
ENTRYPOINT ["bash", "src/finetune.sh"]

# Default command (can be overridden)
CMD []
