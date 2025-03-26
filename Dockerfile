# Use NVIDIA CUDA base image with PyTorch support for GPU acceleration
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    HF_HOME=/app/huggingface

# Install system dependencies and CV stuff
RUN apt-get update && apt-get install -y \
    python3.9 \
    wget \
    git \
    curl \
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for managing environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set Conda path
ENV PATH="/opt/conda/bin:$PATH"

# Clone ControlNet repository
RUN git clone https://github.com/lllyasviel/ControlNet.git /app/ControlNet

# Set working directory
WORKDIR /app/ControlNet

# Create and activate ControlNet Conda environment
RUN conda env create -f environment.yaml && conda clean -afy
RUN echo "source activate control" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Set PYTHONPATH to include ControlNet
ENV PYTHONPATH="/app/ControlNet:$PYTHONPATH"

# Ensure models directory exists
RUN mkdir -p /app/ControlNet/models

# Uncomment below to download the model using HuggingFace CLI while building container - untested
# RUN pip install huggingface-hub
# RUN huggingface-cli download lllyasviel/ControlNet models/control_sd15_canny.pth --local-dir /app/ControlNet

# Copy models folder into ControlNet's models directory
COPY models/ /app/ControlNet/models/

# Copy additional project files
WORKDIR /app
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY test_imgs/ /app/test_imgs/
COPY demo/ /app/demo/

# Expose API port
EXPOSE 8000

# Run the API
CMD ["bash", "-c", "source activate control && python demo/app.py"]
