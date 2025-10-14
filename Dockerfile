# Official MQ-Det Docker Environment
# Base image with CUDA 11.8 and PyTorch pre-installed
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support (official MQ-Det compatible)
RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install essential Python packages
RUN pip3 install \
    cython \
    ninja \
    yacs \
    opencv-python \
    pycocotools \
    matplotlib \
    pillow \
    tqdm \
    numpy==1.24.3 \
    scipy \
    scikit-learn \
    transformers==4.21.3 \
    timm==0.6.7 \
    tensorboard \
    wandb

# Set working directory
WORKDIR /workspace

# Clone MQ-Det repository
RUN git clone https://github.com/YifanXu74/MQ-Det.git . && \
    chmod +x tools/*.py

# Install maskrcnn-benchmark from source with proper CUDA support
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git /tmp/maskrcnn && \
    cd /tmp/maskrcnn && \
    python setup.py build develop && \
    cd /workspace && \
    rm -rf /tmp/maskrcnn

# Create necessary directories
RUN mkdir -p MODEL DATASET OUTPUT configs/custom

# Download pre-trained GLIP model
RUN cd MODEL && \
    wget -q https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "🚀 MQ-Det Official Environment Ready!"\n\
echo "CUDA Version: $(nvcc --version | grep release)"\n\
echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA Available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
if [ "$#" -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]