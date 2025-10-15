# Official MQ-Det Docker Environment - Paper Implementation (Optimized for smaller disk)
# Exact paper environment: python==3.9, torch==2.0.1, GCC==8.3.1, CUDA==11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables for CUDA 11.7
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1

# Install system dependencies and Python 3.9 (Ubuntu 20.04 default)
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
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
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default python and install pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Upgrade pip for Python 3.9
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.0.1 with CUDA 11.7 support (exact paper implementation)
RUN python3.9 -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install essential Python packages
RUN python3.9 -m pip install \
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

# Install maskrcnn-benchmark from GLIP (more compatible with MQ-Det)
# Set additional CUDA environment variables for compilation
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_BIN_PATH=/usr/local/cuda/bin
ENV CUDA_INCLUDE_DIRS=/usr/local/cuda/include
ENV CUDA_LIBRARIES=/usr/local/cuda/lib64

# Verify CUDA installation before building
RUN nvcc --version && \
    python3.9 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install additional build dependencies for GLIP
RUN python3.9 -m pip install \
    pybind11 \
    build \
    wheel

# Build GLIP's bundled maskrcnn_benchmark (more compatible with GLIP/MQ-Det)
RUN git clone --depth 1 https://github.com/microsoft/GLIP.git /tmp/GLIP && \
    cd /tmp/GLIP && \
    # Install GLIP without PEP 517 to avoid build issues
    python3.9 -m pip install -e . --no-build-isolation && \
    cd /workspace && \
    rm -rf /tmp/GLIP

# Create necessary directories
RUN mkdir -p MODEL DATASET OUTPUT configs/custom

# Download pre-trained GLIP model
RUN cd MODEL && \
    wget -q https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

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