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

# ---- toolchain & headers needed by maskrcnn_benchmark ----
RUN apt-get update && apt-get install -y \
    gcc-8 g++-8 libjpeg-dev zlib1g-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

# Use GCC 8 for CUDA 11.x extensions
ENV CC=/usr/bin/gcc-8
ENV CXX=/usr/bin/g++-8
ENV CUDAHOSTCXX=/usr/bin/g++-8
ENV CXXFLAGS="-O3 -std=c++14"

# Ensure pip uses the current env (where torch is already installed)
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_USE_PEP517=0

# (Optional but harmless) CUDA hints
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_BIN_PATH=/usr/local/cuda/bin
ENV CUDA_INCLUDE_DIRS=/usr/local/cuda/include
ENV CUDA_LIBRARIES=/usr/local/cuda/lib64

# ---- Install GLIP (which builds maskrcnn_benchmark) from repo root ----
RUN git clone --recurse-submodules https://github.com/microsoft/GLIP.git /tmp/GLIP && \
    python3.9 -m pip install -v -e /tmp/GLIP --no-build-isolation --config-settings editable_mode=compat && \
    rm -rf /tmp/GLIP
# ----------------------------------------------------------------------


# Create necessary directories
RUN mkdir -p MODEL DATASET OUTPUT configs/custom

# Download pre-trained GLIP model
RUN cd MODEL && \
    wget -q https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

# Create entrypoint script
RUN printf '#!/bin/bash\n\
echo "🚀 MQ-Det Official Environment Ready!"\n\
echo "CUDA Version: $(nvcc --version | grep release)"\n\
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")"\n\
echo "CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")"\n\
if [ "$#" -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]