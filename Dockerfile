# MQ-Det container (stable build path)
# Uses CUDA 11.3 + cuDNN8 + Ubuntu 20.04 and PyTorch 1.10.1+cu113
# This combo provides the legacy THC headers needed by maskrcnn_benchmark.

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# --- CUDA / Torch tuning ---
# T4 (sm_75). Add more archs if you run on other GPUs.
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1

# --- Base OS deps + Python 3.9 (via deadsnakes), dev tools, image libs ---
RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates gnupg \
    git wget curl vim pkg-config \
    build-essential cmake ninja-build \
    libjpeg-dev zlib1g-dev libpng-dev \
    gcc-8 g++-8 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Make Python 3.9 the default and bootstrap pip cleanly
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Pin setuptools to a friendly version for older build systems
RUN python3.9 -m pip install --upgrade pip setuptools==68.2.2 wheel

# --- PyTorch 1.10.1 + cu113 (torchvision/torchaudio matched) ---
RUN python3.9 -m pip install \
    torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# --- Common Python deps for MQ-Det ---
RUN python3.9 -m pip install \
    cython ninja yacs opencv-python pycocotools matplotlib pillow tqdm \
    numpy==1.24.3 scipy scikit-learn \
    transformers==4.21.3 timm==0.6.7 tensorboard wandb

# Workdir and MQ-Det repo
WORKDIR /workspace
RUN git clone https://github.com/YifanXu74/MQ-Det.git . && \
    chmod +x tools/*.py

# --- Build GLIP (this compiles its vendored maskrcnn_benchmark with THC headers under torch 1.10) ---
# Toolchain/env for CUDA extensions
ENV CC=/usr/bin/gcc-8
ENV CXX=/usr/bin/g++-8
ENV CUDAHOSTCXX=/usr/bin/g++-8
ENV CXXFLAGS="-O3 -std=c++14"
# Ensure pip sees the current env (where torch is already installed)
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_USE_PEP517=0

RUN git clone --recurse-submodules https://github.com/microsoft/GLIP.git /tmp/GLIP && \
    python3.9 -m pip install -v -e /tmp/GLIP --no-build-isolation --config-settings editable_mode=compat && \
    # Avoid shadowing: MQ-Det has a source folder named maskrcnn_benchmark — move it aside
    [ -d /workspace/maskrcnn_benchmark ] && mv /workspace/maskrcnn_benchmark /workspace/maskrcnn_benchmark_src || true && \
    rm -rf /tmp/GLIP

# Model weights dir
RUN mkdir -p MODEL DATASET OUTPUT configs/custom && \
    wget -q -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth \
      https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth

# PYTHONPATH so MQ-Det imports resolve
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Simple entrypoint that prints env health, then opens a shell (or runs the passed command)
RUN printf '%s\n' '#!/bin/bash' \
  'echo "🚀 MQ-Det (Torch 1.10.1 + cu113) container ready"' \
  'nvcc --version | sed -n "s/.*release/release/p"' \
  'python - <<PY' \
  'import torch; import os' \
  'print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())' \
  'PY' \
  'exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
