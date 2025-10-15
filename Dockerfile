# MQ-Det container (stable build path)
# CUDA 11.3 + cuDNN8 + Ubuntu 20.04 + PyTorch 1.12.1+cu113
# (works with MQ-Det + GLIP + patched maskrcnn_benchmark)

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# -- GPU / CUDA env --
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# T4 (SM 7.5). Add more if you need other GPUs later.
ENV TORCH_CUDA_ARCH_LIST=7.5
ENV FORCE_CUDA=1

# -- OS deps + Python 3.9 + toolchain --
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

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

RUN python3.9 -m pip install --upgrade pip setuptools==68.2.2 wheel

# -- PyTorch 1.12.1 + cu113 (matching torchvision/torchaudio) --
RUN python3.9 -m pip install \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Ensure runtime can find torch CUDA libs
RUN echo "/usr/local/lib/python3.9/dist-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

# -- Core Python deps for MQ-Det / GLIP / GroundingDINO(new) --
RUN python3.9 -m pip install -U \
    cython ninja yacs pycocotools \
    opencv-python matplotlib pillow tqdm \
    numpy==1.24.3 scipy scikit-learn \
    transformers==4.21.3 timm==0.6.7 tensorboard wandb \
    # GroundingDINO(new) extras that caused your error
    addict einops einops-exts==0.0.4 ftfy pandas prettytable pymongo shapely tensorboardX \
    supervision==0.4.0 \
    # modern requests stack (avoid old chardet warning)
    "requests==2.31.*" charset-normalizer

# Workspace
WORKDIR /workspace
# Copy everything (you’re bind-mounting in compose, but this enables docker build tests)
COPY . /workspace

# Patch old ATen include moved in newer torch
# (ROIAlign/ROIPool include was the common build break)
RUN set -eux; \
  sed -i 's@<ATen/ceil_div.h>@<c10/util/ceil_div.h>@' \
      maskrcnn_benchmark/csrc/cuda/ROIAlign_cuda.cu \
      maskrcnn_benchmark/csrc/cuda/ROIPool_cuda.cu || true

# Build the CUDA/C++ extensions
ENV CC=/usr/bin/gcc-8
ENV CXX=/usr/bin/g++-8
ENV CUDAHOSTCXX=/usr/bin/g++-8
ENV CXXFLAGS="-O3 -std=c++14"
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_USE_PEP517=0

RUN python3.9 setup_glip.py clean --all || true && \
    rm -f maskrcnn_benchmark/_C*.so && \
    python3.9 setup_glip.py build_ext --inplace

# Sanity check native extension
RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace")
from maskrcnn_benchmark import _C
print("✅ _C import OK")
PY

# PYTHONPATH for local sources
ENV PYTHONPATH=/workspace

# Model dirs, weights convenience (skip if you bind mount MODEL/)
RUN mkdir -p MODEL DATASET OUTPUT configs/custom

# Simple entrypoint that prints env health, then execs your cmd/shell
RUN printf '%s\n' '#!/bin/bash' \
  'echo "🚀 MQ-Det container ready (CUDA 11.3 / Torch 1.12.1+cu113)"' \
  'nvcc --version | sed -n "s/.*release/release/p"' \
  'python - <<PY' \
  'import torch' \
  'print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())' \
  'PY' \
  'exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
