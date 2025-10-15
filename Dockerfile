# MQ-Det container (stable build path)
# CUDA 11.3 + cuDNN8 + Ubuntu 20.04 + Python 3.9 + GCC-8
# PyTorch 1.12.1+cu113 (works with your successful native build)

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# --- CUDA + build env ---
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# T4 (compute capability 7.5)
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV FORCE_CUDA=1

# --- Base OS deps + Python 3.9 + GCC-8 ---
RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates gnupg \
    git wget curl vim pkg-config \
    build-essential cmake ninja-build \
    libjpeg-dev zlib1g-dev libpng-dev \
    libgl1 libglib2.0-0 \
    gcc-8 g++-8 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Keep build tooling friendly for older extensions
RUN python3.9 -m pip install --upgrade pip setuptools==68.2.2 wheel

# --- Torch (matches your working env) ---
RUN python3.9 -m pip install \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Make sure the dynamic loader can see torch .so's
RUN echo "/usr/local/lib/python3.9/dist-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

# --- Core Python deps for MQ-Det / GLIP path ---
RUN python3.9 -m pip install -U \
    cython ninja yacs pycocotools \
    opencv-python matplotlib pillow tqdm \
    numpy==1.24.3 scipy scikit-learn \
    timm==0.6.7 transformers==4.31.0 \
    tensorboard wandb

# GroundingDINO & dataset helpers used by MQ-Det (install without pulling Torch 2.x)
RUN python3.9 -m pip install -U --no-cache-dir \
    einops einops-exts==0.0.4 ftfy pandas prettytable pymongo shapely tensorboardX \
    supervision==0.4.0 yapf \
    "requests==2.31.*" charset-normalizer

# --- Bring in your working tree (build context should be the repo root) ---
WORKDIR /workspace
COPY . /workspace

# Patch legacy includes for newer ATen headers (safe no-op if already patched)
# Also fix relative import for native extension to avoid shadowing issues.
RUN set -eux; \
    sed -i 's@<ATen/ceil_div.h>@<c10/util/ceil_div.h>@' \
        maskrcnn_benchmark/csrc/cuda/ROIAlign_cuda.cu \
        maskrcnn_benchmark/csrc/cuda/ROIPool_cuda.cu || true; \
    sed -i 's@<ATen/cuda/ThrustAllocator.h>@<ATen/cuda/CUDAThrustAllocator.h>@' \
        maskrcnn_benchmark/csrc/cuda/ml_nms.cu || true; \
    if grep -qE '^[[:space:]]*import[[:space:]]+_C' maskrcnn_benchmark/__init__.py; then \
        sed -i 's/^[[:space:]]*import[[:space:]]\+_C/from . import _C/' maskrcnn_benchmark/__init__.py; \
    fi

# Toolchain for CUDA builds
ENV CC=/usr/bin/gcc-8
ENV CXX=/usr/bin/g++-8
ENV CUDAHOSTCXX=/usr/bin/g++-8
ENV CXXFLAGS="-O3 -std=c++14"
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_USE_PEP517=0
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Build the native extension now so the container is "ready-to-run"
RUN python3.9 setup_glip.py clean --all || true && \
    rm -f maskrcnn_benchmark/_C*.so && \
    python3.9 setup_glip.py build_ext --inplace

# Sanity import
RUN python - <<'PY'
import os, maskrcnn_benchmark, maskrcnn_benchmark._C as _C
print("maskrcnn_benchmark from:", maskrcnn_benchmark.__file__)
print("native _C import OK; modeling dir exists:",
      os.path.isdir(os.path.join(os.path.dirname(maskrcnn_benchmark.__file__), "modeling")))
PY

# Friendly entrypoint
RUN printf '%s\n' '#!/bin/bash' \
  'echo "🚀 MQ-Det container ready (CUDA 11.3 / Torch 1.12.1+cu113)"' \
  'nvcc --version | sed -n "s/.*release/release/p"' \
  'python - <<PY' \
  'import torch' \
  'print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())' \
  'PY' \
  'exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
