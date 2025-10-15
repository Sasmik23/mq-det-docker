#!/bin/bash
# GCP Setup Script for MQ-Det on Debian 12 host
# Host: NVIDIA driver + Docker + NVIDIA Container Toolkit
# Container: CUDA 11.3 + Ubuntu 20.04 + Python 3.9 + GCC-8 + Torch 1.12.1+cu113

set -euo pipefail

echo "🚀 Setting up MQ-Det on Google Cloud Platform (Debian 12)..."

# 0) Update
sudo apt-get update -y

# 1) NVIDIA driver (skip if already working)
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null; then
  echo "✅ NVIDIA driver detected:"
  nvidia-smi
else
  echo "📋 Enabling non-free repos for Debian 12…"
  if [ ! -f /etc/apt/sources.list.d/debian-nonfree.sources ]; then
    sudo tee /etc/apt/sources.list.d/debian-nonfree.sources >/dev/null <<'EOF'
Types: deb
URIs: http://deb.debian.org/debian/
Suites: bookworm bookworm-updates
Components: main contrib non-free non-free-firmware

Types: deb
URIs: http://security.debian.org/debian-security
Suites: bookworm-security
Components: main contrib non-free non-free-firmware
EOF
  fi
  sudo apt-get update -y
  sudo apt-get install -y nvidia-driver
  echo "🔄 Reboot required to load the kernel modules:"
  echo "   sudo reboot"
  exit 0
fi

# 2) Docker (if missing)
if ! command -v docker >/dev/null; then
  echo "📦 Installing Docker..."
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker "$USER"
  rm -f get-docker.sh
fi

# 3) NVIDIA Container Toolkit (for --gpus all)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "🐳 Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# 4) Quick runtime check with CUDA 11.3 image (matches our Dockerfile)
echo "🧪 Testing NVIDIA Docker runtime..."
sudo docker run --rm --gpus all nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 nvidia-smi

# 5) Prepare host dirs
mkdir -p MODEL OUTPUT DATASET

# 6) Build image
echo "🔨 Building mq-det image..."
sudo docker build -t mq-det .

# 7) Smoke test
echo "🧪 Container smoke test..."
sudo docker run --rm --gpus all mq-det python - <<'PY'
import torch, sys
print("✅ Python:", sys.version.split()[0])
print("✅ Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

cat <<'EONEXT'

🎉 Setup complete.

Next:
  1) Put your data in ./DATASET, weights in ./MODEL (or let the container download).
  2) Start services:   docker compose up -d
  3) Shell into it:    docker compose exec mq-det /bin/bash
  4) Run:              ./extract_queries.sh   (then ./train.sh, ./evaluate.sh)

Tips:
  - If you edit CUDA/C++ sources, re-run:
      python3.9 setup_glip.py clean --all && python3.9 setup_glip.py build_ext --inplace
  - Keep TORCH_CUDA_ARCH_LIST="7.5" for T4. (T4 = compute capability 7.5.) 
EONEXT
