#!/bin/bash
# GCP Setup Script for MQ-Det (Debian 12 host)

set -e
echo "🚀 Setting up MQ-Det on Google Cloud Platform (Debian 12)..."

sudo apt-get update

# NVIDIA driver check / (re)load if needed
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "✅ NVIDIA driver detected:"
  nvidia-smi
else
  echo "⚠️ NVIDIA driver not active. Installing and asking for reboot..."
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
  sudo apt-get update
  sudo apt-get install -y nvidia-driver
  echo "🔄 Reboot required. After reboot, rerun: ./gcp_setup.sh"
  exit 0
fi

# Docker
if ! command -v docker &>/dev/null; then
  echo "📦 Installing Docker..."
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  rm get-docker.sh
fi

# NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "🐳 Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# Quick runtime test
echo "🧪 Testing NVIDIA Docker runtime (CUDA 11.3 devel image)..."
sudo docker run --rm --gpus all nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 nvidia-smi

# Build image
echo "🔨 Building mq-det image…"
sudo docker build -t mq-det .

# Smoke test
echo "🧪 Container smoke test..."
sudo docker run --rm --gpus all mq-det /bin/bash -c 'python - <<PY
import torch, sys
print(f"✅ Python: {sys.version.split()[0]}")
print(f"✅ Torch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
PY'

echo "🎉 Setup complete.

Next:
  1) Put your data in ./DATASET, weights in ./MODEL (or let the container download).
  2) Start services:   docker compose up -d
  3) Shell into it:    docker compose exec mq-det /bin/bash
  4) Run:              ./extract_queries.sh   (then ./train.sh, ./evaluate.sh)
"
