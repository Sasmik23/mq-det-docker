#!/bin/bash
# GCP Setup Script for MQ-Det Docker Deployment
# Configures CUDA 11.8, Docker, and NVIDIA Container Toolkit

set -e

echo "🚀 Setting up MQ-Det on Google Cloud Platform..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update

# Verify CUDA installation (should be pre-installed on Deep Learning image)
echo "🔍 Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver found:"
    nvidia-smi
else
    echo "❌ NVIDIA driver not found, installing..."
    # Install NVIDIA driver
    sudo apt-get install -y nvidia-driver-470
    echo "🔄 Reboot required after driver installation"
    exit 1
fi

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✅ Docker already installed"
else
    echo "📦 Installing Docker..."
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install NVIDIA Container Toolkit if not present
echo "🐳 Setting up NVIDIA Docker..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    # Add NVIDIA Container Toolkit repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ NVIDIA Container Toolkit installed"
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# Test NVIDIA Docker
echo "🧪 Testing NVIDIA Docker setup..."
if sudo docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi; then
    echo "✅ NVIDIA Docker working correctly"
else
    echo "❌ NVIDIA Docker test failed"
    exit 1
fi

# Create required directories
echo "📁 Creating directories..."
mkdir -p MODEL OUTPUT

# Download GLIP-T model if not present
echo "📥 Checking for GLIP-T model..."
if [ ! -f "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth" ]; then
    echo "⬇️  Downloading GLIP-T model (this may take a few minutes)..."
    wget -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth \
        "https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth"
    echo "✅ GLIP-T model downloaded"
else
    echo "✅ GLIP-T model already exists"
fi

# Build Docker image
echo "🔨 Building MQ-Det Docker image..."
sudo docker build -t mq-det .

# Test container creation
echo "🧪 Testing container creation..."
if sudo docker run --rm --gpus all mq-det python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
    echo "✅ Container test successful"
else
    echo "❌ Container test failed"
    exit 1
fi

# Display setup completion
echo ""
echo "🎉 GCP Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Upload your dataset: scp -r DATASET/ user@vm-ip:~/mq-det-docker/"
echo "   2. Start container: sudo docker-compose up -d"
echo "   3. Enter container: sudo docker exec -it mq-det-docker_mq-det_1 /bin/bash"
echo "   4. Extract queries: ./extract_queries.sh"
echo "   5. Train model: ./train.sh"
echo "   6. Evaluate: ./evaluate.sh"
echo ""
echo "💰 Estimated Training Cost on GCP:"
echo "   - Regular T4: ~$1.50-2.00 total"
echo "   - Preemptible T4: ~$0.30 total"
echo ""
echo "🔍 Monitor with: watch -n 1 nvidia-smi"