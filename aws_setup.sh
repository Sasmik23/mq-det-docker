#!/bin/bash
# AWS EC2 + Docker Setup Script for Official MQ-Det

set -e
echo "🚀 Setting up Official MQ-Det on AWS EC2..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Update system
print_status "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install Docker
print_status "Installing Docker..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Step 3: Install NVIDIA Docker
print_status "Installing NVIDIA Docker support..."
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Step 4: Test NVIDIA Docker
print_status "Testing NVIDIA Docker installation..."
if sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi; then
    print_success "NVIDIA Docker working correctly!"
else
    print_error "NVIDIA Docker installation failed!"
    exit 1
fi

# Step 5: Clone MQ-Det repository
print_status "Cloning MQ-Det repository..."
if [ ! -d "MQ-Det" ]; then
    git clone https://github.com/YifanXu74/MQ-Det.git
fi
cd MQ-Det

# Step 6: Create necessary directories
print_status "Creating project structure..."
mkdir -p DATASET OUTPUT configs/custom

# Step 7: Build Docker image
print_status "Building MQ-Det Docker image (this may take 15-20 minutes)..."
sudo docker build -t mq-det:official .

# Step 8: Test Docker container
print_status "Testing MQ-Det Docker container..."
if sudo docker run --rm --gpus all mq-det:official python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    print_success "MQ-Det Docker image built successfully!"
else
    print_error "Docker image build failed!"
    exit 1
fi

print_success "🎉 AWS EC2 + Docker setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your dataset to ./DATASET/"
echo "2. Run: sudo docker-compose up -d"
echo "3. Execute: sudo docker exec -it mq-det_mq-det_1 /bin/bash"
echo "4. Start official MQ-Det training!"
echo ""
echo "💡 Estimated costs:"
echo "   - p3.2xlarge: ~$3/hour"
echo "   - Training time: 2-4 hours"
echo "   - Total cost: $6-12"