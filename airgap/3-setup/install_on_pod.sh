#!/bin/bash
# Installation for HMC Pod - NO PYTORCH UPGRADE
# Works with existing PyTorch 1.14 in base image

set -e

echo "========================================="
echo "🚀 MQ-Det Installation (HMC Pod)"
echo "   Using existing PyTorch 1.14"
echo "========================================="
echo ""

# Configuration
BUNDLE_DIR="/home/2300488/mik/mq-det-offline-bundle"

# Step 1: Verify bundle location
echo "📂 Step 1: Verifying bundle location..."
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "❌ ERROR: Bundle not found at $BUNDLE_DIR"
    exit 1
fi

cd "$BUNDLE_DIR"
echo "✅ Bundle found at: $(pwd)"
echo ""

# Step 2: Check current environment
echo "🐍 Step 2: Checking current environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Step 3: Keep existing PyTorch, just install dependencies
echo "📦 Step 3: Installing MQ-Det dependencies (keeping PyTorch 1.14)..."
echo "Note: groundingdino-new dependency warnings are OK - we'll handle them"
echo ""

sudo pip install --no-cache-dir \
    transformers \
    timm \
    ninja \
    yacs \
    cython \
    matplotlib \
    opencv-python \
    tqdm \
    pycocotools \
    cityscapesscripts \
    scipy \
    Pillow \
    ftfy \
    pymongo \
    shapely \
    tensorboardX \
    yapf \
    regex

# Note: Skipping supervision==0.4.0 as it may conflict
echo ""
echo "✅ Core dependencies installed"
echo ""

# Step 4: Set environment variables
echo "🌍 Step 4: Setting environment variables..."
export TORCH_HOME="$BUNDLE_DIR/MODEL"
export TRANSFORMERS_CACHE="$BUNDLE_DIR/hf_cache"
export TIMM_CACHE="$BUNDLE_DIR/timm_cache"
export PYTHONPATH="$BUNDLE_DIR:$PYTHONPATH"

# Make persistent (check if we have write permission)
if [ -w ~/.bashrc ]; then
    if ! grep -q "MQ-Det Environment" ~/.bashrc; then
        cat >> ~/.bashrc << EOF

# MQ-Det Environment
export TORCH_HOME="$BUNDLE_DIR/MODEL"
export TRANSFORMERS_CACHE="$BUNDLE_DIR/hf_cache"
export TIMM_CACHE="$BUNDLE_DIR/timm_cache"
export PYTHONPATH="$BUNDLE_DIR:\$PYTHONPATH"
EOF
        echo "✅ Environment variables added to ~/.bashrc"
    else
        echo "✅ Environment variables already in ~/.bashrc"
    fi
else
    echo "⚠️ Cannot write to ~/.bashrc (permission denied)"
    echo "Environment variables set for this session only."
    echo "To make permanent, manually add to your shell config:"
    echo "  export TORCH_HOME=\"$BUNDLE_DIR/MODEL\""
    echo "  export TRANSFORMERS_CACHE=\"$BUNDLE_DIR/hf_cache\""
    echo "  export TIMM_CACHE=\"$BUNDLE_DIR/timm_cache\""
    echo "  export PYTHONPATH=\"$BUNDLE_DIR:\$PYTHONPATH\""
fi

echo ""

# Step 5: Build maskrcnn_benchmark with CUDA extensions
echo "🔨 Step 5: Building maskrcnn_benchmark CUDA extensions..."
echo "This will take 10-15 minutes..."
echo ""

cd "$BUNDLE_DIR"

# Clean previous builds
if [ -d "build" ]; then
    sudo rm -rf build
fi
sudo rm -rf *.egg-info

# Build and install (skip dependency checks for air-gapped environment)
echo "Building maskrcnn_benchmark (ignoring dependency checks)..."
sudo python setup.py build develop --no-deps

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ maskrcnn_benchmark built successfully"
else
    echo ""
    echo "❌ Build failed! Check errors above"
    exit 1
fi

# Also build groundingdino_new
echo ""
echo "🔨 Building groundingdino_new..."
cd "$BUNDLE_DIR/groundingdino_new"
sudo rm -rf build *.egg-info
sudo python setup.py build develop --no-deps

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ groundingdino_new built successfully"
else
    echo ""
    echo "❌ Build failed! Check errors above"
    exit 1
fi

echo ""

# Step 6: Cache transformer models
echo "📥 Step 6: Caching transformer models..."
python -c "
try:
    from transformers import BertTokenizer, BertModel
    print('Loading BERT...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print('✅ BERT cached')
except Exception as e:
    print(f'⚠️ BERT cache: {e}')
    print('Will use from cache or download on first use')
"

echo ""

python -c "
try:
    import timm
    print('Loading Swin Transformer...')
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    print('✅ Swin Transformer cached')
except Exception as e:
    print(f'⚠️ Swin cache: {e}')
    print('Will use from cache or download on first use')
"

echo ""

# Step 7: Verify installation
echo "✅ Step 7: Verifying installation..."
echo ""

echo "=== Python & PyTorch ==="
python -c "
import sys
import torch
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== MQ-Det Components ==="
python -c "
import os
import sys

try:
    import maskrcnn_benchmark
    print('✅ maskrcnn_benchmark imported')
    from maskrcnn_benchmark import _C
    print('✅ CUDA extensions compiled and working')
except Exception as e:
    print(f'❌ maskrcnn_benchmark import failed: {e}')
    sys.exit(1)

try:
    from transformers import BertModel
    print('✅ transformers available')
except Exception as e:
    print(f'⚠️ transformers: {e}')

try:
    import timm
    print('✅ timm available')
except Exception as e:
    print(f'⚠️ timm: {e}')

# Check GLIP model
model_path = os.path.join(os.environ.get('TORCH_HOME', 'MODEL'), 'glip_tiny_model_o365_goldg_cc_sbu.pth')
if os.path.exists(model_path):
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f'✅ GLIP-T model found: {size_gb:.2f} GB')
else:
    print(f'⚠️ GLIP-T model not found at: {model_path}')
"

echo ""
echo "========================================="
echo "🎉 Installation Complete!"
echo "========================================="
echo ""
echo "Working directory: $BUNDLE_DIR"
echo "PyTorch version: 1.14 (from base image)"
echo ""
echo "⚠️ NOTE: groundingdino-new dependency warnings are OK"
echo "The version differences don't affect MQ-Det core functionality."
echo ""
echo "Next steps:"
echo "1. Source environment (for new terminals):"
echo "   source ~/.bashrc"
echo ""
echo "2. Test inference:"
echo "   cd $BUNDLE_DIR/tools"
echo "   python test_net.py --help"
echo ""
echo "3. Start training:"
echo "   cd $BUNDLE_DIR"
echo "   bash train.sh"
echo ""
