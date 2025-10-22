#!/bin/bash
# Official MQ-Det Training Script for Connectors Dataset

set -e

echo "🚀 Starting Official MQ-Det Training for Connectors Dataset..."

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Verify query bank exists
if [ ! -f "MODEL/connectors_query_5000_sel_tiny.pth" ]; then
    echo "❌ Query bank not found! Run extract_queries.sh first."
    exit 1
fi

echo "✅ Query bank found, starting training..."

# Create output directory
mkdir -p OUTPUT/MQ-GLIP-TINY-CONNECTORS

# Run official MQ-Det modulated training
echo "🔍 Starting MQ-Det Training..."

# Train with modulated query (single GPU - no distributed training needed)
python3.9 tools/train_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --use-tensorboard \
    OUTPUT_DIR 'OUTPUT/MQ-GLIP-TINY-CONNECTORS/' \
    2>&1 | tee OUTPUT/training.log

# Check training results
if [ -d "OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS" ] && [ "$(ls -A OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS)" ]; then
    echo "✅ Training completed successfully!"
    
    # List generated models
    echo "📄 Generated models:"
    ls -lh OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS/*.pth 2>/dev/null || echo "No .pth files found"
    
    # Extract training metrics if available
    if [ -f "OUTPUT/training.log" ]; then
        echo "📊 Training summary:"
        grep -i "accuracy\|loss\|epoch" OUTPUT/training.log | tail -10 || echo "No metrics found in log"
    fi
else
    echo "❌ Training may have failed - check OUTPUT/training.log"
    exit 1
fi

echo "🎉 Official MQ-Det training completed!"