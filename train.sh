#!/bin/bash
# Official MQ-Det Training Script for Docker Environment

set -e

echo "🚀 Starting Official MQ-Det Training..."

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Verify query bank exists
if [ ! -f "MODEL/connectors_query_official.pth" ]; then
    echo "❌ Query bank not found! Run extract_queries.sh first."
    exit 1
fi

echo "✅ Query bank found, starting training..."

# Create output directory
mkdir -p OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS

# Run official MQ-Det training
echo "🎯 Starting official MQ-Det training with vision queries..."
python tools/train_net.py \
    --config-file configs/custom/mq-glip-official.yaml \
    OUTPUT_DIR 'OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS/' \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.MAX_EPOCH 20 \
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