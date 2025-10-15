#!/bin/bash
# Official MQ-Det Vision Query Extraction for Connectors Dataset

set -e

echo "🧠 Starting Official MQ-Det Vision Query Extraction for Connectors Dataset..."

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Check CUDA availability
python3.9 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "🔍 Extracting vision queries for connectors dataset..."

# Extract vision queries for training (5000 queries per class for query bank)
python3.9 tools/train_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --extract_query \
    VISION_QUERY.QUERY_BANK_PATH "" \
    VISION_QUERY.QUERY_BANK_SAVE_PATH MODEL/connectors_query_5000_sel_tiny.pth \
    VISION_QUERY.MAX_QUERY_NUMBER 5000

echo "✅ Vision query bank created: MODEL/connectors_query_5000_sel_tiny.pth"

# Extract vision queries for evaluation (5 queries per class for evaluation)
python3.9 tools/extract_vision_query.py \
    --config_file configs/pretrain/mq-glip-t_connectors.yaml \
    --dataset connectors \
    --num_vision_queries 5 \
    --add_name tiny

echo "✅ Evaluation vision queries created: MODEL/connectors_query_5_pool7_sel_tiny.pth"
echo "🎯 Vision query extraction completed successfully!"

# Test maskrcnn-benchmark
echo "🔧 Testing maskrcnn-benchmark..."
python3.9 -c "
try:
    import maskrcnn_benchmark
    print('✅ maskrcnn-benchmark imported successfully')
    from maskrcnn_benchmark.layers import nms
    print('✅ CUDA extensions available')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "✅ ✅ ✅ VISION QUERY EXTRACTION COMPLETE! ✅ ✅ ✅"
echo ""
echo "📊 Generated files:"
echo "  - MODEL/connectors_query_5000_sel_tiny.pth (training query bank)"
echo "  - MODEL/connectors_query_5_pool7_sel_tiny.pth (evaluation queries)"
echo ""
echo "🎯 Next steps:"
echo "  1. Run training: ./train.sh"
echo "  2. Run evaluation: ./evaluate.sh"
echo ""
