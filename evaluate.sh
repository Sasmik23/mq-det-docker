#!/bin/bash
# Official MQ-Det Evaluation Script for Connectors Dataset

set -e

echo "📊 Starting Official MQ-Det Evaluation for Connectors Dataset..."

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Check if trained model exists
TRAINED_MODEL="OUTPUT/MQ-GLIP-TINY-CONNECTORS/model_final.pth"
if [ ! -f "$TRAINED_MODEL" ]; then
    echo "❌ Trained model not found at $TRAINED_MODEL"
    echo "Please ensure training is completed first by running train.sh"
    exit 1
fi

# Check if evaluation query bank exists
EVAL_QUERY_BANK="MODEL/connectors_query_5_pool7_sel_tiny.pth"
if [ ! -f "$EVAL_QUERY_BANK" ]; then
    echo "❌ Evaluation query bank not found at $EVAL_QUERY_BANK"
    echo "Please ensure query extraction is completed first by running extract_queries.sh"
    exit 1
fi

echo "✅ Found trained model and query bank, starting evaluation..."

# Create evaluation output directory
mkdir -p OUTPUT/connectors_evaluation

# Run evaluation with vision queries
echo "🎯 Running finetuning-free evaluation with vision queries..."
python -m torch.distributed.launch --nproc_per_node=1 \
    tools/test_grounding_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --additional_model_config configs/vision_query_5shot/connectors.yaml \
    VISION_QUERY.QUERY_BANK_PATH MODEL/connectors_query_5_pool7_sel_tiny.pth \
    MODEL.WEIGHT OUTPUT/MQ-GLIP-TINY-CONNECTORS/model_final.pth \
    TEST.IMS_PER_BATCH 1 \
    OUTPUT_DIR OUTPUT/connectors_evaluation/ \
    2>&1 | tee OUTPUT/evaluation.log

echo "✅ Evaluation completed!"

# Display results
if [ -f "OUTPUT/evaluation.log" ]; then
    echo "📋 Evaluation Results:"
    grep -E "(AP|AR|mAP)" OUTPUT/evaluation.log | tail -10 || echo "No AP/AR metrics found in log"
fi

echo "📁 Check OUTPUT/connectors_evaluation/ for detailed results"