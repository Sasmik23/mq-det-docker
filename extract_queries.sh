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

# Create official configuration
echo "📝 Creating official MQ-Det configuration..."
cat > configs/custom/mq-glip-official.yaml << 'EOF'
MODEL:
  META_ARCHITECTURE: "GeneralizedVLRCNN_New"
  WEIGHT: "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
  RPN_ONLY: True
  RPN_ARCHITECTURE: "VLDYHEAD"

  BACKBONE:
    CONV_BODY: "SWINT-FPN-RETINANET"
    OUT_CHANNELS: 256
    FREEZE_CONV_BODY_AT: -1

  LANGUAGE_BACKBONE:
    FREEZE: False
    TOKENIZER_TYPE: "bert-base-uncased"
    MODEL_TYPE: "bert-base-uncased"
    MASK_SPECIAL: False

  DYHEAD:
    CHANNELS: 256
    NUM_CONVS: 6
    USE_GN: True
    USE_DYRELU: True
    USE_DFCONV: True
    USE_DYFUSE: True
    TOPK: 9
    SCORE_AGG: "MEAN"
    LOG_SCALE: 0.0

    FUSE_CONFIG:
      EARLY_FUSE_ON: True
      TYPE: "MHA-B"
      USE_CLASSIFICATION_LOSS: False
      USE_TOKEN_LOSS: False
      USE_CONTRASTIVE_ALIGN_LOSS: False
      USE_DOT_PRODUCT_TOKEN_LOSS: True
      USE_LAYER_SCALE: True
      CLAMP_MIN_FOR_UNDERFLOW: True
      CLAMP_MAX_FOR_OVERFLOW: True
      USE_VISION_QUERY_LOSS: True
      VISION_QUERY_LOSS_WEIGHT: 10

DATASETS:
  TRAIN: ("connectors_grounding_train",)
  TEST: ("connectors_grounding_val",)
  FEW_SHOT: 0

INPUT:
  MIN_SIZE_TRAIN: (800,)
  MAX_SIZE_TRAIN: 1333
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1333

SOLVER:
  OPTIMIZER: "ADAMW"
  BASE_LR: 0.0001
  WEIGHT_DECAY: 0.0001
  STEPS: (0.67, 0.89)
  MAX_EPOCH: 20
  IMS_PER_BATCH: 8
  WARMUP_ITERS: 1000
  USE_AMP: True
  CHECKPOINT_PERIOD: 99999999
  CHECKPOINT_PER_EPOCH: 2.0

VISION_QUERY:
  ENABLED: True
  QUERY_BANK_PATH: 'MODEL/connectors_query_official.pth'
  PURE_TEXT_RATE: 0.
  TEXT_DROPOUT: 0.4
  VISION_SCALE: 1.0
  NUM_QUERY_PER_CLASS: 5
  MAX_QUERY_NUMBER: 50

OUTPUT_DIR: "OUTPUT/MQ-GLIP-OFFICIAL-CONNECTORS/"
EOF

# Register dataset (assuming it's already set up)
echo "📋 Checking dataset registration..."
if ! grep -q "connectors_grounding_train" maskrcnn_benchmark/config/paths_catalog.py; then
    echo "⚠️ Dataset not registered. Please ensure your dataset is properly registered in paths_catalog.py"
fi

# Step 1: Extract vision queries using official method
echo "🧠 Extracting vision queries using official MQ-Det method..."
python3.9 tools/train_net.py \
    --config-file configs/custom/mq-glip-official.yaml \
    --extract_query \
    VISION_QUERY.QUERY_BANK_PATH "" \
    VISION_QUERY.QUERY_BANK_SAVE_PATH MODEL/connectors_query_official.pth \
    VISION_QUERY.MAX_QUERY_NUMBER 50

if [ -f "MODEL/connectors_query_official.pth" ]; then
    echo "✅ Vision query extraction successful!"
    
    # Check query bank
    python3.9 -c "
import torch
query_bank = torch.load('MODEL/connectors_query_official.pth', map_location='cpu')
if isinstance(query_bank, dict):
    print(f'📊 Query bank contains {len(query_bank.get(\"queries\", []))} queries')
    print(f'🏷️ Categories: {query_bank.get(\"categories\", \"Unknown\")}')
else:
    print(f'📊 Query bank size: {query_bank.shape if hasattr(query_bank, \"shape\") else \"Unknown\"}')
"
else
    echo "❌ Vision query extraction failed!"
    exit 1
fi

echo "✅ Official vision query extraction completed successfully!"