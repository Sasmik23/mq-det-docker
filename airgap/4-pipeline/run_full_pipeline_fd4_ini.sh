#!/bin/bash
# Complete MQ-Det Pipeline for HMC Pod - FD4_INI Dataset
# Runs: Vision Query Extraction → Training → Evaluation

set -e

echo "========================================="
echo "🚀 MQ-Det Complete Pipeline"
echo "   HMC Air-Gapped Pod Edition"
echo "   Dataset: FD4_INI"
echo "========================================="
echo ""

# Navigate to bundle directory
cd /home/2300488/mik/mq-det-offline-bundle

# Source environment
source /home/2300488/mik/setup_mqdet.sh

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Configuration
DATASET_NAME="fd4_ini"
CONFIG_FILE="configs/pretrain/mq-glip-t_fd4_ini.yaml"
MODEL_NAME="MQ-GLIP-TINY-FD4-INI"
OUTPUT_BASE="OUTPUT"

echo "📋 Pipeline Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_BASE/$MODEL_NAME"
echo ""

# ============================================================================
# PHASE 1: ENVIRONMENT CHECK
# ============================================================================
echo "========================================="
echo "Phase 1: Environment Verification"
echo "========================================="
echo ""

# Check CUDA
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# Check maskrcnn_benchmark
python -c "
from maskrcnn_benchmark import _C
print('✅ maskrcnn_benchmark CUDA extensions: OK')
"

# Check GLIP model
if [ -f "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth" ]; then
    MODEL_SIZE=$(du -h MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth | cut -f1)
    echo "✅ GLIP-T model: $MODEL_SIZE"
else
    echo "❌ GLIP-T model not found!"
    exit 1
fi

# Check dataset
if [ -d "DATASET/$DATASET_NAME" ]; then
    echo "✅ Dataset found: DATASET/$DATASET_NAME"
    
    # Check for required annotation files
    if [ -f "DATASET/$DATASET_NAME/annotations/instances_train_fd4.json" ]; then
        echo "✅ Training annotations found"
    else
        echo "❌ Training annotations not found: DATASET/$DATASET_NAME/annotations/instances_train_fd4.json"
        exit 1
    fi
    
    if [ -f "DATASET/$DATASET_NAME/annotations/instances_val_fd4.json" ]; then
        echo "✅ Validation annotations found"
    else
        echo "❌ Validation annotations not found: DATASET/$DATASET_NAME/annotations/instances_val_fd4.json"
        exit 1
    fi
    
    # Check for image directories
    if [ -d "DATASET/$DATASET_NAME/images/train" ]; then
        TRAIN_IMAGES=$(ls -1 DATASET/$DATASET_NAME/images/train | wc -l)
        echo "✅ Training images: $TRAIN_IMAGES files"
    else
        echo "❌ Training images directory not found"
        exit 1
    fi
    
    if [ -d "DATASET/$DATASET_NAME/images/val" ]; then
        VAL_IMAGES=$(ls -1 DATASET/$DATASET_NAME/images/val | wc -l)
        echo "✅ Validation images: $VAL_IMAGES files"
    else
        echo "❌ Validation images directory not found"
        exit 1
    fi
else
    echo "❌ Dataset not found at DATASET/$DATASET_NAME"
    echo "Please ensure your dataset is in COCO format at that location"
    exit 1
fi

echo ""
echo "✅ Environment check passed!"
echo ""
sleep 2

# ============================================================================
# PHASE 2: VISION QUERY EXTRACTION
# ============================================================================
echo "========================================="
echo "Phase 2: Vision Query Extraction"
echo "========================================="
echo ""

# Check if query banks already exist
if [ -f "MODEL/${DATASET_NAME}_query_5000_sel_tiny.pth" ] && [ -f "MODEL/${DATASET_NAME}_query_5_pool7_sel_tiny.pth" ]; then
    echo "⚠️  Query banks already exist:"
    echo "  - MODEL/${DATASET_NAME}_query_5000_sel_tiny.pth"
    echo "  - MODEL/${DATASET_NAME}_query_5_pool7_sel_tiny.pth"
    read -p "Skip extraction? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Skipping query extraction"
        goto_training=true
    else
        goto_training=false
    fi
else
    goto_training=false
fi

if [ "$goto_training" != "true" ]; then
    echo "🧠 Extracting vision queries (5000 per class for training)..."
    python tools/train_net.py \
        --config-file $CONFIG_FILE \
        --extract_query \
        VISION_QUERY.QUERY_BANK_PATH "" \
        VISION_QUERY.QUERY_BANK_SAVE_PATH MODEL/${DATASET_NAME}_query_5000_sel_tiny.pth \
        VISION_QUERY.MAX_QUERY_NUMBER 5000
    
    echo ""
    echo "✅ Training query bank created!"
    echo ""
    
    echo "🔍 Extracting evaluation queries (5 per class)..."
    python tools/extract_vision_query.py \
        --config_file $CONFIG_FILE \
        --dataset $DATASET_NAME \
        --num_vision_queries 5 \
        --add_name tiny
    
    echo ""
    echo "✅ Evaluation query bank created!"
fi

echo ""
echo "📊 Query Bank Summary:"
ls -lh MODEL/${DATASET_NAME}_query_*.pth
echo ""
sleep 2

# ============================================================================
# PHASE 3: TRAINING
# ============================================================================
echo "========================================="
echo "Phase 3: Training"
echo "========================================="
echo ""

# Check if model already trained
if [ -f "$OUTPUT_BASE/$MODEL_NAME/model_final.pth" ]; then
    echo "⚠️  Trained model already exists: $OUTPUT_BASE/$MODEL_NAME/model_final.pth"
    read -p "Skip training? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Skipping training"
        goto_evaluation=true
    else
        goto_evaluation=false
        echo "🗑️  Backing up existing output..."
        mv "$OUTPUT_BASE/$MODEL_NAME" "$OUTPUT_BASE/${MODEL_NAME}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
else
    goto_evaluation=false
fi

if [ "$goto_evaluation" != "true" ]; then
    # Create output directory
    mkdir -p $OUTPUT_BASE/$MODEL_NAME
    
    echo "🚂 Starting MQ-Det training..."
    echo "This may take several hours depending on dataset size..."
    echo ""
    
    # Run training
    python tools/train_net.py \
        --config-file $CONFIG_FILE \
        --use-tensorboard \
        OUTPUT_DIR "$OUTPUT_BASE/$MODEL_NAME/" \
        2>&1 | tee $OUTPUT_BASE/training_fd4_ini.log
    
    echo ""
    echo "✅ Training completed!"
    echo ""
    
    # Show training results
    echo "📊 Training Results:"
    ls -lh $OUTPUT_BASE/$MODEL_NAME/*.pth 2>/dev/null || echo "No model files found"
    
    if [ -f "$OUTPUT_BASE/training_fd4_ini.log" ]; then
        echo ""
        echo "📈 Final training metrics:"
        grep -E "accuracy|loss|eta|iter" $OUTPUT_BASE/training_fd4_ini.log | tail -20 || echo "No metrics found"
    fi
fi

echo ""
sleep 2

# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================
echo "========================================="
echo "Phase 4: Evaluation"
echo "========================================="
echo ""

# Verify required files
TRAINED_MODEL="$OUTPUT_BASE/$MODEL_NAME/model_final.pth"
EVAL_QUERY_BANK="MODEL/${DATASET_NAME}_query_5_pool7_sel_tiny.pth"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "❌ Trained model not found: $TRAINED_MODEL"
    exit 1
fi

if [ ! -f "$EVAL_QUERY_BANK" ]; then
    echo "❌ Evaluation query bank not found: $EVAL_QUERY_BANK"
    exit 1
fi

echo "✅ Found trained model and query bank"
echo ""

# Create evaluation output directory
EVAL_OUTPUT="$OUTPUT_BASE/${DATASET_NAME}_evaluation"
mkdir -p $EVAL_OUTPUT

echo "🎯 Running evaluation with MQ-Det..."
echo ""

# Run evaluation
python tools/test_grounding_net.py \
    --config-file $CONFIG_FILE \
    --additional_model_config configs/vision_query_5shot/${DATASET_NAME}.yaml \
    VISION_QUERY.QUERY_BANK_PATH $EVAL_QUERY_BANK \
    MODEL.WEIGHT $TRAINED_MODEL \
    TEST.IMS_PER_BATCH 1 \
    OUTPUT_DIR $EVAL_OUTPUT/ \
    2>&1 | tee $OUTPUT_BASE/evaluation_fd4_ini.log

echo ""
echo "✅ Evaluation completed!"
echo ""

# Display results
if [ -f "$OUTPUT_BASE/evaluation_fd4_ini.log" ]; then
    echo "========================================="
    echo "📊 Evaluation Results"
    echo "========================================="
    echo ""
    grep -E "AP|mAP|Recall|Precision" $OUTPUT_BASE/evaluation_fd4_ini.log || \
        tail -50 $OUTPUT_BASE/evaluation_fd4_ini.log
fi

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================
echo ""
echo "========================================="
echo "🎉 MQ-Det Pipeline Complete!"
echo "========================================="
echo ""
echo "📁 Output Files:"
echo "  Query Banks:"
echo "    - MODEL/${DATASET_NAME}_query_5000_sel_tiny.pth"
echo "    - MODEL/${DATASET_NAME}_query_5_pool7_sel_tiny.pth"
echo ""
echo "  Trained Model:"
echo "    - $TRAINED_MODEL"
echo ""
echo "  Logs:"
echo "    - $OUTPUT_BASE/training_fd4_ini.log"
echo "    - $OUTPUT_BASE/evaluation_fd4_ini.log"
echo ""
echo "  Evaluation Results:"
echo "    - $EVAL_OUTPUT/"
echo ""
echo "========================================="
echo ""

# Summary
echo "✅ Phase 1: Environment Check - PASSED"
echo "✅ Phase 2: Vision Query Extraction - COMPLETED"
echo "✅ Phase 3: Training - COMPLETED"
echo "✅ Phase 4: Evaluation - COMPLETED"
echo ""
echo "🎯 All phases completed successfully!"
