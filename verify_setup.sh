#!/bin/bash
# Configuration Verification Script for MQ-Det Connectors Setup

echo "🔍 Verifying MQ-Det Connectors Configuration..."

# Check dataset registration
echo "📁 Checking dataset registration in paths_catalog.py..."
if grep -q "connectors_grounding_train" maskrcnn_benchmark/config/paths_catalog.py; then
    echo "✅ connectors_grounding_train registered"
else
    echo "❌ connectors_grounding_train NOT registered"
fi

if grep -q "connectors_grounding_val" maskrcnn_benchmark/config/paths_catalog.py; then
    echo "✅ connectors_grounding_val registered"  
else
    echo "❌ connectors_grounding_val NOT registered"
fi

# Check factory registration
echo "🏭 Checking factory registration..."
if grep -q "connectors_grounding_train.*connectors_grounding_val" maskrcnn_benchmark/config/paths_catalog.py; then
    echo "✅ Factory registration correct"
else
    echo "❌ Factory registration missing or incorrect"
fi

# Check configuration files
echo "⚙️  Checking configuration files..."
if [ -f "configs/pretrain/mq-glip-t_connectors.yaml" ]; then
    echo "✅ Training config exists"
else
    echo "❌ Training config missing"
fi

if [ -f "configs/vision_query_5shot/connectors.yaml" ]; then
    echo "✅ Evaluation config exists"
else
    echo "❌ Evaluation config missing"
fi

# Check script files
echo "📜 Checking script files..."
if [ -f "extract_queries.sh" ] && [ -x "extract_queries.sh" ]; then
    echo "✅ extract_queries.sh exists and executable"
else
    echo "❌ extract_queries.sh missing or not executable"
fi

if [ -f "train.sh" ] && [ -x "train.sh" ]; then
    echo "✅ train.sh exists and executable"
else
    echo "❌ train.sh missing or not executable"
fi

if [ -f "evaluate.sh" ] && [ -x "evaluate.sh" ]; then
    echo "✅ evaluate.sh exists and executable"
else
    echo "❌ evaluate.sh missing or not executable"
fi

# Check dataset directory structure
echo "🗂️  Checking dataset structure..."
if [ -d "DATASET/connectors" ]; then
    echo "✅ DATASET/connectors directory exists"
    if [ -d "DATASET/connectors/images/train" ] && [ -d "DATASET/connectors/images/val" ]; then
        echo "✅ Image directories exist"
        echo "   Train images: $(find DATASET/connectors/images/train -name '*.jpg' -o -name '*.png' | wc -l)"
        echo "   Val images: $(find DATASET/connectors/images/val -name '*.jpg' -o -name '*.png' | wc -l)"
    else
        echo "❌ Image directories missing"
    fi
    
    if [ -f "DATASET/connectors/annotations/instances_train_connectors.json" ] && [ -f "DATASET/connectors/annotations/instances_val_connectors.json" ]; then
        echo "✅ Annotation files exist"
    else
        echo "❌ Annotation files missing"
    fi
else
    echo "❌ DATASET/connectors directory missing"
fi

# Check model directory
echo "📦 Checking model directory..."
if [ -d "MODEL" ]; then
    echo "✅ MODEL directory exists"
    if [ -f "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth" ]; then
        echo "✅ GLIP-T model exists"
    else
        echo "❌ GLIP-T model missing - download required"
    fi
else
    echo "❌ MODEL directory missing"
fi

echo ""
echo "🎯 Configuration Summary:"
echo "   This setup follows the official MQ-Det implementation for custom datasets"
echo "   Dataset: connectors (electrical connector detection)"
echo "   Model: MQ-GLIP-T with vision query extraction"
echo "   Method: Modulated pre-training + finetuning-free evaluation"
echo ""
echo "📋 Next Steps:"
echo "   1. Ensure DATASET/connectors is properly structured"
echo "   2. Download GLIP-T model if missing"
echo "   3. Run ./extract_queries.sh for vision query extraction"
echo "   4. Run ./train.sh for modulated training"
echo "   5. Run ./evaluate.sh for finetuning-free evaluation"