# 🚀 Running Full MQ-Det Pipeline on HMC Pod

## Prerequisites Checklist

Before running the pipeline, ensure you have:

✅ **Installation complete** (from previous steps)
✅ **Dataset prepared** in COCO format
✅ **GLIP-T model weights** (~3.5GB)
✅ **Environment setup script** (`setup_mqdet.sh`)

---

## 📁 Dataset Structure Required

Your dataset should be at: `/home/2300488/mik/mq-det-offline-bundle/DATASET/connectors/`

Expected structure:
```
DATASET/
└── connectors/
    ├── annotations/
    │   ├── instances_train_connectors.json
    │   └── instances_val_connectors.json
    └── images/
        ├── train/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        └── val/
            ├── 000001.jpg
            └── ...
```

---

## ⚡ Quick Start

### Transfer Pipeline Script

Transfer `run_full_pipeline_pod.sh` to your pod:

```bash
# On pod
cd /home/2300488/mik/mq-det-offline-bundle
# Upload run_full_pipeline_pod.sh here
chmod +x run_full_pipeline_pod.sh
```

### Run Complete Pipeline

```bash
# Source environment
source /home/2300488/mik/setup_mqdet.sh

# Run complete pipeline (extraction → training → evaluation)
./run_full_pipeline_pod.sh
```

**Time estimate:**
- Phase 1 (Check): ~30 seconds
- Phase 2 (Query Extraction): ~30-60 minutes
- Phase 3 (Training): ~2-8 hours (depends on dataset size)
- Phase 4 (Evaluation): ~30-60 minutes

---

## 🔧 Run Individual Phases

If you want to run phases separately:

### Phase 2: Vision Query Extraction Only

```bash
source /home/2300488/mik/setup_mqdet.sh
cd /home/2300488/mik/mq-det-offline-bundle

# Training query bank (5000 queries per class)
python tools/train_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --extract_query \
    VISION_QUERY.QUERY_BANK_PATH "" \
    VISION_QUERY.QUERY_BANK_SAVE_PATH MODEL/connectors_query_5000_sel_tiny.pth \
    VISION_QUERY.MAX_QUERY_NUMBER 5000

# Evaluation query bank (5 queries per class)
python tools/extract_vision_query.py \
    --config_file configs/pretrain/mq-glip-t_connectors.yaml \
    --dataset connectors \
    --num_vision_queries 5 \
    --add_name tiny
```

### Phase 3: Training Only

```bash
source /home/2300488/mik/setup_mqdet.sh
cd /home/2300488/mik/mq-det-offline-bundle

# Ensure query bank exists
ls -lh MODEL/connectors_query_5000_sel_tiny.pth

# Train
mkdir -p OUTPUT/MQ-GLIP-TINY-CONNECTORS
python tools/train_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --use-tensorboard \
    OUTPUT_DIR 'OUTPUT/MQ-GLIP-TINY-CONNECTORS/' \
    2>&1 | tee OUTPUT/training.log
```

### Phase 4: Evaluation Only

```bash
source /home/2300488/mik/setup_mqdet.sh
cd /home/2300488/mik/mq-det-offline-bundle

# Ensure trained model and eval query bank exist
ls -lh OUTPUT/MQ-GLIP-TINY-CONNECTORS/model_final.pth
ls -lh MODEL/connectors_query_5_pool7_sel_tiny.pth

# Evaluate
mkdir -p OUTPUT/connectors_evaluation
python tools/test_grounding_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    --additional_model_config configs/vision_query_5shot/connectors.yaml \
    VISION_QUERY.QUERY_BANK_PATH MODEL/connectors_query_5_pool7_sel_tiny.pth \
    MODEL.WEIGHT OUTPUT/MQ-GLIP-TINY-CONNECTORS/model_final.pth \
    TEST.IMS_PER_BATCH 1 \
    OUTPUT_DIR OUTPUT/connectors_evaluation/ \
    2>&1 | tee OUTPUT/evaluation.log
```

---

## 📊 Monitoring Training

While training is running:

```bash
# Monitor training log (in another terminal/session)
tail -f OUTPUT/training.log

# Check GPU utilization
nvidia-smi

# Check output directory
ls -lh OUTPUT/MQ-GLIP-TINY-CONNECTORS/
```

---

## 🎯 Expected Outputs

After successful pipeline run:

```
OUTPUT/
├── MQ-GLIP-TINY-CONNECTORS/
│   ├── model_0001000.pth
│   ├── model_0002000.pth
│   ├── ...
│   ├── model_final.pth        ← Your trained model!
│   └── last_checkpoint
├── connectors_evaluation/
│   ├── bbox.json
│   ├── predictions.pth
│   └── result.txt              ← Evaluation metrics
├── training.log                ← Training details
└── evaluation.log              ← Evaluation metrics

MODEL/
├── glip_tiny_model_o365_goldg_cc_sbu.pth
├── connectors_query_5000_sel_tiny.pth     ← Training query bank
└── connectors_query_5_pool7_sel_tiny.pth  ← Eval query bank
```

---

## ⚠️ Troubleshooting

### Dataset Not Found

**Error:** `Dataset not found at DATASET/connectors`

**Fix:**
```bash
# Create dataset directory
mkdir -p DATASET/connectors/annotations
mkdir -p DATASET/connectors/images/train
mkdir -p DATASET/connectors/images/val

# Copy your dataset there
# Ensure COCO format with instances_train_connectors.json and instances_val_connectors.json
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Fix:**
```bash
# Reduce batch size in config
# Edit configs/pretrain/mq-glip-t_connectors.yaml
# Change:
#   SOLVER.IMS_PER_BATCH: 16  → 8 or 4
#   TEST.IMS_PER_BATCH: 8     → 4 or 1
```

### Training Interrupted

If training gets interrupted, it will resume from last checkpoint:

```bash
# Just run training again - it auto-resumes
python tools/train_net.py \
    --config-file configs/pretrain/mq-glip-t_connectors.yaml \
    OUTPUT_DIR 'OUTPUT/MQ-GLIP-TINY-CONNECTORS/'
```

### Query Extraction Fails

**Error:** Issues during query extraction

**Fix:**
```bash
# Ensure GLIP model loaded correctly
python -c "
import torch
model = torch.load('MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth')
print('Model keys:', len(model))
"

# Check dataset annotations are valid COCO format
python -c "
import json
with open('DATASET/connectors/annotations/instances_train_connectors.json') as f:
    data = json.load(f)
    print('Images:', len(data['images']))
    print('Annotations:', len(data['annotations']))
    print('Categories:', len(data['categories']))
"
```

---

## 🚀 After Pipeline Completes

1. **Check evaluation metrics:**
   ```bash
   cat OUTPUT/evaluation.log | grep -E "AP|mAP"
   ```

2. **Use trained model for inference:**
   ```bash
   python tools/test_grounding_net.py \
       --config-file configs/pretrain/mq-glip-t_connectors.yaml \
       MODEL.WEIGHT OUTPUT/MQ-GLIP-TINY-CONNECTORS/model_final.pth \
       # ... add your test images
   ```

3. **Backup your results:**
   ```bash
   tar -czf mqdet_results_$(date +%Y%m%d).tar.gz \
       OUTPUT/ \
       MODEL/connectors_query*.pth
   ```

---

## 📞 Questions?

- Training too slow? Check GPU utilization with `nvidia-smi`
- Need different dataset? Modify dataset name in script
- Want to tune hyperparameters? Edit YAML configs

Happy training! 🎉
