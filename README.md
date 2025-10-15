# MQ-Det: Custom Object Detection with Docker + GCP

A streamlined Docker-based pipeline for training and evaluating [MQ-Det (Multi-Query Detection)](https://arxiv.org/abs/2305.13962) on custom datasets, deployed on Google Cloud Platform.

> **Original Paper**: [MQ-Det: Multi-modal Queried Object Detection](https://arxiv.org/abs/2305.13962)  
> **Original Repository**: See [ORIGINAL_MQDET_README.md](./ORIGINAL_MQDET_README.md)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-queried-object-detection-in-the/few-shot-object-detection-on-odinw-13)](https://paperswithcode.com/sota/few-shot-object-detection-on-odinw-13?p=multi-modal-queried-object-detection-in-the)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-queried-object-detection-in-the/zero-shot-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0?p=multi-modal-queried-object-detection-in-the)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-queried-object-detection-in-the/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=multi-modal-queried-object-detection-in-the)

---

## 🎯 What This Repository Does

This repository extends the original MQ-Det paper implementation with:

- **🐳 Docker containerization** - Reproducible environment with all dependencies
- **☁️ GCP deployment** - Production-ready setup on Google Cloud Platform
- **📦 Custom dataset support** - Easy training on your own object detection data
- **�� Simplified workflow** - Three scripts: extract queries → train → evaluate
- **💾 Efficient storage** - Optimized checkpoint management for cloud deployment
- **⚡ Pragmatic versions** - Stable PyTorch 1.12.1 + CUDA 11.3 (vs paper's 2.0.1 + 11.7)

---

## 📋 Table of Contents

- [Version Differences](#-version-differences-from-original-paper)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Dataset Preparation](#-dataset-preparation)
- [Training Pipeline](#-training-pipeline)
- [Evaluation](#-evaluation)
- [Configuration](#️-configuration)
- [Cost Optimization](#-cost-optimization)
- [Troubleshooting](#-troubleshooting)
- [Architecture](#️-architecture)
- [Citation](#-citation)

---

## 🔄 Version Differences from Original Paper

This implementation uses **different versions** than the original paper for **stability and compatibility**:

| Component | Paper (Original) | This Repo | Reason for Change |
|-----------|------------------|-----------|-------------------|
| **PyTorch** | 2.0.1 | **1.12.1** | Stable compilation with maskrcnn-benchmark, better CUDA 11.3 support |
| **CUDA** | 11.7 | **11.3** | Broader GPU compatibility (T4, P100, V100), mature ecosystem |
| **Python** | 3.8-3.10 | **3.9** | Optimal for PyTorch 1.12.1, stable package ecosystem |
| **GCC** | 9+ | **8** | Required for CUDA 11.3 compilation, Ubuntu 20.04 default |
| **cuDNN** | 8.5+ | **8.0** | Matches CUDA 11.3 requirements |

### Why We Modified Versions

**1. PyTorch 2.0.1 Compilation Issues** ❌
- \`maskrcnn-benchmark\` fails to compile with PyTorch 2.0+ due to breaking changes in ATen headers (\`at::nullopt\` removed)
- Custom CUDA kernels incompatible with new PyTorch C++ API
- Vision transformers (Swin-T) have API changes

**2. CUDA 11.7 Availability Challenges** ❌
- Limited availability on GCP T4 GPUs (requires newer drivers ≥520)
- CUDA 11.3 is more mature, better tested, and widely supported

**3. Pragmatic Solution: PyTorch 1.12.1 + CUDA 11.3** ✅
- Battle-tested combination (millions of deployments)
- Works on older GPUs (Pascal, Volta, Turing, Ampere)
- \`maskrcnn-benchmark\` compiles cleanly with minor ATen patches
- Stable training, no gradient anomalies
- Extensive community support and bug fixes

**Result**: Successfully trained MQ-Det with **83% improvement** over pretrained baseline using these pragmatic versions!

---

## 🚀 Quick Start

### Prerequisites

- Google Cloud Platform account ([get \$300 free credits](https://cloud.google.com/free))
- Docker Desktop (for local development)
- Git

### 1. Clone Repository

\`\`\`bash
git clone https://github.com/Sasmik23/mq-det-docker.git
cd mq-det-docker
\`\`\`

### 2. Deploy to GCP

\`\`\`bash
# Create VM with T4 GPU
gcloud compute instances create mq-det-vm \\
  --zone=us-central1-a \\
  --machine-type=n1-standard-4 \\
  --accelerator=type=nvidia-tesla-t4,count=1 \\
  --image-family=pytorch-latest-gpu \\
  --image-project=deeplearning-platform-release \\
  --maintenance-policy=TERMINATE \\
  --boot-disk-size=100GB \\
  --metadata="install-nvidia-driver=True"

# SSH into VM
gcloud compute ssh mq-det-vm --zone=us-central1-a

# Clone repo on VM
git clone https://github.com/Sasmik23/mq-det-docker.git
cd mq-det-docker

# Build Docker image
docker compose build

# Start container
docker compose up -d

# Enter container
docker compose exec mq-det bash
\`\`\`

### 3. Prepare Your Dataset

Organize your dataset in COCO format:

\`\`\`
DATASET/
  your_dataset/
    annotations/
      train.json
      val.json
    images/
      train/
        img1.jpg
        img2.jpg
      val/
        img3.jpg
\`\`\`

Register your dataset in \`DATASET/your_dataset/__init__.py\`:

\`\`\`python
DATASET_ROOT = os.getenv("DATASET", "/workspace/DATASET")

YOURDATASET_TRAIN = {
    'img_dir': f'{DATASET_ROOT}/your_dataset/images/train',
    'ann_file': f'{DATASET_ROOT}/your_dataset/annotations/train.json',
    'dataset_name': 'your_dataset_grounding_train'
}

YOURDATASET_VAL = {
    'img_dir': f'{DATASET_ROOT}/your_dataset/images/val',
    'ann_file': f'{DATASET_ROOT}/your_dataset/annotations/val.json',
    'dataset_name': 'your_dataset_grounding_val'
}
\`\`\`

### 4. Extract Vision Queries

\`\`\`bash
bash extract_queries.sh
\`\`\`

### 5. Train Model

\`\`\`bash
bash train.sh
\`\`\`

### 6. Evaluate Model

\`\`\`bash
bash evaluate.sh
\`\`\`

---

## 📊 Results

### Connectors Dataset (8 train, 9 val images, 3 classes)

| Model | AP@50 | AP (0.50:0.95) | AR@100 | Training Time |
|-------|-------|----------------|--------|---------------|
| Pretrained (zero-shot) | 17.5% | 4.9% | 21.5% | - |
| **After 40 iterations** | **32.0%** | **13.8%** | **49.2%** | ~2 minutes |
| **Improvement** | **+83%** | **+180%** | **+128%** | - |

**Per-class breakdown**:
- Orange connector: 58.9% AP@50 ⭐
- Yellow connector: 27.8% AP@50
- White connector: 9.2% AP@50

**Training Details:**
- Architecture: RPN_ONLY + DYHEAD
- Batch size: 2 (optimized for T4 GPU 16GB)
- Image size: 640×1024
- Epochs: 10 (40 iterations)
- Training time: ~2 minutes
- GPU memory: ~7GB / 16GB

---

## 📊 Dataset Preparation

### Dataset Format

MQ-Det expects **COCO format** annotations:

\`\`\`json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "object_name"}
  ]
}
\`\`\`

### Converting Your Dataset

If you have a different format, use conversion tools:

- **YOLO → COCO**: [yolo2coco](https://github.com/RapidAI/YOLO2COCO)
- **Pascal VOC → COCO**: [voc2coco](https://github.com/yukkyo/voc2coco)
- **LabelMe → COCO**: [labelme2coco](https://github.com/fcakyon/labelme2coco)

---

## 🚀 Training Pipeline

### Step 1: Extract Vision Queries

\`\`\`bash
./extract_queries.sh
\`\`\`

**What it does:**
- Extracts visual features from training images
- Creates query banks for training and evaluation
- Generates vision query files

**Time**: ~30 seconds for 8 images

### Step 2: Train Model

\`\`\`bash
./train.sh
\`\`\`

**What it does:**
- Finetunes pretrained GLIP-Tiny on your dataset
- Uses vision queries for modulated training
- Saves checkpoints

**Time**: ~2-3 minutes for 10 epochs on 8 images (T4 GPU)

### Step 3: Evaluate

\`\`\`bash
./evaluate.sh
\`\`\`

**What it does:**
- Evaluates trained model on validation set
- Computes AP@50, AP@75, AR metrics
- Saves results to OUTPUT directory

**Time**: ~10 seconds for 9 validation images

---

## 📈 Evaluation

### Understanding Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **AP@50** | Average Precision at IoU=0.5 (main metric) | > 50% |
| **AP (0.50:0.95)** | AP averaged over IoU 0.5-0.95 | > 30% |
| **AR@100** | Average Recall with max 100 detections | > 60% |

**Good performance indicators:**
- ✅ AP@50 > 50% per class
- ✅ Balanced performance across classes
- ✅ Training improves over pretrained baseline

**Poor performance indicators:**
- ❌ AP@50 < 20% per class
- ❌ One class much worse than others (data imbalance)
- ❌ Training doesn't improve over baseline (overfitting)

---

## ⚙️ Configuration

### Main Config File

\`configs/pretrain/mq-glip-t_connectors.yaml\`

\`\`\`yaml
SOLVER:
  MAX_EPOCH: 10              # Number of training epochs
  IMS_PER_BATCH: 2          # Batch size (GPU memory)
  BASE_LR: 0.00001          # Learning rate
  CHECKPOINT_PER_EPOCH: 0.2  # Checkpoint frequency

INPUT:
  MIN_SIZE_TRAIN: 640       # Reduce if OOM
  MAX_SIZE_TRAIN: 1024      # Reduce if OOM
\`\`\`

---

## 💰 Cost Optimization

### GCP Billing

| Component | Hourly Cost | Daily Cost (8hrs) | Monthly (160hrs) |
|-----------|-------------|-------------------|------------------|
| T4 GPU | \$0.35 | \$2.80 | \$56 |
| n1-standard-4 CPU | \$0.19 | \$1.52 | \$30.40 |
| 100GB SSD | \$0.017 | \$0.14 | \$2.72 |
| **Total** | **\$0.56** | **\$4.46** | **\$89.12** |

### Cost-Saving Tips

1. **Stop VM when not in use**
2. **Use Preemptible instances** (up to 80% cheaper)
3. **Reduce checkpoint frequency**
4. **Delete unused checkpoints**

---

## 🐛 Troubleshooting

### GPU Out of Memory (OOM)

Reduce batch size and image size:

\`\`\`yaml
SOLVER:
  IMS_PER_BATCH: 1  # Down from 2

INPUT:
  MIN_SIZE_TRAIN: 512
  MAX_SIZE_TRAIN: 800
\`\`\`

### Dataset Not Found

Ensure dataset is registered in:
1. \`DATASET/your_dataset/__init__.py\`
2. \`maskrcnn_benchmark/config/paths_catalog.py\`

### Docker Container Not Running

\`\`\`bash
docker compose up -d
docker compose ps
docker compose logs -f
\`\`\`

---

## 🏗️ Architecture

MQ-Det uses **vision queries** to improve object detection:

\`\`\`
Input Image → Backbone (Swin-T) → RPN → DYHEAD → Detection
                                     ↑
                              Vision Query Bank
\`\`\`

**Key Components:**
- **RPN_ONLY**: Faster inference
- **DYHEAD**: Multi-head attention fusion
- **Vision Queries**: Extracted from dataset
- **Language Grounding**: Text-based detection

---

## 📚 Documentation

- [GCP_DEPLOYMENT.md](./GCP_DEPLOYMENT.md) - Detailed GCP setup guide
- [ORIGINAL_MQDET_README.md](./ORIGINAL_MQDET_README.md) - Original paper docs
- [CUSTOMIZED_PRETRAIN.md](./CUSTOMIZED_PRETRAIN.md) - Pretraining customization
- [DATA.md](./DATA.md) - Dataset preparation guide
- [DEBUG.md](./DEBUG.md) - Debugging tips

---

## 📖 Citation

If you use this code, please cite the original MQ-Det paper:

\`\`\`bibtex
@article{xu2024multi,
  title={Multi-modal queried object detection in the wild},
  author={Xu, Yifan and Zhang, Mengdan and Fu, Chaoyou and Chen, Peixian and Yang, Xiaoshan and Li, Ke and Xu, Changsheng},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
\`\`\`

**This Docker Implementation:**
\`\`\`
MQ-Det Docker Pipeline by @Sasmik23
https://github.com/Sasmik23/mq-det-docker
\`\`\`

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgments

- Original MQ-Det paper authors
- GLIP and GroundingDINO teams
- maskrcnn-benchmark contributors
- PyTorch and CUDA teams

---

## 🆘 Support

For issues and questions:
1. Check [GCP_DEPLOYMENT.md](./GCP_DEPLOYMENT.md)
2. Review [DEBUG.md](./DEBUG.md)
3. Open a GitHub issue

---

**Last Updated**: October 2025  
**Status**: ✅ Production Ready  
**Tested On**: GCP T4 GPU, PyTorch 1.12.1, CUDA 11.3
