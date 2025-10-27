# MQ-Det: Multi-Query Detection Pipeline

Complete implementation of MQ-Det with three deployment paths:

- **Original**: Research paper implementation
- **GCP**: Docker deployment with internet access
- **Air-Gapped**: Kubernetes pod deployment for secure environments

**Validated**: 62.6% AP achieved on custom connectors dataset (111 images, A100 GPU)

---

## 📁 Repository Structure

```
mq-det-docker/
├── README.md                          # This file
├── ORIGINAL_MQDET_README.md          # Original MQ-Det documentation
│
├── gcp/                               # GCP Docker deployment (original)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── gcp_setup.sh
│   ├── init.sh
│   ├── extract_queries.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── GCP_DEPLOYMENT.md
│
├── airgap/                            # Air-gapped deployment (HMC pod)
│   ├── 1-prepare/
│   │   ├── prepare_offline_bundle.bat      # Run on Windows with internet
│   │   └── PREPARE_BUNDLE.md               # Bundle preparation guide
│   │
│   ├── 3-setup/
│   │   ├── install_on_pod.sh               # Installation script
│   │   ├── setup_environment.sh            # Environment setup
│   │   └── SETUP_GUIDE.md                  # Setup instructions
│   │
│   ├── 4-pipeline/
│   │   ├── run_full_pipeline.sh            # Complete pipeline
│   │   ├── PIPELINE_GUIDE.md               # Usage guide
│   │   └── fix_coco_eval_error.sh          # Optional evaluation fix
│   │
│   └── AIRGAP_DEPLOYMENT.md                # Complete air-gap workflow
│
├── configs/                           # Model configurations
├── tools/                             # Training/evaluation tools
├── maskrcnn_benchmark/                # Core framework
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Quick Start

### Option 1: GCP Deployment (with Internet)

```bash
# 1. Setup
./gcp/gcp_setup.sh

# 2. Extract queries
./gcp/extract_queries.sh

# 3. Train
./gcp/train.sh

# 4. Evaluate
./gcp/evaluate.sh
```

See: [gcp/GCP_DEPLOYMENT.md](gcp/GCP_DEPLOYMENT.md)

---

### Option 2: Air-Gapped Deployment (HMC Pod)

#### Step 1: Prepare Bundle (on machine with internet)

```bat
cd airgap\1-prepare
prepare_offline_bundle.bat
```

#### Step 2: Transfer to Pod

```bash
# Copy mq-det-offline-bundle.zip to pod
# See airgap/AIRGAP_DEPLOYMENT.md
```

#### Step 3: Setup on Pod

```bash
cd /home/2300488/mik/mq-det-offline-bundle
chmod +x airgap/3-setup/install_on_pod.sh
./airgap/3-setup/install_on_pod.sh
```

#### Step 4: Run Pipeline

```bash
source ~/setup_mqdet.sh
./run_full_pipeline_<your_dataset>.sh
```

See: [airgap/AIRGAP_DEPLOYMENT.md](airgap/AIRGAP_DEPLOYMENT.md)

---

## 📊 Workflows

### GCP Workflow

```
Internet → Docker → Extract Queries → Train → Evaluate
```

### Air-Gap Workflow

```
Prepare Bundle → Transfer → Setup Pod → Extract Queries → Train → Evaluate
```

---

## 🎯 Key Features

- **GCP Deployment**: Full Docker-based pipeline with internet access
- **Air-Gap Deployment**: Complete offline installation for secure environments
- **3-Stage Pipeline**:
  1. Vision Query Extraction (5000 training + 5 eval queries)
  2. Model Training with modulated query bank
  3. Evaluation with mAP metrics

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `gcp/GCP_DEPLOYMENT.md` | GCP Docker deployment guide |
| `airgap/AIRGAP_DEPLOYMENT.md` | Complete air-gap workflow |
| `airgap/1-prepare/PREPARE_BUNDLE.md` | Bundle preparation steps |
| `airgap/3-setup/SETUP_GUIDE.md` | Pod installation guide |
| `airgap/4-pipeline/PIPELINE_GUIDE.md` | Pipeline usage guide |
| `ORIGINAL_MQDET_README.md` | Original MQ-Det docs |

---

## 🔧 Requirements

### GCP

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- Internet connection

### Air-Gap (HMC Pod)

- Kubernetes pod with NVIDIA GPU (A100)
- PyTorch 1.13.1 + CUDA 11.7
- No internet required after bundle transfer

---

## 📝 License

See [LICENSE](LICENSE)

---

## 🤝 Contributing

This repository contains:

- Original MQ-Det framework
- GCP deployment scripts
- HMC air-gapped deployment solution

For questions about the core MQ-Det algorithm, see [ORIGINAL_MQDET_README.md](ORIGINAL_MQDET_README.md).
