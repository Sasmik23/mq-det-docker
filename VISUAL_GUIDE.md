# 🎯 MQ-Det Deployment - Quick Visual Guide

## Choose Your Path

```
                        ┌─────────────────────────────┐
                        │   MQ-Det Deployment         │
                        │   Which environment?        │
                        └──────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐           ┌───────────────────────┐
    │   GCP / WITH INTERNET │           │  AIR-GAP / NO INTERNET│
    │   ✅ Docker           │           │  ✅ Kubernetes Pod    │
    │   ✅ Internet access  │           │  ✅ A100 GPU          │
    │   ✅ Simpler setup    │           │  ✅ Secure env        │
    └───────┬───────────────┘           └───────┬───────────────┘
            │                                   │
            │                                   │
            ▼                                   ▼
    ┌───────────────────┐           ┌─────────────────────────┐
    │  📁 Use: gcp/     │           │  📁 Use: airgap/        │
    └───────┬───────────┘           └───────┬─────────────────┘
            │                               │
            ▼                               ▼
```

---

## 🐳 GCP Path (Simple, with Internet)

```
📦 gcp/
│
├── 1️⃣  Setup
│   └── ./gcp_setup.sh
│       └── Builds Docker, downloads models (30 min)
│
├── 2️⃣  Extract Queries
│   └── ./extract_queries.sh
│       └── Creates query banks (30-60 min)
│
├── 3️⃣  Train
│   └── ./train.sh
│       └── Trains model (2-8 hours)
│
└── 4️⃣  Evaluate
    └── ./evaluate.sh
        └── Gets metrics (30-60 min)

✨ DONE! Model at OUTPUT/model_final.pth
```

---

## 🔒 Air-Gap Path (Secure, no Internet)

```
📦 airgap/
│
├── 1️⃣  Prepare (Windows with Internet)
│   ├── 📁 1-prepare/
│   └── prepare_offline_bundle.bat
│       └── Downloads everything (30-60 min)
│       └── Creates: mq-det-offline-bundle.tar.gz (5 GB)
│
├── 2️⃣  Transfer (Network/USB)
│   ├── 📁 2-transfer/
│   └── scp bundle.tar.gz to pod
│       └── Extract on pod (5-10 min)
│
├── 3️⃣  Setup (On Pod, One-Time)
│   ├── 📁 3-setup/
│   ├── ./install_on_pod.sh
│   │   └── Installs packages, compiles CUDA (15-30 min)
│   └── source setup_environment.sh
│       └── Loads environment variables
│
└── 4️⃣  Pipeline (On Pod, Training)
    ├── 📁 4-pipeline/
    └── ./run_full_pipeline.sh
        ├── Phase 1: Check environment (30s)
        ├── Phase 2: Extract queries (30-60 min)
        ├── Phase 3: Train model (2-8 hours)
        └── Phase 4: Evaluate (30-60 min)

✨ DONE! Model at OUTPUT/model_final.pth
```

---

## 📊 Side-by-Side Comparison

| Aspect | GCP | Air-Gap |
|--------|-----|---------|
| **Setup** | Docker build | Manual install |
| **Internet** | ✅ Required | ❌ Not needed |
| **Time to Start** | 30 min | 2-3 hours |
| **Complexity** | ⭐⭐ Simple | ⭐⭐⭐⭐ Advanced |
| **Security** | Standard | 🔒 High |
| **Best For** | Development, Testing | Production, Secure |
| **GPU** | Any NVIDIA | A100 optimized |
| **Updates** | Easy (docker pull) | Manual bundle |

---

## 🎯 Decision Tree

```
Do you have internet on training machine?
│
├── YES → Use GCP path (gcp/)
│         ├── Faster setup
│         ├── Easier updates
│         └── Good for development
│
└── NO  → Use Air-Gap path (airgap/)
          ├── One-time bundle prep
          ├── Secure deployment
          └── Production-ready
```

---

## 🚀 Quick Commands

### GCP
```bash
cd gcp
./gcp_setup.sh && ./extract_queries.sh && ./train.sh && ./evaluate.sh
```

### Air-Gap
```bash
# Windows (with internet)
cd airgap\1-prepare
prepare_offline_bundle.bat

# Pod (after transfer)
cd airgap/3-setup && ./install_on_pod.sh
source setup_environment.sh
cd ../4-pipeline && ./run_full_pipeline.sh
```

---

## 📚 Documentation Map

```
📖 README.md                          → Start here
│
├── 🐳 GCP Deployment
│   └── gcp/GCP_DEPLOYMENT.md        → Complete GCP guide
│
└── 🔒 Air-Gap Deployment
    └── airgap/AIRGAP_DEPLOYMENT.md  → Master air-gap guide
        ├── 1-prepare/PREPARE_BUNDLE.md    → How to create bundle
        ├── 2-transfer/TRANSFER_GUIDE.md   → How to transfer
        ├── 3-setup/SETUP_GUIDE.md         → How to setup pod
        └── 4-pipeline/PIPELINE_GUIDE.md   → How to train
```

---

## ✅ Success Checklist

### GCP
- [ ] Docker & Docker Compose installed
- [ ] NVIDIA GPU available
- [ ] Internet connected
- [ ] Run: `cd gcp && ./gcp_setup.sh`
- [ ] Dataset in `DATASET/connectors/`
- [ ] Run pipeline: `./extract_queries.sh && ./train.sh && ./evaluate.sh`
- [ ] Check results: `cat OUTPUT/evaluation.log`

### Air-Gap
- [ ] Windows PC with internet (for prep)
- [ ] Bundle created: `prepare_offline_bundle.bat`
- [ ] Bundle transferred to pod
- [ ] Run: `./install_on_pod.sh`
- [ ] Environment loaded: `source setup_environment.sh`
- [ ] Dataset in `DATASET/connectors/`
- [ ] Run pipeline: `./run_full_pipeline.sh`
- [ ] Check results: `cat OUTPUT/evaluation.log`

---

## 🎓 Tips

### GCP Users
- Start with small dataset to test
- Monitor with `docker-compose logs -f`
- GPU usage: `nvidia-smi` inside container

### Air-Gap Users
- Prepare bundle on fast internet
- Keep bundle for future re-deployments
- Test installation on one pod first
- Monitor with `tail -f OUTPUT/training.log`

---

## 🆘 Need Help?

| Issue | GCP | Air-Gap |
|-------|-----|---------|
| Docker fails | See `gcp/GCP_DEPLOYMENT.md` | N/A |
| CUDA errors | Check GPU drivers | Recompile: `TORCH_CUDA_ARCH_LIST="8.0"` |
| Network errors | Check internet | Enable offline mode |
| Out of memory | Reduce batch size | Reduce batch size |

Full troubleshooting in respective guides!

---

## 🎉 You're Ready!

1. **Choose your path** (GCP or Air-Gap)
2. **Follow the guide** in that directory
3. **Train your model**
4. **Get results**

**Happy training!** 🚀
