# MQ-Det Air-Gapped Kubernetes Pod Deployment Guide

## Overview
This guide provides a consolidated, step-by-step workflow for deploying and running MQ-Det on an air-gapped Kubernetes pod, including how to add and configure new datasets.

---

## 🚀 Steps 1-4: Quick Deployment

### 1. Extract the Bundle
- Copy `mq-det-offline-bundle.zip` to your VM.
- SSH into your pod and extract:
  ```bash
  unzip mq-det-offline-bundle.zip
  cd mq-det-offline-bundle
  ```

### 2. Run Installation
- Fix line endings and run the installer:
  ```bash
  find . -name "*.sh" -type f -exec sed -i 's/\r$//' {} \;
  chmod +x airgap/3-setup/install_on_pod.sh
  ./airgap/3-setup/install_on_pod.sh
  ```

### 3. Set Up Environment
- Create and source the environment script:
  ```bash
  cd ~
  cat > setup_mqdet.sh << 'EOF'
  #!/bin/bash
  export TORCH_HOME=~/mq-det-offline-bundle/MODEL
  export TRANSFORMERS_CACHE=~/mq-det-offline-bundle/hf_cache
  export TIMM_CACHE=~/mq-det-offline-bundle/timm_cache
  export HF_HOME=~/mq-det-offline-bundle/hf_cache
  export PYTHONPATH=~/mq-det-offline-bundle:$PYTHONPATH
  export MPLCONFIGDIR=~/mq-det-offline-bundle/.matplotlib
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TORCH_CUDA_ARCH_LIST="8.0"
  export CUDA_VISIBLE_DEVICES=0
  echo "✅ MQ-Det environment loaded"
  EOF
  chmod +x setup_mqdet.sh
  source ~/setup_mqdet.sh
  ```

### 4. Run the Pipeline
- Copy the pipeline script for your dataset (e.g., fd4_ini):
  ```bash
  cp airgap/4-pipeline/run_full_pipeline_fd4_ini.sh .
  chmod +x run_full_pipeline_fd4_ini.sh
  ./run_full_pipeline_fd4_ini.sh
  ```

---

## 🆕 Adding a New Dataset

1. **Create your dataset folder:**
   - Place images in `DATASET/<your_dataset>/images/train/` and `images/val/`.
   - Place COCO-format annotation files in `DATASET/<your_dataset>/annotations/`.

2. **Register the dataset:**
   - Edit `maskrcnn_benchmark/config/paths_catalog.py` to add entries for your dataset (see fd4_ini as example).

3. **Create config files:**
   - Copy and adapt a config in `configs/pretrain/` (e.g., `mq-glip-t_fd4_ini.yaml`).
   - Update:
     - `DATASETS.TRAIN` and `DATASETS.TEST`
     - `VISION_QUERY.QUERY_BANK_PATH`
     - `OUTPUT_DIR`
     - `SOLVER.IMS_PER_BATCH` (reduce if CUDA OOM)
   - Optionally, add a 5-shot config in `configs/vision_query_5shot/`.

4. **Add a pipeline script:**
   - Copy and adapt `airgap/4-pipeline/run_full_pipeline_fd4_ini.sh` for your dataset.
   - Update dataset/config paths as needed.

5. **Update the bundle:**
   - Add your new dataset, configs, and pipeline script before zipping.

---

## ⚙️ Key Parameters to Modify
- `SOLVER.IMS_PER_BATCH`: Lower if you get CUDA OOM errors.
- `OUTPUT_DIR`: Set a unique output directory for each dataset.
- `VISION_QUERY.QUERY_BANK_PATH`: Path for vision query bank file.
- `DATASETS.TRAIN`/`TEST`: Dataset registration names.

---

## 🛠️ Troubleshooting
- **CUDA OOM:** Lower `IMS_PER_BATCH` in config.
- **Annotation errors:** Ensure `file_name` in annotation JSON matches image filenames (no extra subfolders).
- **Line endings:** Always run the `sed` command after transfer from Windows.
- **Import errors:** Re-run the install script and check environment variables.

---

## 📞 Quick Reference
- Load environment: `source ~/setup_mqdet.sh`
- Run pipeline: `./run_full_pipeline_<your_dataset>.sh`
- Monitor: `tail -f OUTPUT/training_<your_dataset>.log`
- Check results: `cat OUTPUT/<your_dataset>_evaluation/result.txt`

---

**For more details, see the full COMPLETE_VM_SETUP_GUIDE.md in the project root.**
