# Setup Environment 
export TORCH_HOME=/home/2300488/mik/mq-det-offline-bundle/torch_cache 
export TRANSFORMERS_CACHE=/home/2300488/mik/mq-det-offline-bundle/hf_cache 
export TIMM_CACHE=/home/2300488/mik/mq-det-offline-bundle/timm_cache 
export HF_HOME=/home/2300488/mik/mq-det-offline-bundle/hf_cache 
export PYTHONPATH=/home/2300488/mik/mq-det-offline-bundle:$PYTHONPATH 
export MPLCONFIGDIR=/home/2300488/mik/mq-det-offline-bundle/.matplotlib 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TORCH_CUDA_ARCH_LIST="8.0" 
echo "✅ MQ-Det environment loaded" 
