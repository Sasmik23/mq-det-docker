# Air-Gapped Deployment Guide 
 
Complete workflow for deploying MQ-Det in air-gapped HMC pod environment. 
 
## Workflow 
 
1. **Prepare Bundle** (on Windows with internet): Run `1-prepare/prepare_offline_bundle.bat` 
2. **Transfer**: Copy bundle to pod (see `2-transfer/TRANSFER_GUIDE.md`) 
3. **Setup**: Run `3-setup/install_on_pod.sh` 
4. **Run Pipeline**: Execute `4-pipeline/run_full_pipeline.sh` 
 
See subdirectories for detailed guides. 
