@echo off
REM prepare_offline_bundle.bat
REM Windows CMD script to prepare MQ-Det for air-gapped deployment
REM Run this on a Windows machine WITH internet access

setlocal enabledelayedexpansion

echo ==================================================
echo MQ-Det Air-Gapped Deployment Bundle Preparation
echo ==================================================
echo.

REM Configuration
set BUNDLE_NAME=mq-det-offline-bundle
set OUTPUT_DIR=%BUNDLE_NAME%
set PYTHON_VERSION=38
set CUDA_VERSION=cu118

echo Creating bundle directory: %OUTPUT_DIR%
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM ============================================
REM Step 1: Download Pretrained Model Weights
REM ============================================
echo.
echo Step 1/6: Downloading pretrained model weights...

if not exist "%OUTPUT_DIR%\MODEL" mkdir "%OUTPUT_DIR%\MODEL"

if not exist "%OUTPUT_DIR%\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth" (
    echo   -^> Downloading GLIP-T ^(Tiny^) model ^(~260MB^)...
    curl -L -o "%OUTPUT_DIR%\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth" ^
        "https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth"
    
    if !errorlevel! equ 0 (
        echo     [OK] GLIP-T downloaded
    ) else (
        echo     [ERROR] Failed to download GLIP-T
        echo     Please download manually from:
        echo     https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth
        pause
    )
) else (
    echo   [SKIP] GLIP-T already exists
)

REM Uncomment if you need GLIP-L (large model)
REM if not exist "%OUTPUT_DIR%\MODEL\glip_large_model.pth" (
REM     echo   -^> Downloading GLIP-L ^(Large^) model ^(~1.5GB^)...
REM     curl -L -o "%OUTPUT_DIR%\MODEL\glip_large_model.pth" ^
REM         "https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth"
REM     echo     [OK] GLIP-L downloaded
REM )

REM ============================================
REM Step 2: Download Python Package Wheels
REM ============================================
echo.
echo Step 2/6: Downloading Python packages for CUDA 11.8...

if not exist "%OUTPUT_DIR%\python_packages" mkdir "%OUTPUT_DIR%\python_packages"

echo   -^> Downloading PyTorch for CUDA 11.8...
pip download torch==2.0.1 torchvision==0.15.2 ^
    --dest "%OUTPUT_DIR%\python_packages" ^
    --platform manylinux2014_x86_64 ^
    --python-version %PYTHON_VERSION% ^
    --extra-index-url https://download.pytorch.org/whl/cu118 ^
    --no-deps

echo   -^> Downloading other requirements...
if exist "requirements.txt" (
    pip download -r requirements.txt ^
        --dest "%OUTPUT_DIR%\python_packages" ^
        --platform manylinux2014_x86_64 ^
        --python-version %PYTHON_VERSION%
    echo     [OK] All packages downloaded
) else (
    echo     [WARNING] requirements.txt not found in current directory
    echo     Creating minimal requirements...
    
    (
        echo numpy
        echo einops
        echo shapely
        echo timm==0.6.7
        echo yacs
        echo tensorboardX
        echo ftfy
        echo prettytable
        echo pymongo
        echo transformers==4.21.3
        echo pycocotools
        echo scipy
        echo opencv-python
        echo einops-exts
        echo addict
        echo yapf
        echo supervision==0.4.0
        echo pandas
    ) > "%OUTPUT_DIR%\minimal_requirements.txt"
    
    pip download -r "%OUTPUT_DIR%\minimal_requirements.txt" ^
        --dest "%OUTPUT_DIR%\python_packages" ^
        --platform manylinux2014_x86_64 ^
        --python-version %PYTHON_VERSION%
)

REM ============================================
REM Step 3: Cache Hugging Face Models
REM ============================================
echo.
echo Step 3/6: Caching Hugging Face transformers models...

if not exist "%OUTPUT_DIR%\hf_cache" mkdir "%OUTPUT_DIR%\hf_cache"

set HF_HOME=%OUTPUT_DIR%\hf_cache
set TRANSFORMERS_CACHE=%OUTPUT_DIR%\hf_cache

echo   -^> Downloading BERT base uncased...
python -c "import os; os.environ['HF_HOME'] = r'%OUTPUT_DIR%\hf_cache'; os.environ['TRANSFORMERS_CACHE'] = r'%OUTPUT_DIR%\hf_cache'; from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); model = AutoModel.from_pretrained('bert-base-uncased'); save_path = r'%OUTPUT_DIR%\hf_cache\bert-base-uncased'; tokenizer.save_pretrained(save_path); model.save_pretrained(save_path); print('    [OK] BERT cached to', save_path)"

if !errorlevel! neq 0 (
    echo     [WARNING] Failed to cache transformers
    echo     You may need to install transformers first: pip install transformers
)

REM ============================================
REM Step 4: Cache timm Models
REM ============================================
echo.
echo Step 4/6: Caching timm ^(PyTorch Image Models^) weights...

if not exist "%OUTPUT_DIR%\timm_cache" mkdir "%OUTPUT_DIR%\timm_cache"

set TORCH_HOME=%OUTPUT_DIR%\timm_cache

echo   -^> Downloading Swin Transformer Tiny...
python -c "import os; os.environ['TORCH_HOME'] = r'%OUTPUT_DIR%\timm_cache'; import timm; import torch; model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True); print('    [OK] Swin-T cached')"

if !errorlevel! neq 0 (
    echo     [WARNING] Failed to cache timm models
    echo     You may need to install timm first: pip install timm torch
)

REM ============================================
REM Step 5: Copy Source Code
REM ============================================
echo.
echo Step 5/6: Copying source code...

REM Copy directories
for %%D in (configs groundingdino_new maskrcnn_benchmark odinw tools utils DATASET) do (
    if exist "%%D" (
        xcopy /E /I /Y /Q "%%D" "%OUTPUT_DIR%\%%D" > nul
        echo   [OK] Copied: %%D
    ) else (
        echo   [SKIP] Not found: %%D
    )
)

REM Copy files
for %%F in (setup_glip.py setup.py init.sh train.sh evaluate.sh extract_queries.sh docker-compose.yml Dockerfile requirements.txt) do (
    if exist "%%F" (
        copy /Y "%%F" "%OUTPUT_DIR%\" > nul
        echo   [OK] Copied: %%F
    ) else (
        echo   [SKIP] Not found: %%F
    )
)

REM Copy all markdown files
for %%F in (*.md) do (
    copy /Y "%%F" "%OUTPUT_DIR%\" > nul
    echo   [OK] Copied: %%F
)

REM ============================================
REM Step 6: Create Modified Files for Air-Gap
REM ============================================
echo.
echo Step 6/6: Creating air-gapped configuration files...

REM Create docker-compose.airgap.yml
(
echo version: '3.8'
echo.
echo services:
echo   mq-det:
echo     build:
echo       context: .
echo       dockerfile: Dockerfile.airgap
echo     image: mq-det:air-gapped
echo     container_name: mq-det-container
echo     runtime: nvidia
echo     environment:
echo       - NVIDIA_VISIBLE_DEVICES=all
echo       - CUDA_VISIBLE_DEVICES=0
echo       # Offline mode for Hugging Face
echo       - TRANSFORMERS_CACHE=/workspace/hf_cache
echo       - HF_HOME=/workspace/hf_cache
echo       - TRANSFORMERS_OFFLINE=1
echo       - HF_DATASETS_OFFLINE=1
echo       # Offline mode for timm
echo       - TORCH_HOME=/workspace/timm_cache
echo       # Python path
echo       - PYTHONPATH=/workspace
echo     volumes:
echo       - ./:/workspace
echo       - ./MODEL:/workspace/MODEL
echo       - ./DATASET:/workspace/DATASET
echo       - ./OUTPUT:/workspace/OUTPUT
echo       - ./hf_cache:/workspace/hf_cache
echo       - ./timm_cache:/workspace/timm_cache
echo     shm_size: '8gb'
echo     stdin_open: true
echo     tty: true
echo     working_dir: /workspace
echo     command: /bin/bash
) > "%OUTPUT_DIR%\docker-compose.airgap.yml"

echo   [OK] Created docker-compose.airgap.yml

REM Create Dockerfile.airgap
(
echo # Air-Gapped MQ-Det Dockerfile
echo # Assumes base image with: Ubuntu 20.04, Python 3.8, CUDA 11.8, cuDNN 8.7
echo.
echo # IMPORTANT: Replace this with your actual air-gapped base image
echo # FROM your-internal-registry/ubuntu20.04-cuda11.8-cudnn8.7-python3.8:latest
echo FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
echo.
echo ENV DEBIAN_FRONTEND=noninteractive
echo ENV CUDA_HOME=/usr/local/cuda
echo ENV PATH=/usr/local/cuda/bin:$PATH
echo ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo ENV TORCH_CUDA_ARCH_LIST=7.5
echo ENV FORCE_CUDA=1
echo.
echo # Install system dependencies ^(if not in base image^)
echo RUN apt-get update ^&^& apt-get install -y \
echo     build-essential cmake ninja-build \
echo     git wget curl vim pkg-config \
echo     gcc-8 g++-8 \
echo     python3.8 python3.8-dev python3-pip \
echo     libjpeg-dev zlib1g-dev libpng-dev \
echo     libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
echo  ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo # Set Python 3.8 as default
echo RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
echo  ^&^& update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
echo.
echo RUN python3.8 -m pip install --upgrade pip setuptools wheel
echo.
echo WORKDIR /workspace
echo.
echo # Copy all files
echo COPY . /workspace
echo.
echo # Install Python packages from local wheels ^(OFFLINE^)
echo RUN pip install --no-index --find-links=/workspace/python_packages \
echo     torch torchvision ^&^& \
echo     pip install --no-index --find-links=/workspace/python_packages -r requirements.txt
echo.
echo # Patch ATen headers for newer PyTorch compatibility
echo RUN sed -i 's@^<ATen/ceil_div.h^>@^<c10/util/ceil_div.h^>@' \
echo     maskrcnn_benchmark/csrc/cuda/ROIAlign_cuda.cu \
echo     maskrcnn_benchmark/csrc/cuda/ROIPool_cuda.cu ^|^| true
echo.
echo # Build CUDA extensions
echo ENV CC=/usr/bin/gcc-8
echo ENV CXX=/usr/bin/g++-8
echo ENV CUDAHOSTCXX=/usr/bin/g++-8
echo ENV CXXFLAGS="-O3 -std=c++14"
echo ENV PIP_NO_BUILD_ISOLATION=1
echo ENV PIP_USE_PEP517=0
echo.
echo RUN python setup_glip.py clean --all ^|^| true ^&^& \
echo     rm -f maskrcnn_benchmark/_C*.so ^&^& \
echo     python setup_glip.py build_ext --inplace
echo.
echo # Verify build
echo RUN python -c "import sys; sys.path.insert^(0, '/workspace'^); from maskrcnn_benchmark import _C; print^('OK _C import OK'^)"
echo.
echo # Set environment for offline operation
echo ENV PYTHONPATH=/workspace
echo ENV TRANSFORMERS_CACHE=/workspace/hf_cache
echo ENV HF_HOME=/workspace/hf_cache
echo ENV TORCH_HOME=/workspace/timm_cache
echo ENV TRANSFORMERS_OFFLINE=1
echo ENV HF_DATASETS_OFFLINE=1
echo.
echo # Create output directories
echo RUN mkdir -p OUTPUT DATASET
echo.
echo # Entry point
echo CMD ["/bin/bash"]
) > "%OUTPUT_DIR%\Dockerfile.airgap"

echo   [OK] Created Dockerfile.airgap

REM Create INSTALL_AIRGAP.bat for easy installation on the VM
(
echo @echo off
echo REM Air-Gapped Installation Script
echo echo Installing MQ-Det in air-gapped environment...
echo echo.
echo echo Step 1: Building Docker image...
echo docker build -f Dockerfile.airgap -t mq-det:air-gapped .
echo if %%errorlevel%% neq 0 ^(
echo     echo [ERROR] Docker build failed!
echo     pause
echo     exit /b 1
echo ^)
echo echo   [OK] Docker image built successfully
echo echo.
echo echo Step 2: Starting container...
echo docker compose -f docker-compose.airgap.yml up -d
echo if %%errorlevel%% neq 0 ^(
echo     echo [ERROR] Failed to start container!
echo     pause
echo     exit /b 1
echo ^)
echo echo   [OK] Container started
echo echo.
echo echo Step 3: Verifying installation...
echo docker exec mq-det-container python -c "import torch; print^('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available^(^)^)"
echo docker exec mq-det-container python -c "from maskrcnn_benchmark import _C; print^('maskrcnn-benchmark: OK'^)"
echo echo.
echo echo ================================================
echo echo Installation complete!
echo echo ================================================
echo echo.
echo echo To enter the container:
echo echo   docker exec -it mq-det-container bash
echo echo.
echo echo To stop the container:
echo echo   docker compose -f docker-compose.airgap.yml down
echo echo.
echo pause
) > "%OUTPUT_DIR%\INSTALL_AIRGAP.bat"

echo   [OK] Created INSTALL_AIRGAP.bat

REM ============================================
REM Create Bundle Archive
REM ============================================
echo.
echo Creating final ZIP archive...

REM Check if tar is available (Windows 10+)
where tar >nul 2>nul
if !errorlevel! equ 0 (
    echo   -^> Using tar to create archive...
    tar -czf "%BUNDLE_NAME%.tar.gz" "%BUNDLE_NAME%"
    if !errorlevel! equ 0 (
        for %%A in ("%BUNDLE_NAME%.tar.gz") do set BUNDLE_SIZE=%%~zA
        set /a BUNDLE_SIZE_MB=!BUNDLE_SIZE! / 1048576
        echo   [OK] Created %BUNDLE_NAME%.tar.gz ^(!BUNDLE_SIZE_MB! MB^)
    )
) else (
    echo   [INFO] tar not found, using PowerShell to create ZIP...
    powershell -Command "Compress-Archive -Path '%BUNDLE_NAME%' -DestinationPath '%BUNDLE_NAME%.zip' -Force"
    if !errorlevel! equ 0 (
        for %%A in ("%BUNDLE_NAME%.zip") do set BUNDLE_SIZE=%%~zA
        set /a BUNDLE_SIZE_MB=!BUNDLE_SIZE! / 1048576
        echo   [OK] Created %BUNDLE_NAME%.zip ^(!BUNDLE_SIZE_MB! MB^)
    )
)

echo.
echo ==================================================
echo Bundle preparation complete!
echo ==================================================
echo.
echo Bundle location: %CD%\%BUNDLE_NAME%.zip ^(or .tar.gz^)
echo.
echo Next steps:
echo   1. Transfer the bundle to your air-gapped VM
echo   2. Extract the bundle
echo   3. Run INSTALL_AIRGAP.bat ^(or follow AIRGAP_INSTALLATION.md^)
echo.
echo Bundle contents:
echo   [OK] Pretrained model weights ^(MODEL/^)
echo   [OK] Python package wheels ^(python_packages/^)
echo   [OK] Hugging Face model cache ^(hf_cache/^)
echo   [OK] timm model cache ^(timm_cache/^)
echo   [OK] Source code and configs
echo   [OK] Modified Dockerfile and docker-compose for air-gap
echo   [OK] Installation instructions and scripts
echo.
echo Have a successful deployment!
echo ==================================================
echo.
pause
