#!/usr/bin/env bash
# setup_env.sh
# Recreates the autoware_cp conda environment on a fresh RunPod pod.
# Tested on: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# CUDA: 12.8 (system), 11.8 (PyTorch compile target — intentional, runtime works)
#
# Usage:
#   bash setup_env.sh 2>&1 | tee setup_env.log
#
# Time estimate: ~60-90 min (mmcv source build is the slow step)
#
# Key lessons:
#   - conda activate does not persist in bash scripts; use explicit $PIP/$PYTHON paths
#   - mmcv must be installed from openmmlab CDN (PyPI source build fails: CUDA 12.8 vs 11.8)
#   - openmmlab download URLs work from RunPod with the -f flag

set -e

WORKSPACE=/workspace
CONDA_DIR=$WORKSPACE/miniconda3
ENV_NAME=autoware_cp
MMDET3D_DIR=$WORKSPACE/mmdetection3d

# Explicit paths — never rely on conda activate inside a script
PIP=$CONDA_DIR/envs/$ENV_NAME/bin/pip
PYTHON=$CONDA_DIR/envs/$ENV_NAME/bin/python

# ── 1. Miniconda ─────────────────────────────────────────────────────────────
echo "[1/7] Installing Miniconda..."
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
echo "Miniconda ready."

# ── 1b. Accept Anaconda ToS (required on fresh installs) ─────────────────────
echo "[1b/7] Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ── 2. Python 3.8 env ────────────────────────────────────────────────────────
echo "[2/7] Creating conda env: $ENV_NAME (Python 3.8)..."
conda create -n "$ENV_NAME" python=3.8 -y

# Activate so subsequent steps can detect installed packages (e.g. torch for mmcv)
# $PIP/$PYTHON are still used explicitly for all installs
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "Conda env created and activated."

# ── 3. PyTorch 2.1.0 + CUDA 11.8 ─────────────────────────────────────────────
# cu118 is intentional: matches the mmcv/mmdet3d stack.
# System CUDA 12.8 runtime is backward-compatible for inference.
echo "[3/7] Installing PyTorch 2.1.0+cu118..."
$PIP install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. mmcv 2.1.0 ─────────────────────────────────────────────────────────────
# Must use openmmlab CDN — PyPI source build fails because mmcv's CUDA extension
# builder detects system CUDA 12.8 vs PyTorch CUDA 11.8 and raises RuntimeError.
# The CDN provides a pre-built wheel for exactly torch2.1.0+cu118.
echo "[4/7] Installing mmcv 2.1.0 (pre-built wheel from openmmlab CDN)..."
$PIP install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# ── 5. mmdet 3.2.0 ────────────────────────────────────────────────────────────
echo "[5/7] Installing mmdet 3.2.0..."
$PIP install mmdet==3.2.0

# ── 6. mmdetection3d 1.3.0 (autowarefoundation fork) ─────────────────────────
echo "[6/7] Cloning + installing autowarefoundation/mmdetection3d..."
if [ ! -d "$MMDET3D_DIR" ]; then
    git clone https://github.com/autowarefoundation/mmdetection3d.git \
        "$MMDET3D_DIR"
fi
cd "$MMDET3D_DIR"
$PIP install -e . --no-deps

# ── 7. Remaining packages ─────────────────────────────────────────────────────
echo "[7/7] Installing remaining packages..."
$PIP install \
    onnx==1.17.0 \
    onnxruntime==1.19.2 \
    mlflow==2.17.0 \
    nuscenes-devkit \
    numpy==1.23.5 \
    numba \
    tqdm \
    rich \
    open3d \
    lyft_dataset_sdk \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh

# ── activate_env.sh ───────────────────────────────────────────────────────────
cat > "$WORKSPACE/activate_env.sh" << 'ENVEOF'
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate autoware_cp
python -c "
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'
print(f'[autoware_cp] torch {torch.__version__} | GPU: {gpu}')
"
ENVEOF
echo "activate_env.sh written to $WORKSPACE/activate_env.sh"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Environment setup complete."
echo ""
echo "Next steps:"
echo "  1. source /workspace/activate_env.sh"
echo "  2. Download + extract nuScenes to /data/nuscenes/"
echo "  3. ln -s /data/nuscenes $MMDET3D_DIR/data/nuscenes"
echo "  4. cp /workspace/dropbox/*.pth /workspace/data/centerpoint/"
echo "  5. cp /workspace/dropbox/*.pkl /workspace/data/centerpoint/"
echo "  5. Verify: $PYTHON -c 'import mmdet3d; print(mmdet3d.__version__)'"