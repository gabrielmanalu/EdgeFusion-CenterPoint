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

set -e

WORKSPACE=/workspace
CONDA_DIR=$WORKSPACE/miniconda3
ENV_NAME=autoware_cp
MMDET3D_DIR=$WORKSPACE/mmdetection3d

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

# ── 2. Python 3.8 env ────────────────────────────────────────────────────────
echo "[2/7] Creating conda env: $ENV_NAME (Python 3.8)..."
conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"

# ── 3. PyTorch 2.1.0 + CUDA 11.8 ─────────────────────────────────────────────
# cu118 is used deliberately: matches the mmcv/mmdet3d stack.
# System CUDA 12.8 runtime is backward-compatible for GPU ops.
echo "[3/7] Installing PyTorch 2.1.0+cu118..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. mmcv 2.1.0 ─────────────────────────────────────────────────────────────
# Key lesson: openmmlab download URLs are unreachable from RunPod.
# Use plain `pip install mmcv` which builds from PyPI source.
# mmcv must be <2.2.0 for mmdet 3.2.0 compatibility.
echo "[4/7] Installing mmcv 2.1.0 (source build — ~20-40 min)..."
pip install mmcv==2.1.0

# ── 5. mmdet 3.2.0 ────────────────────────────────────────────────────────────
echo "[5/7] Installing mmdet 3.2.0..."
pip install mmdet==3.2.0

# ── 6. mmdetection3d 1.3.0 (autowarefoundation fork) ─────────────────────────
echo "[6/7] Cloning + installing autowarefoundation/mmdetection3d..."
if [ ! -d "$MMDET3D_DIR" ]; then
    git clone https://github.com/autowarefoundation/mmdetection3d.git \
        "$MMDET3D_DIR"
fi
cd "$MMDET3D_DIR"
pip install -e . --no-deps

# ── 7. Remaining packages ─────────────────────────────────────────────────────
echo "[7/7] Installing remaining packages..."
pip install \
    onnx==1.17.0 \
    onnxruntime==1.19.2 \
    mlflow==2.17.0 \
    nuscenes-devkit \
    numpy==1.23.5 \
    numba \
    tqdm \
    rich \
    open3d

# ── activate_env.sh ───────────────────────────────────────────────────────────
cat > "$WORKSPACE/activate_env.sh" << 'EOF'
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate autoware_cp
EOF
echo "activate_env.sh written to $WORKSPACE/activate_env.sh"

# ── data symlink (after nuScenes is extracted to /data/nuscenes/) ─────────────
echo ""
echo "========================================"
echo "Environment setup complete."
echo ""
echo "Next steps:"
echo "  1. source /workspace/activate_env.sh"
echo "  2. Download + extract nuScenes to /data/nuscenes/"
echo "  3. ln -s /data/nuscenes $MMDET3D_DIR/data/nuscenes"
echo "  4. Restore checkpoints + pkl files to /workspace/data/centerpoint/"
echo "  5. Verify: python -c 'import mmdet3d; print(mmdet3d.__version__)'"
echo "========================================"