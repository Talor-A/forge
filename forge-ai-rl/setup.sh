#!/bin/bash
# Setup script for the Forge RL AI Python environment
# Usage: source forge-ai-rl/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_DIR="$SCRIPT_DIR/src/main/python"

echo "=== Forge RL AI Setup ==="

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$PYTHON_DIR/requirements.txt"

# Verify CUDA
echo ""
echo "=== Environment Info ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_mem / 1024**3:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available. Training will be slow on CPU.')
print()

# Estimate memory usage
import sys
sys.path.insert(0, '$PYTHON_DIR')
from model.gpu_config import auto_detect_profile, estimate_memory_usage
profile = auto_detect_profile()
print(f'GPU Profile: {profile.name}')
print(f'Recommended batch size: {profile.batch_size}')
print(f'Mixed precision (AMP): {profile.use_amp}')
mem = estimate_memory_usage(profile.batch_size)
print(f'Estimated VRAM usage: {mem[\"total_gb\"]:.2f} GB')
"

echo ""
echo "=== Setup Complete ==="
echo "Virtual environment: $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo ""
echo "Quick start:"
echo "  1. Start model server:  python3 $PYTHON_DIR/serving/model_server.py --device cuda"
echo "  2. Train decisions:     python3 $PYTHON_DIR/training/train_decisions.py --device cuda"
echo "  3. PPO self-play:       python3 $PYTHON_DIR/training/ppo_trainer.py --device cuda"
