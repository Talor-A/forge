#!/bin/bash
# Step 8: Export trained model to ONNX for Java inference
# Usage: 08_export_onnx.sh [checkpoint] [output_dir]
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

CKPT_DIR=/home/maustin/forge/rl_data/checkpoints
if [ -z "$1" ]; then
    if [ -f "$CKPT_DIR/best_ppo_model.pt" ]; then
        CKPT="$CKPT_DIR/best_ppo_model.pt"
    else
        CKPT="$CKPT_DIR/model_with_decisions.pt"
    fi
else
    CKPT="$1"
fi
OUTPUT=${2:-/home/maustin/forge/rl_data/models}

echo "Exporting ONNX models..."
echo "  Checkpoint: $CKPT"
echo "  Output: $OUTPUT"

pip install onnxruntime -q 2>/dev/null || true

python tools/export_onnx.py \
    --checkpoint "$CKPT" \
    --output-dir "$OUTPUT" \
    --device cpu

# Copy to Forge data directory for GUI access
FORGE_DIR="$HOME/.forge/res/rl/models"
mkdir -p "$FORGE_DIR"
cp "$OUTPUT"/*.onnx "$FORGE_DIR/"
echo "Copied ONNX files to $FORGE_DIR"
