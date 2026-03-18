#!/bin/bash
# Train the value network on collected trajectory data.
# Usage: ./train.sh [epochs]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"
SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"
LOG_DIR="$PROJECT_ROOT/rl_data/runs"
EPOCHS="${1:-50}"

if [ ! -f "$PYTHON" ]; then
    echo "Python venv not found. Run setup.sh first."
    exit 1
fi

# Check data exists
NUM_FILES=$(ls "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No trajectory data found in $DATA_DIR"
    echo "Run collect_data.sh first."
    exit 1
fi
echo "Found $NUM_FILES trajectory files"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "=== Training Value Network ==="
echo "Data: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "Checkpoints: $SAVE_DIR"
echo "TensorBoard: $LOG_DIR"
echo ""

cd "$PYTHON_DIR"
"$PYTHON" training/train_value.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --log-dir "$LOG_DIR/value_train" \
    --epochs "$EPOCHS" \
    --lr 1e-3

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $SAVE_DIR"
echo "View training: tensorboard --logdir $LOG_DIR"
