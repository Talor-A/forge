#!/bin/bash
# Step 3: Train value network (game state encoder), headless.
# Usage: 03_train_value.sh [epochs] [batch_size]
#
# Env overrides:
#   FORGE_ROOT  repo root          (default: /workspace/forge-ai-investigation)
#   DATA_DIR    trajectory dir     (default: $FORGE_ROOT/rl_data/trajectories)
#   SAVE_DIR    checkpoint dir     (default: /workspace/checkpoints)
set -e
export MPLBACKEND=Agg

ROOT="${FORGE_ROOT:-/workspace/forge-ai-investigation}"
DATA_DIR="${DATA_DIR:-$ROOT/rl_data/trajectories}"
SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints}"
VENV="$ROOT/forge-ai-rl/venv"

[ -d "$VENV" ] || { echo "venv missing at $VENV — run runpod_bootstrap.sh with INSTALL_TRAIN_DEPS=1"; exit 1; }

cd "$ROOT/forge-ai-rl/src/main/python"
mkdir -p "$SAVE_DIR/logs"

EPOCHS=${1:-100}
BATCH=${2:-256}
LOG="$SAVE_DIR/logs/value_$(date +%Y%m%d_%H%M%S).log"

echo "Training value network for $EPOCHS epochs, batch=$BATCH"
echo "  data: $DATA_DIR"
echo "  out:  $SAVE_DIR"
echo "  log:  $LOG"

"$VENV/bin/python" training/train_value.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" 2>&1 | tee "$LOG"
