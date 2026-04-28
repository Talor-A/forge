#!/bin/bash
# Step 4: Train attack/block/priority decision heads (imitation learning), headless.
# Usage: 04_train_decisions.sh [epochs] [batch_size] [encoder] [heads] [--joint] [--model-size SIZE]
# heads: "all" (default), or comma-separated: "priority", "attack,block", etc.
# --joint: train all heads simultaneously with unfrozen encoder
# --model-size: small (512/23M), medium (768/45M), large (1024/73M), xl (1024/107M)
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

EPOCHS=${1:-10}
BATCH=${2:-256}
CKPT_DIR="$SAVE_DIR"
HEADS=${4:-all}
JOINT_FLAG=""
MODEL_SIZE_FLAG=""
if [[ "$*" == *"--joint"* ]]; then
    JOINT_FLAG="--joint"
    echo "JOINT MODE: training all heads with unfrozen encoder"
fi
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--model-size" ]]; then
        MODEL_SIZE_FLAG="--model-size $arg"
        echo "MODEL SIZE: $arg"
    fi
    prev="$arg"
done

# Auto-select best available checkpoint if not explicitly provided
if [ -z "$3" ]; then
    # Chain order: model_with_decisions > best_block > best_attack > best_priority > best_value
    if [ -f "$CKPT_DIR/model_with_decisions.pt" ]; then
        ENCODER="$CKPT_DIR/model_with_decisions.pt"
    elif [ -f "$CKPT_DIR/best_block_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_block_model.pt"
    elif [ -f "$CKPT_DIR/best_attack_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_attack_model.pt"
    elif [ -f "$CKPT_DIR/best_priority_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_priority_model.pt"
    else
        ENCODER="$CKPT_DIR/best_value_model.pt"
    fi
    echo "Auto-selected encoder: $(basename "$ENCODER")"
else
    ENCODER="$3"
fi

# Warn if using base value model when trained heads exist
if [[ "$ENCODER" == *"best_value_model"* ]]; then
    for f in best_priority_model.pt best_attack_model.pt best_block_model.pt model_with_decisions.pt; do
        if [ -f "$CKPT_DIR/$f" ]; then
            echo "WARNING: Using best_value_model.pt but $f exists!"
            echo "  This will DISCARD trained head weights. Use $f instead?"
            echo "  Press Ctrl+C to abort, or Enter to continue anyway."
            read -r
            break
        fi
    done
fi

LOG="$SAVE_DIR/logs/decisions_$(date +%Y%m%d_%H%M%S).log"
echo "Training decision heads for $EPOCHS epochs, batch=$BATCH, heads=$HEADS"
echo "  encoder: $ENCODER"
echo "  data:    $DATA_DIR"
echo "  out:     $SAVE_DIR"
echo "  log:     $LOG"

"$VENV/bin/python" training/train_decisions.py \
    --data-dir "$DATA_DIR" \
    --encoder-checkpoint "$ENCODER" \
    --save-dir "$SAVE_DIR" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --heads "$HEADS" \
    $JOINT_FLAG \
    $MODEL_SIZE_FLAG 2>&1 | tee "$LOG"
