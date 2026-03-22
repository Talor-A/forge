#!/bin/bash
# Step 4: Train attack/block/priority decision heads (imitation learning)
# Usage: 04_train_decisions.sh [epochs] [batch_size] [encoder] [heads]
# heads: "all" (default), or comma-separated: "priority", "attack,block", etc.
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

EPOCHS=${1:-10}
BATCH=${2:-256}
CKPT_DIR=/home/maustin/forge/rl_data/checkpoints
HEADS=${4:-all}

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
    echo "Auto-selected encoder: $(basename $ENCODER)"
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
            read
            break
        fi
    done
fi

echo "Training decision heads for $EPOCHS epochs, batch=$BATCH, heads=$HEADS..."
echo "Encoder: $ENCODER"
python training/train_decisions_ui.py \
    --data-dir /home/maustin/forge/rl_data/trajectories \
    --encoder-checkpoint "$ENCODER" \
    --save-dir /home/maustin/forge/rl_data/checkpoints \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --heads "$HEADS"
