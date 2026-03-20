#!/bin/bash
# Step 4: Train attack/block decision heads (imitation learning)
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

EPOCHS=${1:-50}
BATCH=${2:-256}
ENCODER=${3:-/home/maustin/forge/rl_data/checkpoints/best_value_model.pt}

echo "Training decision heads for $EPOCHS epochs, batch=$BATCH..."
echo "Encoder: $ENCODER"
python training/train_decisions_ui.py \
    --data-dir /home/maustin/forge/rl_data/trajectories \
    --encoder-checkpoint "$ENCODER" \
    --save-dir /home/maustin/forge/rl_data/checkpoints \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH"
