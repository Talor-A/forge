#!/bin/bash
# Step 3: Train value network (game state encoder)
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

EPOCHS=${1:-100}
BATCH=${2:-256}

echo "Training value network for $EPOCHS epochs, batch=$BATCH..."
python training/training_ui.py \
    --data-dir /home/maustin/forge/rl_data/trajectories \
    --save-dir /home/maustin/forge/rl_data/checkpoints \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH"
