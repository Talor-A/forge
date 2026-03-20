#!/bin/bash
# Step 7: Visualize game states and model predictions
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

MODEL=${1:-/home/maustin/forge/rl_data/checkpoints/model_with_decisions.pt}
DATA=${2:-/home/maustin/forge/rl_data/trajectories}

echo "Launching visualizer..."
echo "  Model: $MODEL"
echo "  Data:  $DATA"
python training/visualize_game_state.py \
    --data-dir "$DATA" \
    --model "$MODEL" \
    --device cuda
