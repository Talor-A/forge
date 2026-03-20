#!/bin/bash
# Step 6: PPO training with dashboard
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

pkill -f "model_server\|ppo_ui" 2>/dev/null || true
sleep 1

MODEL=${1:-/home/maustin/forge/rl_data/checkpoints/model_with_decisions.pt}
ROUNDS=${2:-10}
GAMES=${3:-100}

echo "PPO Training: $ROUNDS rounds, $GAMES games/round"
echo "Model: $MODEL"

python training/ppo_ui.py \
    --checkpoint "$MODEL" \
    --save-dir /home/maustin/forge/rl_data/checkpoints \
    --device cuda \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 30 \
    --ppo-epochs 4 \
    --batch-size 32 \
    --lr 1e-5 \
    --port 0
