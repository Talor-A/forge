#!/bin/bash
# Step 2b: Preprocess trajectory data (JSONL -> memory-mapped numpy)
# Run after data collection (02), before training (03/04).
# Usage: 02b_preprocess_data.sh [data_dir] [output_dir]
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

DATA_DIR=${1:-/home/maustin/forge/rl_data/trajectories}
OUTPUT_DIR=${2:-/home/maustin/forge/rl_data/preprocessed}

echo "Preprocessing trajectories: $DATA_DIR -> $OUTPUT_DIR"
python training/preprocess_trajectories.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
