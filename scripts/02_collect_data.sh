#!/bin/bash
# Step 2: Collect trajectory data + preprocess with live dashboard
# Usage: ./02_collect_data.sh [games] [--clean]
#   --clean: delete old trajectories and preprocessed data first
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

GAMES=${1:-1000}
shift 2>/dev/null || true

echo "Launching collection dashboard for $GAMES games..."
python training/collect_ui.py --games "$GAMES" "$@"
