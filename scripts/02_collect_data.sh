#!/bin/bash
# Step 2: Collect trajectory data with live dashboard
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

GAMES=${1:-1000}

echo "Launching collection dashboard for $GAMES games..."
python training/collect_ui.py --games "$GAMES"
