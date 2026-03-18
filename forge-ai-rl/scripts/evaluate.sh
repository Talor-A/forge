#!/bin/bash
# Evaluate RL AI against heuristic AI.
# Requires model server running (./serve_model.sh).
# Usage: ./evaluate.sh [num_games]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FORGE_JAR="$PROJECT_ROOT/forge-gui-desktop/target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar"
DATA_DIR="$PROJECT_ROOT/rl_data/eval_trajectories"
NUM_GAMES="${1:-100}"

mkdir -p "$DATA_DIR"

echo "=== Evaluating RL AI vs Heuristic ==="
echo "Games: $NUM_GAMES"
echo ""

cd "$PROJECT_ROOT/forge-gui-desktop"
java -Xmx4096m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain evaluate \
    -d green_stompy.dck \
    -d white_weenie.dck \
    -n "$NUM_GAMES" \
    -o "$DATA_DIR" \
    -host localhost \
    -port 50051
