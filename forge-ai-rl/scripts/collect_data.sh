#!/bin/bash
# Collect imitation learning data by running heuristic AI vs AI games.
# Usage: ./collect_data.sh [num_games]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FORGE_JAR="$PROJECT_ROOT/forge-gui-desktop/target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar"
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"
NUM_GAMES="${1:-1000}"

if [ ! -f "$FORGE_JAR" ]; then
    echo "Forge jar not found. Building..."
    cd "$PROJECT_ROOT"
    mvn package -pl forge-gui-desktop -am \
        -Denforcer.skip=true -Dcheckstyle.skip=true \
        -DskipTests -q
fi

mkdir -p "$DATA_DIR"

echo "=== Collecting $NUM_GAMES games ==="
echo "Output: $DATA_DIR"
echo ""

cd "$PROJECT_ROOT/forge-gui-desktop"
java -Xmx4096m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain collect \
    -d green_stompy.dck \
    -d white_weenie.dck \
    -d blue_tempo.dck \
    -d red_aggro.dck \
    -n "$NUM_GAMES" \
    -o "$DATA_DIR" \
    -q

echo ""
echo "Files collected:"
ls -lh "$DATA_DIR" | tail -5
echo "Total: $(ls "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l) trajectory files"
