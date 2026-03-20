#!/bin/bash
# Step 2: Collect trajectory data (heuristic vs heuristic)
set -e
cd /home/maustin/forge

GAMES=${1:-1000}
OUTPUT_DIR="rl_data/trajectories"
JAR="forge-gui-desktop/target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar"

echo "Collecting $GAMES games into $OUTPUT_DIR..."
cd forge-gui-desktop
java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar \
    rltrain collect \
    -d green_stompy.dck -d white_weenie.dck -d blue_tempo.dck -d red_aggro.dck \
    -n "$GAMES" -t 16 \
    -o "/home/maustin/forge/$OUTPUT_DIR" -q

echo ""
echo "=== Data verification ==="
cd /home/maustin/forge
echo "Files: $(ls $OUTPUT_DIR/traj_*.jsonl | wc -l)"
grep -oh '"decisionType":"[^"]*"' $OUTPUT_DIR/traj_*.jsonl | sort | uniq -c | sort -rn

# Verify tapped flag is NOT leaking into attack candidates
echo ""
echo "=== Tapped flag check ==="
source forge-ai-rl/venv/bin/activate && python3 -c "
import json
from pathlib import Path
tapped_attacked = 0
total = 0
for f in sorted(Path('$OUTPUT_DIR').glob('traj_*.jsonl'))[:200]:
    for line in open(f):
        try:
            r = json.loads(line)
            if r.get('decisionType') != 'DECLARE_ATTACKERS': continue
            for i, cf in enumerate(r.get('candidateFeatures', [])):
                total += 1
                if cf[17] > 0.5 and i in r.get('selectedIndices', []):
                    tapped_attacked += 1
        except: pass
print(f'Tapped+attacked: {tapped_attacked}/{total} (should be 0 if fix works)')
"
