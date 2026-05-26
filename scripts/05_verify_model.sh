#!/bin/bash
# Step 5: Verify model works in live games
# Checks: server alive, RL decisions real, creatures played, attack probs vary
set -e
cd /home/maustin/forge

MODEL=${1:-rl_data/checkpoints/model_with_decisions.pt}
GAMES=${2:-10}
PORT=50051

# Kill any existing server
pkill -f "model_server" 2>/dev/null || true
sleep 1

# Start model server in a subshell
echo "Starting model server with $MODEL..."
(
    cd forge-ai-rl/src/main/python
    source /home/maustin/forge/forge-ai-rl/venv/bin/activate
    exec python -c "
from serving.model_server import ModelServer
from model.mtg_model import MTGModel
import threading, time
model = MTGModel.load('/home/maustin/forge/$MODEL', device='cuda')
server = ModelServer(model, port=$PORT, device='cuda')
t = threading.Thread(target=server.start, daemon=True)
t.start()
print('Server started', flush=True)
while True: time.sleep(1)
"
) &
SERVER_PID=$!
sleep 5

# Verify alive
echo "=== SERVER CHECK ==="
lsof -i :$PORT 2>/dev/null | grep LISTEN && echo "SERVER: ALIVE" || { echo "SERVER: DEAD"; kill $SERVER_PID 2>/dev/null; exit 1; }

# Run eval
echo "Running $GAMES games..."
FORGE_JAR=$(ls /home/maustin/forge/forge-gui-desktop/target/forge-gui-desktop-*-jar-with-dependencies.jar 2>/dev/null | tail -n1)
cd /home/maustin/forge/forge-gui-desktop
java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain evaluate \
    -d "Green Stompy.dck" -d "White Weenie.dck" -d "Blue Tempo.dck" -d "Red Aggro.dck" \
    -n "$GAMES" -o /tmp/rl_verify \
    -host localhost -port $PORT -q 2>&1 | grep -E "RL Wins|Complete"

echo ""
echo "=== SERVER STILL ALIVE? ==="
lsof -i :$PORT 2>/dev/null | grep LISTEN && echo "YES" || echo "NO"

echo ""
echo "=== DECISION TYPES ==="
grep -oh '"decisionType":"[^"]*"' /tmp/rl_verify/traj_*.jsonl 2>/dev/null | sort | uniq -c | sort -rn

echo ""
echo "=== CREATURE COUNTS ==="
grep "main1_turn" /tmp/rl_verify/traj_*.jsonl 2>/dev/null | head -5 | while IFS= read -r line; do
echo "$line" | cut -d: -f2- | python3 -c "
import sys,json
r=json.loads(sys.stdin.read())
gf=r.get('globalFeatures',[])
mc=gf[23] if len(gf)>23 else -1
print(f'  {r.get(\"contextInfo\",\"\"):20s} myCreatures={mc*20:.0f}')
" 2>/dev/null
done

echo ""
echo "=== ATTACK PROBS (should vary, not all 0.01) ==="
grep "RL_MODEL_ATTACK" /tmp/rl_verify/traj_*.jsonl 2>/dev/null | head -5 || \
  cd /home/maustin/forge/forge-gui-desktop && java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain evaluate \
    -d "Green Stompy.dck" -d "White Weenie.dck" \
    -n 1 -o /tmp/rl_verify_probs \
    -host localhost -port $PORT 2>&1 | grep "RL_MODEL_ATTACK" | head -5

# Cleanup
kill $SERVER_PID 2>/dev/null
echo ""
echo "=== DONE ==="
