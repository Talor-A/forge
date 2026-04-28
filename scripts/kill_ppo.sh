#!/usr/bin/env bash
# Kill PPO training run and its child processes (model server, Forge JVMs).
set -u
patterns=(ppo_ui.py ppo_trainer.py model_server.py forge-gui-desktop)
for p in "${patterns[@]}"; do
    pkill -f "$p" 2>/dev/null || true
done
sleep 2
remaining=$(ps aux | grep -E "ppo_ui|ppo_trainer|model_server|forge-gui-desktop" | grep -v grep | grep -v kill_ppo)
if [ -n "$remaining" ]; then
    echo "still running, sending SIGKILL:"
    echo "$remaining"
    for p in "${patterns[@]}"; do
        pkill -9 -f "$p" 2>/dev/null || true
    done
    sleep 1
fi
echo "done"
ps aux | grep -E "ppo_ui|ppo_trainer|model_server" | grep -v grep | grep -v kill_ppo || echo "no PPO processes running"
