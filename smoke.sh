#!/usr/bin/env bash
# smoke.sh — end-to-end RL smoke test: collect a few self-play games, train
# the value head, and assert that training loss drops substantially.
# Usage: ./smoke.sh [GAMES] [EPOCHS]    (defaults: 10 games, 30 epochs)
# Prerequisites: ./ci.sh has been run (forge jar built, Python venv populated).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$ROOT/forge-ai-rl"
VENV_DIR="$RL_DIR/venv"
PY_DIR="$RL_DIR/src/main/python"
OUT_DIR="${SMOKE_OUT:-$(mktemp -d -t forge-smoke)}"

GAMES="${1:-10}"
EPOCHS="${2:-30}"

step() { printf '\n=== %s ===\n' "$1"; }

FORGE_JAR=$(ls "$ROOT/forge-gui-desktop/target/"forge-gui-desktop-*-jar-with-dependencies.jar 2>/dev/null | tail -n1)
[ -n "$FORGE_JAR" ] && [ -f "$FORGE_JAR" ] || {
    echo "forge jar not found under forge-gui-desktop/target/ — run ./ci.sh first" >&2
    exit 1
}
[ -d "$VENV_DIR" ] || {
    echo "Python venv missing at $VENV_DIR — run ./ci.sh first" >&2
    exit 1
}

JAVA_BIN="${JAVA_HOME:+$JAVA_HOME/bin/java}"
[ -x "${JAVA_BIN:-}" ] || JAVA_BIN="/opt/homebrew/opt/openjdk@21/bin/java"
[ -x "$JAVA_BIN" ] || JAVA_BIN="java"

mkdir -p "$OUT_DIR/trajectories" "$OUT_DIR/preprocessed" "$OUT_DIR/checkpoints" "$OUT_DIR/logs"
echo "output dir: $OUT_DIR"

step "Collect $GAMES self-play games"
echo "  jar:  $FORGE_JAR"
echo "  out:  $OUT_DIR/trajectories"
( cd "$ROOT/forge-gui-desktop" && "$JAVA_BIN" -Xmx4g \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain collect \
    -d 'Green Stompy.dck' -d 'White Weenie.dck' -d 'Red Aggro.dck' \
    -n "$GAMES" -t 4 -q \
    -o "$OUT_DIR/trajectories" ) > "$OUT_DIR/logs/collect.log" 2>&1
collected=$(ls "$OUT_DIR/trajectories"/traj_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "  collected $collected trajectory files"
[ "$collected" -gt 0 ] || { echo "no trajectories produced — see $OUT_DIR/logs/collect.log" >&2; exit 1; }

step "Preprocess trajectories → mmap arrays"
( cd "$PY_DIR" && "$VENV_DIR/bin/python" training/preprocess_trajectories.py \
    --data-dir "$OUT_DIR/trajectories" \
    --output-dir "$OUT_DIR/preprocessed" ) > "$OUT_DIR/logs/preprocess.log" 2>&1
[ -f "$OUT_DIR/preprocessed/metadata.json" ] || {
    echo "preprocessing produced no metadata — see $OUT_DIR/logs/preprocess.log" >&2
    exit 1
}

step "Train value head ($EPOCHS epochs)"
( cd "$PY_DIR" && "$VENV_DIR/bin/python" training/train_value.py \
    --data-dir "$OUT_DIR/preprocessed" \
    --save-dir "$OUT_DIR/checkpoints" \
    --log-dir "$OUT_DIR/logs" \
    --device cpu \
    --epochs "$EPOCHS" --batch-size 64 --val-split 0.1 \
    --save-every "$EPOCHS" \
    --early-stop-train-acc 0.99 --early-stop-patience 2 ) | tee "$OUT_DIR/logs/train.log"

step "Verdict"
# Extract first and last epoch's training loss from the table rows.
read -r first_loss last_loss < <(
    awk -F'│' '
        /^[[:space:]]+[0-9]+\/[0-9]+[[:space:]]*$/ { next }
        /^[[:space:]]+[0-9]+\/[0-9]+[[:space:]]*│/ {
            gsub(/[[:space:]]/, "", $2)
            if (first == "") first = $2
            last = $2
        }
        END { print first, last }
    ' "$OUT_DIR/logs/train.log"
)
echo "  first epoch train loss: ${first_loss:-?}"
echo "  last epoch  train loss: ${last_loss:-?}"
if [ -z "${first_loss:-}" ] || [ -z "${last_loss:-}" ]; then
    echo "  could not parse training loss — check $OUT_DIR/logs/train.log" >&2
    exit 1
fi
# Require at least a 5x drop in train loss to call it a pass.
awk -v a="$first_loss" -v b="$last_loss" 'BEGIN { exit !(a/b >= 5) }' || {
    echo "  FAIL: train loss did not drop ≥5x (${first_loss} → ${last_loss})" >&2
    exit 1
}
echo "  PASS: train loss dropped $(awk -v a="$first_loss" -v b="$last_loss" 'BEGIN{printf "%.1f", a/b}')x"
