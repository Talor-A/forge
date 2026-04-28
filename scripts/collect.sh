#!/usr/bin/env bash
# Phase 1 — heuristic-vs-heuristic trajectory collection.
# Run on the pod, after runpod_bootstrap.sh has succeeded.
#
# Usage:
#   ./scripts/collect.sh -n 10000 -t 16 -j 4 -o /workspace/trajectories_v3
#   ./scripts/collect.sh --smoke
#
# Flags:
#   -n GAMES        total games across all JVMs (default 1000)
#   -t THREADS      threads per JVM (default 16)
#   -j JVMS         parallel JVMs (default 1)
#   -o OUTPUT_DIR   root output dir on /workspace (required unless --smoke)
#   --smoke         shorthand for -n 100 -j 1 -o /workspace/trajectories_smoke
#   --force         skip the tmux warning
#
# Decks (hardcoded for now):
#   White Weenie, Red Aggro, Green Stompy

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/forge-ai-investigation}"
JVM_XMX_MB="${JVM_XMX_MB:-8192}"
DECKS=("White Weenie.dck" "Red Aggro.dck" "Green Stompy.dck")

GAMES=1000
THREADS=16
JVMS=1
OUTPUT_DIR=""
SMOKE=0
FORCE=0

log()  { printf '\n\033[1;36m[collect]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[collect WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[collect FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ─── Parse args ──────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        -n) GAMES="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -j|--jvms) JVMS="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
done

if [ "$SMOKE" = "1" ]; then
    GAMES=100
    JVMS=1
    OUTPUT_DIR="/workspace/trajectories_smoke"
fi

[ -n "$OUTPUT_DIR" ] || die "-o OUTPUT_DIR required (or use --smoke)"

# ─── Pre-flight: repo & jar ──────────────────────────────────────────
log "Pre-flight"
DESKTOP_DIR="$REPO_DIR/forge-gui-desktop"
[ -d "$DESKTOP_DIR" ] || die "$DESKTOP_DIR missing — run runpod_bootstrap.sh"

JAR=$(ls "$DESKTOP_DIR"/target/forge-gui-desktop-*-jar-with-dependencies.jar 2>/dev/null | head -1 || true)
[ -n "$JAR" ] || die "fat jar missing in $DESKTOP_DIR/target — run runpod_bootstrap.sh"
echo "  jar: $JAR"

# res symlink
RES="$DESKTOP_DIR/res"
[ -L "$RES" ] || die "$RES is not a symlink — bootstrap should have created it"
[ -e "$RES" ] || die "$RES symlink target broken — bootstrap is incomplete"
echo "  res: $(readlink "$RES")"

# Java 21
JAVA_VER=$(java -version 2>&1 | head -1)
echo "  java: $JAVA_VER"
echo "$JAVA_VER" | grep -q '"21' || die "Java 21 required, got: $JAVA_VER"

# Decks
DECK_DIR="$HOME/.forge/decks/constructed"
for d in "${DECKS[@]}"; do
    [ -f "$DECK_DIR/$d" ] || die "missing deck: $DECK_DIR/$d (rerun runpod_bootstrap.sh)"
done
echo "  decks: ${DECKS[*]}"

# Output dir on /workspace
case "$OUTPUT_DIR" in
    /workspace/*) ;;
    *) die "OUTPUT_DIR must be under /workspace (got $OUTPUT_DIR)" ;;
esac

# Disk free vs estimate (~150 KB/game heuristic from existing data)
EST_KB=$(( GAMES * 150 ))
FREE_KB=$(df -k /workspace | awk 'NR==2 {print $4}')
echo "  disk: free=${FREE_KB}KB  estimate=${EST_KB}KB"
if [ "$EST_KB" -gt 0 ] && [ "$FREE_KB" -lt $((EST_KB * 2)) ]; then
    die "low disk: free ${FREE_KB}KB < 2× estimate ${EST_KB}KB on /workspace"
fi

# Memory budget: jvms × Xmx ≤ 80% of MemTotal
MEM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
NEEDED_KB=$(( JVMS * JVM_XMX_MB * 1024 ))
LIMIT_KB=$(( MEM_TOTAL_KB * 80 / 100 ))
echo "  mem:  total=${MEM_TOTAL_KB}KB  needed=${NEEDED_KB}KB  limit=${LIMIT_KB}KB"
if [ "$NEEDED_KB" -gt "$LIMIT_KB" ]; then
    die "memory: $JVMS × ${JVM_XMX_MB}MB exceeds 80% of system RAM"
fi

# Competing JVMs
if pgrep -f forge-gui-desktop >/dev/null; then
    warn "another forge-gui-desktop process is already running"
fi

# tmux
if [ -z "${TMUX:-}" ] && [ "$GAMES" -gt 1000 ] && [ "$FORCE" != "1" ]; then
    die "you are not inside tmux and GAMES=$GAMES is large.
       wrap this with:  tmux new -s collect -d '$0 $*'
       or rerun with --force to ignore."
fi

# ─── Plan ────────────────────────────────────────────────────────────
PER_JVM=$(( (GAMES + JVMS - 1) / JVMS ))   # ceil-divide
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

cat > "$OUTPUT_DIR/run_metadata.json" <<EOF
{
  "started_at":   "$TIMESTAMP",
  "games_total":  $GAMES,
  "games_per_jvm": $PER_JVM,
  "jvms":         $JVMS,
  "threads":      $THREADS,
  "decks":        $(printf '%s\n' "${DECKS[@]}" | jq -R . | jq -s .),
  "git_sha":      "$(git -C "$REPO_DIR" rev-parse HEAD)",
  "jar":          "$JAR"
}
EOF
log "Wrote $OUTPUT_DIR/run_metadata.json"

# ─── Build deck args once ────────────────────────────────────────────
DECK_ARGS=()
for d in "${DECKS[@]}"; do DECK_ARGS+=(-d "$d"); done

# ─── Launch JVMs ─────────────────────────────────────────────────────
cd "$DESKTOP_DIR"

PIDS=()
for i in $(seq 0 $((JVMS - 1))); do
    OUT_I="$OUTPUT_DIR/jvm_$i"
    LOG_I="$LOG_DIR/collect_${TIMESTAMP}_jvm${i}.log"
    mkdir -p "$OUT_I"
    log "JVM $i  -> -n $PER_JVM  out=$OUT_I  log=$LOG_I"

    java -Xmx${JVM_XMX_MB}m \
        --add-opens java.base/java.lang=ALL-UNNAMED \
        --add-opens java.base/java.util=ALL-UNNAMED \
        --add-opens java.base/java.text=ALL-UNNAMED \
        --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
        --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
        -jar "$JAR" \
        rltrain collect \
        "${DECK_ARGS[@]}" \
        -n "$PER_JVM" -t "$THREADS" \
        -o "$OUT_I" -q \
        > "$LOG_I" 2>&1 &
    PIDS+=($!)
done

log "Launched ${#PIDS[@]} JVMs (pids: ${PIDS[*]}). Waiting…"
START_S=$(date +%s)
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        warn "JVM pid $pid exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done
END_S=$(date +%s)
ELAPSED=$((END_S - START_S))

# ─── Recombine per-JVM subdirs into flat OUTPUT_DIR ─────────────────
# preprocess_trajectories.py uses a non-recursive glob (traj_*.jsonl),
# so flatten now. Prefix filenames with jvm<i>_ to avoid collisions
# (gameId counter is per-JVM, timestamps can repeat).
log "Flattening jvm_*/ into $OUTPUT_DIR"
for i in $(seq 0 $((JVMS - 1))); do
    sub="$OUTPUT_DIR/jvm_$i"
    [ -d "$sub" ] || continue
    for f in "$sub"/traj_*.jsonl; do
        [ -e "$f" ] || break
        base=$(basename "$f")          # traj_<rest>.jsonl
        rest="${base#traj_}"           # <rest>.jsonl
        mv "$f" "$OUTPUT_DIR/traj_jvm${i}_${rest}"
    done
    rmdir "$sub" 2>/dev/null || warn "  $sub not empty, leaving in place"
done

# ─── Summary ─────────────────────────────────────────────────────────
PRODUCED=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'traj_*.jsonl' | wc -l | tr -d ' ')
TOTAL_KB=$(du -sk "$OUTPUT_DIR" | awk '{print $1}')
RATE="n/a"
[ "$ELAPSED" -gt 0 ] && RATE=$(awk -v p="$PRODUCED" -v e="$ELAPSED" 'BEGIN{printf "%.2f", p/e}')

log "Done in ${ELAPSED}s.  files=$PRODUCED  size=${TOTAL_KB}KB  rate=${RATE} files/s"
[ "$FAIL" -eq 0 ] || warn "$FAIL JVM(s) failed; inspect logs in $LOG_DIR"

# Update metadata with end state
python3 - "$OUTPUT_DIR/run_metadata.json" "$ELAPSED" "$PRODUCED" "$TOTAL_KB" "$FAIL" <<'PY'
import json, sys
path, elapsed, produced, total_kb, fail = sys.argv[1:]
with open(path) as f: d = json.load(f)
d.update(elapsed_s=int(elapsed), produced_files=int(produced),
         total_kb=int(total_kb), failed_jvms=int(fail))
with open(path, "w") as f: json.dump(d, f, indent=2)
PY

cat <<EOF

Next:
  python3 $REPO_DIR/forge-ai-rl/src/main/python/training/preprocess_trajectories.py \\
      --data-dir $OUTPUT_DIR \\
      --output-dir ${OUTPUT_DIR%/*}/preprocessed_${OUTPUT_DIR##*/}

EOF

[ "$FAIL" -eq 0 ]
