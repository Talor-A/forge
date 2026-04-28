#!/usr/bin/env bash
# Run from your Mac. Resolves the RunPod SSH endpoint via runpodctl,
# uploads decks + the bootstrap script, then runs the bootstrap on the pod.
# Idempotent: re-running just re-syncs the decks/script and re-runs bootstrap
# (which is itself idempotent).
#
# Usage:
#   ./scripts/runpod_setup_from_mac.sh                # auto-detect single running pod
#   ./scripts/runpod_setup_from_mac.sh <pod-id>       # explicit pod
#
# Env overrides:
#   DECK_SRC   local deck dir (default: ~/.forge/decks/constructed)
#   FORCE_REBUILD=1   pass through to remote bootstrap

set -euo pipefail

DECK_SRC="${DECK_SRC:-$HOME/.forge/decks/constructed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP="$SCRIPT_DIR/runpod_bootstrap.sh"

log()  { printf '\n\033[1;36m[mac]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[mac WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[mac FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v runpodctl >/dev/null || die "runpodctl not on PATH"
command -v jq >/dev/null        || die "jq not on PATH (brew install jq)"
[ -f "$BOOTSTRAP" ]              || die "bootstrap script missing: $BOOTSTRAP"
[ -d "$DECK_SRC" ]                || die "deck dir not found: $DECK_SRC"

# ─── 1. Pick a pod ───────────────────────────────────────────────────
POD_ID="${1:-}"
if [ -z "$POD_ID" ]; then
    log "Looking for running pod"
    POD_ID=$(runpodctl pod list 2>/dev/null \
        | jq -r '[.[] | select(.desiredStatus=="RUNNING")] | if length==1 then .[0].id else empty end')
    if [ -z "$POD_ID" ]; then
        runpodctl pod list
        die "Specify pod id explicitly: $0 <pod-id>"
    fi
    log "Auto-selected pod: $POD_ID"
fi

# ─── 2. Resolve SSH endpoint ─────────────────────────────────────────
log "Resolving SSH info"
SSH_JSON=$(runpodctl ssh info "$POD_ID")
IP=$(echo "$SSH_JSON"   | jq -r '.ip')
PORT=$(echo "$SSH_JSON" | jq -r '.port')
KEY=$(echo "$SSH_JSON"  | jq -r '.ssh_key.path')
[ -n "$IP" ] && [ -n "$PORT" ] && [ -n "$KEY" ] || die "couldn't parse ssh info: $SSH_JSON"
log "  $IP:$PORT  key=$KEY"

SSH_OPTS=(-i "$KEY" -p "$PORT" \
          -o StrictHostKeyChecking=accept-new \
          -o ServerAliveInterval=30)
SSH="ssh ${SSH_OPTS[*]} root@$IP"
SCP_OPTS=(-i "$KEY" -P "$PORT" \
          -o StrictHostKeyChecking=accept-new)

# ─── 3. Ensure remote dirs ───────────────────────────────────────────
log "Creating /workspace/decks and /workspace/scripts on pod"
$SSH 'mkdir -p /workspace/decks /workspace/scripts'

# ─── 4. Upload decks ─────────────────────────────────────────────────
DECK_COUNT=$(find "$DECK_SRC" -maxdepth 1 -name '*.dck' | wc -l | tr -d ' ')
if [ "$DECK_COUNT" = "0" ]; then
    warn "No .dck files in $DECK_SRC — skipping deck upload"
else
    log "Uploading $DECK_COUNT decks from $DECK_SRC"
    scp "${SCP_OPTS[@]}" "$DECK_SRC"/*.dck "root@$IP:/workspace/decks/"
fi

# ─── 5. Upload bootstrap script ──────────────────────────────────────
log "Uploading bootstrap script"
scp "${SCP_OPTS[@]}" "$BOOTSTRAP" "root@$IP:/workspace/scripts/runpod_bootstrap.sh"
$SSH 'chmod +x /workspace/scripts/runpod_bootstrap.sh'

# ─── 6. Run bootstrap on pod ─────────────────────────────────────────
log "Running bootstrap on pod (this may take 5–10 min on first run)"
REMOTE_ENV=""
if [ "${FORCE_REBUILD:-0}" = "1" ]; then
    REMOTE_ENV="FORCE_REBUILD=1"
fi
$SSH "$REMOTE_ENV bash /workspace/scripts/runpod_bootstrap.sh"

log "Done."
echo
echo "SSH in with:"
echo "  ssh -i $KEY -p $PORT root@$IP"
