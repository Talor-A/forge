#!/usr/bin/env bash
# Stop a RunPod pod (releases CPU/RAM, keeps disk + volume).
# Tomorrow: runpodctl pod start <pod-id> to resume with files intact.
#
# Usage:
#   ./scripts/runpod_stop.sh                # auto-detect single running pod
#   ./scripts/runpod_stop.sh <pod-id>       # explicit pod
#   ./scripts/runpod_stop.sh -y             # skip confirmation

set -euo pipefail

log()  { printf '\n\033[1;36m[stop]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[stop WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[stop FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v runpodctl >/dev/null || die "runpodctl not on PATH"
command -v jq >/dev/null         || die "jq not on PATH"

YES=0
DRY_RUN=0
POD_ID=""
for arg in "$@"; do
    case "$arg" in
        -y|--yes) YES=1 ;;
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) POD_ID="$arg" ;;
    esac
done

if [ -z "$POD_ID" ]; then
    POD_ID=$(runpodctl pod list 2>/dev/null \
        | jq -r '[.[] | select(.desiredStatus=="RUNNING")] | if length==1 then .[0].id else empty end')
    [ -n "$POD_ID" ] || { runpodctl pod list; die "specify pod id explicitly"; }
fi

# Show current state + cost telemetry
DETAIL=$(runpodctl pod get "$POD_ID")
NAME=$(echo "$DETAIL"   | jq -r '.name')
COST=$(echo "$DETAIL"   | jq -r '.costPerHr')
UPTIME=$(echo "$DETAIL" | jq -r '.uptimeSeconds')
HRS=$(awk -v s="$UPTIME" 'BEGIN{printf "%.2f", s/3600}')
SPENT=$(awk -v c="$COST" -v h="$HRS" 'BEGIN{printf "%.2f", c*h}')

log "Pod: $NAME ($POD_ID)"
echo "  uptime:   ${HRS}h"
echo "  cost/hr:  \$$COST"
echo "  spent:    \$$SPENT this session"

# Sanity-check: warn if anything heavy is running
log "Checking for in-flight work"
SSH_INFO=$(runpodctl ssh info "$POD_ID")
IP=$(echo "$SSH_INFO"   | jq -r '.ip')
PORT=$(echo "$SSH_INFO" | jq -r '.port')
KEY=$(echo "$SSH_INFO"  | jq -r '.ssh_key.path')
if [ "$IP" != "null" ] && [ "$PORT" != "null" ]; then
    BUSY=$(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new \
                -o ConnectTimeout=5 root@"$IP" \
                'pgrep -af "java|mvn|git|python" | head -10' 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        warn "Active processes on the pod:"
        echo "$BUSY" | sed 's/^/    /'
        warn "Stopping now will SIGTERM these. Data on /workspace + container disk is preserved."
    else
        log "  no notable processes running"
    fi
else
    warn "could not reach pod over ssh; skipping busy-check"
fi

# Confirm
if [ "$YES" != "1" ]; then
    read -r -p $'\n\033[1;33mStop the pod now? [y/N] \033[0m' ans
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { log "aborted"; exit 0; }
fi

if [ "$DRY_RUN" = "1" ]; then
    log "DRY RUN — would stop $POD_ID"
    echo "  rerun without --dry-run to actually stop"
    exit 0
fi

log "Stopping $POD_ID"
runpodctl pod stop "$POD_ID"

log "Done. To resume tomorrow:"
echo "  runpodctl pod start $POD_ID"
echo "  ./scripts/runpod_setup_from_mac.sh   # idempotent re-sync"
