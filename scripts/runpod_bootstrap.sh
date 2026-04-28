#!/usr/bin/env bash
# RunPod bootstrap for forge-ai-investigation Phase 1 (data generation).
# Idempotent: safe to re-run. Each step skips if already done.
#
# Usage:
#   curl -sSL <raw-url-of-this-script> | bash
#   # or
#   bash runpod_bootstrap.sh
#
# Env overrides:
#   REPO_URL     git URL to clone (default: https://github.com/austinio7116/forge.git)
#   REPO_DIR     where to clone (default: /workspace/forge-ai-investigation)
#   FORGE_BRANCH branch to check out (default: leave HEAD alone)
#   FORCE_REBUILD=1  rebuild fat jar even if it exists

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/austinio7116/forge.git}"
REPO_DIR="${REPO_DIR:-/workspace/forge-ai-investigation}"
FORGE_BRANCH="${FORGE_BRANCH:-}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

DECK_DIR="$HOME/.forge/decks/constructed"
JAR_GLOB="$REPO_DIR/forge-gui-desktop/target/forge-gui-desktop-*-jar-with-dependencies.jar"

log()  { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[bootstrap WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[bootstrap FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ─── 1. System packages ──────────────────────────────────────────────
log "Step 1/6  System packages"
NEED_PKGS=()
for pkg in openjdk-21-jdk maven git python3-pip python3-venv htop tmux curl; do
    dpkg -s "$pkg" >/dev/null 2>&1 || NEED_PKGS+=("$pkg")
done
if [ ${#NEED_PKGS[@]} -gt 0 ]; then
    log "  installing: ${NEED_PKGS[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends "${NEED_PKGS[@]}"
else
    log "  all packages already present"
fi

java -version 2>&1 | head -1
mvn -v | head -1

# ─── 2. Persistent volume sanity ─────────────────────────────────────
log "Step 2/6  Persistent volume"
if [ ! -d /workspace ]; then
    die "/workspace does not exist — attach a network volume mounted at /workspace and rerun."
fi
df -h /workspace | tail -1

# ─── 3. Repo clone / update ──────────────────────────────────────────
log "Step 3/6  Repo at $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    log "  cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "  already cloned, fetching"
    git -C "$REPO_DIR" fetch --quiet
fi
if [ -n "$FORGE_BRANCH" ]; then
    git -C "$REPO_DIR" checkout "$FORGE_BRANCH"
fi
git -C "$REPO_DIR" log -1 --oneline

# ─── 4. res symlink (per CLAUDE.md gotcha) ───────────────────────────
log "Step 4/6  res symlink"
RES_LINK="$REPO_DIR/forge-gui-desktop/res"
RES_TARGET="../forge-gui/res"
if [ -L "$RES_LINK" ]; then
    log "  symlink exists -> $(readlink "$RES_LINK")"
elif [ -e "$RES_LINK" ]; then
    warn "  $RES_LINK exists but is not a symlink — leaving alone"
else
    ln -s "$RES_TARGET" "$RES_LINK"
    log "  created $RES_LINK -> $RES_TARGET"
fi

# ─── 5. Build fat jar ────────────────────────────────────────────────
log "Step 5/6  Maven fat jar"
EXISTING_JAR=$(ls $JAR_GLOB 2>/dev/null | head -1 || true)
if [ -n "$EXISTING_JAR" ] && [ "$FORCE_REBUILD" != "1" ]; then
    log "  jar exists: $EXISTING_JAR  (set FORCE_REBUILD=1 to rebuild)"
else
    log "  building (first run takes 5–10 min)"
    cd "$REPO_DIR"
    mvn package -pl forge-gui-desktop -am \
        -Denforcer.skip=true -Dcheckstyle.skip=true -DskipTests -q
    cd - >/dev/null
    EXISTING_JAR=$(ls $JAR_GLOB | head -1)
    log "  built: $EXISTING_JAR"
fi

# ─── 6. Decks ────────────────────────────────────────────────────────
log "Step 6/6  Decks at $DECK_DIR"
mkdir -p "$DECK_DIR"

# CLAUDE.md references these four deck names; they are not in the repo,
# so we look for them under a few likely locations and copy if found.
WANTED=("Green Stompy.dck" "White Weenie.dck" "Blue Tempo.dck" "Red Aggro.dck")
SEARCH_ROOTS=(
    "$REPO_DIR/rl_data/decks"
    "$REPO_DIR/forge-ai-rl/decks"
    "$REPO_DIR/decks"
    "$REPO_DIR"
)

found_any=0
for name in "${WANTED[@]}"; do
    if [ -f "$DECK_DIR/$name" ]; then
        log "  $name  (already in $DECK_DIR)"
        found_any=1
        continue
    fi
    src=""
    for root in "${SEARCH_ROOTS[@]}"; do
        [ -d "$root" ] || continue
        cand=$(find "$root" -maxdepth 5 -type f -name "$name" 2>/dev/null | head -1)
        if [ -n "$cand" ]; then src="$cand"; break; fi
    done
    if [ -n "$src" ]; then
        cp "$src" "$DECK_DIR/$name"
        log "  copied $name from $src"
        found_any=1
    else
        warn "  $name not found in repo — copy it to $DECK_DIR manually before running 'rltrain collect'"
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────
log "Bootstrap complete."
cat <<EOF

Next steps:

  1.  Smoke test (100 games, ~few minutes):
      tmux new -s smoke
      cd $REPO_DIR/forge-gui-desktop
      java -Xmx8192m \\
          --add-opens java.base/java.lang=ALL-UNNAMED \\
          --add-opens java.base/java.util=ALL-UNNAMED \\
          --add-opens java.base/java.text=ALL-UNNAMED \\
          --add-opens java.base/java.lang.reflect=ALL-UNNAMED \\
          --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \\
          -jar target/forge-gui-desktop-*-jar-with-dependencies.jar \\
          rltrain collect \\
          -d "Green Stompy.dck" -d "White Weenie.dck" \\
          -d "Blue Tempo.dck" -d "Red Aggro.dck" \\
          -n 100 -t 16 \\
          -o /workspace/trajectories_smoke -q

  2.  Full Phase 1: same command, larger -n, output dir like
      /workspace/trajectories_v3 (keep on the persistent volume).

  3.  Phase 2 (preprocess): run from forge-ai-rl/src/main/python:
      python training/preprocess_trajectories.py \\
          --data-dir /workspace/trajectories_v3 \\
          --output-dir /workspace/preprocessed_v3

EOF
