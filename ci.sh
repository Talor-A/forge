#!/usr/bin/env bash
# ci.sh — sanity check that the Forge codebase builds and the Python RL bridge is importable.
# Usage: ./ci.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$ROOT/forge-ai-rl"
VENV_DIR="$RL_DIR/venv"
PY_DIR="$RL_DIR/src/main/python"

run_step() {
    local name="$1"
    local fn="$2"
    printf '=== %s ... ' "$name"
    local log
    log=$(mktemp)
    if "$fn" >"$log" 2>&1; then
        printf 'OK\n'
        rm -f "$log"
    else
        local rc=$?
        printf 'FAILED (exit %d)\n' "$rc"
        cat "$log"
        rm -f "$log"
        exit "$rc"
    fi
}

step_maven() {
    cd "$ROOT"
    mvn package \
        -pl forge-gui-desktop -am \
        -Denforcer.skip=true \
        -Dcheckstyle.skip=true \
        -DskipTests=true \
        -q
}

step_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    # requirements.txt targets Linux+CUDA: pins --index-url to PyTorch CUDA wheels
    # and asks for onnxruntime-gpu. macOS has neither, so rewrite on Darwin to use
    # the default PyPI index and the CPU-flavored onnxruntime package.
    if [ "$(uname -s)" = "Darwin" ]; then
        sed -e '/^--index-url/d' -e 's/^onnxruntime-gpu/onnxruntime/' \
            "$PY_DIR/requirements.txt" | pip install --quiet -r /dev/stdin
    else
        pip install --quiet -r "$PY_DIR/requirements.txt"
    fi
}

step_bridge_smoke() {
    PYTHONPATH="$PY_DIR" python3 - <<'PY'
import torch
from model.mtg_model import MTGModel
from serving.model_server import ModelServer

model = MTGModel()
server = ModelServer(model, host="127.0.0.1", port=0, device="cpu")
print(f"torch {torch.__version__} | MTGModel + ModelServer imported OK")
PY
}

step_unit_tests() {
    PYTHONPATH="$PY_DIR" python3 -m unittest discover \
        -s "$RL_DIR/src/test/python" -p 'test_*.py' -v
}

run_step "Maven build (forge-gui-desktop + deps)" step_maven
run_step "Python venv"                            step_venv
run_step "Python bridge import smoke test"        step_bridge_smoke
run_step "Python unit tests"                      step_unit_tests

printf '=== CI OK ===\n'
