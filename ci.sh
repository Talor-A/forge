#!/usr/bin/env bash
# ci.sh — sanity check that the Forge codebase builds and the Python RL bridge is importable.
# Usage: ./ci.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$ROOT/forge-ai-rl"
VENV_DIR="$RL_DIR/venv"
PY_DIR="$RL_DIR/src/main/python"

step() { printf '\n=== %s ===\n' "$1"; }

step "Maven build (forge-gui-desktop + deps)"
cd "$ROOT"
mvn package \
    -pl forge-gui-desktop -am \
    -Denforcer.skip=true \
    -Dcheckstyle.skip=true \
    -q

step "Python venv"
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

step "Python bridge import smoke test"
PYTHONPATH="$PY_DIR" python3 - <<'PY'
import torch
from model.mtg_model import MTGModel
from serving.model_server import ModelServer

model = MTGModel()
server = ModelServer(model, host="127.0.0.1", port=0, device="cpu")
print(f"torch {torch.__version__} | MTGModel + ModelServer imported OK")
PY

step "Python unit tests"
PYTHONPATH="$PY_DIR" python3 -m unittest discover \
    -s "$RL_DIR/src/test/python" -p 'test_*.py' -v

step "CI OK"
