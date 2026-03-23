# Forge RL AI — Development Notes

## Project Structure

- `forge-ai-rl/` — RL AI module (Java + Python)
- `forge-gui-desktop/` — Desktop app, includes `SimulateRLTraining.java` headless runner
- `rl_data/` — Training data, checkpoints, deck files (not committed)
- `RLAI_PLAN.md` — Full architecture plan

## Running Headless Games

### CWD must be `forge-gui-desktop/`
The game engine resolves `res/` relative to CWD. A symlink `forge-gui-desktop/res -> ../forge-gui/res` is required for the git checkout layout (because `BuildInfo.getVersionString()` doesn't contain "git", so `getAssetsDir()` returns `""` not `"../forge-gui/"`).

### Always use 16 threads for data collection
```bash
cd forge-gui-desktop
java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar \
    rltrain collect \
    -d "Green Stompy.dck" -d "White Weenie.dck" -d "Blue Tempo.dck" -d "Red Aggro.dck" \
    -n 1000 -t 16 \
    -o /path/to/trajectories -q
```

### Build command (from project root)
```bash
cd /home/maustin/forge
mvn package -pl forge-gui-desktop -am -Denforcer.skip=true -Dcheckstyle.skip=true -DskipTests -q
```

## Critical Gotchas

### DO NOT subclass LobbyPlayerAi
Subclassing `LobbyPlayerAi` causes instant turn-0 game termination. The game engine likely does identity checks (`instanceof` or class equality) that fail for subclasses. Use plain `LobbyPlayerAi` and attach recording externally.

### DO NOT add fields to PlayerControllerAi
Adding a `decisionListener` field to `PlayerControllerAi` breaks the fat jar. The modified `forge-ai` class conflicts with `forge-game` expectations at runtime, causing silent turn-0 failures. All recording must be done via the EventBus (`game.subscribeToEvents()`).

### DO NOT use DataLoader num_workers > 0 with large in-memory datasets
Python's fork-based DataLoader workers duplicate the entire process memory. With 3-5GB of trajectory data loaded, each worker adds another 5GB. This causes swap thrashing and apparent hangs after epoch 1. Always use `num_workers=0`.

### Localizer Java 21 fix
`ResourceBundle.getBundle()` fails in Java 21 with the original fallback code `new Locale("en_US")`. Fixed in `Localizer.java` to use `PropertyResourceBundle` direct file loading as last resort.

### Deck files must be in Forge's constructed directory
Copy `.dck` files to `~/.forge/decks/constructed/` or use absolute paths. The `sim` command prepends the constructed deck directory path.

## Training

### Python venv
```bash
source forge-ai-rl/venv/bin/activate
```

### Training UI (Tkinter dashboard with live charts)
```bash
cd forge-ai-rl/src/main/python
python training/training_ui.py --data-dir /path/to/trajectories --device cuda --epochs 100
```

### GPU: RTX 3080 (10GB VRAM)
- Model: 11M params, 42MB, uses ~51MB VRAM for inference
- Training batch 64 with AMP: ~0.4GB VRAM
- num_workers=0 mandatory (see gotcha above)

## Trajectory Data Format

JSONL files, one per player per game. Header line + decision records:
```json
{"gameId":"...","won":true,"totalDecisions":15,"durationMs":3500}
{"decisionType":"DECLARE_ATTACKERS","contextInfo":"attack_2_of_5","candidateFeatures":[[...256 floats...],[...]],"selectedIndices":[0,2],"globalFeatures":[...96 floats...],"gameStateFlat":[...37216 floats...]}
```

Decision types with action data:
- `DECLARE_ATTACKERS` — candidateFeatures = all creatures, selectedIndices = which attacked
- `DECLARE_BLOCKERS` — candidateFeatures = all creatures, selectedIndices = which blocked
- `PRIORITY_ACTION` — spell/land played (contextInfo has card name)
- Phase snapshots — `main1_turn_N` state snapshots

## Back up trajectory data before regenerating
Previous versions stored in `rl_data/trajectories_v1_endgame/` and `rl_data/trajectories_v2_midgame/`.
