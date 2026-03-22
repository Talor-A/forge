# Reinforcement Learning AI for Forge MTG — Architecture Plan

## Codebase Summary

The Forge codebase is well-structured for this project:

- **`PlayerController`** is the clean abstraction layer — 72+ abstract methods define every decision point (mulligan, cast spell, choose targets, declare attackers/blockers, pay costs, etc.)
- **`PlayerControllerAi`** implements all of these with heuristics via `AiController`
- **`GameSimulator`** already copies full game state for lookahead
- **32,300+ cards** implemented via a text-based scripting DSL
- **204 effect types**, **152 AI ability handlers**, **47 cost types**
- The game is fully deterministic given decisions — perfect for RL

The key insight: **we don't need to touch the game engine at all**. We create a new `PlayerControllerRL` that implements the same interface, and the game engine treats it like any other player.

---

## The Core Challenge

MTG is uniquely hard for RL because:

1. **Massive action space** — at any priority window, dozens of spells/abilities may be legal, each with different targets, modes, costs
2. **Variable-length games** — 5 to 50+ turns with thousands of micro-decisions
3. **Hidden information** — opponent's hand, library order
4. **Enormous card variety** — 32,300 cards with unique rules text
5. **Combinatorial explosion** — card interactions create emergent complexity
6. **Sparse rewards** — you only win or lose, many decisions earlier are what mattered

A single monolithic neural network won't work. We need a **hierarchical, modular architecture**.

---

## Proposed Architecture: Hierarchical RL with Specialized Decision Models

### Layer 1: Game State Encoder (shared foundation)

A single neural network that converts the raw game state into a dense vector representation. All decision models consume this.

**Inputs** (encoded as feature vectors):
- **Board state**: each permanent's power/toughness/keywords/types/tapped status/counters/attachments
- **Hand**: cards in hand with mana costs, types, key abilities
- **Life totals**: both players
- **Mana available**: lands untapped, mana pool contents
- **Graveyard/exile**: key cards and counts
- **Stack**: spells/abilities currently resolving
- **Phase/turn**: current phase, turn number, who's active
- **Combat state**: declared attackers/blockers if in combat
- **Game metadata**: cards in library (count), cards in opponent's hand (count)

**Architecture**: Transformer-based encoder with attention over sets of cards (since board state is a variable-size set of objects). Similar to how DeepMind handles StarCraft units.

**Output**: Fixed-size game state embedding (e.g., 512-1024 dimensions)

### Layer 2: Strategic Value Network

A network that evaluates "how good is this game state for me?" — analogous to the existing `GameStateEvaluator` but learned.

- Takes game state embedding → outputs win probability estimate
- Trained from game outcomes (Monte Carlo returns)
- Used by all decision models as a baseline/critic
- Also used for MCTS-style lookahead when time permits

### Layer 3: Specialized Decision Heads

Rather than one model for all decisions, we use **specialized heads** for each major decision category. Each head shares the game state encoder but has its own policy network.

#### Decision Head 1: **Priority Action Selection** (most critical)
- **When**: Every time the player gets priority
- **Decides**: Play a spell/ability from the available options, or pass priority
- **Architecture**:
  - Encode each available action (spell/ability) as a feature vector
  - Cross-attention between game state and available actions
  - Output: probability distribution over actions + "pass" option
- **Key features per action**: mana cost, card type, effect type (from ApiType), target requirements, whether it's a creature/removal/draw/etc.

#### Decision Head 2: **Target Selection**
- **When**: A spell/ability requires choosing targets
- **Decides**: Which legal target(s) to select
- **Architecture**:
  - Encode each legal target (card or player) as a feature vector
  - Pointer network that selects from the candidate set
  - Handles multi-target by iterative selection

#### Decision Head 3: **Combat — Attackers**
- **When**: Declare attackers step
- **Decides**: Which creatures attack, and which opponent/planeswalker they attack
- **Architecture**:
  - For each creature: binary attack/don't-attack decision
  - Encode creatures as feature vectors with combat-relevant stats
  - Joint decision (not independent per creature — attacking patterns matter)
  - Use combinatorial action representation

#### Decision Head 4: **Combat — Blockers**
- **When**: Declare blockers step
- **Decides**: Which creatures block which attackers
- **Architecture**:
  - Assignment problem: each blocker maps to zero or one attacker
  - Attention-based matching network
  - Must respect blocking restrictions/requirements

#### Decision Head 5: **Card Selection** (general purpose)
- **When**: Choose cards for effects (discard, sacrifice, scry, etc.)
- **Decides**: Which card(s) to select from a set
- **Architecture**:
  - Pointer network over candidate cards
  - Context includes the reason for selection (effect type)
  - Handles variable min/max selection counts

#### Decision Head 6: **Mulligan & Game Start**
- **When**: Pre-game mulligan decisions
- **Decides**: Keep or mulligan, which cards to bottom (London mulligan)
- **Architecture**:
  - Hand evaluation network
  - Trained on correlation between opening hands and win rates

#### Decision Head 7: **Binary/Numeric Choices**
- **When**: confirmAction, chooseNumber, chooseBinary, etc.
- **Decides**: Yes/no or numeric value
- **Architecture**: Simple MLP head on game state embedding + context

---

## Module Structure (new `forge-ai-rl` module)

```
forge-ai-rl/
├── src/main/java/forge/ai/rl/
│   ├── PlayerControllerRL.java          # Implements PlayerController
│   ├── RLController.java                # Orchestrates decision heads
│   ├── GameStateEncoder.java            # Converts game state → feature vectors
│   ├── ActionEncoder.java               # Encodes available actions
│   ├── CardEncoder.java                 # Encodes individual cards
│   │
│   ├── decisions/                        # Decision head interfaces
│   │   ├── PriorityDecision.java
│   │   ├── TargetDecision.java
│   │   ├── AttackDecision.java
│   │   ├── BlockDecision.java
│   │   ├── CardSelectDecision.java
│   │   ├── MulliganDecision.java
│   │   └── BinaryDecision.java
│   │
│   ├── model/                            # Neural network integration
│   │   ├── ModelServer.java             # gRPC/socket client to Python model server
│   │   ├── InferenceRequest.java
│   │   ├── InferenceResponse.java
│   │   └── ModelConfig.java
│   │
│   ├── training/                         # Training infrastructure
│   │   ├── GameRunner.java              # Runs AI vs AI games at scale
│   │   ├── ExperienceBuffer.java        # Stores game trajectories
│   │   ├── TrajectoryRecorder.java      # Records state-action-reward tuples
│   │   ├── RewardShaper.java            # Intermediate reward signals
│   │   └── SelfPlayManager.java         # Manages self-play population
│   │
│   └── features/                         # Feature extraction
│       ├── BoardFeatures.java
│       ├── CardFeatures.java
│       ├── CombatFeatures.java
│       ├── ManaFeatures.java
│       └── FeatureNormalizer.java
│
├── src/main/python/                      # Python ML side
│   ├── model/
│   │   ├── game_state_encoder.py        # Transformer encoder
│   │   ├── value_network.py             # Win probability estimator
│   │   ├── priority_head.py             # Action selection policy
│   │   ├── target_head.py               # Target selection policy
│   │   ├── combat_attack_head.py        # Attacker declaration policy
│   │   ├── combat_block_head.py         # Blocker declaration policy
│   │   ├── card_select_head.py          # Card selection policy
│   │   ├── mulligan_head.py             # Mulligan policy
│   │   └── binary_head.py              # Yes/no decisions
│   │
│   ├── training/
│   │   ├── trainer.py                   # PPO/IMPALA training loop
│   │   ├── self_play.py                 # Self-play orchestration
│   │   ├── curriculum.py                # Curriculum learning scheduler
│   │   ├── elo_tracker.py               # Track model strength over time
│   │   └── replay_buffer.py             # Prioritized experience replay
│   │
│   ├── serving/
│   │   ├── model_server.py              # gRPC server for inference
│   │   └── batch_inference.py           # Batch requests for throughput
│   │
│   └── evaluation/
│       ├── benchmark.py                 # Evaluate vs heuristic AI
│       ├── card_understanding.py        # Test card-specific knowledge
│       └── visualize.py                 # Training curves, attention maps
```

---

## Training Strategy

### Phase 1: Bootstrap — Learn from the Heuristic AI (Imitation Learning)

Before any RL, we **imitate the existing heuristic AI** to get a reasonable starting policy.

1. Run thousands of heuristic AI vs AI games
2. Record every decision point: (game_state, available_actions, chosen_action)
3. Train each decision head via supervised learning to predict the heuristic AI's choices
4. This gives us a "warm start" — a policy that's roughly as good as the heuristic AI

**Why**: Starting RL from scratch with random play would take astronomically long. The heuristic AI is already decent — we start there and improve.

### Phase 2: Curriculum Learning — Start Simple

Don't train on all 32,300 cards at once. Use a curriculum:

1. **Stage A**: Vanilla creatures only (no abilities). Learn combat math, life total management, mana curve
2. **Stage B**: Add common keywords (flying, trample, first strike, deathtouch, lifelink)
3. **Stage C**: Add removal spells, combat tricks
4. **Stage D**: Add card draw, counterspells, stack interaction
5. **Stage E**: Add planeswalkers, enchantments, complex abilities
6. **Stage F**: Full card pool (or competitive format subsets like Standard/Modern)

At each stage, cards are restricted to a curated pool. The AI learns fundamentals before complexity.

### Phase 3: Self-Play RL (PPO + Population-Based Training)

Once bootstrapped, the RL agent plays against itself and a pool of opponents:

**Algorithm**: Proximal Policy Optimization (PPO) with:
- **Population-based training**: Maintain a population of ~10-20 agents with different hyperparameters
- **League training** (AlphaStar-style):
  - "Main agents" that train against all opponents
  - "Exploiter agents" that specifically target weaknesses in main agents
  - "League exploiters" that target the full history of agents
- **Historical snapshots**: Keep checkpoints of past agents as training opponents to prevent forgetting

**Reward Shaping** (critical for sparse-reward games):
- +1.0 for winning, -1.0 for losing (terminal reward)
- Small intermediate rewards to guide learning:
  - +0.01 per point of life advantage gained
  - +0.05 per card advantage gained
  - +0.02 per creature advantage on board
  - -0.1 for illegal action attempts (shouldn't happen but safety net)
  - Rewards decay over training — eventually rely only on win/loss

**Discount factor**: γ = 0.999 (very long horizon — early decisions matter for late game outcomes)

### Phase 4: Targeted Training Against Heuristic AI

Periodically pit the RL agent against the heuristic AI to:
- Measure improvement (Elo tracking)
- Identify specific weaknesses (does it lose to aggro? control? combo?)
- Generate hard examples for focused training

---

## Java-Python Bridge

The game engine runs in Java; the neural networks run in Python. We need efficient communication.

### Option A: gRPC Service (recommended for training)
- Python model server exposes gRPC endpoints for each decision type
- Java client sends encoded game state, receives action
- Supports batching for throughput during training
- ~1-5ms latency per decision (acceptable for AI vs AI)

### Option B: ONNX Runtime (for deployment)
- Export trained PyTorch models to ONNX format
- Load ONNX models directly in Java via ONNX Runtime
- Zero inter-process communication overhead
- Use this for the final deployed model (no Python dependency)

### Recommended approach: gRPC during training, ONNX for deployment.

---

## Card Representation

Each card is represented as:

1. **Structural features** (fixed, interpretable):
   - Mana cost (CMC + color breakdown)
   - Card types (creature, instant, sorcery, enchantment, artifact, planeswalker, land)
   - Subtypes (encoded categorically)
   - Power/toughness (for creatures)
   - Keywords (binary vector for ~100 common keywords)
   - Zone it's currently in
   - Tapped/untapped, summoning sick, counters

2. **Ability features** (extracted from card script):
   - ApiType of each ability (DealDamage, Draw, Counter, ChangeZone, etc.)
   - Target types
   - Numeric parameters (damage amount, cards drawn, etc.)
   - Cost to activate

3. **Learned embedding** (trainable):
   - Each unique card gets a trainable embedding vector (like word embeddings)
   - Initialized randomly, learned during training
   - Captures subtle card interactions that aren't in structural features

4. **Card text embedding** (optional, advanced):
   - Encode Oracle text with a small language model
   - Helps generalize to unseen cards

The final card representation is the **concatenation of all four**, projected to a fixed size.

---

## Handling MTG Complexity

### Stack Interactions
- When the RL agent has priority with spells on the stack, the stack contents are encoded as part of game state
- The priority decision head sees what's on the stack and can choose to respond or pass
- Train specifically on counter-spell scenarios

### Hidden Information
- The RL agent only sees what a legal player would see (no peeking at opponent's hand/library)
- Use the game's existing visibility rules
- The value network must learn to estimate under uncertainty

### Variable Action Spaces
- Each decision presents a different set of legal actions
- Pointer networks / attention over action sets handle this naturally
- "Pass" is always an option during priority

### Multi-step Decisions
- Some effects require sequences of choices (e.g., cast a spell → choose targets → choose modes → pay costs)
- Model as a sequence of decision head invocations, each conditioned on previous choices

---

## Implementation Phases

### Phase 1: Infrastructure (weeks 1-4) ✅ COMPLETE
1. ✅ Create `forge-ai-rl` Maven module
2. ✅ Implement `PlayerControllerRL` — extends PlayerControllerAi, overrides combat + priority decisions
3. ✅ Build `GameStateEncoder` — 37,216-float game state (96 global + 145×256 card features)
4. ✅ Build `CardFeatures` (256-dim) + `ActionEncoder` (64-dim) — card and spell representations
5. ✅ Build `TrajectoryRecorder` — JSONL trajectory files with full state + action features
6. ✅ Build `SimulateRLTraining` — headless parallel game runner (16 threads, 1.3 games/sec)
7. ✅ JSON-over-TCP bridge with batched inference server
8. ✅ Python project with PyTorch, MTGModel (11M params), training dashboards

### Phase 2: Imitation Learning (weeks 5-8) 🔄 IN PROGRESS
1. ✅ Run 1,000 heuristic AI vs AI games, recording all decisions (~137K records)
2. ✅ Train game state encoder + value network (stopped early, overfitting observed)
   - *Note: Previous 99.6% accuracy was from a version with feature leakage (tapped flag leaked attack decisions). Not comparable to current run.*
3. 🔄 Train decision heads with 256-dim card features (training in progress)
   - Priority: 126,172 samples (CE softmax) — TBD accuracy
   - Attack: 8,576 samples (BCE per-creature) — TBD accuracy
   - Block: 2,478 samples (CE per-blocker assignment) — TBD accuracy
4. ⬜ Validate RL agent win rate vs heuristic
   - *Note: Previous 53% win rate was from 128-dim leaked-feature model. Not comparable.*

### Phase 3: RL Training — Simple Cards (weeks 9-14) ⬜ NOT STARTED (with 256-dim model)
1. ⬜ Curriculum card pools (currently using 4 aggro decks)
2. ✅ PPO training loop implemented (ppo_trainer.py + ppo_ui.py)
3. ✅ Reward shaping implemented (life/card/board advantage signals with decay)
4. ⬜ PPO training with 256-dim model (after decision heads converge)
5. ⬜ Progress through curriculum stages
6. ⬜ Elo tracking and benchmarking

### Phase 4: RL Training — Full Complexity (weeks 15-22)
1. Train on full card pools (Standard, Modern, or curated sets)
2. Implement league training with exploiter agents
3. Implement population-based training for hyperparameter search
4. Run large-scale self-play (millions of games)
5. Regular benchmarking vs heuristic AI at different difficulty levels

### Phase 5: Deployment & Polish (weeks 23-26)
1. Export best models to ONNX for Java-native inference
2. Integrate into Forge as a selectable AI opponent ("RL AI" option)
3. Add difficulty scaling (use earlier checkpoints = easier, latest = hardest)
4. Performance optimization for acceptable game speed
5. Add AI personality profiles that bias the RL agent's style

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Game simulation too slow for millions of training games | Profile and optimize headless game execution; strip UI/logging; parallelize across cores |
| Action space too large for effective learning | Hierarchical decisions + curriculum learning + imitation pre-training |
| Card interactions create long-tail edge cases | Focus on competitive format card pools (Standard ~2000 cards); expand gradually |
| gRPC latency slows training | Batch inference; async game execution; eventually ONNX in-process |
| RL agent learns degenerate strategies | League training with exploiters; periodic evaluation on diverse matchups |
| New card sets invalidate training | Card embedding approach generalizes; fine-tune on new sets; structural features transfer |

## Success Metrics

1. **Phase 2 target**: RL agent (imitation) wins 45-55% vs heuristic AI (parity)
2. **Phase 3 target**: RL agent wins 60%+ vs heuristic AI on simple card pools
3. **Phase 4 target**: RL agent wins 65%+ vs heuristic AI on full card pools
4. **Stretch goal**: RL agent discovers non-obvious strategies that surprise experienced players


claude --resume 97af074f-b317-4e20-b335-03a1e87a07ae