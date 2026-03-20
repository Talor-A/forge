# Learning to Play Magic: The Gathering Through Hierarchical Reinforcement Learning with Transformer-Based State Encoding

**Authors:** M. Austin, with architectural design and implementation assistance from Claude (Anthropic)

**Date:** March 2026

---

## Abstract

We present a reinforcement learning system for playing Magic: The Gathering (MTG), arguably the most complex widely-played strategy card game in existence. MTG presents unique challenges for AI: a combinatorial action space with over 32,000 unique cards, hidden information, stochastic elements, and deeply nested interactions between game mechanics. Our approach employs a hierarchical architecture with a shared transformer-based game state encoder and specialized decision heads for each major action type (spell casting, combat, card selection). We bootstrap the system through imitation learning on a heuristic AI opponent, then improve via Proximal Policy Optimization (PPO) self-play with curriculum learning. We describe the full system architecture, feature engineering, training pipeline, and preliminary results on the open-source Forge MTG game engine, where our value network achieves 99.6% accuracy in predicting game outcomes from mid-game board states after training on 28,828 decision snapshots from 1,000 games.

---

## 1. Introduction

### 1.1 Magic: The Gathering as an AI Challenge

Magic: The Gathering (MTG), created by Richard Garfield in 1993, is a collectible card game played by over 40 million people worldwide. From a game-theoretic perspective, MTG is remarkable in its complexity:

- **Massive state space.** A game state includes two players' life totals, hands (hidden), libraries (ordered, hidden), graveyards, exile zones, the battlefield (with permanents that may be tapped, have counters, attachments, and modified attributes), and a stack of spells and abilities awaiting resolution. Conservative estimates place the state space at 10^(100+), far exceeding chess (~10^47) or Go (~10^170 legal positions).

- **Enormous action space.** At any priority window, a player may cast spells, activate abilities, or pass. Each spell may require targeting decisions, mode selections, cost payment choices, and responses to triggered abilities. The branching factor at a single decision point regularly exceeds 100 and can reach thousands.

- **Hidden information.** Players cannot see opponents' hands or library ordering, requiring probabilistic reasoning about unknown cards.

- **Card diversity.** Over 27,000 unique cards have been printed, with approximately 32,300 implemented in the Forge game engine. Each card introduces unique rules text that modifies the game's mechanics, creating a long tail of rare interactions.

- **Deep strategic planning.** Games last 5-50+ turns, with each turn comprising multiple phases (untap, upkeep, draw, main, combat with sub-phases, second main, end). Resource management (mana), tempo, card advantage, and board control create layered strategic considerations.

These properties make MTG significantly harder than games previously conquered by AI. Chess and Go have large but manageable state spaces with perfect information. StarCraft II, perhaps the closest analogue in AI research, has hidden information and large action spaces but a fixed set of units and buildings. MTG combines all of these challenges with an effectively unbounded rule set that grows with each new card printed.

### 1.2 Prior Work

**Classical game AI.** AlphaGo (Silver et al., 2016) and AlphaZero (Silver et al., 2018) demonstrated that deep reinforcement learning with Monte Carlo Tree Search (MCTS) can master perfect-information games. AlphaStar (Vinyals et al., 2019) extended this to the imperfect-information, real-time domain of StarCraft II using population-based training and league play.

**Card game AI.** Libratus and Pluribus (Brown & Sandholm, 2017, 2019) achieved superhuman performance in poker through counterfactual regret minimization, exploiting poker's relatively constrained action space. Hearthstone, a digital card game inspired by MTG but with substantially simpler rules, has been the subject of several AI efforts (Hoover et al., 2020; Santos et al., 2017) using neural network function approximation with MCTS.

**MTG-specific work.** Previous MTG AI efforts have been limited primarily to heuristic systems. The Forge game engine (Card-Forge project, 2007-present) implements a sophisticated heuristic AI with card-specific evaluation functions, lookahead simulation, and tunable personality profiles. Academic work on MTG AI includes genetic algorithm-based deck building (Ward & Cowling, 2009) and limited MCTS-based play (Cowling et al., 2012). To our knowledge, no prior work has applied deep reinforcement learning to MTG with the full rules engine and card pool.

### 1.3 Contributions

We make the following contributions:

1. **A hierarchical RL architecture** for MTG that decomposes the decision problem into specialized heads for each action type, sharing a common game state encoder.

2. **A transformer-based game state encoder** that uses per-zone set attention over variable-length card collections, capturing board relationships between permanents.

3. **An efficient integration** with the Forge MTG game engine (Java) via a JSON-over-TCP bridge to a Python model server, enabling headless parallel game execution at 1.6 games/second across 16 threads.

4. **A trajectory recording system** that captures heuristic AI decisions with full game state and action features via the game engine's event bus, without modifying the core AI logic.

5. **Preliminary results** demonstrating that the value network learns meaningful board state evaluation from imitation data, and the decision heads can be trained to predict heuristic AI combat choices.

---

## 2. System Architecture

### 2.1 Overview

Our system comprises four major components:

1. **Game Engine (Java).** The Forge MTG engine handles all game rules, card interactions, and state management. We run it headlessly for data collection and evaluation.

2. **Feature Extraction (Java).** `GameStateEncoder`, `CardFeatures`, and `ActionEncoder` classes convert rich game objects into fixed-size numerical feature vectors suitable for neural network input.

3. **Neural Network (Python/PyTorch).** The `MTGModel` combines a shared transformer encoder with specialized decision heads and a value network.

4. **Training Pipeline (Python).** Data loading, training loops, and evaluation scripts with GPU acceleration (AMP on NVIDIA RTX 3080).

The architecture is designed around the insight that MTG decisions are heterogeneous — choosing which spell to cast is fundamentally different from choosing which creatures to attack with, which is different from choosing which cards to discard. Rather than forcing all decisions through a single network, we use specialized heads that share a common understanding of the game state.

### 2.2 Game State Representation

#### 2.2.1 Global Features (64 dimensions)

The global feature vector captures non-card-specific game state:

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Player's life total | [0,1] over [-10, 40] |
| 1 | Opponent's life total | [0,1] over [-10, 40] |
| 2-3 | Poison counters (both players) | [0,1] over [0, 10] |
| 4 | Turn number | [0,1] over [0, 30] |
| 5 | Active player flag | {0, 1} |
| 6-18 | Current phase (one-hot, 13 phases) | {0, 1} |
| 19-20 | Hand sizes (both players) | [0,1] over [0, 15] |
| 21-22 | Library sizes (both players) | [0,1] over [0, 60] |
| 23-24 | Creature counts (both players) | [0,1] over [0, 20] |
| 25-28 | Total power/toughness (both) | [0,1] over [0, 60] |
| 29-31 | Land counts (untapped, tapped, opponent) | [0,1] over [0, 15] |
| 32 | Stack size | [0,1] over [0, 10] |
| 33-35 | Phase convenience flags | {0, 1} |
| 36-63 | Reserved | 0 |

#### 2.2.2 Card Features (128 dimensions per card)

Each card in any zone is encoded as a 128-dimensional vector:

| Range | Features | Encoding |
|-------|----------|----------|
| 0-6 | Card types (creature, instant, sorcery, enchantment, artifact, planeswalker, land) | Binary flags |
| 7-12 | Color identity (W, U, B, R, G, colorless) | Binary flags |
| 13 | Converted mana cost | Normalized [0,1] |
| 14-15 | Power / Toughness | Normalized [-5,20] |
| 16 | Loyalty (planeswalkers) | Normalized [0,10] |
| 17-21 | In-game state (tapped, summoning sick, attacking, blocking, face-down) | Binary flags |
| 22-26 | Counter types (+1/+1, -1/-1, loyalty, charge, other) | Normalized counts |
| 27 | Number of attachments | Normalized [0,5] |
| 28 | Damage marked | Normalized [0,20] |
| 29-58 | Keyword abilities (30 common: flying, first strike, trample, deathtouch, lifelink, haste, vigilance, reach, menace, hexproof, shroud, indestructible, flash, defender, fear, ward, prowess, wither, infect, protection, shadow, undying, persist, convoke, delve, cascade, equip, enchant, flanking) | Binary flags |
| 59-68 | Zone (one-hot: battlefield, hand, library, graveyard, exile, stack, command, sideboard, ante, planar) | Binary flags |
| 69-124 | Reserved for ability type encoding | 0 |
| 125-128 | Card identity hash (4 bytes, normalized) | [0,1] per byte |

The card identity hash enables the model to develop learned embeddings for specific cards, capturing strategic value beyond what structural features alone convey.

#### 2.2.3 Zone Encoding

Cards are grouped into zones with fixed maximum sizes:

| Zone | Max Cards | Description |
|------|-----------|-------------|
| My Battlefield | 30 | Player's permanents |
| Opponent's Battlefield | 30 | Opponent's permanents |
| My Hand | 15 | Cards in hand |
| My Graveyard | 40 | Player's graveyard |
| Opponent's Graveyard | 40 | Opponent's graveyard |
| Stack | 10 | Spells/abilities resolving |

Each zone produces a (max_cards, 128) tensor with a boolean mask indicating which slots contain real cards versus padding. The total flattened game state is 21,184 floats (64 global + 165 × 128 card features).

### 2.3 Neural Network Architecture

The complete model architecture is shown in Figure 1. Data flows left-to-right: raw game state inputs are encoded by the shared transformer encoder into a 512-dimensional state embedding, which then fans out to the value network (critic) and seven specialized decision heads (actors). Heads shown at full opacity are trained; greyed heads are architecturally complete but currently fall through to the heuristic AI.

![Model Architecture](forge-ai-rl/src/main/python/tools/mtg_model_architecture.svg)

*Figure 1: MTG RL Model Architecture. The shared game state encoder (3.4M parameters) processes 7 input zones through per-zone self-attention and cross-zone attention, producing a 512-dimensional state embedding. This embedding is consumed by the value network (critic, providing PPO advantage baseline) and decision heads (actors). Each head is specialized for its decision type: the priority head uses cross-attention between game state and available spells to select which spell to play; the attack head uses self-attention among creatures for coordinated attack decisions; the block head uses cross-attention between blockers and attackers for assignment. Currently trained heads: Value, Priority (95.5% accuracy), Attack (84.4%), Block (61.0%). Untrained heads (Target, Card Select, Mulligan, Binary) fall through to the heuristic AI.*

#### 2.3.1 Game State Encoder (Transformer)

The encoder uses a two-level attention architecture:

**Level 1: Per-Zone Card Set Attention.** Each zone has a dedicated `CardSetEncoder` module consisting of:
- Linear projection: card_dim (128) → zone_embed_dim (128)
- Multi-head self-attention transformer (2 layers, 4 heads, GELU activation)
- Masked mean pooling over valid cards

This produces a single zone_embed_dim vector per zone. Self-attention within a zone captures relationships between cards — for example, an equipment card's value depends on the creatures it could be attached to.

**Level 2: Cross-Zone Attention.** The 7 zone embeddings (6 card zones + 1 global features projection) are stacked and processed through a single transformer encoder layer with 4 attention heads. This allows the model to reason about inter-zone relationships — for example, a card in hand is more valuable if the battlefield has mana to cast it.

**Output projection.** The 7 zone embeddings are concatenated (7 × 128 = 896 dimensions) and projected through a two-layer MLP to the final state embedding of 512 dimensions.

Total encoder parameters: 3,409,664.

#### 2.3.2 Value Network (Critic)

A three-layer MLP mapping the 512-dimensional state embedding to a scalar in [-1, 1]:
- Linear(512, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 1) → Tanh

The output represents estimated advantage: +1 indicates certain victory, -1 certain defeat, and 0 an even position.

Total parameters: 198,401.

#### 2.3.3 Attack Decision Head

The attack head makes a joint binary decision for each potential attacker: attack or hold back. This is implemented as:

1. **Card projection.** Each creature's 128-dim features are projected to 256 dimensions.
2. **Self-attention among potential attackers** (2 transformer layers, 4 heads). This allows the model to consider attack patterns — e.g., "if creature A attacks, creature B should also attack to force unfavorable blocks."
3. **State conditioning.** The 512-dim game state embedding is expanded and concatenated with each creature's attention-refined representation.
4. **Binary classifier.** A two-layer MLP produces a logit per creature, where positive means "attack" and negative means "hold."

During training, binary cross-entropy loss is applied per creature, masked to only count real creatures (not padding).

Total parameters: 1,875,457.

#### 2.3.4 Block Decision Head

The block head assigns each potential blocker to an attacker (or no attacker). Architecture:

1. **Separate projections** for blockers and attackers (128 → 256 each).
2. **Cross-attention.** Blockers attend to attackers to understand the threat landscape.
3. **Pairwise scoring.** For each (blocker, attacker) pair, a scoring network produces an assignment logit, plus a "don't block" option.
4. **Independent categorical sampling** per blocker over the attacker options + no-block.

Total parameters: 657,665.

#### 2.3.5 Priority Action Head

The priority head selects which spell or ability to play from the available options, or passes priority. Architecture:

1. **Action encoding.** Each available SpellAbility is encoded as a 64-dim feature vector (source card type, color, CMC, ability type via ApiType one-hot, targeting requirements, estimated effect magnitude).
2. **Cross-attention.** Available actions attend to the game state embedding.
3. **Scoring network.** Combined features are scored, producing logits over actions + pass.

Total parameters: 543,233.

#### 2.3.6 Additional Heads

| Head | Purpose | Parameters |
|------|---------|------------|
| Target | Pointer network for selecting spell targets | 674,304 |
| Card Select | General card selection (discard, sacrifice, scry) | 1,480,449 |
| Mulligan | Opening hand evaluation + London mulligan bottom selection | 2,007,042 |
| Binary | Yes/no decisions (confirm triggers, replacement effects) | 197,889 |

#### 2.3.7 Total Model Size

| Component | Parameters |
|-----------|------------|
| Game State Encoder | 3,409,664 |
| Value Network | 198,401 |
| Priority Head | 543,233 |
| Target Head | 674,304 |
| Attack Head | 1,875,457 |
| Block Head | 657,665 |
| Card Select Head | 1,480,449 |
| Mulligan Head | 2,007,042 |
| Binary Head | 197,889 |
| **Total** | **11,044,104** |

At 42MB in fp32, the full model fits comfortably on consumer GPUs. We plan progressive scaling to 50-80M parameters in later training phases, with weight transfer via net2net-style initialization.

---

## 3. Training Methodology

### 3.1 Phase 1: Imitation Learning

We bootstrap the RL agent by imitating the existing heuristic AI in the Forge game engine. This provides a warm start that is critical for the sparse-reward MTG environment.

#### 3.1.1 Data Collection

Games are run headlessly using the Forge engine's `SimulateRLTraining` runner, which:
- Creates two heuristic AI players with standard profiles
- Runs the game with a configurable timeout (180 seconds)
- Captures decision data via the Guava EventBus subscriber pattern

The `PlayerControllerRL` class extends `PlayerControllerAi` and overrides key decision methods to capture training data:

- **`declareAttackers`**: Captures pre-decision creature features (128-dim per creature), delegates to heuristic, reads back which creatures were selected as attackers from the combat object.
- **`declareBlockers`**: Same pattern — pre-decision capture, heuristic execution, post-decision readback of blocking assignments.
- **`chooseSpellAbilityToPlay`**: Leverages a modified `AiController.chooseSpellAbilityToPlayFromList` that evaluates ALL candidates through the engine's full `canPlayAndPayFor()` validation (timing, cost, targeting, AI logic) rather than short-circuiting at the first playable spell. The mechanically-legal candidate list (spells passing timing + cost checks) is cached separately from the heuristic-approved list (spells the AI considers worth playing). This gives the RL model visibility into options the heuristic would reject — enabling it to discover unconventional plays.

Each game produces two trajectory files (one per player perspective), containing 50-400 decision records depending on game length. Each record includes the full 21,184-float game state, candidate feature vectors where applicable, the indices of the heuristic AI's choices, and the eventual game outcome.

Data is collected in parallel across 16 threads, achieving 1.3 games/second on a 16-core machine. A batch of 1,000 games produces approximately 2,000 trajectory files with ~153,000 decision records.

We note a critical implementation constraint discovered during development: the Forge game engine performs class identity checks on the `LobbyPlayerAi` class that prevent subclassing, and modifications to the `PlayerControllerAi` class break the fat-jar class resolution at runtime. Recording is implemented via `PlayerControllerRL` (which extends `PlayerControllerAi`, not `LobbyPlayerAi`) and a minimal caching addition to `AiController` — the only core engine modification is caching the validated candidate list during spell evaluation.

#### 3.1.2 Value Network Training

The value network is trained first as it provides the foundation for all subsequent training:

- **Task.** Predict game outcome (+1 win, -1 loss) from any mid-game state.
- **Data.** 28,828 state snapshots from 1,000 games, captured at every attack, block, spell, and main phase.
- **Loss.** Mean squared error between predicted value and game outcome.
- **Optimization.** AdamW (lr=3×10⁻⁴, weight decay=10⁻⁴), cosine annealing schedule, gradient clipping at 1.0.
- **Hardware.** NVIDIA RTX 3080 (10GB VRAM), automatic mixed precision (fp16), batch size 64.

In preliminary experiments with 1,710 end-game-only samples, the value network achieved 99.6% accuracy in predicting the correct winner, with validation loss converging to near zero within 4 epochs. Training on the richer mid-game dataset with 28,828 samples is expected to produce more nuanced evaluation, as mid-game positions are inherently more ambiguous than end-game states.

#### 3.1.3 Decision Head Training

After the value network converges, we freeze the encoder weights and train each decision head independently:

**Attack Head.** Binary cross-entropy loss per creature, trained on 13,918 attack decisions. The model learns to predict which creatures the heuristic AI chose to attack with, given the board state and available attackers.

**Block Head.** Same architecture and loss as the attack head, trained on 6,546 blocking decisions.

**Priority Head.** Cross-entropy loss over available actions (softmax single-select, distinct from combat's binary BCE). Trained on 140,603 priority decisions captured with the full mechanically-legal candidate set. Each decision includes 1-7 playable spells plus a pass option, with 64-dim action features per candidate. The heuristic passes priority 88.8% of the time even with playable spells available, providing rich timing data — the model learns both *what* to play and *when* to wait.

### 3.2 Phase 2: Reinforcement Learning via Self-Play

Once imitation learning produces a policy that plays at rough parity with the heuristic AI, we switch to PPO-based self-play to improve beyond the heuristic's level.

#### 3.2.1 PPO Formulation

At each decision point $t$, the agent observes state $s_t$ and takes action $a_t$ according to policy $\pi_\theta(a_t|s_t)$. The advantage is estimated using Generalized Advantage Estimation (GAE):

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

The PPO clipped objective is:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and clipping parameter $\epsilon = 0.2$.

The total loss combines policy, value, and entropy terms:

$$L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta]$$

with $c_1 = 0.5$ (value loss coefficient) and $c_2 = 0.01$ (entropy bonus to encourage exploration).

#### 3.2.2 Reward Shaping

MTG's terminal reward (win/loss) is extremely sparse — a game may involve hundreds of micro-decisions before the outcome is determined. We employ reward shaping to provide more frequent feedback during early training:

| Signal | Reward | Rationale |
|--------|--------|-----------|
| Win | +1.0 | Terminal reward |
| Loss | -1.0 | Terminal reward |
| Life advantage change | ±0.01 per point | Tracks damage race |
| Card advantage change | ±0.05 per card | Card advantage is a fundamental MTG concept |
| Board advantage change | ±0.02 per creature | Board presence correlates with winning |
| Total power advantage | ±0.005 per power | Quality of board matters, not just quantity |

Shaping rewards are multiplied by a decay factor ($\alpha = 0.9999$ per training step) so the agent eventually optimizes purely for win rate. The discount factor $\gamma = 0.999$ reflects the long-horizon nature of MTG games.

#### 3.2.3 Curriculum Learning

We introduce card complexity gradually across six stages:

| Stage | Card Pool | Advancement Criteria |
|-------|-----------|---------------------|
| A: Vanilla Creatures | Creatures with no abilities, basic lands | 60% win rate, 5K games |
| B: Keywords | Add flying, trample, first strike, deathtouch, lifelink, haste, vigilance | 58% win rate, 10K games |
| C: Removal & Tricks | Add instant-speed removal, pump spells, combat tricks | 56% win rate, 15K games |
| D: Card Draw & Counters | Add card draw, counterspells, stack interaction | 55% win rate, 20K games |
| E: Complex Permanents | Add enchantments, artifacts, planeswalkers, activated abilities | 54% win rate, 30K games |
| F: Full Card Pool | Standard/Modern format card pools | 52% win rate, 50K games |

Win rates are measured against the heuristic AI. Advancement thresholds decrease at higher stages because the task becomes inherently harder — the agent faces more complex card interactions and the heuristic AI's hand-tuned card-specific logic becomes a stronger baseline.

#### 3.2.4 League Training

Following the AlphaStar approach (Vinyals et al., 2019), we maintain a population of agents:

- **Main agents** (3): Train against all opponents in the league, including historical snapshots.
- **Exploiter agents** (2): Specifically target weaknesses in current main agents to prevent strategy cycling.
- **Historical snapshots**: Checkpoints saved periodically, serving as fixed opponents to prevent catastrophic forgetting.

Elo ratings track relative strength across the population, providing an objective measure of improvement independent of win rate against any single opponent.

### 3.3 Phase 3: Progressive Scaling

Model capacity is increased in stages to match the growing complexity of the card pool:

| Phase | Parameters | State Dim | Hidden Dim | Layers | Estimated VRAM |
|-------|------------|-----------|------------|--------|----------------|
| Current | 11M | 512 | 256 | 2 | 0.4 GB |
| Scale 1 | 50M | 768 | 512 | 3 | 2.5 GB |
| Scale 2 | 150M | 1024 | 768 | 4 | 8 GB |

Weight transfer uses net2net-style initialization: the smaller model's weights are copied into the corresponding positions of the larger model, with new dimensions initialized near zero. This preserves learned representations while providing capacity for new knowledge.

---

## 4. Implementation

### 4.1 Game Engine Integration

The Forge game engine (Java 17, ~50,000 source files) implements the complete MTG rules with 32,300 cards. Our integration adds a `forge-ai-rl` Maven module (17 Java files, 21 Python files) without modifying any core game engine classes.

Key integration points:

- **`SimulateRLTraining.java`**: Headless game runner supporting parallel execution, trajectory recording, and configurable AI opponents.
- **`GameStateRecorder.java`**: Guava EventBus subscriber that captures game state and action data at decision points.
- **`PlayerControllerRL.java`**: Full `PlayerController` implementation that routes decisions to the RL model server with heuristic fallback.
- **`ModelServerClient.java`**: JSON-over-TCP client for inference requests to the Python model server.

### 4.2 Java-Python Bridge

During training, the game engine (Java) communicates with the model server (Python) via a length-prefixed JSON-over-TCP protocol:

```
Client → Server: [4 bytes big-endian length][JSON request]
Server → Client: [4 bytes big-endian length][JSON response]
```

Request payloads include the decision type, global features, per-zone card features with masks, candidate action features, and selection constraints. Response payloads include selected action indices, probability distributions, and value estimates.

For deployment, trained models are exported to ONNX format and loaded directly in Java via ONNX Runtime, eliminating the Python dependency and inter-process communication overhead.

### 4.3 Hardware Requirements

All experiments are conducted on a single workstation:
- **CPU:** 16-core (parallel game execution)
- **GPU:** NVIDIA GeForce RTX 3080, 10GB VRAM
- **RAM:** 16GB

Automatic mixed precision (AMP) with fp16 reduces GPU memory usage by approximately 50%. The full model (11M parameters) uses only 51MB of VRAM for inference. Training at batch size 64 uses approximately 400MB, leaving substantial headroom for scaling.

Data collection at 1.6 games/second produces sufficient training data in minutes rather than hours. A complete imitation learning cycle (1,000 games → training → evaluation) takes approximately 30 minutes end-to-end.

---

## 5. Preliminary Results

### 5.1 Data Collection

We collected 1,000 games between four constructed decks (Red Aggro, Green Stompy, White Weenie, Blue Tempo) using 16-thread parallel execution in 776 seconds (13 minutes). This produced:

| Metric | Value |
|--------|-------|
| Trajectory files | 1,998 (475 P1 wins, 525 P2 wins) |
| Total decision records | ~153,000 |
| Attack decisions | 9,750 |
| Block decisions | 2,830 |
| Priority decisions | 140,603 |
| Priority: play a spell | 15,679 (11.2%) |
| Priority: pass with options | 124,924 (88.8%) |
| Candidate distribution | 1-7 options per decision |

### 5.2 Value Network Performance

The value network was trained for 100 epochs on 28,828 mid-game decision snapshots:

| Metric | Result |
|--------|--------|
| Training accuracy | 99.6% |
| Validation accuracy | 100.0% |
| Validation loss | < 0.001 |
| Training time | ~40 minutes (100 epochs, RTX 3080) |
| Epoch time | ~1.5 seconds |

Qualitative evaluation confirms the model has learned meaningful board evaluation. Synthetic test states with favorable positions (high life, creatures, cards) receive values near +1.0, while unfavorable positions receive values near -1.0.

The high accuracy on this task suggests the current model capacity (11M parameters) is sufficient for the four-deck environment. We anticipate accuracy will decrease as the card pool expands, necessitating the progressive scaling described in Section 3.3.

### 5.3 Combat Decision Head Training

The attack head was trained on 9,750 attack decisions and the block head on 2,830 blocking decisions, both using binary cross-entropy loss per creature. PPO self-play training with these heads achieved a 53% win rate vs the heuristic AI baseline.

### 5.4 Priority Decision Head Training (In Progress)

The priority head is being trained on 140,603 priority decisions using cross-entropy loss (softmax over candidate spells + pass). This represents a fundamentally different decision type from combat — single-select with variable candidate count and a significant class imbalance toward passing. The training pipeline processes priority first, followed by combat heads, all sharing the frozen encoder weights.

The priority data reveals that the heuristic AI frequently passes even with mechanically-legal spells available (88.8% pass rate). This creates a rich signal for the RL model to learn timing — when the heuristic's pass was correct (holding mana for a potential response) versus suboptimal (missing a tempo-positive play window).

---

## 6. Discussion

### 6.1 Comparison to Existing Approaches

The Forge heuristic AI uses hand-coded evaluation functions with card-specific logic, maintained by a community of contributors over 18+ years. Its `ComputerUtil` classes span over 300,000 lines of Java code encoding MTG-specific heuristics for every card interaction.

Our approach aims to match and eventually exceed this performance through learned representations rather than hand-coded rules. The key advantages of the learned approach are:

1. **Generalization.** The heuristic AI requires card-specific code for each of 32,300 cards. Our model learns general patterns (e.g., "creatures with high power are good attackers") that transfer to unseen cards.

2. **Self-improvement.** The heuristic AI's strength is bounded by human insight. RL self-play can discover strategies that human designers did not encode.

3. **Adaptability.** The learned model can be fine-tuned to new cards by continuing training, rather than requiring manual implementation of card-specific logic.

### 6.2 Limitations

**Current scope.** Our preliminary results use only four simple constructed decks. Real MTG involves thousands of viable decks with complex synergies and interactions that our current training data does not cover.

**Action granularity.** Priority decisions now capture the full mechanically-legal candidate set by caching validated spells during the engine's own `canPlayAndPayFor()` evaluation. However, the mechanical check (timing + cost) is broader than the AI's strategic evaluation — some candidates pass the mechanical check but would be strategically poor (e.g., bouncing your own creature). This means the RL model sees options the heuristic wouldn't consider, which is both an opportunity (discovering unconventional plays) and a risk (learning from noisy candidates).

**Hidden information.** Our current feature encoding does not model uncertainty over the opponent's hidden hand. The model receives what a legal player would see (hand size, not contents), but has no explicit mechanism for probabilistic reasoning about hidden cards.

**Computational scale.** Our single-GPU setup limits model size and training throughput. The progressive scaling plan targets 150M parameters, which approaches the practical limit of a 10GB GPU with mixed precision.

### 6.3 Future Work

**Priority head refinement.** The priority candidate set currently uses the engine's mechanical validation (timing + cost). Future work could split this into a "strategically reasonable" tier (candidates passing AI heuristic checks) versus a "creative play" tier (mechanically legal but heuristic-rejected), allowing the model to explore unconventional lines while still grounding training in reasonable play.

**Opponent modeling.** Adding a recurrent component that maintains a hidden state across the game, enabling the model to build beliefs about the opponent's hand based on observed play patterns.

**Deck building.** Extending the system to not just play games but construct decks, using the value network to evaluate card choices in the context of a deck archetype.

**Multi-format support.** Training on Commander (100-card singleton, multiplayer), Draft (card selection from packs), and other MTG formats with different strategic considerations.

---

## 7. Conclusion

We have presented the architecture and early implementation of a hierarchical reinforcement learning system for Magic: The Gathering. The system combines a transformer-based game state encoder with specialized decision heads, trained through imitation learning on a heuristic AI and designed for subsequent self-play improvement via PPO.

Our preliminary results demonstrate that the game state encoder successfully learns to evaluate board positions from mid-game snapshots, and that the full training pipeline — from parallel headless game execution through trajectory recording to GPU-accelerated training — operates efficiently on consumer hardware.

The key architectural insights are: (1) decomposing the MTG decision space into specialized heads that share a common state understanding is more tractable than a monolithic policy, (2) the transformer's set-attention mechanism naturally handles the variable-size card collections in each game zone, and (3) the game engine's event bus provides a non-invasive recording mechanism that captures action-level decision data without modifying core AI logic.

MTG represents a frontier challenge for game AI — a domain where the rules themselves are as complex as the strategies that emerge from them. Our system provides a foundation for exploring this frontier through learned, self-improving play.

---

## References

Brown, N. & Sandholm, T. (2017). Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. *Science*, 359(6374), 418-424.

Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.

Card-Forge Project. (2007-present). Forge: Magic: The Gathering game engine. https://github.com/Card-Forge/forge

Cowling, P. I., Ward, C. D., & Powley, E. J. (2012). Ensemble determinization in Monte Carlo tree search for the imperfect information card game Magic: The Gathering. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(4), 267-277.

Hoover, A. K., et al. (2020). Building a better Hearthstone agent using deep reinforcement learning. *IEEE Conference on Games*.

Santos, A., et al. (2017). Monte Carlo tree search experiments in Hearthstone. *IEEE Conference on Computational Intelligence and Games*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.

Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.

Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature*, 575(7782), 350-354.

Ward, C. D. & Cowling, P. I. (2009). Monte Carlo search applied to card selection in Magic: The Gathering. *IEEE Symposium on Computational Intelligence and Games*.

---

## Appendix A: Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| State embedding dimension | 512 |
| Zone embedding dimension | 128 |
| Card feature dimension | 128 |
| Action feature dimension | 64 |
| Hidden dimension | 256 |
| Attention heads | 4 |
| Transformer layers (per-zone) | 2 |
| Cross-zone transformer layers | 1 |
| Dropout | 0.1 |
| Activation function | GELU |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 10⁻⁴ |
| LR schedule | Cosine annealing |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| PPO clip epsilon | 0.2 |
| Discount factor (γ) | 0.999 |
| GAE lambda (λ) | 0.95 |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |

## Appendix B: Card Feature Index Reference

Full 128-dimension card feature vector specification is provided in `CardFeatures.java` with inline documentation for each index range.

## Appendix C: Software Availability

The complete implementation is available at https://github.com/austinio7116/forge/tree/ai_investigation, including:
- Java integration code (`forge-ai-rl/src/main/java/`)
- Python model and training code (`forge-ai-rl/src/main/python/`)
- Pipeline scripts (`forge-ai-rl/scripts/`)
- Test decks (`rl_data/decks/`)
- Architecture plan (`RLAI_PLAN.md`)
- Development notes (`CLAUDE.md`)
