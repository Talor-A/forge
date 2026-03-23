# Learning to Play Magic: The Gathering Through Hierarchical Reinforcement Learning with Transformer-Based State Encoding

**Authors:** M. Austin, with architectural design and implementation assistance from Claude (Anthropic)

**Date:** March 2026

---

## Abstract

We present a reinforcement learning system for playing Magic: The Gathering (MTG), arguably the most complex widely-played strategy card game in existence. MTG presents unique challenges for AI: a combinatorial action space with over 32,000 unique cards, hidden information, stochastic elements, and deeply nested interactions between game mechanics. Our approach employs a hierarchical architecture with a shared transformer-based game state encoder and specialized decision heads for each major action type (spell casting, combat, card selection). We bootstrap the system through imitation learning on a heuristic AI opponent, then improve via Proximal Policy Optimization (PPO) self-play with curriculum learning. We describe the full system architecture, feature engineering, training pipeline, and results on the open-source Forge MTG game engine. From 1,000 heuristic AI games we collect ~127,000 decision records across all 7 decision types. All decision heads achieve strong imitation accuracy (priority 95.7%, attack 82.8%, target 74.3%, block 64.2%, mulligan 99.0%, binary 80.9%, card select 76.5%). The trained model is deployed via ONNX for real-time play in the Forge GUI without Python dependency.

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

5. **Results** demonstrating that all 7 decision heads learn to predict heuristic AI choices with high accuracy (64-99%), and the model can be deployed via ONNX for real-time play in the Forge GUI.

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

#### 2.2.1 Global Features (96 dimensions)

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
| 36-41 | Available mana in pool by color (WUBRGC) | [0,1] over [0, 10] |
| 42-47 | Producible mana from untapped permanents (WUBRGC) | {0, 1} |
| 48 | Total available mana (pool + untapped lands) | [0,1] over [0, 15] |
| 49 | Spells cast this turn | [0,1] over [0, 10] |
| 50 | Lands played this turn | [0,1] over [0, 2] |
| 51 | Opponent lands untapped | [0,1] over [0, 15] |
| 52-53 | Nonland permanent counts (both players) | [0,1] over [0, 30] |
| 54 | Is sorcery speed (main phase + active player) | {0, 1} |
| 55-63 | Reserved | 0 |
| 64-69 | Color devotion (WUBRGC) | [0,1] over [0, 15] |
| 70 | Castable cards in hand | [0,1] over [0, 10] |
| 72-75 | Enchantment/artifact counts (both players) | [0,1] over [0, 10] |
| 76-95 | Reserved | 0 |

#### 2.2.2 Card Features (256 dimensions per card)

Each card in any zone is encoded as a 256-dimensional vector:

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
| 59-68 | Zone (one-hot) | Binary flags |
| 69-98 | Primary ability ApiType flags (30 ability types: Mana, ManaReflected, Pump, PumpAll, DealDamage, Destroy, Counter, Draw, ChangeZone, Token, Attach, Animate, Protection, Regenerate, Sacrifice, Tap, Untap, Proliferate, Gain/LoseLife, Mill, Discard, Fight, Explore, Scry, Transform, Bounce, Copy, Exile, Surveil, Adapt) | Binary flags |
| 99-102 | Ability summary (has_activated, has_triggered, has_mana, n_abilities) | Mixed |
| 103-106 | Primary ability effects (est. damage, cards drawn, life gained, tokens) | Normalized |
| 107-108 | Targeting (requires_target, targets_creatures) | Binary flags |
| 109-138 | Secondary ability ApiType flags (same 30 types) | Binary flags |
| 139-168 | Extended keywords (30 more: horsemanship, intimidate, skulk, annihilator, absorb, bushido, exalted, melee, modular, toxic, afflict, phasing, cumulative_upkeep, echo, fading, vanishing, storm, affinity, changeling, devoid, emerge, improvise, spectacle, revolt, riot, entwine, companion, foretell, disturb, daybound) | Binary flags |
| 169-173 | Mana production (produces W, U, B, R, G) | Binary flags |
| 174-177 | Spell speed (instant_speed, has_flash, is_modal, has_kicker) | Binary flags |
| 178-181 | Trigger summary (ETB, death, combat, upkeep) | Binary flags |
| 182-189 | Mana cost breakdown (W, U, B, R, G, generic, total, has_X) | Normalized / Binary |
| 190-251 | Reserved | 0 |
| 252-255 | Card identity hash (4 bytes, normalized) | [0,1] per byte |

The expanded encoding allows the model to see what cards *do* — their ability types, effects, mana production, and triggers — rather than just their static characteristics. The card identity hash enables learned embeddings for specific cards.

#### 2.2.3 Zone Encoding

Cards are grouped into zones with fixed maximum sizes:

| Zone | Max Cards | Description |
|------|-----------|-------------|
| My Battlefield | 40 | Player's permanents |
| Opponent's Battlefield | 40 | Opponent's permanents |
| My Hand | 15 | Cards in hand |
| My Graveyard | 20 | Player's graveyard |
| Opponent's Graveyard | 20 | Opponent's graveyard |
| Stack | 10 | Spells/abilities resolving |

Each zone produces a (max_cards, 256) tensor with a boolean mask indicating which slots contain real cards versus padding. The total flattened game state is 37,216 floats (96 global + 145 × 256 card features). Board sizes were increased to 40 to accommodate token-heavy strategies, while graveyards were reduced to 20 since most relevant graveyard information is captured in the most recent cards.

### 2.3 Neural Network Architecture

The complete model architecture is shown in Figure 1. Data flows left-to-right: raw game state inputs are encoded by the shared transformer encoder into a 512-dimensional state embedding, which then fans out to the value network (critic) and seven specialized decision heads (actors). Heads shown at full opacity are trained; greyed heads are architecturally complete but currently fall through to the heuristic AI.

![Model Architecture](forge-ai-rl/src/main/python/tools/mtg_model_architecture.svg)

*Figure 1: MTG RL Model Architecture. The shared game state encoder (3.5M parameters) processes 7 input zones through per-zone self-attention and cross-zone attention, producing a 512-dimensional state embedding. This embedding is consumed by the value network (critic, providing PPO advantage baseline) and 7 decision heads (actors). Each head is specialized for its decision type: the priority head uses cross-attention between game state and available spells; the attack head uses self-attention among creatures for coordinated attacks; the block head uses cross-attention between blockers and attackers. All heads are trained and active — no decisions fall through to the heuristic AI. The model is deployed via 9 ONNX files for Java-native inference.*

#### 2.3.1 Game State Encoder (Transformer)

The encoder uses a two-level attention architecture:

**Level 1: Per-Zone Card Set Attention.** Each zone has a dedicated `CardSetEncoder` module consisting of:
- Linear projection: card_dim (256) → zone_embed_dim (128)
- Multi-head self-attention transformer (2 layers, 4 heads, GELU activation)
- Masked mean pooling over valid cards

This produces a single zone_embed_dim vector per zone. Self-attention within a zone captures relationships between cards — for example, an equipment card's value depends on the creatures it could be attached to.

**Level 2: Cross-Zone Attention.** The 7 zone embeddings (6 card zones + 1 global features projection) are stacked and processed through a single transformer encoder layer with 4 attention heads. This allows the model to reason about inter-zone relationships — for example, a card in hand is more valuable if the battlefield has mana to cast it.

**Output projection.** The 7 zone embeddings are concatenated (7 × 128 = 896 dimensions) and projected through a two-layer MLP to the final state embedding of 512 dimensions.

Total encoder parameters: 3,512,064.

#### 2.3.2 Value Network (Critic)

A three-layer MLP mapping the 512-dimensional state embedding to a scalar in [-1, 1]:
- Linear(512, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 1) → Tanh

The output represents the estimated discounted return: values near +1 indicate near-certain victory (late game, winning position), values near -1 indicate near-certain defeat, and values near 0 indicate either an even position or high uncertainty (typical of early-game states where the outcome depends on many future decisions).

Total parameters: 198,401.

#### 2.3.3 Attack Decision Head

The attack head makes a joint binary decision for each potential attacker: attack or hold back. This is implemented as:

1. **Card projection.** Each creature's 256-dim features are projected to 256 dimensions.
2. **Self-attention among potential attackers** (2 transformer layers, 4 heads). This allows the model to consider attack patterns — e.g., "if creature A attacks, creature B should also attack to force unfavorable blocks."
3. **State conditioning.** The 512-dim game state embedding is expanded and concatenated with each creature's attention-refined representation.
4. **Binary classifier.** A two-layer MLP produces a logit per creature, where positive means "attack" and negative means "hold."

During training, binary cross-entropy loss is applied per creature, masked to only count real creatures (not padding).

Total parameters: 1,908,225.

#### 2.3.4 Block Decision Head

The block head assigns each potential blocker to an attacker (or no attacker). Architecture:

1. **Separate projections** for blockers and attackers (256 → 256 each).
2. **Cross-attention.** Blockers attend to attackers to understand the threat landscape.
3. **Pairwise scoring.** For each (blocker, attacker) pair, a scoring network produces an assignment logit, plus a "don't block" option.
4. **Independent categorical sampling** per blocker over the attacker options + no-block.

Total parameters: 723,201.

#### 2.3.5 Priority Action Head

The priority head selects which spell or ability to play from the available options, or passes priority. Architecture:

1. **Action encoding.** Each available SpellAbility is encoded as a 64-dim feature vector (source card type, color, CMC, ability type via ApiType one-hot, targeting requirements, estimated effect magnitude).
2. **Cross-attention.** Available actions attend to the game state embedding.
3. **Scoring network.** Combined features are scored, producing logits over actions + pass.

Total parameters: 543,233.

#### 2.3.6 Target Selection Head

The target head selects spell targets from a variable-size candidate set using a pointer attention mechanism with autoregressive multi-target support:

1. **Target projection.** Each candidate's 256-dim card features are projected to 256 dimensions.
2. **Pointer attention.** The 512-dim game state embedding is projected to a query vector; candidate projections serve as keys. Scaled dot-product attention produces logits over candidates, masked to valid targets only.
3. **Multi-target via GRU.** For spells requiring multiple targets (e.g., Searing Blaze), a GRU cell updates the query context after each selection, conditioning subsequent picks on previous choices. Selected targets are masked out to prevent re-selection.

The head handles both single-target (softmax + argmax) and multi-target (autoregressive sampling) modes based on `minTargets`/`maxTargets` from the spell's `TargetRestrictions`.

Total parameters: 723,456.

#### 2.3.7 Card Selection Head

The card selection head handles general "choose N cards" effects (scry top/bottom, discard, sacrifice). Architecture:

1. **Card projection.** Each candidate's 256-dim features are projected to 256 dimensions.
2. **Self-attention among candidates.** A 1-layer transformer encoder (4 heads) allows candidates to attend to each other — e.g., when discarding, the relative value of cards in hand matters.
3. **State-conditioned scoring.** Each candidate's attention-refined embedding is concatenated with the 512-dim game state and scored by a two-layer MLP producing a selection logit per candidate.
4. **Multi-card selection via GRU.** For effects requiring multiple selections, a GRU cell updates context after each pick, enabling sequential "pick the worst, then the next worst" reasoning.

Total parameters: 1,513,217.

#### 2.3.8 Mulligan Head

The mulligan head makes two related decisions: keep or mulligan the opening hand, and which cards to put on bottom (London mulligan). Architecture:

1. **Hand projection.** Each card's 256-dim features projected to 256 dimensions.
2. **Hand evaluation via self-attention.** A 2-layer transformer encoder (4 heads) lets cards attend to each other — the value of a card depends on what else is in hand (e.g., a 3-drop is better with a land-heavy hand). Attention-refined card embeddings are mean-pooled to a single hand representation.
3. **Keep/mulligan classifier.** The pooled hand embedding is concatenated with the 512-dim game state and passed through a two-layer MLP producing a keep logit (positive = keep, negative = mulligan).
4. **Bottom card scorer.** For London mulligan, each card's attention-refined embedding (concatenated with game state) is scored independently. Cards with the highest bottom scores are put on bottom of library.

Total parameters: 2,039,810.

#### 2.3.9 Binary Decision Head

The binary head handles simple yes/no decisions (trigger confirmations, replacement effects, optional costs). Architecture:

1. **Three-layer MLP.** Maps the 512-dim game state embedding directly to a single logit:
   - Linear(512, 256) → GELU → LayerNorm → Dropout(0.1)
   - Linear(256, 256) → GELU → Dropout(0.1)
   - Linear(256, 1)

No candidate features are needed — binary decisions depend only on the board state. The sigmoid of the output gives the probability of "yes."

Total parameters: 197,889.

#### 2.3.10 Total Model Size

| Component | Parameters |
|-----------|------------|
| Game State Encoder | 3,512,064 |
| Value Network | 198,401 |
| Priority Head | 543,233 |
| Target Head | 723,456 |
| Attack Head | 1,908,225 |
| Block Head | 723,201 |
| Card Select Head | 1,513,217 |
| Mulligan Head | 2,039,810 |
| Binary Head | 197,889 |
| **Total** | **11,359,496** |

At ~43MB in fp32, the full model fits comfortably on consumer GPUs (~51MB VRAM for inference). For deployment, the model is exported as 9 separate ONNX files (one encoder + one per head + value network), totalling ~42MB on disk. Each head receives the 512-dim state embedding from the shared encoder, so inference runs the encoder once per decision and then only the relevant head.

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

- **`declareAttackers`**: Captures pre-decision creature features (256-dim per creature), delegates to heuristic, reads back which creatures were selected as attackers from the combat object.
- **`declareBlockers`**: Same pattern — pre-decision capture, heuristic execution, post-decision readback of blocking assignments.
- **`chooseSpellAbilityToPlay`**: Leverages a modified `AiController.chooseSpellAbilityToPlayFromList` that evaluates ALL candidates through the engine's full `canPlayAndPayFor()` validation (timing, cost, targeting, AI logic) rather than short-circuiting at the first playable spell. The mechanically-legal candidate list (spells passing timing + cost checks) is cached separately from the heuristic-approved list (spells the AI considers worth playing). This gives the RL model visibility into options the heuristic would reject — enabling it to discover unconventional plays.

Each game produces two trajectory files (one per player perspective), containing 50-400 decision records depending on game length. Each record includes the full 37,216-float game state, candidate feature vectors where applicable, the indices of the heuristic AI's choices, intermediate reward signals (life/card/board advantage changes), and the terminal outcome.

During preprocessing, discounted returns are computed backward through each trajectory using $G_t = r_t + \gamma G_{t+1}$ with $\gamma = 0.95$, where $r_t$ includes both the intermediate reward shaping signal and the terminal reward (+1/-1) on the final step. This means early-game decisions have value targets closer to 0 (high uncertainty about outcome), while late-game decisions have targets closer to ±1 (outcome nearly determined). The value network learns to predict these discounted returns rather than flat binary outcomes, giving it a calibrated sense of game-state certainty.

Data is collected in parallel across 16 threads, achieving 1.3 games/second on a 16-core machine. A batch of 1,000 games produces approximately 2,000 trajectory files with ~153,000 decision records.

We note a critical implementation constraint discovered during development: the Forge game engine performs class identity checks on the `LobbyPlayerAi` class that prevent subclassing, and modifications to the `PlayerControllerAi` class break the fat-jar class resolution at runtime. Recording is implemented via `PlayerControllerRL` (which extends `PlayerControllerAi`, not `LobbyPlayerAi`) and a minimal caching addition to `AiController` — the only core engine modification is caching the validated candidate list during spell evaluation.

#### 3.1.2 Value Network Training

The value network is trained first as it provides the foundation for all subsequent training:

- **Task.** Predict discounted return $G_t$ from any mid-game state, where $G_t = r_t + \gamma G_{t+1}$ ($\gamma = 0.95$). Early-game targets are near 0 (uncertain outcome); late-game targets approach ±1 (determined outcome). This gives the value network calibrated confidence — it learns that a turn-2 board state is inherently uncertain, while a turn-12 state with one player at 2 life is nearly decided.
- **Data.** ~127,000 decision snapshots from 1,000 games, captured at every decision point across all 7 types.
- **Loss.** Mean squared error between predicted value and discounted return.
- **Optimization.** AdamW (lr=3×10⁻⁴, weight decay=10⁻⁴), cosine annealing schedule, gradient clipping at 1.0.
- **Hardware.** NVIDIA RTX 3080 (10GB VRAM), automatic mixed precision (fp16), batch size 256.

The feature encoding captures pre-decision state (creatures are recorded as untapped before attack declaration) to prevent information leakage. Train/val splits use game-level grouping so both player perspectives from the same game stay in the same split.

#### 3.1.3 Decision Head Training

After the value network converges, we freeze the encoder weights and train each decision head independently, chaining checkpoints so each head inherits all previously trained weights:

**Priority Head.** Cross-entropy loss over available actions (softmax single-select). Trained on ~104,500 priority decisions with the full mechanically-legal candidate set. Each decision includes 1-7 playable spells plus a pass option, with 64-dim action features per candidate. The heuristic passes priority ~85% of the time even with playable spells available, providing rich timing data — the model learns both *what* to play and *when* to wait.

**Attack Head.** Binary cross-entropy loss per creature, trained on ~9,500 attack decisions. The model learns to predict which creatures the heuristic AI chose to attack with, given the board state and available attackers.

**Target Head.** Cross-entropy loss over candidate targets (pointer attention), trained on ~5,000 target decisions with 256-dim card features per candidate.

**Card Select Head.** Binary cross-entropy per candidate, trained on ~650 card selection decisions (scry top/bottom).

**Block Head.** Cross-entropy per blocker over attacker assignment options, trained on ~2,800 blocking decisions.

**Mulligan Head.** Binary cross-entropy (keep/mulligan), trained on ~2,600 mulligan decisions with hand card features.

**Binary Head.** Binary cross-entropy (yes/no), trained on ~2,000 trigger confirmation decisions.

### 3.2 Phase 2: Reinforcement Learning via Self-Play

Once imitation learning produces a policy that plays at rough parity with the heuristic AI, we switch to PPO-based self-play to improve beyond the heuristic's level.

#### 3.2.1 PPO Formulation

At each decision point $t$, the agent observes state $s_t$ and takes action $a_t$ according to policy $\pi_\theta(a_t|s_t)$. The advantage is estimated using Generalized Advantage Estimation (GAE):

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

The PPO clipped objective is:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and clipping parameter $\epsilon = 0.1$.

The total loss combines policy, value, and entropy terms:

$$L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta]$$

with $c_1 = 0.5$ (value loss coefficient) and $c_2 = 0.005$ (entropy bonus to encourage exploration).

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
- **`PlayerControllerRL.java`**: Full `PlayerController` implementation that routes all 7 decision types to the RL model (ONNX or GRPC) with heuristic fallback.
- **`ONNXModelClient.java`**: Loads 9 ONNX model files for Java-native inference without Python dependency. Handles input padding, masking, and zone tensor reconstruction matching the Python `parse_game_state()`.
- **`ModelServerClient.java`**: JSON-over-TCP client for inference requests to the Python model server (used during training).

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

We collected 1,000 games between four constructed decks (Red Aggro, Green Stompy, White Weenie, Blue Tempo) using 16-thread parallel execution in 742 seconds (12 minutes). This produced:

| Metric | Value |
|--------|-------|
| Trajectory files | 2,000 |
| Unique games | 987 |
| Total decision records | 127,156 |
| Priority decisions | 104,514 |
| Attack decisions | 9,562 |
| Target decisions | 4,995 |
| Block decisions | 2,768 |
| Mulligan decisions | 2,635 |
| Binary decisions | 2,032 |
| Card select decisions | 650 |
| Average turns per game | 13.9 |
| P1/P2 win balance | 483/517 |
| Preprocessed disk usage | 18.8 GB |

### 5.2 Value Network Performance

The value network is trained on ~127,000 mid-game decision snapshots from 1,000 games using 256-dim card features. The value target for each decision is a discounted return ($\gamma = 0.95$) computed backward through the trajectory, incorporating intermediate reward signals (life/card/board advantage changes) and the terminal outcome. This gives early-game states targets near 0 (uncertain outcome) and late-game states targets near ±1.

The feature encoding captures pre-decision state to prevent information leakage (e.g., creatures are recorded as untapped before attack declaration). Train/val splits use game-level grouping so both P1 and P2 perspectives from the same game stay in the same split, preventing the model from seeing both sides of a game across train and val sets.

### 5.3 Decision Head Training (Imitation Learning)

After the value network converges, the encoder weights are frozen and each decision head is trained independently. All heads use the same 512-dimensional state embedding from the shared encoder. Heads are trained sequentially, chaining checkpoints so each subsequent head inherits all previously trained weights.

| Head | Samples | Loss Function | Val Accuracy | Epochs |
|------|---------|---------------|-------------|--------|
| Priority | 104,514 | CrossEntropy (softmax) | 95.7% | 10 |
| Attack | 9,562 | BCE (per-creature sigmoid) | 82.8% | 10 |
| Target | 4,995 | CrossEntropy (pointer) | 74.3% | 10 |
| Card Select | 650 | BCE (multi-select sigmoid) | 76.5% | 10 |
| Block | 2,768 | CE per-blocker (assignment) | 64.2% | 10 |
| Mulligan | 2,635 | BCE (keep/mulligan) | 99.0% | 10 |
| Binary | 2,032 | BCE (yes/no) | 80.9% | 10 |

The ~85% pass rate in priority training data means a naive "always pass" baseline achieves ~85%, so the priority head's 95.7% accuracy represents significant learning of spell timing. The mulligan head's 99.0% accuracy reflects that the heuristic almost always keeps 7-card hands, making the baseline high.

Train/val splits use game-level grouping (both P1/P2 perspectives in the same split) to prevent data leakage.

### 5.4 PPO Self-Play Training

After imitation learning, we will apply Proximal Policy Optimization (PPO) to improve beyond the heuristic baseline. The RL agent plays against the heuristic AI, collecting trajectories with action probabilities for importance sampling, then performs clipped policy gradient updates.

**Implementation.** We store the old policy's action probabilities (`actionProbabilities`) in each trajectory record during data collection. During PPO updates, the importance sampling ratio $r = \pi_{new} / \pi_{old}$ is computed and clipped to $[1-\epsilon, 1+\epsilon]$ with $\epsilon = 0.1$. Advantages are computed via GAE ($\gamma = 0.999$, $\lambda = 0.95$) using per-decision intermediate rewards from the reward shaper. The encoder is frozen during PPO to prevent representation drift; heads use lr=3e-5 and the value network uses lr=1e-4. The entropy coefficient is set to 0.005.

The RL model handles all decisions autonomously during PPO self-play — all 7 heads are active, targeting uses `decideTargets()` directly with legal candidates, and multi-target spells use actual min/max counts from `TargetRestrictions`. No decisions fall through to the heuristic AI.

**Results.** PPO training is pending after imitation learning retraining on the current data.

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

**Action granularity.** Priority decisions capture the full mechanically-legal candidate set. The mechanical check (timing + cost) is broader than the AI's strategic evaluation — some candidates pass the mechanical check but would be strategically poor. This means the RL model sees options the heuristic wouldn't consider, which is both an opportunity (discovering unconventional plays) and a risk (learning from noisy candidates). The RL model now handles its own targeting via `decideTargets()` rather than relying on heuristic targeting, eliminating the previous "targeting gap" where the heuristic would veto RL spell choices.

**Hidden information.** Our current feature encoding does not model uncertainty over the opponent's hidden hand. The model receives what a legal player would see (hand size, not contents), but has no explicit mechanism for probabilistic reasoning about hidden cards.

**Computational scale.** Our single-GPU setup limits model size and training throughput. The progressive scaling plan targets 150M parameters, which approaches the practical limit of a 10GB GPU with mixed precision.

### 6.3 Future Work

**Priority head refinement.** The priority candidate set currently uses the engine's mechanical validation (timing + cost). Future work could split this into a "strategically reasonable" tier (candidates passing AI heuristic checks) versus a "creative play" tier (mechanically legal but heuristic-rejected), allowing the model to explore unconventional lines while still grounding training in reasonable play.

**Opponent modeling.** Adding a recurrent component that maintains a hidden state across the game, enabling the model to build beliefs about the opponent's hand based on observed play patterns.

**Deck building.** Extending the system to not just play games but construct decks, using the value network to evaluate card choices in the context of a deck archetype.

**Multi-format support.** Training on Commander (100-card singleton, multiplayer), Draft (card selection from packs), and other MTG formats with different strategic considerations.

---

## 7. Conclusion

We have presented a hierarchical reinforcement learning system for Magic: The Gathering, implemented as an 11M-parameter model with a shared transformer encoder and 7 specialized decision heads. The system integrates with the Forge game engine (32,300 cards) through a `PlayerControllerRL` that overrides all decision methods, and deploys via 9 ONNX model files for real-time Java-native inference without Python dependency.

Our imitation learning pipeline trains all 7 decision heads from ~127K heuristic AI trajectory records: priority (95.7% accuracy on 104K samples), attack (82.8% on 9.5K), target (74.3% on 5K), card select (76.5% on 650), block (64.2% on 2.8K), mulligan (99.0% on 2.6K), and binary (80.9% on 2K). The model handles the complete decision space — no decisions fall through to the heuristic AI.

Key data quality improvements include game-level train/val splitting (preventing leakage from P1/P2 perspectives of the same game), corrected feature encoding (duplicate ApiType fix, aura targeting via enchant keywords, `is_sorcery_speed` global feature), and multi-target spell support (Searing Blaze).

The key architectural insights are: (1) decomposing the MTG decision space into specialized heads — each with appropriate loss functions (softmax CE for single-select priority, BCE for multi-select combat, per-blocker CE for assignment) — is more tractable than a monolithic policy; (2) the transformer's set-attention mechanism naturally handles variable-size card collections; and (3) recording the full mechanically-legal candidate set (not just the heuristic's choice) gives the RL model visibility into creative plays the heuristic would never consider.

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
| Card feature dimension | 256 |
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
| PPO clip epsilon | 0.1 |
| Discount factor (γ) | 0.999 |
| GAE lambda (λ) | 0.95 |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.005 |

## Appendix B: Card Feature Index Reference

Full 256-dimension card feature vector specification is provided in `CardFeatures.java` with inline documentation for each index range.

## Appendix C: Software Availability

The complete implementation is available at https://github.com/austinio7116/forge/tree/ai_investigation, including:
- Java integration code (`forge-ai-rl/src/main/java/`)
- Python model and training code (`forge-ai-rl/src/main/python/`)
- Pipeline scripts (`forge-ai-rl/scripts/`)
- Test decks (`rl_data/decks/`)
- Architecture plan (`RLAI_PLAN.md`)
- Development notes (`CLAUDE.md`)
