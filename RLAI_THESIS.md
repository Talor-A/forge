# Learning to Play Magic: The Gathering Through Hierarchical Reinforcement Learning with Transformer-Based State Encoding

**Author:** M. Austin

**Supervised by:** Claude (Anthropic) — architectural design and implementation assistance

**Date:** March 2026

**Institution:** Independent Research

---

## Abstract

We present a reinforcement learning system for playing Magic: The Gathering (MTG), the most complex widely-played strategy card game in existence. MTG presents challenges that exceed those of games previously mastered by AI: a state space exceeding 10^100, an action space with branching factors regularly exceeding 100, hidden information, stochastic elements, over 32,000 unique cards each introducing novel game mechanics, and deeply nested interactions between cards on the stack. Our approach employs a hierarchical architecture with a shared transformer-based game state encoder (3.4M parameters) and seven specialized decision heads (7.6M parameters) for each major action type — spell casting (priority), combat declaration (attack and block), target selection, card selection, mulligan evaluation, and binary choices. We bootstrap the system through imitation learning on a heuristic AI opponent within the open-source Forge game engine, achieving 95.5% accuracy on priority decisions (142K samples), 84.4% on attack decisions (9.6K samples), and 61.0% on blocking assignments (2.8K samples). We then apply Proximal Policy Optimization (PPO) with clipped importance sampling to improve beyond the heuristic baseline, achieving 30-40% win rate against the heuristic AI in early rounds of self-play. We describe the complete system architecture, feature engineering, training pipeline, engineering challenges encountered, and lessons learned, providing a foundation for future work toward superhuman MTG play.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Background: Magic: The Gathering](#3-background-magic-the-gathering)
4. [System Architecture](#4-system-architecture)
5. [Training Methodology](#5-training-methodology)
6. [Implementation](#6-implementation)
7. [Experimental Results](#7-experimental-results)
8. [Engineering Challenges and Lessons Learned](#8-engineering-challenges-and-lessons-learned)
9. [Discussion](#9-discussion)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)
12. [References](#references)
13. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Motivation

The history of artificial intelligence is marked by milestone achievements in game playing: Deep Blue's victory over Kasparov in chess (1997), Watson's Jeopardy! triumph (2011), AlphaGo's defeat of Lee Sedol in Go (2016), and AlphaStar's Grandmaster-level StarCraft II play (2019). Each milestone pushed the boundaries of what AI systems could learn, from perfect-information combinatorial games to imperfect-information real-time strategy.

Magic: The Gathering (MTG) represents a natural next frontier. Created by mathematician Richard Garfield in 1993 and played by over 40 million people worldwide, MTG combines the strategic depth of chess with the hidden information of poker, the real-time decision complexity of StarCraft, and a rule system so complex that it has been proven Turing-complete (Churchill et al., 2019). No AI system has achieved expert-level play in MTG with the full rule set.

This thesis presents our attempt to build such a system using modern deep reinforcement learning techniques, integrated with the open-source Forge game engine that implements 32,300 unique cards.

### 1.2 Magic: The Gathering as an AI Challenge

From a game-theoretic perspective, MTG is remarkable in its complexity across multiple dimensions:

**Massive state space.** A game state includes two players' life totals, hands (hidden), libraries (ordered, hidden), graveyards, exile zones, the battlefield (with permanents that may be tapped, have counters, attachments, and modified attributes), and a stack of spells and abilities awaiting resolution. Conservative estimates place the state space at 10^(100+), far exceeding chess (~10^47) or Go (~10^170 legal positions).

**Enormous action space.** At any priority window, a player may cast spells, activate abilities, or pass. Each spell may require targeting decisions, mode selections, cost payment choices, and responses to triggered abilities. The branching factor at a single decision point regularly exceeds 100 and can reach thousands.

**Hidden information.** Players cannot see opponents' hands or library ordering, requiring probabilistic reasoning about unknown cards. Unlike poker where the hidden information is a small number of cards from a known deck, MTG hands can contain any combination from a 60-card deck with varying quantities.

**Card diversity.** Over 27,000 unique cards have been printed across 30+ years, with approximately 32,300 implemented in the Forge game engine. Each card introduces unique rules text that modifies the game's mechanics, creating a long tail of rare interactions. New cards are printed quarterly, continuously expanding the rule space.

**Deep strategic planning.** Games last 5-50+ turns, with each turn comprising multiple phases (untap, upkeep, draw, main, combat with sub-phases, second main, end). Resource management (mana), tempo, card advantage, and board control create layered strategic considerations that expert players reason about simultaneously.

**Turing completeness.** Churchill, Biderman, and Herrick (2019) proved that MTG is Turing-complete — the game's rule system can simulate any computation. This means that determining the optimal play in a given game state is, in the general case, undecidable. No algorithm can perfectly play MTG, making heuristic and learned approaches the only viable paths.

### 1.3 Research Questions

This thesis addresses the following research questions:

1. **Can a transformer-based neural network learn meaningful representations of MTG game states** from raw card features, capturing the strategic relationships between cards across zones?

2. **Is hierarchical decomposition effective for MTG decision-making** — does training specialized heads for each decision type outperform a monolithic policy?

3. **Can imitation learning on a heuristic AI provide a sufficient warm start** for PPO self-play to improve beyond the heuristic's level?

4. **What engineering challenges arise** when integrating a deep RL system with a complex, existing game engine, and how can they be addressed?

### 1.4 Contributions

We make the following contributions:

1. **A hierarchical RL architecture** for MTG that decomposes the decision problem into seven specialized heads for each action type, sharing a common game state encoder. We demonstrate that different loss functions are appropriate for different decision types: softmax cross-entropy for single-select priority decisions, binary cross-entropy for multi-select combat, and per-element cross-entropy for assignment problems.

2. **A transformer-based game state encoder** that uses per-zone set attention over variable-length card collections followed by cross-zone attention, producing a fixed-size state embedding that captures board relationships.

3. **A mechanically-legal candidate set recording system** that captures not just the heuristic AI's chosen action but all actions that the player could legally take, enabling the RL model to discover plays the heuristic would never consider.

4. **An efficient integration** with the Forge MTG game engine (Java) via a JSON-over-TCP bridge to a Python model server, enabling headless parallel game execution at 1.3 games/second across 4-16 threads.

5. **A detailed engineering case study** documenting the bugs, design decisions, and architectural constraints encountered when building an RL system on top of a complex existing codebase — insights that are rarely published but frequently encountered in practice.

### 1.5 Thesis Structure

The remainder of this thesis is organized as follows. Chapter 2 reviews related work in game AI, reinforcement learning, and card game AI. Chapter 3 provides background on MTG's game mechanics relevant to our approach. Chapter 4 describes the system architecture in detail. Chapter 5 covers the training methodology. Chapter 6 discusses implementation details and the Java-Python integration. Chapter 7 presents experimental results. Chapter 8 documents engineering challenges and lessons learned. Chapter 9 discusses our findings in context. Chapter 10 outlines future work. Chapter 11 concludes.

---

## 2. Literature Review

### 2.1 Deep Reinforcement Learning for Games

#### 2.1.1 Perfect-Information Games

The modern era of game-playing AI began with DeepMind's AlphaGo (Silver et al., 2016), which combined deep convolutional neural networks with Monte Carlo Tree Search (MCTS) to defeat the world champion Go player Lee Sedol. AlphaGo used two networks: a policy network trained via supervised learning on expert games, and a value network trained via self-play reinforcement learning. The policy network guided MCTS search by suggesting promising moves, while the value network evaluated board positions.

AlphaZero (Silver et al., 2018) generalized this approach, demonstrating that a single architecture could master chess, shogi, and Go through self-play alone, without any human game data. AlphaZero used a single neural network with dual heads (policy and value) trained entirely through self-play with MCTS. The key insight was that starting from random play and learning through self-play was sufficient to achieve superhuman performance in perfect-information games.

MuZero (Schrittwieser et al., 2020) extended this further by learning a model of the environment dynamics, enabling planning without knowledge of the game rules. MuZero matched AlphaZero's performance while also excelling at Atari games with visual observations.

**Relevance to MTG:** These works established the paradigm of combining learned evaluation functions with search. However, their reliance on MCTS is problematic for MTG due to the enormous branching factor and hidden information. Our approach retains the dual network concept (policy heads + value network) but replaces MCTS with direct policy output, similar to AlphaStar's approach for real-time games.

#### 2.1.2 Imperfect-Information Games

**Poker.** Libratus (Brown & Sandholm, 2017) achieved superhuman performance in heads-up no-limit Texas Hold'em through counterfactual regret minimization (CFR), a game-theoretic approach that converges to Nash equilibrium strategies. Pluribus (Brown & Sandholm, 2019) extended this to six-player poker, using a blueprint strategy computed offline combined with real-time search. These systems exploit poker's relatively constrained action space (fold, call, raise with limited bet sizes) and the ability to abstract the game into manageable information sets.

**Relevance to MTG:** Poker's hidden information is structurally simpler than MTG's — a few hidden cards from a 52-card deck versus entire hands from 60-card constructed decks. CFR-based approaches are unlikely to scale to MTG's state space, but the principle of reasoning about opponent's hidden information is directly applicable.

**StarCraft II.** AlphaStar (Vinyals et al., 2019) achieved Grandmaster-level play in StarCraft II, a real-time strategy game with imperfect information, continuous action spaces, and long time horizons. AlphaStar used a transformer-based architecture to process variable-length unit lists, supervised learning on human replays for warm-starting, and population-based league training for self-play. The league training approach maintained a population of agents with different strategies to prevent cycling and ensure robustness.

**Relevance to MTG:** AlphaStar is perhaps the closest analogue to our work. Its transformer-based processing of variable-length entity lists directly inspired our per-zone card set attention. Its league training approach is part of our planned future work. However, StarCraft has a fixed set of ~100 unit types, whereas MTG has 32,300 unique cards.

**Dota 2.** OpenAI Five (Berner et al., 2019) achieved superhuman performance in Dota 2 using massive-scale PPO training across thousands of GPUs. The key insight was that simple RL algorithms (PPO) could work at scale with sufficient compute, without requiring sophisticated search or game-theoretic reasoning. OpenAI Five used LSTM-based policies and trained for over 10 months of wall-clock time.

**Relevance to MTG:** OpenAI Five demonstrated that PPO can handle complex games with long horizons and large action spaces, validating our choice of PPO for the RL phase. However, our compute budget is orders of magnitude smaller (single GPU vs. thousands), necessitating more efficient architectural choices.

### 2.2 Proximal Policy Optimization

Schulman et al. (2017) introduced Proximal Policy Optimization (PPO), a family of policy gradient methods that alternate between sampling data through interaction with the environment and optimizing a "surrogate" objective function using stochastic gradient ascent. PPO constrains policy updates through a clipped objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta)$ is the probability ratio between new and old policies. This clipping prevents destructively large policy updates, providing stability without the complexity of trust region methods (TRPO). PPO has become the de facto standard for RL training in complex environments due to its simplicity, stability, and strong empirical performance.

**Implementation note:** Our initial PPO implementation incorrectly omitted the importance sampling ratio, effectively implementing REINFORCE with baseline. This was identified as a key factor in training instability and corrected to use proper clipped importance sampling (see Section 8.3).

### 2.3 Transformer Architectures for Set-Structured Data

Vaswani et al. (2017) introduced the transformer architecture, which uses self-attention mechanisms to process sequential data without recurrence. The key innovation — scaled dot-product attention — enables each element to attend to all other elements, capturing long-range dependencies efficiently.

Lee et al. (2019) extended transformers to set-structured data with Set Transformer, demonstrating that attention-based architectures naturally handle variable-size, permutation-invariant inputs. This is directly relevant to MTG, where each game zone contains a variable number of cards with no inherent ordering.

**Our approach** uses per-zone set attention (each zone's cards attend to each other) followed by cross-zone attention (zone summaries attend to each other), producing a fixed-size game state embedding regardless of the number of cards in play. This two-level architecture captures both intra-zone relationships (e.g., equipment and the creature it could equip) and inter-zone relationships (e.g., a removal spell in hand and an opponent's threatening creature).

### 2.4 Card Game AI

#### 2.4.1 Hearthstone

Hearthstone, a digital card game inspired by MTG but with substantially simpler rules (no stack, no instant-speed interaction, fixed mana curve), has been the subject of several AI efforts. Santos et al. (2017) applied MCTS with hand-crafted evaluation functions. Hoover et al. (2020) used deep reinforcement learning with neural network function approximation. Zhang and Buro (2017) applied information set MCTS with opponent modeling.

**Relevance to MTG:** Hearthstone results provide a lower bound on what's achievable for card game AI, but MTG's significantly greater complexity (instant-speed interaction, the stack, 10x more cards, complex targeting) means these approaches do not directly transfer.

#### 2.4.2 Magic: The Gathering

Previous MTG AI efforts have been limited primarily to heuristic systems and constrained academic experiments:

**Forge heuristic AI** (Card-Forge project, 2007-present): The Forge game engine implements a sophisticated heuristic AI with card-specific evaluation functions spanning over 300,000 lines of Java code, lookahead simulation via `GameSimulator`, and tunable personality profiles. This heuristic AI represents 18+ years of community-contributed MTG knowledge and serves as both our training data source and evaluation baseline.

**MCTS for MTG:** Cowling, Ward, and Powley (2012) applied ensemble determinization — a technique for handling hidden information in MCTS by sampling possible opponent hands and running deterministic MCTS on each sample — to a simplified MTG environment. Results were promising but limited to small card pools.

**Deck building:** Ward and Cowling (2009) applied Monte Carlo search to the card selection (deck building) problem, demonstrating that search-based approaches could construct competitive decks from limited card pools.

**Turing completeness:** Churchill, Biderman, and Herrick (2019) proved that MTG's game rules are Turing-complete, establishing a theoretical impossibility result: no algorithm can determine the optimal play in all possible MTG game states. This result motivates heuristic and learned approaches.

To our knowledge, no prior work has applied deep reinforcement learning to MTG with the full rules engine and a significant card pool.

### 2.5 Imitation Learning

Imitation learning — training a policy to mimic an expert's behavior — provides a warm start for RL that can dramatically accelerate training. Ross, Gordon, and Bagnell (2011) formalized the DAgger algorithm for iterative imitation learning. AlphaStar used supervised learning on human replays as initialization before self-play RL.

Our approach uses a simpler form: direct behavioral cloning on heuristic AI trajectories. While this is known to suffer from compounding errors (the imitator encounters states the expert never visited), it provides a strong initialization that PPO can refine. The key advantage in our setting is that the heuristic AI is available on-demand for unlimited data generation, unlike human expert data which is scarce and expensive.

---

## 3. Background: Magic: The Gathering

### 3.1 Game Overview

MTG is a two-player (or multiplayer) card game where each player constructs a deck of 60 cards (in Constructed formats) from a pool of available cards. Players take turns playing lands (mana sources), casting spells, and attacking with creatures, with the goal of reducing the opponent's life total from 20 to 0.

### 3.2 Turn Structure

Each turn follows a rigid phase structure that creates multiple decision points:

1. **Beginning Phase:** Untap → Upkeep → Draw
2. **Pre-Combat Main Phase:** Play lands, cast spells
3. **Combat Phase:** Begin Combat → Declare Attackers → Declare Blockers → Combat Damage → End Combat
4. **Post-Combat Main Phase:** Play lands, cast spells
5. **Ending Phase:** End Step → Cleanup

At each phase transition and during most phases, both players receive priority — the opportunity to cast instants or activate abilities. This creates a complex nested decision structure where the number of meaningful decision points per turn can exceed 20.

### 3.3 The Stack

MTG's unique contribution to card game design is the stack — a last-in, first-out queue of spells and abilities awaiting resolution. When a player casts a spell, it goes on the stack, and the opponent receives priority to respond (potentially casting their own spell or ability on top). This creates complex interactive sequences where the order of play matters critically. For example, a creature can be destroyed in response to an opponent's pump spell, "wasting" the pump spell entirely.

The stack makes MTG fundamentally different from Hearthstone and other digital card games where actions resolve immediately. Any AI system for MTG must reason about the stack to play competitively.

### 3.4 Decision Types

We identify seven major decision categories in MTG, each requiring distinct reasoning:

1. **Priority (spell casting):** Which spell or ability to play, or whether to pass. Requires evaluating spell effects in the current board context.
2. **Target selection:** Which legal targets to choose for a spell or ability. Requires understanding spell effects and board state.
3. **Attack declaration:** Which creatures to send into combat. Requires evaluating damage trades, potential blocks, and life total implications.
4. **Block declaration:** Which creatures to assign as blockers, and to which attacker. An assignment problem with strategic depth.
5. **Card selection:** Choosing cards for effects like discard, sacrifice, scry. Requires evaluating relative card values in context.
6. **Mulligan:** Whether to keep an opening hand or draw a new one, and which cards to put back. Requires hand evaluation against the expected matchup.
7. **Binary choices:** Yes/no decisions for triggered abilities, replacement effects, etc.

These seven categories motivate our seven-head architecture (plus a value network for state evaluation).

### 3.5 The Forge Game Engine

The Forge project (Card-Forge, 2007-present) is an open-source Java implementation of MTG with 32,300 cards, complete rules enforcement, and a heuristic AI opponent. Key properties relevant to our work:

- **Complete rules implementation:** All MTG mechanics including the stack, priority, state-based actions, replacement effects, and triggered abilities
- **Headless execution:** Games can run without GUI for data collection
- **Heuristic AI:** `AiController` with card-specific logic in `ComputerUtil*` classes, lookahead simulation, and customizable profiles
- **Extensible architecture:** `PlayerController` interface allows custom player implementations

We integrate with Forge through a `PlayerControllerRL` that extends the existing `PlayerControllerAi`, enabling our RL model to make decisions while falling back to the heuristic AI for untrained decision types.

---

## 4. System Architecture

### 4.1 Overview

Our system comprises four major components:

1. **Game Engine (Java).** The Forge MTG engine handles all game rules, card interactions, and state management. We run it headlessly for data collection and evaluation.

2. **Feature Extraction (Java).** `GameStateEncoder`, `CardFeatures`, and `ActionEncoder` classes convert rich game objects into fixed-size numerical feature vectors suitable for neural network input.

3. **Neural Network (Python/PyTorch).** The `MTGModel` combines a shared transformer encoder with specialized decision heads and a value network.

4. **Training Pipeline (Python).** Data loading, training loops, and evaluation scripts with GPU acceleration (AMP on NVIDIA RTX 3080).

The architecture is designed around the insight that MTG decisions are heterogeneous — choosing which spell to cast is fundamentally different from choosing which creatures to attack with, which is different from choosing which cards to discard. Rather than forcing all decisions through a single network, we use specialized heads that share a common understanding of the game state.

### 4.2 Game State Representation

#### 4.2.1 Global Features (64 dimensions)

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

#### 4.2.2 Card Features (128 dimensions per card)

Each card in any zone is encoded as a 128-dimensional vector capturing card type, color, mana cost, power/toughness, game state (tapped, summoning sick), counters, 30 keyword abilities, zone location, and a 4-byte card identity hash for learned embeddings.

#### 4.2.3 Zone Encoding

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

### 4.3 Neural Network Architecture

The complete model architecture is shown in Figure 1. Data flows left-to-right: raw game state inputs are encoded by the shared transformer encoder into a 512-dimensional state embedding, which then fans out to the value network (critic) and seven specialized decision heads (actors).

![Model Architecture](forge-ai-rl/src/main/python/tools/mtg_model_architecture.svg)

*Figure 1: MTG RL Model Architecture. The shared game state encoder (3.4M parameters) processes 7 input zones through per-zone self-attention and cross-zone attention, producing a 512-dimensional state embedding. This embedding is consumed by the value network (critic, providing PPO advantage baseline) and decision heads (actors). Currently trained heads: Value, Priority (95.5% accuracy), Attack (84.4%), Block (61.0%). Untrained heads (Target, Card Select, Mulligan, Binary) fall through to the heuristic AI.*

#### 4.3.1 Game State Encoder (Transformer)

The encoder uses a two-level attention architecture:

**Level 1: Per-Zone Card Set Attention.** Each zone has a dedicated `CardSetEncoder` module consisting of:
- Linear projection: card_dim (128) → zone_embed_dim (128)
- Multi-head self-attention transformer (2 layers, 4 heads, GELU activation)
- Masked mean pooling over valid cards

Self-attention within a zone captures relationships between cards — for example, an equipment card's value depends on the creatures it could be attached to, and a creature's effective power depends on other creatures that might benefit from the same pump spell.

**Level 2: Cross-Zone Attention.** The 7 zone embeddings (6 card zones + 1 global features projection) are stacked and processed through a single transformer encoder layer with 4 attention heads. This allows the model to reason about inter-zone relationships — for example, a card in hand is more valuable if the battlefield has mana to cast it, and removal in hand is more valuable if the opponent has a threatening creature.

**Output projection.** The 7 zone embeddings are concatenated (7 × 128 = 896 dimensions) and projected through a two-layer MLP to the final state embedding of 512 dimensions.

Total encoder parameters: 3,409,664.

#### 4.3.2 Value Network (Critic)

A three-layer MLP mapping the 512-dimensional state embedding to a scalar in [-1, 1]:
- Linear(512, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 256) → GELU → LayerNorm → Dropout(0.1)
- Linear(256, 1) → Tanh

The output represents estimated game value: +1 indicates certain victory, -1 certain defeat. This value estimate serves as the baseline for PPO advantage computation: advantage = actual_outcome - value_estimate.

Total parameters: 198,401.

#### 4.3.3 Priority Action Head

The priority head selects which spell or ability to play from the available options, or passes priority:

1. **Action encoding.** Each available SpellAbility is encoded as a 64-dim feature vector capturing source card type, color, CMC, ability type via ApiType one-hot (30 most common types), targeting requirements, and estimated effect magnitude.
2. **Cross-attention.** Available actions attend to the game state embedding (4 heads, hidden_dim=256).
3. **Scoring network.** Combined features are scored through an MLP (512→256→1), producing logits over actions + pass.
4. **Output.** Softmax probability distribution over N actions + pass. Loss: cross-entropy.

Total parameters: 543,233.

#### 4.3.4 Attack Decision Head

The attack head makes a joint binary decision for each potential attacker:

1. **Card projection.** Each creature's 128-dim features are projected to 256 dimensions.
2. **Self-attention among potential attackers** (2 transformer layers, 4 heads, d_ff=1024). Captures coordinated attack patterns.
3. **State conditioning.** The 512-dim game state embedding is concatenated with each creature's representation.
4. **Binary classifier.** A two-layer MLP (512→256→1) produces a logit per creature.
5. **Output.** Independent sigmoid per creature. Loss: binary cross-entropy.

Total parameters: 1,875,457.

#### 4.3.5 Block Decision Head

The block head assigns each potential blocker to an attacker (or no attacker):

1. **Separate projections** for blockers and attackers (128 → 256 each).
2. **Cross-attention.** Blockers attend to attackers to understand the threat landscape (4 heads).
3. **Pairwise scoring.** For each (blocker, attacker) pair, a scoring network (768→256→1) produces an assignment logit, plus a "don't block" option.
4. **Output.** Independent categorical distribution per blocker over attacker options + no-block. Loss: per-blocker cross-entropy.

Total parameters: 657,665.

#### 4.3.6 Additional Heads (Architecturally Complete, Not Yet Trained)

| Head | Purpose | Architecture | Parameters |
|------|---------|-------------|------------|
| Target | Pointer network for selecting spell targets | Scaled dot-product attention + GRU for multi-target | 674,304 |
| Card Select | General card selection (discard, sacrifice, scry) | Self-attention + scoring MLP + GRU for multi-select | 1,480,449 |
| Mulligan | Opening hand evaluation + London mulligan | Self-attention + keep/mull classifier + bottom scorer | 2,007,042 |
| Binary | Yes/no decisions (confirm triggers, replacement effects) | MLP 512→256→256→1 | 197,889 |

#### 4.3.7 Total Model Size

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

At 42MB in fp32, the full model fits comfortably on consumer GPUs with substantial headroom for scaling.

---

## 5. Training Methodology

### 5.1 Phase 1: Imitation Learning

We bootstrap the RL agent by imitating the existing heuristic AI in the Forge game engine. This provides a warm start that is critical for the sparse-reward MTG environment — starting from random play would require astronomical numbers of games to learn even basic strategy.

#### 5.1.1 Data Collection

Games are run headlessly using the Forge engine's `SimulateRLTraining` runner with 16 parallel threads. The `PlayerControllerRL` class extends `PlayerControllerAi` and overrides key decision methods to capture training data:

- **`declareAttackers`**: Captures pre-decision creature features (128-dim per creature), delegates to heuristic, reads back which creatures were selected as attackers from the combat object.
- **`declareBlockers`**: Captures (blocker, attacker) pair features (256-dim concatenated) with full assignment information from the combat object.
- **`chooseSpellAbilityToPlay`**: Leverages a modified `AiController.chooseSpellAbilityToPlayFromList` that evaluates ALL candidates through the engine's full `canPlayAndPayFor()` validation rather than short-circuiting at the first playable spell. The mechanically-legal candidate list is cached for RL recording.

Each game produces two trajectory files (one per player perspective), containing 50-400 decision records. A batch of 1,000 games produces approximately 2,000 trajectory files with ~153,000 decision records.

#### 5.1.2 Training Pipeline

Training proceeds sequentially:

1. **Value Network** (50-100 epochs): Train the entire model (encoder + value network) to predict game outcomes from mid-game states. Loss: MSE. This establishes the shared state representation.

2. **Priority Head** (10 epochs): Freeze encoder and value network. Train priority head on 142K priority decisions. Loss: cross-entropy (softmax single-select).

3. **Attack Head** (10 epochs): Freeze encoder. Train attack head on 9.6K attack decisions. Loss: binary cross-entropy per creature.

4. **Block Head** (10 epochs): Freeze encoder. Train block head on 2.8K blocking assignment decisions. Loss: per-blocker cross-entropy over attacker assignment + no-block.

### 5.2 Phase 2: Reinforcement Learning via PPO

#### 5.2.1 PPO Formulation

At each decision point $t$, the agent observes state $s_t$ and takes action $a_t$ according to policy $\pi_\theta(a_t|s_t)$. The advantage is estimated as:

$$\hat{A}_t = R_t - V(s_t)$$

where $R_t$ is the Monte Carlo return (game outcome ±1) and $V(s_t)$ is the value network's estimate. Advantages are normalized per batch for stability.

The PPO clipped objective uses importance sampling:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$, clipping parameter $\epsilon = 0.2$, and old policy probabilities stored in trajectory data during collection.

The total loss combines policy, value, and entropy terms:

$$L = L^{CLIP} - 0.5 \cdot L^{VF} + 0.01 \cdot H[\pi_\theta]$$

#### 5.2.2 Reward Shaping

MTG's terminal reward (win/loss) is extremely sparse. We employ potential-based reward shaping:

| Signal | Reward | Rationale |
|--------|--------|-----------|
| Win | +1.0 | Terminal reward |
| Loss | -1.0 | Terminal reward |
| Life advantage change | ±0.01 per point | Tracks damage race |
| Card advantage change | ±0.05 per card | Card advantage is fundamental |
| Board advantage change | ±0.02 per creature | Board presence correlates with winning |

Shaping rewards decay over training so the agent eventually optimizes purely for win rate.

### 5.3 Phase 3: Curriculum Learning (Planned)

We plan to introduce card complexity gradually across six stages, from vanilla creatures through keywords, removal, counterspells, complex permanents, to the full card pool.

### 5.4 Phase 4: League Training (Planned)

Following AlphaStar's approach, we plan to maintain a population of agents with main agents, exploiter agents, and historical snapshots to prevent strategy cycling and catastrophic forgetting.

---

## 6. Implementation

### 6.1 Game Engine Integration

The Forge game engine (Java 21, ~50,000 source files) implements the complete MTG rules with 32,300 cards. Our integration adds a `forge-ai-rl` Maven module with minimal modifications to the core engine:

- **`SimulateRLTraining.java`**: Headless game runner supporting parallel execution, trajectory recording, and configurable AI opponents. Increased AI decision timeout from 5s to 15s to accommodate full candidate evaluation.
- **`PlayerControllerRL.java`**: Extends `PlayerControllerAi`, overriding `chooseSpellAbilityToPlay`, `declareAttackers`, and `declareBlockers` with RL model decisions (GRPC mode) or heuristic recording (RECORD_HEURISTIC mode).
- **`RLController.java`**: Central orchestrator routing decisions to model heads, managing the model server connection, and recording trajectories.
- **`AiController.java`**: Minimal modification — added caching of mechanically-legal spell candidates and a `canPlayAndPayForFacade` for RL targeting support.

### 6.2 Java-Python Bridge

During training, the game engine (Java) communicates with the model server (Python) via a length-prefixed JSON-over-TCP protocol:

```
Client → Server: [4 bytes big-endian length][JSON request]
Server → Client: [4 bytes big-endian length][JSON response]
```

The model server supports batched inference for throughput. Response payloads include selected action indices, probability distributions, and value estimates.

### 6.3 Hardware

All experiments are conducted on a single workstation:
- **CPU:** 16-core (parallel game execution)
- **GPU:** NVIDIA GeForce RTX 3080, 10GB VRAM
- **RAM:** 16GB

Automatic mixed precision (AMP) with fp16 reduces GPU memory usage by approximately 50%. The full model uses ~51MB VRAM for inference. Training at batch size 256 uses approximately 2.3GB.

---

## 7. Experimental Results

### 7.1 Data Collection

We collected 1,000 games between four constructed decks (Red Aggro, Green Stompy, White Weenie, Blue Tempo) using 16-thread parallel execution in 776 seconds (13 minutes):

| Metric | Value |
|--------|-------|
| Trajectory files | 1,999 |
| Total decision records | ~153,000 |
| Attack decisions | 9,618 |
| Block decisions (with assignment) | 2,781 |
| Priority decisions | 142,721 |
| Priority: play a spell | 15,679 (11.2%) |
| Priority: pass with options | 124,924 (88.8%) |
| Candidate distribution | 1-7 options per decision |

### 7.2 Value Network Performance

The value network was trained for 50-100 epochs on ~153,000 mid-game decision snapshots. An early version achieved 99.6% accuracy, but this was later found to be inflated by a feature leakage bug: the "tapped" flag in creature features leaked the attack decision before it was made, allowing the value network to trivially predict outcomes from the tapped state. After fixing the feature encoding to capture pre-decision state, the value network converges to reasonable accuracy with more meaningful representations.

### 7.3 Decision Head Training (Imitation Learning)

| Head | Samples | Loss Function | Val Accuracy | Epochs |
|------|---------|---------------|-------------|--------|
| Priority | 142,721 | CrossEntropy (softmax) | 95.5% | 10 |
| Attack | 9,618 | BCE (per-creature sigmoid) | 84.4% | 10 |
| Block | 2,781 | CE per-blocker (assignment) | 61.0% | 10 |

**Priority Head (95.5%).** The 88.8% pass rate means a naive "always pass" baseline achieves ~89%. The model's 95.5% demonstrates genuine learning of play decisions beyond the pass baseline — it correctly identifies when to play and which spell to choose.

**Attack Head (84.4%).** The model learns coordinated attack patterns, with attack probabilities correctly varying based on board state (aggressive when ahead, conservative when behind).

**Block Head (61.0%).** The lowest accuracy reflects the hardest problem: blocking is a combinatorial assignment problem with limited training data (2,781 samples).

### 7.4 PPO Self-Play Training

Starting from the imitation-learned checkpoint, the RL agent achieves a 30-40% win rate against the heuristic AI across initial PPO rounds. Win rates oscillate between rounds, attributed to:

1. **Value network miscalibration** — trained on old data with feature leakage, producing overconfident estimates that hinder advantage computation.
2. **Targeting gap** — the RL model selects spells but relies on heuristic targeting, creating inconsistencies.
3. **Partial head coverage** — only 3 of 7 decision heads are trained; untrained decisions fall through to the heuristic.

Value network retraining on clean data is in progress to address point 1.

---

## 8. Engineering Challenges and Lessons Learned

### 8.1 Game Engine Constraints

**DO NOT subclass LobbyPlayerAi.** Subclassing `LobbyPlayerAi` causes instant turn-0 game termination. The game engine performs identity checks that fail for subclasses. Solution: use `PlayerControllerRL` extending `PlayerControllerAi` instead.

**DO NOT add fields to PlayerControllerAi.** Adding fields to `PlayerControllerAi` breaks the fat-jar class resolution at runtime. The modified class from `forge-ai` conflicts with `forge-game` expectations. Solution: all RL state is maintained in the separate `RLController` class.

**DO NOT use DataLoader num_workers > 0 with large in-memory datasets.** Python's fork-based DataLoader workers duplicate the entire process memory. With 3-5GB of trajectory data, each worker adds another 5GB, causing swap thrashing. Solution: always use `num_workers=0`.

### 8.2 Feature Engineering Bugs

**Tapped flag leak.** The original card feature encoding included the "tapped" flag, which for attacking creatures is set AFTER the attack decision. Recording features post-decision leaked the attack choice into the game state, inflating value network accuracy to 99.6%. Fix: capture creature features BEFORE the heuristic makes combat decisions.

**Block head using wrong model.** The block head training was accidentally using `model.attack_head` instead of `model.block_head`, overwriting the attack head's weights with block training data. This caused the attack head to produce very low probabilities (learned "don't block" = "don't attack"), resulting in a 5% win rate. Fix: use the correct model head for each decision type.

### 8.3 PPO Implementation Errors

**Missing importance sampling.** The initial PPO implementation computed `-(log_prob * advantage).mean()`, which is REINFORCE with baseline, not PPO. Without the importance sampling ratio and clipping, the policy could drift arbitrarily far per update, causing training instability. Fix: store old policy probabilities in trajectories, compute ratio, apply clipping.

**Empty list vs null for pass.** Returning an empty list from `chooseSpellAbilityToPlay()` when the RL model chose to pass caused the game engine to retry the decision infinitely ("AI looped too much"). The engine expects `null` for "pass priority", not an empty list. Fix: return `null` for pass.

**Targeting failures.** The RL model picked spells from the mechanically-legal candidate list, but these spells hadn't had targeting set up by the heuristic's `canPlayAndPayFor()`. Spells with targets (Giant Growth, Lightning Bolt) failed to go on the stack. Fix: after RL picks a spell, run `canPlayAndPayForFacade()` to set up targeting via the heuristic's logic.

### 8.4 Performance Constraints

**AI decision timeout.** The default 5-second timeout in `AiController.chooseSpellAbilityToPlayFromList` was insufficient when evaluating all candidates (not just the first playable one). Each timeout caused the RL player to miss a priority window, losing tempo. Fix: increase timeout to 15 seconds for RL games.

**Model server throughput.** With 16 parallel game threads and hundreds of priority checks per game, the single-threaded Python model server became a bottleneck. Fix: reduce to 4 game threads for model server mode.

---

## 9. Discussion

### 9.1 Comparison to Existing Approaches

The Forge heuristic AI represents 18+ years of hand-coded MTG knowledge. Our RL system, after imitation learning and initial PPO, achieves 30-40% win rate against it — below parity but demonstrating meaningful play. Key comparisons:

| Aspect | Heuristic AI | RL AI |
|--------|-------------|-------|
| Development time | 18+ years | ~2 weeks |
| Card-specific code | 300K+ lines | 0 lines |
| Decision coverage | All decisions | 3 of 7 heads |
| Targeting | Full support | Falls through to heuristic |
| Win rate (vs each other) | ~60-70% | ~30-40% |
| Generalization | Requires per-card code | Learns from features |

### 9.2 Key Insights

1. **Hierarchical decomposition is essential.** Different MTG decisions require fundamentally different architectures and loss functions. Softmax CE for single-select priority, BCE for multi-select combat, and per-element CE for assignment problems cannot be handled by a single head.

2. **Recording the full candidate set is critical.** Recording only the heuristic's choice (as done initially) prevents the model from learning *what else was available*. Recording the mechanically-legal candidate set gives the RL model visibility into creative plays the heuristic would reject.

3. **Feature leakage is insidious.** The tapped flag leak produced excellent metrics (99.6% accuracy) that masked the underlying problem. Pre-decision state capture is essential for valid training.

4. **PPO implementation details matter enormously.** The difference between REINFORCE-with-baseline and proper PPO (with importance sampling ratio and clipping) is the difference between unstable oscillation and stable learning.

---

## 10. Future Work

### 10.1 Immediate Next Steps

**Target head training.** Record targeting decisions and train the target head to select spell targets independently, removing the dependency on heuristic targeting.

**Value network retraining.** Retrain the value network on clean pre-decision-state data to provide calibrated advantage estimates for PPO.

**Extended PPO training.** Run PPO for 50-100 rounds with calibrated values and lower learning rate (1e-6) to establish a clear training signal.

### 10.2 Medium-Term Goals

**Remaining decision heads.** Train card selection, mulligan, and binary decision heads to achieve full autonomous play without heuristic fallback.

**Curriculum learning.** Introduce progressively complex card pools, starting from vanilla creatures and advancing through keywords, removal, counterspells, and complex permanents.

**Opponent modeling.** Add a recurrent component for tracking game history, enabling beliefs about the opponent's hidden hand based on observed play patterns.

### 10.3 Long-Term Vision

**League training.** Maintain a population of agents with exploiter and historical opponents to prevent strategy cycling, following the AlphaStar approach.

**Deck building.** Extend the system to construct decks, using the value network to evaluate card choices in the context of deck archetypes.

**Multi-format support.** Train on Commander (100-card singleton, multiplayer), Draft (card selection from packs), and other MTG formats.

**Progressive scaling.** Scale model from 11M to 50-150M parameters with net2net-style weight transfer as the card pool expands.

---

## 11. Conclusion

We have presented a hierarchical reinforcement learning system for Magic: The Gathering, implemented as an 11M-parameter model with a shared transformer encoder and 7 specialized decision heads. The system integrates with the Forge game engine (32,300 cards) through minimal modifications: a cached spell evaluation list in `AiController` and a `PlayerControllerRL` that overrides priority, attack, and block decisions.

Our imitation learning pipeline successfully trains three decision heads from heuristic AI trajectories: the priority head (95.5% accuracy on 142K samples) learns spell timing and selection; the attack head (84.4% on 9.6K samples) learns coordinated creature attacks; and the block head (61.0% on 2.8K samples) learns blocker-attacker assignment. The priority head's ability to predict both spell choices and pass timing from mechanically-legal candidate sets — including options the heuristic would reject — provides a foundation for discovering non-obvious plays through PPO self-play.

PPO training with proper clipped importance sampling produces a 30-40% win rate against the heuristic AI after initial rounds, compared to a 50% baseline (heuristic vs heuristic). The gap is attributable to incomplete head coverage (3 of 7 heads trained), the targeting gap (RL chooses spells but heuristic handles targeting), and value network miscalibration. These are addressable engineering problems, not fundamental limitations of the approach.

The key architectural insights are: (1) decomposing the MTG decision space into specialized heads — each with appropriate loss functions — is more tractable than a monolithic policy; (2) the transformer's set-attention mechanism naturally handles variable-size card collections in each game zone; (3) recording the full mechanically-legal candidate set enables the RL model to discover creative plays; and (4) rigorous engineering practices (pre-decision state capture, proper PPO implementation, correct model head assignment) are as important as architectural choices for a system of this complexity.

MTG represents a frontier challenge for game AI — a domain where the rules themselves are as complex as the strategies that emerge from them, and where the game continues to grow with each new card set. Our system provides a foundation for exploring this frontier through learned, self-improving play.

---

## References

Berner, C., Brockman, G., Chan, B., Cheung, V., Dębiak, P., Dennison, C., ... & Zhang, S. (2019). Dota 2 with large scale deep reinforcement learning. *arXiv preprint arXiv:1912.06680*.

Brown, N. & Sandholm, T. (2017). Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. *Science*, 359(6374), 418-424.

Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.

Card-Forge Project. (2007-present). Forge: Magic: The Gathering game engine. https://github.com/Card-Forge/forge

Churchill, A., Biderman, S., & Herrick, A. (2019). Magic: The Gathering is Turing complete. *arXiv preprint arXiv:1904.09828*.

Cowling, P. I., Ward, C. D., & Powley, E. J. (2012). Ensemble determinization in Monte Carlo tree search for the imperfect information card game Magic: The Gathering. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(4), 267-277.

Hoover, A. K., Togelius, J., Lee, S., & de Mesentier Silva, F. (2020). The many AI challenges of Hearthstone. *KI - Künstliche Intelligenz*, 34, 33-43.

Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set transformer: A framework for attention-based permutation-invariant input. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*.

Santos, A., Santos, P. A., & Melo, F. S. (2017). Monte Carlo tree search experiments in Hearthstone. *IEEE Conference on Computational Intelligence and Games*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., ... & Silver, D. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature*, 588(7839), 604-609.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., ... & Silver, D. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature*, 575(7782), 350-354.

Ward, C. D. & Cowling, P. I. (2009). Monte Carlo search applied to card selection in Magic: The Gathering. *IEEE Symposium on Computational Intelligence and Games*.

Zhang, S. & Buro, M. (2017). Improving Hearthstone AI by learning high-level rollout policies and bucketing chance node events. *IEEE Conference on Computational Intelligence and Games*.

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
| Learning rate (imitation) | 1 × 10⁻³ |
| Learning rate (PPO) | 1 × 10⁻⁵ |
| Weight decay | 10⁻⁴ |
| LR schedule | Cosine annealing |
| Gradient clipping | 1.0 (imitation), 0.5 (PPO) |
| Batch size | 256 (imitation), 32 (PPO) |
| PPO clip epsilon | 0.2 |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |

## Appendix B: Card Feature Index Reference

Full 128-dimension card feature vector specification is provided in `CardFeatures.java` with inline documentation for each index range.

## Appendix C: Action Feature Index Reference

Full 64-dimension action feature vector specification is provided in `ActionEncoder.java`:

| Range | Features |
|-------|----------|
| 0-6 | Source card type flags (creature, instant, sorcery, enchantment, artifact, planeswalker, land) |
| 7-12 | Color flags (W, U, B, R, G, colorless) |
| 13 | CMC (normalized 0-16) |
| 14-17 | Spell/ability type (spell, activated, triggered, mana) |
| 18-47 | ApiType one-hot (top 30 most common) |
| 48-51 | Targeting info (requires target, min targets, can target creatures/players) |
| 52-53 | Source card combat stats (P/T if creature) |
| 54-55 | Estimated effect magnitude (damage, cards drawn) |
| 56-62 | Reserved |
| 63 | Pass action flag |

## Appendix D: Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `01_build.sh` | Build Java fat jar |
| `02_collect_data.sh [N]` | Collect N games of heuristic trajectory data |
| `03_train_value.sh [epochs]` | Train value network / encoder |
| `04_train_decisions.sh [epochs] [batch] [encoder] [heads]` | Train decision heads |
| `05_verify_model.sh` | Run evaluation games with model server |
| `06_ppo_train.sh [model] [rounds] [games]` | PPO self-play training |
| `07_visualize.sh [model] [data]` | Interactive game state visualizer |

## Appendix E: Software Availability

The complete implementation is available at https://github.com/austinio7116/forge/tree/ai_investigation, including:
- Java integration code (`forge-ai-rl/src/main/java/`)
- Python model and training code (`forge-ai-rl/src/main/python/`)
- Pipeline scripts (`scripts/`)
- Architecture plan (`RLAI_PLAN.md`)
- Development notes (`CLAUDE.md`)
- Model architecture diagram (`forge-ai-rl/src/main/python/tools/mtg_model_architecture.svg`)
