# RL AI — Potential Improvements

## Current Baseline (2026-03-26)

- **Win rate vs heuristic:** 30-39% (peaked at 39% round 21 of PPO vs heuristic)
- **Per-deck:** White 46%, Green 32%, Red 21-40%, Blue 5-15%
- **Spells/turn:** 0.50-0.57 (less than 1 spell per turn for aggro decks)
- **Attack rate:** 55% of available creatures sent, 31% hold-back rate
- **PPO plateaued** after 43 rounds (~17,200 games) with no upward trend

---

## 1. Frozen Encoder — The Biggest Bottleneck

**Problem:** The `GameStateTransformer` encoder is frozen during PPO (ppo_ui.py:149-151). The 512-dim game state embedding was learned from imitating heuristic play — it encodes "what matters to the heuristic," not what matters for winning. Decision heads can only work with the representations the frozen encoder provides. If winning requires valuing something the heuristic doesn't (holding mana, spell sequencing), the encoder cannot represent it.

**Analogy:** Training a chess engine but freezing the board evaluation learned from watching amateur games. The policy heads try to find winning moves, but their "eyes" are permanently tuned to amateur priorities.

**Recommendation:** Unfreeze the encoder with a much smaller learning rate (e.g., 1/10th of the head LR). Use `lr=1e-5` for heads, `lr=1e-6` for encoder. This lets representations slowly adapt without catastrophic forgetting. Add a KL penalty (0.01-0.02) against the pre-PPO policy to prevent drift.

**Priority: HIGH | Effort: MODERATE**

---

## 2. GAE Hyperparameters — Insufficient Credit Assignment

**Problem:** Current `gamma=0.95, lambda=0.90`. A typical game has ~50 decisions. With gamma=0.95:
- Decision 30 steps from end: `0.95^30 = 0.21` — only 21% of terminal signal reaches it
- Effective GAE horizon: `1/(1 - gamma*lambda) ≈ 7 steps` — decisions >7 steps from game end get near-zero advantage signal
- Turn 1-3 decisions (creature deployment, mana sequencing) get almost no learning signal

This explains why spells/turn isn't improving — the model gets no gradient for early-game deployment decisions.

**Recommendation:** Increase to `gamma=0.99, lambda=0.95`:
- Decision 30 steps from end: `0.99^30 = 0.74`
- Effective horizon: ~20 steps
- Requires a better value function (see #3), otherwise high-gamma GAE amplifies value estimation errors

**Priority: HIGH | Effort: TRIVIAL**

---

## 3. Value Function Quality — The Foundation of Everything

**Problem:** GAE relies entirely on `V(s)` being accurate. If V is wrong, advantages are noisy, and PPO updates in random directions. The current value network:
- Was trained on imitation data (heuristic play trajectories)
- Uses the frozen encoder's representations
- Is updated during PPO with the same batch of 400 games

The value function is probably undertrained and uses wrong representations. V might think a board state with 3 creatures is "good" regardless of context. V can't distinguish "3 creatures with mana up for counter" from "3 creatures tapped out" if the encoder doesn't differentiate these.

**Recommendations:**
- Train value function for more epochs on collected data (currently 4 PPO epochs, try 10-20 for value only)
- Use separate value network optimizer with higher LR than policy
- Consider value function clipping (PPO paper recommends it) to prevent value from jumping between rounds
- Periodically retrain value from scratch on accumulated trajectories (not just latest round)

**Priority: HIGH | Effort: MODERATE**

---

## 4. Entropy and Exploration — Policy Is Too Deterministic

**Problem:** `entropy_coeff = 0.005` is very small. Logged entropy is 0.2-0.3 and declining. Once the policy becomes near-deterministic:
- PPO has no gradient (ratio is always ~1.0)
- No exploration means no discovery of better strategies
- The 31% hold-back attack rate may be policy collapse ("never attack when uncertain") rather than a learned strategy

**Recommendation:** Increase to `entropy_coeff = 0.02-0.05`. This forces more exploration. Optionally decay over training (start 0.05, end 0.01). Monitor entropy — if it drops below 0.1, the policy is effectively frozen.

**Priority: HIGH | Effort: TRIVIAL**

---

## 5. No Temporal Context — Each Decision Is Amnesiac

**Problem:** Every decision is based solely on the current game state snapshot. The model has zero memory of:
- What it did earlier this turn
- What the opponent did in response to prior actions
- Its own strategic plan ("I'm holding Counterspell, so I should pass")

This is why Blue is at 5-15%. Counter-based play requires multi-step planning: "I have Counterspell in hand -> I should pass priority now -> if opponent plays something good, counter it." The model sees "pass" as equivalent to "I have nothing to do" because the game state looks similar in both cases.

**Recommendations (in order of complexity):**

### 5a. Short-term: Feature Engineering
Add explicit features to the game state encoding:
- "Cards in hand that could be played but aren't being played" (opportunity cost signal)
- "Mana being held open relative to castable spells" (intentional pass vs nothing to do)
- "Number of instant-speed responses available" (reactive potential)

### 5b. Medium-term: Decision History Buffer
Feed the last 5-10 decisions (type + selected action) as additional context to each head. Gives the model "I just played a creature" -> "now I should attack with it."

### 5c. Long-term: Sequence Transformer
Replace per-decision MLP heads with a Transformer over the decision sequence within a turn/game. Architecturally significant but would enable genuine planning.

**Priority: MEDIUM-HIGH | Effort: LOW (5a) to HIGH (5c)**

---

## 6. Sample Efficiency — Not Enough Data Per Update

**Problem:** 400 games x ~50 decisions = ~20K decisions, split across 5+ decision types:
- Priority: ~15K (most common)
- Attack: ~2K
- Block: ~500
- Target: ~1.5K
- Binary: ~500

The attack/block heads see very few samples per PPO update. With 4 epochs and batch_size=64, the block head might only do ~30 gradient steps per round. Not enough to learn anything.

**Recommendations:**
- Increase `games_per_round` to 800-1000 (2x-2.5x data per update)
- Reduce eval frequency (eval every 3 rounds instead of every round) to spend more time collecting
- Use experience replay — keep a buffer of recent rounds (2-3) and sample from all during PPO epochs. Technically off-policy but PPO's clipping handles moderate staleness.

**Priority: MEDIUM | Effort: LOW-MODERATE**

---

## 7. Learning Rate and Optimization

**Problem:** `lr=1e-5`, constant, single AdamW for everything.
- Same LR for all heads despite very different sample counts (priority sees 10x more data than block)
- No warmup — early updates with bad value estimates can push policy in wrong direction
- No decay — if we find a good policy, we keep perturbing it

**Recommendations:**
- Per-head learning rates: priority 1e-5, attack 3e-5, block 5e-5 (inversely proportional to sample frequency)
- Cosine LR schedule with warmup: ramp up over 5 rounds, decay over remaining
- Or reduce LR when plateau detected (3 rounds without improvement -> halve LR)

**Priority: MEDIUM | Effort: MODERATE**

---

## 8. Alternative Algorithmic Approaches

### 8a. Expert Iteration (ExIt) — Highest Ceiling

Instead of pure RL, use search to improve decisions:
1. At each decision point, do shallow Monte Carlo rollouts (play 10-50 random games from this state with different choices)
2. Pick the choice that wins most often
3. Train the policy supervised on these search-improved decisions
4. Repeat

This is how AlphaZero works. No need for good value function or credit assignment — just simulate and see what wins.

**Feasibility:** Java engine runs ~100 games/min with heuristic players. For 50 rollouts per decision x 50 decisions/game = 2500 simulations per training game. At 100 games/min, that's 25 min per training game. Too slow for full ExIt, but could be used selectively for high-value decisions (turn 1-3 plays, attack declarations).

### 8b. Decision Transformer / Offline RL

Frame the problem as sequence modeling:
- Collect a large corpus of games (both wins and losses)
- Train a Transformer to predict: "given this game history and desired outcome (win), what action should I take?"
- At inference, condition on "I want to win" and generate actions

Sidesteps the credit assignment problem entirely. Requires lots of diverse data but no reward shaping or value functions.

### 8c. Hindsight Relabeling

For lost games, ask: "Which single decision, if changed, would most likely have changed the outcome?" Use counterfactual reasoning:
- For each decision in a lost game, simulate the alternative
- If the alternative leads to a win in simulation, that decision gets a strong positive advantage

Expensive but gives targeted signal exactly where the model needs it most.

**Priority: MEDIUM-LOW (long-term) | Effort: HIGH**

---

## 9. Curriculum and Deck Strategy

**Problem:** All 4 decks in random matchups from round 1. Blue Tempo at 5% win rate contributes mostly noise. The model sees 25% of its data from a deck it can barely play, diluting the gradient signal from decks where it's actually learning (White at 46%).

**Recommendations:**

### Phase 1 — Aggro fundamentals (rounds 1-20)
Train on Green Stompy + White Weenie + Red Aggro only. These decks reward the basics: play on curve, attack aggressively, use removal. Remove Blue until the model demonstrates fundamentals.

### Phase 2 — Add complexity (rounds 20-50)
Re-introduce Blue Tempo once the model consistently beats heuristic at >45% with aggro. By then, the value function and encoder are better calibrated.

### Phase 3 — Full league
All decks, mixed self-play and vs-heuristic.

### Alternative: Deck-conditioned training
Add deck identity as a global feature (one-hot or learned embedding). Lets the policy specialize per deck without separate models.

**Priority: MEDIUM | Effort: LOW**

---

## 10. What to Try First — Priority Order

Given RTX 3080 constraints and existing infrastructure:

| # | Change | Impact | Effort | Code Change |
|---|--------|--------|--------|-------------|
| 1 | Increase entropy_coeff to 0.03 | Prevents policy collapse | Trivial | One constant |
| 2 | Unfreeze encoder with low LR (1e-6 encoder, 1e-5 heads) | Enables representation learning | Moderate | Param groups + KL |
| 3 | Increase gamma to 0.99, lambda to 0.95 | Better early-game credit | Trivial | Two constants |
| 4 | Drop Blue deck temporarily | Reduces noise in signal | Trivial | Script change |
| 5 | Increase games_per_round to 800 | More data for rare heads | Trivial | Script arg |
| 6 | Add decision history features | Temporal context | Moderate | Java + Python |
| 7 | Per-head learning rates | Better optimization | Moderate | Optimizer groups |
| 8 | ExIt for early-game decisions | Highest ceiling | High | New pipeline |

**Recommended first batch:** Items 1-5 together. These are mostly parameter changes that can be tested in a single run. If the model breaks 45% with these changes, proceed to items 6-8 for the next performance tier.
