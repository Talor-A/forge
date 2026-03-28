package forge.ai.rl.mcts;

import forge.ai.ComputerUtil;
import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameSimulator;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.*;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;

import org.tinylog.Logger;

import java.util.*;
import java.util.concurrent.*;

/**
 * Makes decisions via Monte Carlo rollouts: for each candidate action,
 * copy the game, apply the action, play the rest with heuristic AIs,
 * and pick the candidate with the highest win rate.
 *
 * Used by Expert Iteration (ExIt) to generate search-improved training data.
 */
public class MCTSDecisionMaker {

    private final int rolloutsPerCandidate;
    private final int rolloutTimeoutSec;
    private final Random rng;

    // Stats for logging
    private int totalDecisions = 0;
    private int totalRollouts = 0;
    private long totalRolloutTimeMs = 0;

    public MCTSDecisionMaker(int rolloutsPerCandidate, int rolloutTimeoutSec) {
        this.rolloutsPerCandidate = rolloutsPerCandidate;
        this.rolloutTimeoutSec = rolloutTimeoutSec;
        this.rng = new Random();
    }

    /**
     * Evaluate priority candidates via rollouts.
     * @param game current game state (not modified)
     * @param player the player making the decision
     * @param candidates list of playable SpellAbilities
     * @return index of best candidate, or candidates.size() for pass
     */
    public int decidePriority(Game game, Player player,
                              List<SpellAbility> candidates) {
        if (candidates.isEmpty()) return 0;

        int nCandidates = candidates.size() + 1; // +1 for pass
        float[] winRates = new float[nCandidates];

        for (int c = 0; c < nCandidates; c++) {
            SpellAbility sa = (c < candidates.size()) ? candidates.get(c) : null;
            int wins = 0;
            for (int r = 0; r < rolloutsPerCandidate; r++) {
                if (rollout(game, player, sa)) {
                    wins++;
                }
                totalRollouts++;
            }
            winRates[c] = (float) wins / rolloutsPerCandidate;
        }

        int best = argmax(winRates);
        totalDecisions++;

        if (totalDecisions % 20 == 0) {
            logStats();
        }

        String label = (best < candidates.size())
                ? cardName(candidates.get(best))
                : "PASS";
        Logger.info("MCTS_PRIORITY: n={} best={} ({}) rates={}",
                candidates.size(), best, label, formatRates(winRates));

        return best;
    }

    /**
     * Evaluate attack candidates via rollouts.
     * Tests: all-in attack, no attack, and each individual creature.
     * @return set of indices of creatures that should attack
     */
    public List<Integer> decideAttackers(Game game, Player player,
                                          List<Card> possibleAttackers) {
        if (possibleAttackers.isEmpty()) return List.of();

        // Candidate attack patterns:
        // 0 = no attack, 1 = all-in, 2..N+1 = individual creatures
        int nPatterns = 2 + possibleAttackers.size();
        float[] winRates = new float[nPatterns];

        // Pattern 0: no attack
        winRates[0] = rolloutAttackPattern(game, player, possibleAttackers,
                Collections.emptyList());

        // Pattern 1: all-in
        List<Integer> allIn = new ArrayList<>();
        for (int i = 0; i < possibleAttackers.size(); i++) allIn.add(i);
        winRates[1] = rolloutAttackPattern(game, player, possibleAttackers, allIn);

        // Patterns 2+: individual creatures
        for (int i = 0; i < possibleAttackers.size(); i++) {
            winRates[2 + i] = rolloutAttackPattern(game, player,
                    possibleAttackers, List.of(i));
        }

        int best = argmax(winRates);
        totalDecisions++;

        List<Integer> result;
        if (best == 0) {
            result = List.of();
        } else if (best == 1) {
            result = allIn;
        } else {
            result = List.of(best - 2);
        }

        Logger.info("MCTS_ATTACK: n={} best_pattern={} selected={} rates={}",
                possibleAttackers.size(), best, result,
                formatRates(winRates));

        return result;
    }

    /**
     * Evaluate target candidates via rollouts.
     * @return index of best target
     */
    public int decideTarget(Game game, Player player,
                            List<GameEntity> targets,
                            SpellAbility sourceSpell) {
        if (targets.isEmpty()) return 0;
        if (targets.size() == 1) return 0;

        float[] winRates = new float[targets.size()];

        for (int t = 0; t < targets.size(); t++) {
            int wins = 0;
            for (int r = 0; r < rolloutsPerCandidate; r++) {
                if (rolloutWithTarget(game, player, sourceSpell,
                        targets.get(t))) {
                    wins++;
                }
                totalRollouts++;
            }
            winRates[t] = (float) wins / rolloutsPerCandidate;
        }

        int best = argmax(winRates);
        totalDecisions++;

        Logger.info("MCTS_TARGET: n={} best={} rates={}",
                targets.size(), best, formatRates(winRates));

        return best;
    }

    /**
     * Core rollout: copy game, optionally apply a spell, play to completion.
     * @return true if player won
     */
    private boolean rollout(Game origGame, Player origPlayer,
                            SpellAbility origSa) {
        long t0 = System.currentTimeMillis();
        try {
            Random origRandom = MyRandom.getRandom();
            MyRandom.setRandom(new Random(rng.nextLong()));

            try {
                GameCopier copier = new GameCopier(origGame);
                Game simGame = copier.makeCopy();
                Player simPlayer = (Player) copier.find(origPlayer);

                // Apply the action if not pass
                if (origSa != null) {
                    SpellAbility simSa = findSaInCopy(copier, origSa, simGame);
                    if (simSa != null) {
                        simSa.setActivatingPlayer(simPlayer);
                        ComputerUtil.handlePlayingSpellAbility(simPlayer, simSa, null);
                        GameSimulator.resolveStack(simGame,
                                getOpponent(simGame, simPlayer));
                    }
                }

                // Play to completion with timeout
                if (!simGame.isGameOver()) {
                    if (!runWithTimeout(() ->
                            simGame.getPhaseHandler().mainGameLoop(),
                            rolloutTimeoutSec)) {
                        return false;
                    }
                }

                if (simGame.getOutcome() == null || simGame.getOutcome().isDraw()) {
                    return false;
                }

                return simGame.getOutcome().isWinner(
                        simPlayer.getRegisteredPlayer());
            } finally {
                MyRandom.setRandom(origRandom);
            }
        } finally {
            totalRolloutTimeMs += System.currentTimeMillis() - t0;
        }
    }

    /**
     * Rollout with a specific attack pattern applied.
     */
    private float rolloutAttackPattern(Game origGame, Player origPlayer,
                                        List<Card> possibleAttackers,
                                        List<Integer> attackerIndices) {
        int wins = 0;
        for (int r = 0; r < rolloutsPerCandidate; r++) {
            Random origRandom = MyRandom.getRandom();
            MyRandom.setRandom(new Random(rng.nextLong()));
            try {
                GameCopier copier = new GameCopier(origGame);
                Game simGame = copier.makeCopy();
                Player simPlayer = (Player) copier.find(origPlayer);
                GameEntity defender = simPlayer.getWeakestOpponent();

                if (defender != null) {
                    Combat combat = simGame.getCombat();
                    if (combat == null) {
                        combat = new Combat(simPlayer);
                        simGame.getPhaseHandler().setCombat(combat);
                    }
                    for (int idx : attackerIndices) {
                        if (idx >= 0 && idx < possibleAttackers.size()) {
                            Card origCard = possibleAttackers.get(idx);
                            Card simCard = (Card) copier.find(origCard);
                            if (simCard != null
                                    && CombatUtil.canAttack(simCard, defender)) {
                                combat.addAttacker(simCard, defender);
                            }
                        }
                    }
                }

                // Play to completion
                if (!simGame.isGameOver()) {
                    if (!runWithTimeout(() ->
                            simGame.getPhaseHandler().mainGameLoop(),
                            rolloutTimeoutSec)) {
                        totalRollouts++;
                        continue;
                    }
                }

                if (simGame.getOutcome() != null
                        && !simGame.getOutcome().isDraw()
                        && simGame.getOutcome().isWinner(
                            simPlayer.getRegisteredPlayer())) {
                    wins++;
                }
            } finally {
                MyRandom.setRandom(origRandom);
            }
            totalRollouts++;
        }
        return (float) wins / rolloutsPerCandidate;
    }

    /**
     * Rollout with a specific target for the current spell.
     */
    private boolean rolloutWithTarget(Game origGame, Player origPlayer,
                                       SpellAbility origSa,
                                       GameEntity target) {
        Random origRandom = MyRandom.getRandom();
        MyRandom.setRandom(new Random(rng.nextLong()));
        long t0 = System.currentTimeMillis();
        try {
            GameCopier copier = new GameCopier(origGame);
            Game simGame = copier.makeCopy();
            Player simPlayer = (Player) copier.find(origPlayer);

            if (origSa != null) {
                SpellAbility simSa = findSaInCopy(copier, origSa, simGame);
                if (simSa != null) {
                    simSa.setActivatingPlayer(simPlayer);
                    // Set the target
                    simSa.resetTargets();
                    GameObject simTarget = copier.find(target);
                    if (simTarget instanceof GameEntity) {
                        simSa.getTargets().add((GameEntity) simTarget);
                    }
                    ComputerUtil.handlePlayingSpellAbility(simPlayer, simSa, null);
                    GameSimulator.resolveStack(simGame,
                            getOpponent(simGame, simPlayer));
                }
            }

            if (!simGame.isGameOver()) {
                if (!runWithTimeout(() ->
                        simGame.getPhaseHandler().mainGameLoop(),
                        rolloutTimeoutSec)) {
                    return false;
                }
            }

            if (simGame.getOutcome() == null || simGame.getOutcome().isDraw()) {
                return false;
            }
            return simGame.getOutcome().isWinner(
                    simPlayer.getRegisteredPlayer());
        } finally {
            MyRandom.setRandom(origRandom);
            totalRolloutTimeMs += System.currentTimeMillis() - t0;
        }
    }

    // ── Helpers ──

    private SpellAbility findSaInCopy(GameCopier copier, SpellAbility origSa,
                                       Game simGame) {
        Card origCard = origSa.getHostCard();
        if (origCard == null) return null;
        Card simCard = (Card) copier.find(origCard);
        if (simCard == null) return null;

        // Try to find matching SA on the copied card
        SpellAbility fallback = null;
        for (SpellAbility sa : simCard.getAllSpellAbilities()) {
            if (fallback == null) fallback = sa;
            if (sa.getApi() == origSa.getApi()
                    && Objects.equals(sa.getDescription(), origSa.getDescription())) {
                return sa;
            }
        }
        return fallback;
    }

    private Player getOpponent(Game game, Player player) {
        for (Player p : game.getPlayers()) {
            if (p != player) return p;
        }
        return null;
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    private static String cardName(SpellAbility sa) {
        if (sa == null) return "null";
        Card c = sa.getHostCard();
        return c != null ? c.getName() : "?";
    }

    private static String formatRates(float[] rates) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < rates.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.0f%%", rates[i] * 100));
        }
        sb.append("]");
        return sb.toString();
    }

    private void logStats() {
        double avgMs = totalRollouts > 0
                ? (double) totalRolloutTimeMs / totalRollouts : 0;
        Logger.info("MCTS_STATS: decisions={} rollouts={} avg={:.0f}ms/rollout",
                totalDecisions, totalRollouts, avgMs);
    }

    /**
     * Run a task with a timeout. Returns true if completed, false if timed out.
     */
    private static boolean runWithTimeout(Runnable task, int timeoutSec) {
        ExecutorService exec = Executors.newSingleThreadExecutor();
        Future<?> future = exec.submit(task);
        try {
            future.get(timeoutSec, TimeUnit.SECONDS);
            return true;
        } catch (TimeoutException e) {
            future.cancel(true);
            return false;
        } catch (Exception e) {
            return false;
        } finally {
            exec.shutdownNow();
        }
    }

    public int getTotalDecisions() { return totalDecisions; }
    public int getTotalRollouts() { return totalRollouts; }
    public long getTotalRolloutTimeMs() { return totalRolloutTimeMs; }
}
