package forge.ai.rl;

import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.model.ModelServerClient;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.ai.rl.training.RewardShaper;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * Central orchestrator for the RL AI system.
 * Routes decisions to the appropriate model head, manages the model server connection,
 * and records trajectories for training.
 */
public class RLController {
    private final RLConfig config;
    private final GameStateEncoder stateEncoder;
    private final ModelServerClient modelClient;
    private final TrajectoryRecorder trajectoryRecorder;
    private final RewardShaper rewardShaper;

    private Player player;
    private Game game;

    public RLController(RLConfig config) {
        this.config = config;
        this.stateEncoder = new GameStateEncoder(config);
        this.modelClient = new ModelServerClient(config);
        this.trajectoryRecorder = config.isRecordTrajectories()
                ? new TrajectoryRecorder(config.getTrajectoryOutputDir())
                : null;
        this.rewardShaper = new RewardShaper(1.0, config.getRewardShapingDecay());
    }

    public void setPlayer(Player player) {
        this.player = player;
        this.game = player.getGame();
    }

    public Player getPlayer() { return player; }
    public Game getGame() { return game; }
    public RLConfig getConfig() { return config; }

    /**
     * Check if the model server is reachable.
     * When false, callers should use heuristic fallback for all decisions.
     */
    public boolean isModelServerAvailable() {
        return config.getMode() == RLModelMode.GRPC && modelClient.isConnected();
    }

    /**
     * Start a new game — initialize trajectory recording.
     */
    public void onGameStart(String gameId) {
        rewardShaper.reset();
        if (trajectoryRecorder != null) {
            trajectoryRecorder.startGame(gameId);
        }
        if (config.getMode() == RLModelMode.GRPC) {
            modelClient.connect();
        }
    }

    /**
     * End the game — finalize trajectory and compute terminal reward.
     */
    public void onGameEnd(boolean won) {
        if (trajectoryRecorder != null) {
            trajectoryRecorder.endGame(won);
        }
    }

    /**
     * Make a priority action decision: choose which spell/ability to play, or pass.
     *
     * @param availableActions list of playable SpellAbilities
     * @return index into availableActions, or -1 to pass priority
     */
    public int decidePriorityAction(List<SpellAbility> availableActions) {
        if (availableActions.isEmpty()) return -1;

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        // Encode each available action + pass
        List<float[]> candidates = new ArrayList<>();
        for (SpellAbility sa : availableActions) {
            candidates.add(ActionEncoder.encode(sa));
        }
        candidates.add(ActionEncoder.encodePassAction()); // last index = pass

        DecisionContext context = DecisionContext.singleSelect(
                DecisionType.PRIORITY_ACTION, gameState, candidates, "priority_action");

        DecisionResult result = requestDecision(context);
        if (result == null) return -1; // fallback: pass

        int selectedIdx = result.getSelectedIndex();
        recordDecision(context, result);

        // If selected index is the pass action (last candidate)
        if (selectedIdx >= availableActions.size()) return -1;
        return selectedIdx;
    }

    /**
     * Choose targets for a spell/ability from a list of legal targets.
     *
     * @param targets list of legal target entities
     * @param min minimum targets to select
     * @param max maximum targets to select
     * @return list of indices into targets
     */
    public List<Integer> decideTargets(List<GameEntity> targets, int min, int max) {
        if (targets.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidates = new ArrayList<>();
        for (GameEntity entity : targets) {
            candidates.add(ActionEncoder.encodeTarget(entity));
        }

        DecisionContext context = DecisionContext.multiSelect(
                DecisionType.TARGET_SELECTION, gameState, candidates, min, max, "target_selection");

        DecisionResult result = requestDecision(context);
        if (result == null) return List.of(0); // fallback: first target
        recordDecision(context, result);
        return result.getSelectedIndices();
    }

    /**
     * Choose which creatures to attack with.
     *
     * @param possibleAttackers creatures that can legally attack
     * @return list of indices of creatures that should attack
     */
    public List<Integer> decideAttackers(List<Card> possibleAttackers) {
        if (possibleAttackers.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidates = new ArrayList<>();
        for (Card c : possibleAttackers) {
            candidates.add(forge.ai.rl.features.CardFeatures.encode(c));
        }

        DecisionContext context = DecisionContext.multiSelect(
                DecisionType.DECLARE_ATTACKERS, gameState, candidates,
                0, possibleAttackers.size(), "declare_attackers");

        DecisionResult result = requestDecision(context);
        if (result == null) {
            throw new ModelServerException(
                    "Model server unavailable for attack decision");
        }
        recordDecision(context, result);
        return result.getSelectedIndices();
    }

    /**
     * Choose blocking assignments.
     *
     * @param possibleBlockers creatures that can legally block
     * @param attackers the attacking creatures
     * @return list of pairs (blocker_index, attacker_index), or empty for no blocks
     */
    public List<int[]> decideBlockers(List<Card> possibleBlockers, List<Card> attackers) {
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        // Encode blockers and attackers together as candidates
        // Each candidate is a (blocker, attacker) pair
        List<float[]> candidates = new ArrayList<>();
        List<int[]> pairIndices = new ArrayList<>();

        for (int b = 0; b < possibleBlockers.size(); b++) {
            for (int a = 0; a < attackers.size(); a++) {
                float[] blockerFeats = forge.ai.rl.features.CardFeatures.encode(possibleBlockers.get(b));
                float[] attackerFeats = forge.ai.rl.features.CardFeatures.encode(attackers.get(a));
                // Concatenate blocker + attacker features
                float[] combined = new float[blockerFeats.length + attackerFeats.length];
                System.arraycopy(blockerFeats, 0, combined, 0, blockerFeats.length);
                System.arraycopy(attackerFeats, 0, combined, blockerFeats.length, attackerFeats.length);
                candidates.add(combined);
                pairIndices.add(new int[]{b, a});
            }
        }
        // Add "no block" option
        candidates.add(new float[candidates.get(0).length]); // zero vector = no block

        DecisionContext context = DecisionContext.multiSelect(
                DecisionType.DECLARE_BLOCKERS, gameState, candidates,
                0, possibleBlockers.size(), "declare_blockers");

        DecisionResult result = requestDecision(context);
        if (result == null) {
            throw new ModelServerException(
                    "Model server unavailable for block decision");
        }
        recordDecision(context, result);

        // Convert selected indices to blocker-attacker pairs
        List<int[]> assignments = new ArrayList<>();
        for (int idx : result.getSelectedIndices()) {
            if (idx < pairIndices.size()) {
                assignments.add(pairIndices.get(idx));
            }
        }
        return assignments;
    }

    /**
     * Choose cards from a set (for discard, sacrifice, scry, etc.)
     *
     * @param candidates the cards to choose from
     * @param min minimum to select
     * @param max maximum to select
     * @return indices of selected cards
     */
    public List<Integer> decideCardSelection(CardCollectionView candidates, int min, int max) {
        if (candidates.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidateFeatures = new ArrayList<>();
        for (Card c : candidates) {
            candidateFeatures.add(forge.ai.rl.features.CardFeatures.encode(c));
        }

        DecisionContext context = DecisionContext.multiSelect(
                DecisionType.CARD_SELECTION, gameState, candidateFeatures, min, max, "card_selection");

        DecisionResult result = requestDecision(context);
        if (result == null) {
            // Fallback: select first min cards
            List<Integer> fallback = new ArrayList<>();
            for (int i = 0; i < min && i < candidates.size(); i++) fallback.add(i);
            return fallback;
        }
        recordDecision(context, result);
        return result.getSelectedIndices();
    }

    /**
     * Make a binary (yes/no) decision.
     */
    public boolean decideBinary(String contextInfo) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);
        DecisionContext context = DecisionContext.binary(gameState, contextInfo);

        DecisionResult result = requestDecision(context);
        if (result == null) return false; // fallback: no
        recordDecision(context, result);
        return result.getSelectedIndex() == 1;
    }

    /**
     * Make a mulligan keep/reject decision.
     *
     * @param handCards the current hand
     * @param cardsToReturn number of cards to return (0 = keep)
     * @return true to keep, false to mulligan
     */
    public boolean decideMulligan(CardCollectionView handCards, int cardsToReturn) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);

        List<float[]> candidates = new ArrayList<>();
        for (Card c : handCards) {
            candidates.add(forge.ai.rl.features.CardFeatures.encode(c));
        }

        DecisionContext context = new DecisionContext(
                DecisionType.MULLIGAN, gameState, candidates,
                0, 0, "mulligan_keep_" + cardsToReturn);

        DecisionResult result = requestDecision(context);
        if (result == null) return cardsToReturn <= 1; // fallback: keep if 0-1 cards to return
        recordDecision(context, result);
        return result.getSelectedIndex() == 1; // 1 = keep
    }

    /**
     * Choose a number within a range.
     */
    public int decideNumber(int min, int max, String contextInfo) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);

        List<float[]> candidates = new ArrayList<>();
        for (int i = min; i <= max; i++) {
            float[] feat = new float[4];
            feat[0] = (float)(i - min) / Math.max(1, max - min); // normalized position
            feat[1] = (float) i; // raw value
            feat[2] = (float) min;
            feat[3] = (float) max;
            candidates.add(feat);
        }

        DecisionContext context = DecisionContext.singleSelect(
                DecisionType.CATEGORICAL_CHOICE, gameState, candidates, contextInfo);

        DecisionResult result = requestDecision(context);
        if (result == null) return min; // fallback: minimum
        recordDecision(context, result);
        return min + result.getSelectedIndex();
    }

    // --- Internal helpers ---

    private DecisionResult requestDecision(DecisionContext context) {
        switch (config.getMode()) {
            case GRPC:
                return modelClient.requestDecision(context);
            case ONNX:
                // TODO: implement ONNX local inference
                Logger.warn("ONNX inference not yet implemented, returning null");
                return null;
            case HEURISTIC_FALLBACK:
            case RECORD_HEURISTIC:
            default:
                return null; // caller should use heuristic fallback
        }
    }

    private void recordDecision(DecisionContext context, DecisionResult result) {
        if (trajectoryRecorder == null || result == null) return;

        Player opp = player.getWeakestOpponent();
        trajectoryRecorder.recordDecision(
                context, result,
                player.getLife(),
                opp != null ? opp.getLife() : 0,
                player.getCardsIn(ZoneType.Hand).size(),
                opp != null ? opp.getCardsIn(ZoneType.Hand).size() : 0,
                countCreatures(player),
                opp != null ? countCreatures(opp) : 0
        );
    }

    private int countCreatures(Player p) {
        int count = 0;
        for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) count++;
        }
        return count;
    }
}
