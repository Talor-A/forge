package forge.ai.rl;

import com.google.common.collect.Multimap;
import com.google.common.eventbus.Subscribe;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.CardFeatures;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventAttackersDeclared;
import forge.game.event.GameEventBlockersDeclared;
import forge.game.event.GameEventLandPlayed;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.zone.ZoneType;
import forge.game.GameEntityView;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Subscribes to game events and records state snapshots WITH
 * action data for trajectory collection. Captures:
 * - Attackers declared (which creatures, encoded as features)
 * - Blockers declared (which creatures block which)
 * - Spells resolved (what was cast)
 * - Lands played
 * - Phase transitions (board state at key moments)
 */
public class GameStateRecorder {
    private final Game game;
    private final Player player;
    private final PlayerView playerView;
    private final GameStateEncoder encoder;
    private final TrajectoryRecorder recorder;
    private int lastRecordedTurn = -1;

    public GameStateRecorder(
            Game game, Player player,
            TrajectoryRecorder recorder, RLConfig config) {
        this.game = game;
        this.player = player;
        this.playerView = PlayerView.get(player);
        this.encoder = new GameStateEncoder(config);
        this.recorder = recorder;
    }

    public void register() {
        game.subscribeToEvents(this);
    }

    // ── Attackers declared ────────────────────────

    @Subscribe
    public void onAttackersDeclared(
            GameEventAttackersDeclared event) {
        try {
            if (!event.player().equals(playerView)) {
                return;
            }
            // Get all creatures and identify which attacked
            List<Card> allCreatures = new ArrayList<>(
                    player.getCreaturesInPlay());
            Collection<CardView> attackerViews =
                    event.attackersMap().values();

            List<Card> attackingCards = new ArrayList<>();
            List<Integer> attackerIndices = new ArrayList<>();
            for (int i = 0; i < allCreatures.size(); i++) {
                CardView cv = CardView.get(allCreatures.get(i));
                if (attackerViews.contains(cv)) {
                    attackerIndices.add(i);
                    attackingCards.add(allCreatures.get(i));
                }
            }

            // Temporarily untap attacking creatures so the
            // game state and card features reflect PRE-attack
            // state. The event fires AFTER tapping, but the
            // model must learn from pre-decision features.
            for (Card c : attackingCards) {
                c.setTapped(false);
            }

            // Now encode — game state and candidates will
            // show creatures as untapped (matching inference)
            List<float[]> feats = new ArrayList<>();
            for (Card c : allCreatures) {
                feats.add(CardFeatures.encode(c, player));
            }
            // Record with pre-attack state
            recordWithAction(
                    DecisionType.DECLARE_ATTACKERS,
                    feats, attackerIndices,
                    "attack_" + attackerIndices.size()
                        + "_of_" + allCreatures.size());

            // Re-tap attacking creatures to restore game state
            for (Card c : attackingCards) {
                c.setTapped(true);
            }
        } catch (Exception e) {
            org.tinylog.Logger.warn("GameStateRecorder error: {}", e.getMessage());
        }
    }

    // ── Blockers declared ─────────────────────────

    @Subscribe
    public void onBlockersDeclared(
            GameEventBlockersDeclared event) {
        try {
            if (!event.defendingPlayer().equals(playerView)) {
                return;
            }
            List<Card> allCreatures = new ArrayList<>(
                    player.getCreaturesInPlay());
            List<float[]> feats = new ArrayList<>();
            List<Integer> blockerIndices = new ArrayList<>();

            // Collect all blocker card views
            // The multimap is attacker → [blockers], so values() has the blockers.
            // When no blockers, the engine puts the attacker as a placeholder — filter those out.
            List<CardView> allBlockerViews = new ArrayList<>();
            for (Multimap<CardView, CardView> mm
                    : event.blockers().values()) {
                for (Map.Entry<CardView, CardView> entry : mm.entries()) {
                    CardView attacker = entry.getKey();
                    CardView blocker = entry.getValue();
                    // Skip placeholder entries where attacker == blocker (means unblocked)
                    if (!attacker.equals(blocker)) {
                        allBlockerViews.add(blocker);
                    }
                }
            }

            for (int i = 0; i < allCreatures.size(); i++) {
                Card c = allCreatures.get(i);
                feats.add(CardFeatures.encode(c, player));
                CardView cv = CardView.get(c);
                if (allBlockerViews.contains(cv)) {
                    blockerIndices.add(i);
                }
            }
            recordWithAction(
                    DecisionType.DECLARE_BLOCKERS,
                    feats, blockerIndices,
                    "block_" + blockerIndices.size());
        } catch (Exception e) {
            org.tinylog.Logger.warn("GameStateRecorder error: {}", e.getMessage());
        }
    }

    // ── Spell resolved ────────────────────────────

    @Subscribe
    public void onSpellResolved(
            GameEventSpellResolved event) {
        try {
            if (event.hasFizzled()) {
                return;
            }
            String desc = event.stackDescription();
            if (desc == null) {
                desc = "unknown_spell";
            }
            // Record state snapshot at spell resolution
            recordWithAction(
                    DecisionType.PRIORITY_ACTION,
                    null, List.of(0),
                    "spell_" + desc.substring(0,
                        Math.min(desc.length(), 40)));
        } catch (Exception e) {
            org.tinylog.Logger.warn("GameStateRecorder error: {}", e.getMessage());
        }
    }

    // ── Land played ───────────────────────────────

    @Subscribe
    public void onLandPlayed(GameEventLandPlayed event) {
        try {
            if (!event.player().equals(playerView)) {
                return;
            }
            recordWithAction(
                    DecisionType.PRIORITY_ACTION,
                    null, List.of(0),
                    "land_" + event.land().getName());
        } catch (Exception e) {
            org.tinylog.Logger.warn("GameStateRecorder error: {}", e.getMessage());
        }
    }

    // ── Phase transitions ─────────────────────────

    @Subscribe
    public void onTurnPhase(GameEventTurnPhase event) {
        try {
            PhaseType phase = event.phase();

            // Record at main phase 1 (key decision point)
            if (phase == PhaseType.MAIN1) {
                int turn = game.getPhaseHandler().getTurn();
                if (turn != lastRecordedTurn) {
                    lastRecordedTurn = turn;
                    recordWithAction(
                            DecisionType.PRIORITY_ACTION,
                            null, List.of(),
                            "main1_turn_" + turn);
                }
            }
        } catch (Exception e) {
            org.tinylog.Logger.warn("GameStateRecorder error: {}", e.getMessage());
        }
    }

    // ── Recording helper ──────────────────────────

    private void recordWithAction(
            DecisionType type,
            List<float[]> candidateFeats,
            List<Integer> selectedIndices,
            String info) {
        GameStateFeatures gs = encoder.encode(game, player);
        DecisionContext ctx = new DecisionContext(
                type, gs,
                candidateFeats != null
                    ? candidateFeats : List.of(),
                selectedIndices.size(),
                candidateFeats != null
                    ? candidateFeats.size() : 0,
                info);
        DecisionResult res = new DecisionResult(
                selectedIndices, new float[0], 0f, true);
        Player opp = player.getWeakestOpponent();
        recorder.recordDecision(ctx, res,
                player.getLife(),
                opp != null ? opp.getLife() : 0,
                player.getCardsIn(ZoneType.Hand).size(),
                opp != null
                    ? opp.getCardsIn(ZoneType.Hand).size()
                    : 0,
                countCreatures(player),
                opp != null ? countCreatures(opp) : 0);
    }

    private int countCreatures(Player p) {
        int c = 0;
        for (Card card : p.getCardsIn(ZoneType.Battlefield)) {
            if (card.isCreature()) {
                c++;
            }
        }
        return c;
    }
}
