package forge.ai.rl;

import forge.ai.AiDecisionListener;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.CardFeatures;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * Records AI decisions with full game state and action features
 * for imitation learning. Attached to PlayerControllerAi via
 * the AiDecisionListener interface.
 */
public class TrajectoryDecisionListener
        implements AiDecisionListener {

    private final Player player;
    private final GameStateEncoder encoder;
    private final TrajectoryRecorder recorder;

    public TrajectoryDecisionListener(
            Player player, TrajectoryRecorder recorder,
            RLConfig config) {
        this.player = player;
        this.encoder = new GameStateEncoder(config);
        this.recorder = recorder;
    }

    private void record(DecisionType type,
                        List<float[]> candidateFeats,
                        List<Integer> selectedIndices,
                        String info) {
        try {
            GameStateFeatures gs = encoder.encode(
                    player.getGame(), player);
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
        } catch (Exception e) {
            // Never crash the game
        }
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

    @Override
    public void onSpellChosen(
            List<SpellAbility> chosen, Player p) {
        if (p != player || chosen == null || chosen.isEmpty()) {
            return;
        }
        List<float[]> feats = new ArrayList<>();
        for (SpellAbility sa : chosen) {
            feats.add(ActionEncoder.encode(sa));
        }
        record(DecisionType.PRIORITY_ACTION, feats,
                List.of(0),
                "spell_" + chosen.get(0).getHostCard()
                    .getName());
    }

    @Override
    public void onAttackersDeclared(
            Player attacker, Combat combat) {
        if (attacker != player) {
            return;
        }
        // Encode all creatures, mark which are attacking
        List<Card> creatures = new ArrayList<>(
                attacker.getCreaturesInPlay());
        List<float[]> feats = new ArrayList<>();
        List<Integer> attackerIndices = new ArrayList<>();
        for (int i = 0; i < creatures.size(); i++) {
            feats.add(CardFeatures.encode(creatures.get(i)));
            if (combat.isAttacking(creatures.get(i))) {
                attackerIndices.add(i);
            }
        }
        record(DecisionType.DECLARE_ATTACKERS, feats,
                attackerIndices,
                "attack_" + attackerIndices.size()
                    + "_of_" + creatures.size());
    }

    @Override
    public void onBlockersDeclared(
            Player defender, Combat combat) {
        if (defender != player) {
            return;
        }
        List<Card> creatures = new ArrayList<>(
                defender.getCreaturesInPlay());
        List<float[]> feats = new ArrayList<>();
        List<Integer> blockerIndices = new ArrayList<>();
        for (int i = 0; i < creatures.size(); i++) {
            feats.add(CardFeatures.encode(creatures.get(i)));
            if (combat.isBlocking(creatures.get(i))) {
                blockerIndices.add(i);
            }
        }
        record(DecisionType.DECLARE_BLOCKERS, feats,
                blockerIndices,
                "block_" + blockerIndices.size());
    }

    @Override
    public void onCardsChosen(
            CardCollectionView chosen,
            CardCollectionView options,
            String reason, Player p) {
        if (p != player) {
            return;
        }
        List<float[]> feats = new ArrayList<>();
        for (Card c : options) {
            feats.add(CardFeatures.encode(c));
        }
        List<Integer> indices = new ArrayList<>();
        for (Card c : chosen) {
            int idx = options.indexOf(c);
            if (idx >= 0) {
                indices.add(idx);
            }
        }
        record(DecisionType.CARD_SELECTION, feats,
                indices, reason);
    }

    @Override
    public void onMulliganDecision(
            boolean keep, int cardsToReturn, Player p) {
        if (p != player) {
            return;
        }
        List<float[]> handFeats = new ArrayList<>();
        for (Card c : player.getCardsIn(ZoneType.Hand)) {
            handFeats.add(CardFeatures.encode(c));
        }
        record(DecisionType.MULLIGAN, handFeats,
                List.of(keep ? 1 : 0),
                "mulligan_" + cardsToReturn
                    + (keep ? "_keep" : "_mull"));
    }

    @Override
    public void onBinaryDecision(
            boolean choice, String context, Player p) {
        if (p != player) {
            return;
        }
        record(DecisionType.BINARY_CHOICE, null,
                List.of(choice ? 1 : 0), context);
    }
}
