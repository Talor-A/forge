package forge.ai;

import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * Listener interface for observing AI decisions.
 * Attached to PlayerControllerAi to record what the
 * heuristic AI chooses at each decision point, without
 * modifying the AI logic itself.
 */
public interface AiDecisionListener {

    /** Called when the AI chooses a spell/ability to play. */
    default void onSpellChosen(
            List<SpellAbility> chosen,
            Player player) { }

    /** Called after the AI declares attackers. */
    default void onAttackersDeclared(
            Player attacker, Combat combat) { }

    /** Called after the AI declares blockers. */
    default void onBlockersDeclared(
            Player defender, Combat combat) { }

    /** Called when the AI chooses cards for an effect. */
    default void onCardsChosen(
            CardCollectionView chosen,
            CardCollectionView options,
            String reason,
            Player player) { }

    /** Called on mulligan decision. */
    default void onMulliganDecision(
            boolean keep, int cardsToReturn,
            Player player) { }

    /** Called on any binary (yes/no) decision. */
    default void onBinaryDecision(
            boolean choice, String context,
            Player player) { }
}
