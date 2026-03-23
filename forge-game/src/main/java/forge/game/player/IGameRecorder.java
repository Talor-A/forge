package forge.game.player;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * Interface for recording game decisions (human or AI).
 * Implementations handle feature encoding and trajectory file writing.
 */
public interface IGameRecorder {

    /** Called when the game starts. */
    void onGameStart(String gameId);

    /** Called when the game ends. */
    void onGameEnd(boolean won);

    /** Capture state before a priority decision. */
    void capturePrePriority(List<SpellAbility> candidates);

    /** Record the priority decision after the human chooses. */
    void recordPriorityDecision(List<SpellAbility> candidates, SpellAbility chosen);

    /** Capture state before an attack decision. */
    void capturePreAttack(List<Card> possibleAttackers);

    /** Record the attack decision after combat is set. */
    void recordAttackDecision(List<Card> possibleAttackers, Combat combat);

    /** Record the block decision after combat is set. */
    void recordBlockDecision(List<Card> possibleBlockers,
                             List<Card> attackers, Combat combat);

    /** Record a mulligan decision. */
    void recordMulligan(CardCollectionView hand, boolean kept);

    /** Record a target/card selection decision. */
    void recordTargetDecision(List<?> candidates, int selectedIdx);

    /** Record a binary yes/no decision. */
    void recordBinaryDecision(boolean yes, String context);

    /** Get the count of decisions recorded. */
    int getDecisionCount();
}
