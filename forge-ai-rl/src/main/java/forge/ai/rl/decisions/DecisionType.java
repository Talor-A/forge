package forge.ai.rl.decisions;

/**
 * Enumerates the major decision categories in MTG that the RL agent must handle.
 * Each category maps to a specialized decision head in the neural network.
 */
public enum DecisionType {
    /** Choose which spell/ability to play from available options, or pass priority */
    PRIORITY_ACTION,

    /** Choose targets for a spell or ability */
    TARGET_SELECTION,

    /** Choose which creatures attack and where */
    DECLARE_ATTACKERS,

    /** Choose which creatures block which attackers */
    DECLARE_BLOCKERS,

    /** Choose cards for an effect (discard, sacrifice, scry, etc.) */
    CARD_SELECTION,

    /** Mulligan keep/mulligan and card selection for London mulligan */
    MULLIGAN,

    /** Binary yes/no or simple numeric choices */
    BINARY_CHOICE,

    /** Choose a type, color, card name, or other categorical value */
    CATEGORICAL_CHOICE,

    /** Order cards or abilities (damage assignment, trigger ordering, etc.) */
    ORDERING,

    /** Pay costs (mana decisions, convoke/improvise tapping, etc.) */
    COST_PAYMENT
}
