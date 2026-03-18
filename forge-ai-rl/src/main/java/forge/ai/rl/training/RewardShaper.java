package forge.ai.rl.training;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

/**
 * Computes intermediate rewards based on game state changes.
 * These shaped rewards help guide the RL agent during training by providing
 * more frequent feedback than the sparse win/loss terminal reward.
 *
 * Reward components:
 * - Life advantage: small reward for gaining life advantage
 * - Card advantage: moderate reward for card advantage
 * - Board advantage: small reward for creature advantage
 * - Tempo advantage: reward for efficient mana usage
 *
 * Shaping rewards decay over training so the agent eventually optimizes
 * for win rate alone.
 */
public class RewardShaper {
    private double shapingScale;
    private final double decayRate;

    // Previous state for delta computation
    private int prevLifeDelta;
    private int prevCardDelta;
    private int prevBoardDelta;
    private int prevTotalPowerDelta;
    private boolean initialized = false;

    public RewardShaper(double initialScale, double decayRate) {
        this.shapingScale = initialScale;
        this.decayRate = decayRate;
    }

    /**
     * Compute the shaped reward given the current game state.
     * Returns the incremental reward since the last call.
     */
    public double computeReward(Game game, Player rlPlayer) {
        Player opponent = rlPlayer.getWeakestOpponent();
        if (opponent == null) return 0;

        int lifeDelta = rlPlayer.getLife() - opponent.getLife();
        int myHandSize = rlPlayer.getCardsIn(ZoneType.Hand).size();
        int oppHandSize = opponent.getCardsIn(ZoneType.Hand).size();
        int cardDelta = myHandSize - oppHandSize;

        int myCreatures = 0, oppCreatures = 0;
        int myPower = 0, oppPower = 0;
        for (Card c : rlPlayer.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) { myCreatures++; myPower += c.getNetPower(); }
        }
        for (Card c : opponent.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) { oppCreatures++; oppPower += c.getNetPower(); }
        }
        int boardDelta = myCreatures - oppCreatures;
        int powerDelta = myPower - oppPower;

        if (!initialized) {
            prevLifeDelta = lifeDelta;
            prevCardDelta = cardDelta;
            prevBoardDelta = boardDelta;
            prevTotalPowerDelta = powerDelta;
            initialized = false;
            return 0;
        }

        double reward = 0;
        reward += (lifeDelta - prevLifeDelta) * 0.01;
        reward += (cardDelta - prevCardDelta) * 0.05;
        reward += (boardDelta - prevBoardDelta) * 0.02;
        reward += (powerDelta - prevTotalPowerDelta) * 0.005;

        prevLifeDelta = lifeDelta;
        prevCardDelta = cardDelta;
        prevBoardDelta = boardDelta;
        prevTotalPowerDelta = powerDelta;

        return reward * shapingScale;
    }

    /**
     * Get the terminal reward for game end.
     */
    public double terminalReward(boolean won) {
        return won ? 1.0 : -1.0;
    }

    /**
     * Decay the shaping scale. Call once per training iteration.
     */
    public void decay() {
        shapingScale *= decayRate;
    }

    /**
     * Reset state for a new game.
     */
    public void reset() {
        initialized = false;
        prevLifeDelta = 0;
        prevCardDelta = 0;
        prevBoardDelta = 0;
        prevTotalPowerDelta = 0;
    }

    public double getShapingScale() {
        return shapingScale;
    }
}
