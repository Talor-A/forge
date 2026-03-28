package forge.ai.rl.mcts;

import java.util.List;

/**
 * Result from MCTS rollout evaluation.
 * Contains the selected action, win rates for all candidates (soft targets),
 * and the value estimate (win rate of best candidate).
 */
public class MCTSResult {
    private final int selectedIndex;
    private final float[] winRates;     // per-candidate win rates from rollouts
    private final float valueEstimate;  // win rate of selected candidate

    public MCTSResult(int selectedIndex, float[] winRates, float valueEstimate) {
        this.selectedIndex = selectedIndex;
        this.winRates = winRates;
        this.valueEstimate = valueEstimate;
    }

    public int getSelectedIndex() { return selectedIndex; }
    public float[] getWinRates() { return winRates; }
    public float getValueEstimate() { return valueEstimate; }
}
