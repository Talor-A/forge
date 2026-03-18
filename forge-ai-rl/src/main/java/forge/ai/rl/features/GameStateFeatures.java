package forge.ai.rl.features;

/**
 * Dense feature vector representation of the full game state.
 * This is the input to the game state encoder neural network.
 *
 * All features are normalized to roughly [-1, 1] or [0, 1] range.
 */
public class GameStateFeatures {
    // Global game state features
    private final float[] globalFeatures;

    // Per-card features for each zone (variable length, padded to max)
    private final float[][] myBoardFeatures;      // my permanents on battlefield
    private final float[][] oppBoardFeatures;      // opponent's permanents on battlefield
    private final float[][] myHandFeatures;        // cards in my hand
    private final float[][] myGraveyardFeatures;   // cards in my graveyard
    private final float[][] oppGraveyardFeatures;  // cards in opponent's graveyard
    private final float[][] stackFeatures;         // spells/abilities on the stack

    // Mask arrays to indicate which slots are real vs padding
    private final boolean[] myBoardMask;
    private final boolean[] oppBoardMask;
    private final boolean[] myHandMask;
    private final boolean[] myGraveyardMask;
    private final boolean[] oppGraveyardMask;
    private final boolean[] stackMask;

    public GameStateFeatures(float[] globalFeatures,
                              float[][] myBoardFeatures, boolean[] myBoardMask,
                              float[][] oppBoardFeatures, boolean[] oppBoardMask,
                              float[][] myHandFeatures, boolean[] myHandMask,
                              float[][] myGraveyardFeatures, boolean[] myGraveyardMask,
                              float[][] oppGraveyardFeatures, boolean[] oppGraveyardMask,
                              float[][] stackFeatures, boolean[] stackMask) {
        this.globalFeatures = globalFeatures;
        this.myBoardFeatures = myBoardFeatures;
        this.myBoardMask = myBoardMask;
        this.oppBoardFeatures = oppBoardFeatures;
        this.oppBoardMask = oppBoardMask;
        this.myHandFeatures = myHandFeatures;
        this.myHandMask = myHandMask;
        this.myGraveyardFeatures = myGraveyardFeatures;
        this.myGraveyardMask = myGraveyardMask;
        this.oppGraveyardFeatures = oppGraveyardFeatures;
        this.oppGraveyardMask = oppGraveyardMask;
        this.stackFeatures = stackFeatures;
        this.stackMask = stackMask;
    }

    public float[] getGlobalFeatures() { return globalFeatures; }
    public float[][] getMyBoardFeatures() { return myBoardFeatures; }
    public boolean[] getMyBoardMask() { return myBoardMask; }
    public float[][] getOppBoardFeatures() { return oppBoardFeatures; }
    public boolean[] getOppBoardMask() { return oppBoardMask; }
    public float[][] getMyHandFeatures() { return myHandFeatures; }
    public boolean[] getMyHandMask() { return myHandMask; }
    public float[][] getMyGraveyardFeatures() { return myGraveyardFeatures; }
    public boolean[] getMyGraveyardMask() { return myGraveyardMask; }
    public float[][] getOppGraveyardFeatures() { return oppGraveyardFeatures; }
    public boolean[] getOppGraveyardMask() { return oppGraveyardMask; }
    public float[][] getStackFeatures() { return stackFeatures; }
    public boolean[] getStackMask() { return stackMask; }

    /**
     * Flatten all features into a single float array for serialization.
     * Format: [globalLen, global..., boardLen, board..., ...]
     */
    public float[] flatten() {
        int totalLen = globalFeatures.length;
        for (float[][] zone : new float[][][] {myBoardFeatures, oppBoardFeatures, myHandFeatures,
                myGraveyardFeatures, oppGraveyardFeatures, stackFeatures}) {
            for (float[] card : zone) {
                totalLen += card.length;
            }
        }
        float[] result = new float[totalLen];
        int offset = 0;
        System.arraycopy(globalFeatures, 0, result, offset, globalFeatures.length);
        offset += globalFeatures.length;
        for (float[][] zone : new float[][][] {myBoardFeatures, oppBoardFeatures, myHandFeatures,
                myGraveyardFeatures, oppGraveyardFeatures, stackFeatures}) {
            for (float[] card : zone) {
                System.arraycopy(card, 0, result, offset, card.length);
                offset += card.length;
            }
        }
        return result;
    }

    @Override
    public String toString() {
        int myBoardCount = 0;
        for (boolean b : myBoardMask) if (b) myBoardCount++;
        int oppBoardCount = 0;
        for (boolean b : oppBoardMask) if (b) oppBoardCount++;
        int handCount = 0;
        for (boolean b : myHandMask) if (b) handCount++;
        return String.format("GameState[global=%d, myBoard=%d, oppBoard=%d, hand=%d]",
                globalFeatures.length, myBoardCount, oppBoardCount, handCount);
    }
}
