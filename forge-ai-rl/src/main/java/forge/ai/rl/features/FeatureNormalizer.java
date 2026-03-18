package forge.ai.rl.features;

/**
 * Utility class for feature normalization.
 * Provides consistent normalization across all feature extractors.
 */
public class FeatureNormalizer {

    /**
     * Normalize a value to [0, 1] given expected min/max range.
     * Values outside the range are clamped.
     */
    public static float normalize(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }

    /**
     * Normalize a value to [-1, 1] given expected min/max range.
     */
    public static float normalizeSymmetric(double value, double min, double max) {
        return normalize(value, min, max) * 2f - 1f;
    }

    /**
     * Log-normalize a count value. Useful for quantities that can vary widely
     * (e.g., turn number, total damage dealt).
     */
    public static float logNormalize(double value, double scale) {
        if (value <= 0) return 0f;
        return (float) (Math.log1p(value) / Math.log1p(scale));
    }

    /**
     * One-hot encode an index into a float array.
     */
    public static void oneHot(float[] features, int offset, int index, int numClasses) {
        for (int i = 0; i < numClasses; i++) {
            features[offset + i] = (i == index) ? 1f : 0f;
        }
    }
}
