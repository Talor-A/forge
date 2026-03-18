package forge.ai.rl;

/**
 * Operating mode for the RL AI system.
 */
public enum RLModelMode {
    /** Use gRPC to communicate with Python model server (training + evaluation) */
    GRPC,

    /** Use ONNX Runtime for local inference (deployment, no Python dependency) */
    ONNX,

    /** Fall back to heuristic AI for all decisions (for testing infrastructure) */
    HEURISTIC_FALLBACK,

    /** Record trajectories from heuristic AI decisions (imitation learning data collection) */
    RECORD_HEURISTIC
}
