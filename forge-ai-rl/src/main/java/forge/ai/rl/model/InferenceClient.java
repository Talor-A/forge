package forge.ai.rl.model;

import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;

/**
 * Source of model-policy decisions. Each backend (gRPC server, local ONNX,
 * none-configured) implements this so RLController stays out of the
 * backend-selection switch.
 */
public interface InferenceClient {

    /**
     * Run inference for the given context. Implementations either return a
     * result or throw — never return null on a hard failure, because a silent
     * null in a training run gets quietly swapped for the heuristic and
     * corrupts trajectories. Null is only valid when this client is the
     * "no inference configured" sentinel.
     */
    DecisionResult requestDecision(DecisionContext context);

    /**
     * True if this client can serve a request right now.
     */
    boolean isAvailable();

    /**
     * Eagerly verify the backend is reachable, retrying if needed. Default
     * is a no-op; backends that need an active connection (gRPC) override
     * this to retry-and-throw so misconfigured runs fail fast at startup
     * instead of silently degrading.
     */
    default void ensureAvailable() {}

    /**
     * Optional warmup hook called at game start (e.g. open the socket).
     * Default no-op.
     */
    default void warmUp() {}
}
