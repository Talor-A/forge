package forge.ai.rl.model;

import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;

/**
 * Sentinel inference client for modes that don't run a model (heuristic-only,
 * record-heuristic). Always reports unavailable; never serves a decision.
 * Callers that depend on a real model should check {@link #isAvailable()}
 * before issuing a request.
 */
public final class NoInferenceClient implements InferenceClient {

    @Override
    public DecisionResult requestDecision(DecisionContext context) {
        return null;
    }

    @Override
    public boolean isAvailable() {
        return false;
    }
}
