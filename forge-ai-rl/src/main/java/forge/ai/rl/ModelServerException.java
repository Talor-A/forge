package forge.ai.rl;

/**
 * Thrown when the model server is unavailable or fails to respond.
 * This is a distinct exception type so callers can distinguish server failures
 * from game engine bugs and take appropriate action (e.g., abort the run
 * rather than silently counting as a draw).
 */
public class ModelServerException extends RuntimeException {
    public ModelServerException(String message) {
        super(message);
    }

    public ModelServerException(String message, Throwable cause) {
        super(message, cause);
    }
}
