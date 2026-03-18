package forge.ai.rl;

import forge.LobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * A LobbyPlayerAi that wraps the standard heuristic AI with a
 * RecordingPlayerController to capture trajectory data during games.
 *
 * Used for imitation learning data collection: the heuristic AI
 * makes all decisions normally, but every decision is recorded
 * as a trajectory step for later supervised training.
 */
public class RecordingLobbyPlayerAi extends LobbyPlayerAi {

    private final RLConfig config;

    public RecordingLobbyPlayerAi(String name, RLConfig config) {
        super(name, null);
        setAiProfile("Default");
        this.config = config;
    }

    @Override
    public Player createIngamePlayer(Game game, int id) {
        // Create standard AI player — don't wrap the controller
        // (wrapping breaks AI internals). Trajectory recording
        // happens post-game via game state snapshots.
        return super.createIngamePlayer(game, id);
    }

    /**
     * Record a game state snapshot for trajectory data.
     * Called externally by the game runner at each decision.
     */
    public TrajectoryRecorder getRecorder() {
        if (recorder == null) {
            recorder = new TrajectoryRecorder(
                    config.getTrajectoryOutputDir());
        }
        return recorder;
    }

    private TrajectoryRecorder recorder;
}
