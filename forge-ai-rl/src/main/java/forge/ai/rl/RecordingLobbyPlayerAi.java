package forge.ai.rl;

import forge.ai.LobbyPlayerAi;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * LobbyPlayer that uses standard heuristic AI but provides
 * a TrajectoryRecorder for external state snapshotting.
 *
 * The game runner captures state at turn boundaries and
 * records decisions via the game event system, rather than
 * instrumenting the controller (which breaks AI internals).
 */
public class RecordingLobbyPlayerAi extends LobbyPlayerAi {

    private final RLConfig config;
    private TrajectoryRecorder recorder;

    public RecordingLobbyPlayerAi(
            String name, RLConfig config) {
        super(name, null);
        setAiProfile("Default");
        this.config = config;
        this.recorder = new TrajectoryRecorder(
                config.getTrajectoryOutputDir());
    }

    // Use standard AI — don't override createIngamePlayer

    public TrajectoryRecorder getRecorder() {
        return recorder;
    }

    public RLConfig getConfig() {
        return config;
    }
}
