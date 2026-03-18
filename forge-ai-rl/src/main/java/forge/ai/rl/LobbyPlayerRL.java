package forge.ai.rl;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.IGameEntitiesFactory;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import org.tinylog.Logger;

/**
 * LobbyPlayer implementation for the RL AI.
 * Creates PlayerControllerRL instances for each game.
 */
public class LobbyPlayerRL extends LobbyPlayer implements IGameEntitiesFactory {

    private final RLConfig config;

    public LobbyPlayerRL(String name, RLConfig config) {
        super(name);
        this.config = config;
    }

    @Override
    public Player createIngamePlayer(Game game, int id) {
        Player p = new Player(getName(), game, id);
        PlayerControllerRL controller = new PlayerControllerRL(game, p, this, config);
        p.setFirstController(controller);
        Logger.info("Created RL player '{}' with mode: {}", getName(), config.getMode());
        return p;
    }

    @Override
    public PlayerController createMindSlaveController(Player master, Player slave) {
        // When mind-slaved, use the RL controller for the slave
        return new PlayerControllerRL(slave.getGame(), slave, this, config);
    }

    @Override
    public void hear(LobbyPlayer player, String message) {
        // RL AI is deaf like the heuristic AI
    }

    public RLConfig getConfig() {
        return config;
    }
}
