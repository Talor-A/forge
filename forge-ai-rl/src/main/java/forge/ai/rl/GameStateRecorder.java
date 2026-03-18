package forge.ai.rl;

import com.google.common.eventbus.Subscribe;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;

/**
 * Subscribes to game events and records state snapshots for
 * trajectory data. Registers with the game's EventBus and
 * captures state at key moments:
 * - Start of each main phase (board state for decisions)
 * - After each spell resolves (what was played)
 * - Combat phases (attacker/blocker declarations)
 */
public class GameStateRecorder {
    private final Game game;
    private final Player player;
    private final GameStateEncoder encoder;
    private final TrajectoryRecorder recorder;
    private int lastRecordedTurn = -1;

    public GameStateRecorder(
            Game game, Player player,
            TrajectoryRecorder recorder, RLConfig config) {
        this.game = game;
        this.player = player;
        this.encoder = new GameStateEncoder(config);
        this.recorder = recorder;
    }

    /**
     * Register with the game's event bus.
     */
    public void register() {
        game.subscribeToEvents(this);
    }

    @Subscribe
    public void onTurnPhase(GameEventTurnPhase event) {
        try {
            PhaseType phase = event.phase();

            // Record at main phase 1 (key decision point)
            if (phase == PhaseType.MAIN1) {
                int turn = game.getPhaseHandler().getTurn();
                if (turn != lastRecordedTurn) {
                    lastRecordedTurn = turn;
                    recordSnapshot("main1_turn_" + turn);
                }
            }

            // Record at declare attackers (our attack turn)
            if (phase == PhaseType.COMBAT_DECLARE_ATTACKERS
                    && game.getPhaseHandler().getPlayerTurn()
                            == player) {
                recordSnapshot("pre_attack_turn_"
                        + game.getPhaseHandler().getTurn());
            }

            // Record at declare blockers (we're defending)
            if (phase == PhaseType.COMBAT_DECLARE_BLOCKERS
                    && game.getPhaseHandler().getPlayerTurn()
                            != player) {
                recordSnapshot("pre_block_turn_"
                        + game.getPhaseHandler().getTurn());
            }

            // Record at main phase 2 (post-combat)
            if (phase == PhaseType.MAIN2) {
                recordSnapshot("main2_turn_"
                        + game.getPhaseHandler().getTurn());
            }
        } catch (Exception e) {
            // Never crash the game
        }
    }

    @Subscribe
    public void onSpellResolved(
            GameEventSpellResolved event) {
        try {
            // Record after any spell resolves — use
            // stackDescription which is available from
            // the event record
            if (event.stackDescription() != null
                    && !event.hasFizzled()) {
                recordSnapshot("spell_resolved");
            }
        } catch (Exception e) {
            // Never crash the game
        }
    }

    private void recordSnapshot(String info) {
        GameStateFeatures gs = encoder.encode(game, player);
        DecisionContext ctx = new DecisionContext(
                DecisionType.PRIORITY_ACTION, gs,
                List.of(), 0, 0, info);
        DecisionResult res = new DecisionResult(
                List.of(0), new float[0], 0f, true);
        Player opp = player.getWeakestOpponent();
        recorder.recordDecision(ctx, res,
                player.getLife(),
                opp != null ? opp.getLife() : 0,
                player.getCardsIn(ZoneType.Hand).size(),
                opp != null
                    ? opp.getCardsIn(ZoneType.Hand).size() : 0,
                countCreatures(player),
                opp != null ? countCreatures(opp) : 0);
    }

    private int countCreatures(Player p) {
        int c = 0;
        for (Card card : p.getCardsIn(ZoneType.Battlefield)) {
            if (card.isCreature()) {
                c++;
            }
        }
        return c;
    }
}
