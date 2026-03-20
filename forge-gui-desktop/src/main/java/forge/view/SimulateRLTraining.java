package forge.view;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.time.StopWatch;

import forge.LobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.ai.rl.LobbyPlayerRL;
import forge.ai.rl.ModelServerException;
import forge.ai.rl.PlayerControllerRL;
import forge.ai.rl.RLConfig;
import forge.ai.rl.RLModelMode;
import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.*;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.localinstance.properties.ForgeConstants;
import forge.model.FModel;
import forge.player.GamePlayerUtil;

/**
 * Headless runner for RL training data collection and evaluation.
 *
 * Modes:
 *   collect  - Run heuristic AI vs heuristic AI, recording trajectories for imitation learning
 *   evaluate - Run RL AI vs heuristic AI, measuring win rate
 *   selfplay - Run RL AI vs RL AI for self-play training
 *
 * Usage:
 *   forge rltrain collect -d deck1.dck -d deck2.dck -n 1000
 *   forge rltrain evaluate -d deck1.dck -d deck2.dck -n 100
 *   forge rltrain selfplay -d deck1.dck -d deck2.dck -n 1000
 */
public class SimulateRLTraining {

    public static void simulate(String[] args) {
        FModel.initialize(null, null);

        System.out.println("=== Forge RL Training Mode ===");

        // Parse arguments: rltrain <mode> -d deck1 -d deck2 -n N ...
        final Map<String, List<String>> params = new HashMap<>();
        String mode = "collect";
        int startIdx = 1; // skip "rltrain" at args[0]

        // First non-flag arg is the mode
        if (args.length > 1 && !args[1].startsWith("-")) {
            mode = args[1].toLowerCase();
            startIdx = 2;
        }

        String currentKey = null;
        for (int i = startIdx; i < args.length; i++) {
            final String a = args[i];
            if (a.startsWith("-")) {
                currentKey = a.substring(1);
                if (!params.containsKey(currentKey)) {
                    params.put(currentKey, new ArrayList<>());
                }
            } else if (currentKey != null) {
                params.get(currentKey).add(a);
            }
        }

        int nGames = params.containsKey("n")
                ? Integer.parseInt(params.get("n").get(0)) : 100;
        int timeout = params.containsKey("c")
                ? Integer.parseInt(params.get("c").get(0)) : 180;
        String outputDir = params.containsKey("o")
                ? params.get("o").get(0) : "rl_data/trajectories";
        boolean quiet = params.containsKey("q");
        int threads = params.containsKey("t")
                ? Integer.parseInt(params.get("t").get(0))
                : Runtime.getRuntime().availableProcessors();
        String grpcHost = params.containsKey("host")
                ? params.get("host").get(0) : "localhost";
        int grpcPort = params.containsKey("port")
                ? Integer.parseInt(params.get("port").get(0)) : 50051;

        // Load decks
        List<Deck> decks = new ArrayList<>();
        if (params.containsKey("d")) {
            for (String deckParam : params.get("d")) {
                Deck d = loadDeck(deckParam);
                if (d == null) {
                    System.out.println("Could not load deck: " + deckParam);
                    return;
                }
                decks.add(d);
                System.out.println("Loaded deck: " + d.getName() + " (" + d.getMain().countAll() + " cards)");
            }
        }

        // Load all decks from directory
        if (params.containsKey("D")) {
            String dirPath = params.get("D").get(0);
            File dir = new File(dirPath);
            if (dir.isDirectory()) {
                for (File f : dir.listFiles((d, name) -> name.endsWith(".dck"))) {
                    Deck d = DeckSerializer.fromFile(f);
                    if (d != null) {
                        decks.add(d);
                    }
                }
                System.out.println("Loaded " + decks.size() + " decks from " + dirPath);
            }
        }

        if (decks.size() < 2) {
            System.out.println("Need at least 2 decks. Use -d <deck.dck> or -D <directory>");
            printHelp();
            return;
        }

        System.out.println("Mode: " + mode + " | Games: " + nGames
                + " | Threads: " + threads + " | Timeout: " + timeout + "s");
        System.out.println("Output: " + outputDir);
        System.out.println();

        switch (mode) {
            case "collect":
                runCollectionMode(decks, nGames, timeout, outputDir, quiet, threads);
                break;
            case "evaluate":
                runEvaluationMode(decks, nGames, timeout, outputDir, quiet, grpcHost, grpcPort);
                break;
            case "selfplay":
                runSelfPlayMode(decks, nGames, timeout, outputDir, quiet, grpcHost, grpcPort);
                break;
            default:
                System.out.println("Unknown mode: " + mode);
                printHelp();
        }
    }

    /**
     * Collection mode: heuristic AI vs heuristic AI, recording all decisions as trajectories.
     * This produces training data for imitation learning (Phase 2 of the plan).
     * Runs games in parallel across multiple threads.
     */
    private static void runCollectionMode(List<Deck> decks, int nGames, int timeout,
                                           String outputDir, boolean quiet, int threads) {
        System.out.println("=== Imitation Learning Data Collection ===");
        System.out.println("Using " + threads + " threads");

        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger p1Wins = new AtomicInteger(0);
        AtomicInteger p2Wins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger errors = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            final Deck deck1 = decks.get(i % decks.size());
            final Deck deck2 = decks.get((i + 1) % decks.size());
            final boolean p1First = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                // Each thread gets its own RLConfig to avoid contention
                RLConfig config = new RLConfig();
                config.setMode(RLModelMode.HEURISTIC_FALLBACK);
                config.setRecordTrajectories(true);
                config.setTrajectoryOutputDir(outputDir);

                try {
                    GameResult result = runSingleGame(
                            deck1, deck2, config, config,
                            "P1_" + gameIdx, "P2_" + gameIdx,
                            timeout, p1First);

                    if (result.isDraw) {
                        draws.incrementAndGet();
                    } else if (result.player1Won) {
                        p1Wins.incrementAndGet();
                    } else {
                        p2Wins.incrementAndGet();
                    }
                } catch (Exception e) {
                    errors.incrementAndGet();
                }

                int done = completed.incrementAndGet();
                if (quiet && done % 100 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gps = done * 1000.0 / elapsed;
                    int remaining = nGames - done;
                    double etaSec = remaining / gps;
                    System.out.printf(
                            "Progress: %d/%d (%.1f games/sec) "
                            + "P1:%d P2:%d Draw:%d Err:%d "
                            + "ETA:%.0fs%n",
                            done, nGames, gps,
                            p1Wins.get(), p2Wins.get(),
                            draws.get(), errors.get(), etaSec);
                } else if (!quiet && done % 10 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gps = done * 1000.0 / elapsed;
                    System.out.printf(
                            "Progress: %d/%d (%.1f games/sec) "
                            + "P1:%d P2:%d%n",
                            done, nGames, gps,
                            p1Wins.get(), p2Wins.get());
                }
            }));
        }

        // Wait for all games to complete
        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get(timeout + 30, TimeUnit.SECONDS);
            } catch (Exception e) {
                errors.incrementAndGet();
            }
        }
        executor.shutdown();

        long totalMs = System.currentTimeMillis() - startTime;
        double gamesPerSec = nGames * 1000.0 / totalMs;

        System.out.println();
        System.out.println("=== Collection Complete ===");
        System.out.printf("Games: %d | P1: %d | P2: %d | Draw: %d | Errors: %d%n",
                nGames, p1Wins.get(), p2Wins.get(), draws.get(), errors.get());
        System.out.printf("Total time: %.1fs | %.1f games/sec (%d threads)%n",
                totalMs / 1000.0, gamesPerSec, threads);
        System.out.println("Trajectories saved to: " + outputDir);
    }

    /**
     * Evaluation mode: RL AI vs heuristic AI, measuring win rate.
     */
    private static void runEvaluationMode(List<Deck> decks, int nGames, int timeout,
                                            String outputDir, boolean quiet,
                                            String grpcHost, int grpcPort) {
        System.out.println("=== RL Evaluation Mode ===");

        RLConfig rlConfig = new RLConfig();
        rlConfig.setMode(RLModelMode.GRPC);
        rlConfig.setGrpcHost(grpcHost);
        rlConfig.setGrpcPort(grpcPort);
        rlConfig.setRecordTrajectories(true);
        rlConfig.setTrajectoryOutputDir(outputDir);

        RLConfig heuristicConfig = new RLConfig();
        heuristicConfig.setMode(RLModelMode.HEURISTIC_FALLBACK);
        heuristicConfig.setRecordTrajectories(false);

        AtomicInteger rlWins = new AtomicInteger(0);
        AtomicInteger heuristicWins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger serverErrors = new AtomicInteger(0);
        final int MAX_SERVER_ERRORS = 3;

        for (int i = 0; i < nGames; i++) {
            Deck rlDeck = decks.get(i % decks.size());
            Deck aiDeck = decks.get((i + 1) % decks.size());
            boolean rlFirst = (i % 2 == 0);

            try {
                GameResult result;
                if (rlFirst) {
                    result = runSingleGame(rlDeck, aiDeck, rlConfig, heuristicConfig,
                            "RL_" + i, "Heuristic_" + i, timeout, true);
                } else {
                    result = runSingleGame(aiDeck, rlDeck, heuristicConfig, rlConfig,
                            "Heuristic_" + i, "RL_" + i, timeout, true);
                }

                // Game completed successfully — reset consecutive error count
                serverErrors.set(0);

                if (result.isDraw) {
                    draws.incrementAndGet();
                } else {
                    // Figure out if RL won
                    boolean rlWon = rlFirst ? result.player1Won : !result.player1Won;
                    if (rlWon) rlWins.incrementAndGet();
                    else heuristicWins.incrementAndGet();
                }

                if (!quiet && (i + 1) % 10 == 0) {
                    int total = rlWins.get() + heuristicWins.get();
                    double winRate = total > 0 ? (double) rlWins.get() / total : 0;
                    System.out.printf("Game %d/%d: RL win rate: %.1f%% (%d/%d)%n",
                            i + 1, nGames, winRate * 100, rlWins.get(), total);
                }

            } catch (ModelServerException e) {
                int errCount = serverErrors.incrementAndGet();
                System.out.printf("Game %d MODEL_SERVER_ERROR (%d/%d): %s%n",
                        i + 1, errCount, MAX_SERVER_ERRORS, e.getMessage());
                if (errCount >= MAX_SERVER_ERRORS) {
                    System.out.println();
                    System.out.println("ABORT: Model server failed " + MAX_SERVER_ERRORS
                            + " consecutive games. Server is down or broken.");
                    System.out.println("ABORT: Fix the model server and retry.");
                    break;
                }
            } catch (Exception e) {
                System.out.printf("Game %d FAILED: %s%n", i + 1, e.getMessage());
            }
        }

        int totalDecisive = rlWins.get() + heuristicWins.get();
        double winRate = totalDecisive > 0 ? (double) rlWins.get() / totalDecisive : 0;
        System.out.println();
        if (serverErrors.get() >= MAX_SERVER_ERRORS) {
            System.out.println("=== Evaluation ABORTED (model server down) ===");
        } else {
            System.out.println("=== Evaluation Complete ===");
        }
        System.out.printf("RL Wins: %d (%.1f%%) | Heuristic Wins: %d | Draws: %d | Server Errors: %d%n",
                rlWins.get(), winRate * 100, heuristicWins.get(), draws.get(), serverErrors.get());
    }

    /**
     * Self-play mode: RL AI vs RL AI.
     */
    private static void runSelfPlayMode(List<Deck> decks, int nGames, int timeout,
                                          String outputDir, boolean quiet,
                                          String grpcHost, int grpcPort) {
        System.out.println("=== Self-Play Mode ===");

        RLConfig config = new RLConfig();
        config.setMode(RLModelMode.GRPC);
        config.setGrpcHost(grpcHost);
        config.setGrpcPort(grpcPort);
        config.setRecordTrajectories(true);
        config.setTrajectoryOutputDir(outputDir);

        int serverErrors = 0;
        final int MAX_SERVER_ERRORS = 3;

        for (int i = 0; i < nGames; i++) {
            Deck deck1 = decks.get(i % decks.size());
            Deck deck2 = decks.get((i + 1) % decks.size());

            try {
                GameResult result = runSingleGame(deck1, deck2, config, config,
                        "RL_A_" + i, "RL_B_" + i, timeout, i % 2 == 0);

                serverErrors = 0; // reset on success

                if (!quiet && (i + 1) % 10 == 0) {
                    System.out.printf("Self-play game %d/%d complete%n", i + 1, nGames);
                }
            } catch (ModelServerException e) {
                serverErrors++;
                System.out.printf("Game %d MODEL_SERVER_ERROR (%d/%d): %s%n",
                        i + 1, serverErrors, MAX_SERVER_ERRORS, e.getMessage());
                if (serverErrors >= MAX_SERVER_ERRORS) {
                    System.out.println();
                    System.out.println("ABORT: Model server failed " + MAX_SERVER_ERRORS
                            + " consecutive games. Server is down or broken.");
                    break;
                }
            } catch (Exception e) {
                System.out.printf("Game %d FAILED: %s%n", i + 1, e.getMessage());
            }
        }
        System.out.println("Self-play complete. Trajectories saved to: " + outputDir);
    }

    // ===== Core game execution =====

    private static GameResult runSingleGame(Deck deck1, Deck deck2,
                                              RLConfig config1, RLConfig config2,
                                              String name1, String name2,
                                              int timeoutSec, boolean p1First) {
        // Create players
        LobbyPlayer lobby1 = createPlayer(name1, config1);
        LobbyPlayer lobby2 = createPlayer(name2, config2);

        RegisteredPlayer rp1 = new RegisteredPlayer(deck1);
        rp1.setPlayer(lobby1);
        RegisteredPlayer rp2 = new RegisteredPlayer(deck2);
        rp2.setPlayer(lobby2);

        List<RegisteredPlayer> players = p1First
                ? Arrays.asList(rp1, rp2) : Arrays.asList(rp2, rp1);

        GameRules rules = new GameRules(GameType.Constructed);
        rules.setAppliedVariants(EnumSet.of(GameType.Constructed));
        rules.setGamesPerMatch(1);
        rules.setSimTimeout(timeoutSec);

        Match match = new Match(rules, players, "RL Training");
        Game game = match.createGame();

        // Store player refs before game (lost players get
        // removed from game.getPlayers() after game ends)
        Map<LobbyPlayer, Player> lobbyToPlayer = new HashMap<>();
        String gameId = name1 + "_vs_" + name2;
        for (Player p : game.getPlayers()) {
            lobbyToPlayer.put(p.getLobbyPlayer(), p);
            if (p.getController() instanceof PlayerControllerRL) {
                ((PlayerControllerRL) p.getController())
                    .getRLController()
                    .onGameStart(gameId + "_" + p.getName());
            }
        }

        // Attach decision listeners and state recorders
        // Uses plain LobbyPlayerAi (subclassing breaks game)
        // so we create recorders directly here
        Map<Player, forge.ai.rl.training.TrajectoryRecorder>
                recorders = new HashMap<>();
        RLConfig anyConfig = config1.isRecordTrajectories()
                ? config1 : config2;

        if (anyConfig.isRecordTrajectories()) {
            for (Player p : lobbyToPlayer.values()) {
                forge.ai.rl.training.TrajectoryRecorder rec =
                    new forge.ai.rl.training.TrajectoryRecorder(
                        anyConfig.getTrajectoryOutputDir());
                rec.startGame(gameId + "_" + p.getName());
                recorders.put(p, rec);

                // Decision listener disabled — causes turn-0
                // (likely PlayerControllerAi checkstyle issues
                // with modified class in fat jar)

                // Event-based state snapshots
                forge.ai.rl.GameStateRecorder gsr =
                    new forge.ai.rl.GameStateRecorder(
                        game, p, rec, anyConfig);
                gsr.register();
            }
        }

        // Run the game with timeout
        try {
            TimeLimitedCodeBlock.runWithTimeout(() -> {
                match.startGame(game);
            }, timeoutSec, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            if (!game.isGameOver()) {
                game.setGameOver(GameEndReason.Draw);
            }
        } catch (ModelServerException e) {
            throw e;
        } catch (Exception | StackOverflowError e) {
            // Check if a ModelServerException is wrapped inside
            Throwable cause = e.getCause();
            while (cause != null) {
                if (cause instanceof ModelServerException) {
                    throw (ModelServerException) cause;
                }
                cause = cause.getCause();
            }
            if (!game.isGameOver()) {
                game.setGameOver(GameEndReason.Draw);
            }
        }

        // Stop recording and write trajectory files
        for (Map.Entry<Player,
                forge.ai.rl.training.TrajectoryRecorder> entry
                : recorders.entrySet()) {
            Player p = entry.getKey();
            RegisteredPlayer rp = p.getRegisteredPlayer();
            boolean won = game.getOutcome() != null
                    && !game.getOutcome().isDraw()
                    && rp != null
                    && game.getOutcome().isWinner(rp);
            entry.getValue().endGame(won);
        }

        // Finalize RLController trajectory recorders (records RL model decisions)
        for (Player p : lobbyToPlayer.values()) {
            if (p.getController() instanceof PlayerControllerRL) {
                RegisteredPlayer rp = p.getRegisteredPlayer();
                boolean won = game.getOutcome() != null
                        && !game.getOutcome().isDraw()
                        && rp != null
                        && game.getOutcome().isWinner(rp);
                ((PlayerControllerRL) p.getController())
                    .getRLController()
                    .onGameEnd(won);
            }
        }

        // Build result
        GameResult result = new GameResult();
        if (game.getOutcome() == null || game.getOutcome().isDraw()) {
            result.isDraw = true;
            result.summary = "Draw";
        } else {
            LobbyPlayer winner = game.getOutcome().getWinningLobbyPlayer();
            result.player1Won = winner.equals(lobby1);
            result.summary = winner.getName() + " won (turn " + game.getPhaseHandler().getTurn() + ")";
        }
        result.turns = game.getPhaseHandler().getTurn();

        // Write trajectory files for PPO
        if (anyConfig.isRecordTrajectories()
                && !result.isDraw) {
            for (Player p : lobbyToPlayer.values()) {
                try {
                    boolean won = !result.isDraw
                            && p.getLobbyPlayer().equals(
                                game.getOutcome()
                                .getWinningLobbyPlayer());
                    forge.ai.rl.training.TrajectoryRecorder
                        rec = new forge.ai.rl.training
                            .TrajectoryRecorder(
                                anyConfig
                                .getTrajectoryOutputDir());
                    rec.startGame(
                        gameId + "_" + p.getName());
                    // Minimal record: just global features
                    // and outcome (no encoder, no cards)
                    float[] gf = new float[64];
                    gf[0] = normalize(p.getLife(), -10, 40);
                    Player opp = null;
                    for (Player o : lobbyToPlayer.values()) {
                        if (o != p) { opp = o; break; }
                    }
                    gf[1] = opp != null
                        ? normalize(opp.getLife(), -10, 40)
                        : 0.5f;
                    gf[4] = normalize(result.turns, 0, 30);
                    forge.ai.rl.features.GameStateFeatures
                        gs = new forge.ai.rl.features
                            .GameStateFeatures(
                            gf,
                            new float[30][128],
                            new boolean[30],
                            new float[30][128],
                            new boolean[30],
                            new float[15][128],
                            new boolean[15],
                            new float[40][128],
                            new boolean[40],
                            new float[40][128],
                            new boolean[40],
                            new float[10][128],
                            new boolean[10]);
                    forge.ai.rl.decisions.DecisionContext
                        ctx = new forge.ai.rl.decisions
                            .DecisionContext(
                            forge.ai.rl.decisions
                                .DecisionType
                                .PRIORITY_ACTION,
                            gs, java.util.List.of(),
                            0, 0,
                            "game_end_turn_"
                                + result.turns);
                    forge.ai.rl.decisions.DecisionResult
                        dr = new forge.ai.rl.decisions
                            .DecisionResult(
                            java.util.List.of(
                                won ? 1 : 0),
                            new float[0],
                            won ? 1f : -1f, true);
                    rec.recordDecision(ctx, dr,
                        p.getLife(),
                        opp != null ? opp.getLife() : 0,
                        0, 0, 0, 0);
                    rec.endGame(won);
                } catch (Exception e) {
                    System.err.println("TRAJ: "
                        + e.getMessage());
                }
            }
        }

        return result;
    }

    private static LobbyPlayer createPlayer(String name, RLConfig config) {
        if (config.getMode() == RLModelMode.HEURISTIC_FALLBACK) {
            LobbyPlayerAi aiPlayer;
            // Always use plain LobbyPlayerAi — subclassing
            // breaks the game engine (turn 0 instant win)
            aiPlayer = new LobbyPlayerAi(name, null);
            aiPlayer.setAiProfile("Default");
            return aiPlayer;
        } else {
            // RL mode — use our RL controller
            return new LobbyPlayerRL(name, config);
        }
    }

    private static Deck loadDeck(String deckParam) {
        // Check if it's a file path
        int dotpos = deckParam.lastIndexOf('.');
        if (dotpos > 0 && dotpos == deckParam.length() - 4) {
            // Try as absolute path first
            File f = new File(deckParam);
            if (f.exists()) {
                return DeckSerializer.fromFile(f);
            }
            // Try in constructed deck directory
            f = new File(ForgeConstants.DECK_CONSTRUCTED_DIR + deckParam);
            if (f.exists()) {
                return DeckSerializer.fromFile(f);
            }
        }

        // Try by name in deck storage
        try {
            return FModel.getDecks().getConstructed().get(deckParam);
        } catch (Exception e) {
            return null;
        }
    }

    private static void printHelp() {
        System.out.println();
        System.out.println("Usage: forge rltrain <mode> [options]");
        System.out.println();
        System.out.println("Modes:");
        System.out.println("  collect   - Record heuristic AI games for imitation learning");
        System.out.println("  evaluate  - Evaluate RL AI against heuristic AI");
        System.out.println("  selfplay  - Run RL AI self-play games");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -d <deck>     Deck file (.dck) or deck name (repeat for multiple)");
        System.out.println("  -D <dir>      Load all .dck files from directory");
        System.out.println("  -n <N>        Number of games (default: 100)");
        System.out.println("  -o <dir>      Output directory for trajectories (default: rl_data/trajectories)");
        System.out.println("  -t <threads>  Parallel threads (default: all CPUs)");
        System.out.println("  -c <seconds>  Timeout per game (default: 180)");
        System.out.println("  -host <host>  Model server host (default: localhost)");
        System.out.println("  -port <port>  Model server port (default: 50051)");
        System.out.println("  -q            Quiet mode");
    }

    private static float normalize(double v, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (v - min) / (max - min)));
    }

    private static class GameResult {
        boolean player1Won = false;
        boolean isDraw = false;
        int turns = 0;
        String summary = "";
    }
}
