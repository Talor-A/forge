package forge.ai.rl.training;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import org.tinylog.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.function.IntSupplier;

/**
 * Records game trajectories (sequences of state-action-reward tuples) for training.
 *
 * Each game produces one trajectory file containing all decisions made by the RL player.
 * These files are consumed by the Python training pipeline.
 *
 * File format: one JSON object per line (JSONL), with a header line and one line per decision.
 */
public class TrajectoryRecorder {
    private final String outputDir;
    private final Gson gson;
    private final List<DecisionRecord> currentGame;
    private String gameId;
    private long gameStartTime;
    private IntSupplier logSizeSupplier;

    // Running state for reward shaping
    private int prevLifeAdvantage = 0;
    private int prevCardAdvantage = 0;
    private int prevBoardAdvantage = 0;

    public TrajectoryRecorder(String outputDir) {
        this.outputDir = outputDir;
        this.gson = new GsonBuilder().create();
        this.currentGame = new ArrayList<>();

        // Ensure output directory exists
        try {
            Files.createDirectories(Paths.get(outputDir));
        } catch (IOException e) {
            Logger.error("Failed to create trajectory output directory: {}", e.getMessage());
        }
    }

    /**
     * Start recording a new game.
     */
    public void startGame(String gameId) {
        this.gameId = gameId;
        this.gameStartTime = System.currentTimeMillis();
        this.currentGame.clear();
        this.prevLifeAdvantage = 0;
        this.prevCardAdvantage = 0;
        this.prevBoardAdvantage = 0;
        this.logSizeSupplier = null;
    }

    /** Provide a supplier that returns the current GameLog entry count, used to
     * tag each decision with the count of log entries that preceded it. */
    public void setLogSizeSupplier(IntSupplier supplier) {
        this.logSizeSupplier = supplier;
    }

    /**
     * Record a decision made during the game.
     */
    public void recordDecision(DecisionContext context, DecisionResult result,
                                int myLife, int oppLife,
                                int myHandSize, int oppHandSize,
                                int myCreatureCount, int oppCreatureCount) {
        DecisionRecord record = new DecisionRecord();
        record.turnIndex = currentGame.size();
        record.decisionType = context.getType().name();
        record.contextInfo = context.getContextInfo();
        record.globalFeatures = context.getGameState().getGlobalFeatures();
        record.candidateCount = context.getCandidateFeatures().size();
        record.selectedIndices = result.getSelectedIndices();
        record.actionProbabilities = result.getActionProbabilities();
        record.visitProportions = result.getVisitProportions();
        record.valueEstimate = result.getValueEstimate();
        record.usedFallback = result.isUsedFallback();
        record.logIndex = logSizeSupplier != null ? logSizeSupplier.getAsInt() : -1;

        // Compute intermediate reward from state changes
        int lifeAdv = myLife - oppLife;
        int cardAdv = myHandSize - oppHandSize;
        int boardAdv = myCreatureCount - oppCreatureCount;

        record.intermediateReward = 0;
        record.intermediateReward += (lifeAdv - prevLifeAdvantage) * 0.01;
        record.intermediateReward += (cardAdv - prevCardAdvantage) * 0.05;
        record.intermediateReward += (boardAdv - prevBoardAdvantage) * 0.02;

        prevLifeAdvantage = lifeAdv;
        prevCardAdvantage = cardAdv;
        prevBoardAdvantage = boardAdv;

        // Store flattened game state for training
        record.gameStateFlat = context.getGameState().flatten();

        // Store candidate features
        record.candidateFeatures = context.getCandidateFeatures().toArray(new float[0][]);

        // Store spell features for target decisions
        record.spellFeatures = context.getSpellFeatures();

        currentGame.add(record);
    }

    /**
     * End the game and write the trajectory to disk.
     * @param won true if the RL player won
     */
    public void endGame(boolean won) {
        endGame(won, Collections.emptyList());
    }

    /**
     * End the game and write the trajectory, including a human-readable game log.
     * @param gameLogLines log entries in chronological order
     */
    public void endGame(boolean won, List<String> gameLogLines) {
        if (currentGame.isEmpty()) return;

        double terminalReward = won ? 1.0 : -1.0;

        // Set terminal reward on last decision
        if (!currentGame.isEmpty()) {
            currentGame.get(currentGame.size() - 1).terminalReward = terminalReward;
        }

        // Write trajectory file
        String filename = String.format("traj_%s_%s_%s.jsonl",
                gameId != null ? gameId : UUID.randomUUID().toString().substring(0, 8),
                won ? "W" : "L",
                gameStartTime);

        Path filePath = Paths.get(outputDir, filename);
        try (BufferedWriter writer = Files.newBufferedWriter(filePath)) {
            // Header
            TrajectoryHeader header = new TrajectoryHeader();
            header.gameId = gameId;
            header.won = won;
            header.totalDecisions = currentGame.size();
            header.durationMs = System.currentTimeMillis() - gameStartTime;
            header.gameLog = gameLogLines;
            writer.write(gson.toJson(header));
            writer.newLine();

            // Decision records
            for (DecisionRecord record : currentGame) {
                writer.write(gson.toJson(record));
                writer.newLine();
            }

            Logger.info("Wrote trajectory: {} ({} decisions, {})",
                    filePath.getFileName(), currentGame.size(), won ? "WIN" : "LOSS");

        } catch (IOException e) {
            Logger.error("Failed to write trajectory file: {}", e.getMessage());
        }

        currentGame.clear();
    }

    /**
     * Get the number of decisions recorded in the current game.
     */
    public int getCurrentGameDecisionCount() {
        return currentGame.size();
    }

    // Internal data structures for serialization

    private static class TrajectoryHeader {
        String gameId;
        boolean won;
        int totalDecisions;
        long durationMs;
        List<String> gameLog;
    }

    private static class DecisionRecord {
        int turnIndex;
        String decisionType;
        String contextInfo;
        float[] globalFeatures;
        float[] gameStateFlat;
        float[][] candidateFeatures;
        int candidateCount;
        List<Integer> selectedIndices;
        float[] actionProbabilities;   // win rates (Q-values) per candidate
        float[] visitProportions;      // visit fractions per candidate (search policy)
        float valueEstimate;
        boolean usedFallback;
        double intermediateReward;
        double terminalReward;
        float[] spellFeatures; // 64-dim source spell for target decisions
        int logIndex; // GameLog entry count at decision time (-1 if unavailable)
    }
}
