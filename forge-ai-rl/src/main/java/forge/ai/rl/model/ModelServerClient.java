package forge.ai.rl.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import forge.ai.rl.RLConfig;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import org.tinylog.Logger;

import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Client for communicating with the Python model server.
 *
 * Uses a simple JSON-over-TCP protocol for initial implementation.
 * Can be upgraded to gRPC/protobuf for production performance.
 *
 * Protocol:
 * - Client sends: [4 bytes length (big-endian)] [JSON payload]
 * - Server responds: [4 bytes length (big-endian)] [JSON payload]
 */
public class ModelServerClient {
    private final RLConfig config;
    private final Gson gson;
    private Socket socket;
    private DataInputStream in;
    private DataOutputStream out;
    private boolean connected = false;

    public ModelServerClient(RLConfig config) {
        this.config = config;
        this.gson = new GsonBuilder().create();
    }

    /**
     * Connect to the Python model server.
     */
    public synchronized boolean connect() {
        if (connected) return true;
        try {
            socket = new Socket(config.getGrpcHost(), config.getGrpcPort());
            socket.setSoTimeout(config.getGrpcTimeoutMs());
            in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
            connected = true;
            Logger.info("Connected to RL model server at {}:{}", config.getGrpcHost(), config.getGrpcPort());
            return true;
        } catch (IOException e) {
            Logger.warn("Failed to connect to RL model server: {}", e.getMessage());
            connected = false;
            return false;
        }
    }

    /**
     * Disconnect from the model server.
     */
    public synchronized void disconnect() {
        try {
            if (socket != null) socket.close();
        } catch (IOException e) {
            // ignore
        }
        connected = false;
    }

    /**
     * Send a decision request to the model server and get the result.
     * Returns null if the server is unavailable.
     */
    public synchronized DecisionResult requestDecision(DecisionContext context) {
        if (!connected && !connect()) {
            return null;
        }

        try {
            // Serialize request — send the flat game state array
            // (same format as trajectory files) so the Python server
            // parses it with the exact same parse_game_state() used
            // during training. No encoding mismatch possible.
            InferenceRequest request = new InferenceRequest();
            request.decisionType = context.getType().name();
            request.globalFeatures = context.getGameState().getGlobalFeatures();
            request.gameStateFlat = context.getGameState().flatten();
            request.candidateFeatures = context.getCandidateFeatures().toArray(new float[0][]);
            request.minSelections = context.getMinSelections();
            request.maxSelections = context.getMaxSelections();
            request.contextInfo = context.getContextInfo();

            String json = gson.toJson(request);
            byte[] payload = json.getBytes(StandardCharsets.UTF_8);

            // Send: length prefix + payload
            out.writeInt(payload.length);
            out.write(payload);
            out.flush();

            // Receive: length prefix + payload
            int responseLen = in.readInt();
            byte[] responseBytes = new byte[responseLen];
            in.readFully(responseBytes);
            String responseJson = new String(responseBytes, StandardCharsets.UTF_8);

            InferenceResponse response = gson.fromJson(responseJson, InferenceResponse.class);
            return new DecisionResult(
                    response.selectedIndices != null ? response.selectedIndices : List.of(),
                    response.actionProbabilities != null ? response.actionProbabilities : new float[0],
                    response.valueEstimate,
                    false
            );

        } catch (IOException e) {
            Logger.warn("Model server communication error: {}", e.getMessage());
            connected = false;
            return null;
        }
    }

    /**
     * Check if connected to the model server.
     */
    public boolean isConnected() {
        return connected;
    }

    /**
     * Request structure sent to the Python model server.
     */
    private static class InferenceRequest {
        String decisionType;
        float[] globalFeatures;
        float[] gameStateFlat;
        float[][] candidateFeatures;
        int minSelections;
        int maxSelections;
        String contextInfo;
    }

    /**
     * Response structure from the Python model server.
     */
    static class InferenceResponse {
        List<Integer> selectedIndices;
        float[] actionProbabilities;
        float valueEstimate;
    }
}
