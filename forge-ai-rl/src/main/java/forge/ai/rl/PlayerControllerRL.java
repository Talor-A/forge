package forge.ai.rl;

import forge.LobbyPlayer;
import forge.game.*;
import forge.game.card.*;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.*;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * PlayerController for the Reinforcement Learning AI.
 *
 * Extends PlayerControllerAi so ALL non-combat decisions use the proven
 * heuristic AI (ComputerUtil, AiController, etc.) without ClassCastException.
 *
 * declareAttackers and declareBlockers are overridden to either:
 * - Use the RL model (GRPC mode, server available)
 * - Or delegate to heuristic AND record the decision with pre-decision
 *   state for training data collection
 */
public class PlayerControllerRL extends forge.ai.PlayerControllerAi {

    private final RLController rl;

    public PlayerControllerRL(Game game, Player p, LobbyPlayer lp, RLConfig config) {
        super(game, p, lp instanceof forge.ai.LobbyPlayerAi
                ? (forge.ai.LobbyPlayerAi) lp
                : createFallbackLobby(lp.getName()));
        this.rl = new RLController(config);
        this.rl.setPlayer(p);
    }

    private static forge.ai.LobbyPlayerAi createFallbackLobby(String name) {
        forge.ai.LobbyPlayerAi lp = new forge.ai.LobbyPlayerAi(name, null);
        lp.setAiProfile("Default");
        return lp;
    }

    public RLController getRLController() {
        return rl;
    }

    // ===== COMBAT — RL model or heuristic with recording =====

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        GameEntity defender = attacker.getWeakestOpponent();
        if (defender == null) return;

        // Build candidate list (creatures that can legally attack)
        CardCollection possibleAttackers = new CardCollection();
        for (Card c : attacker.getCreaturesInPlay()) {
            if (CombatUtil.canAttack(c, defender)) {
                possibleAttackers.add(c);
            }
        }
        if (possibleAttackers.isEmpty()) return;

        if (rl.getConfig().getMode() == RLModelMode.GRPC && rl.isModelServerAvailable()) {
            // RL model makes the decision
            List<Integer> attackerIndices = rl.decideAttackers(possibleAttackers);
            for (int idx : attackerIndices) {
                if (idx >= 0 && idx < possibleAttackers.size()) {
                    Card c = possibleAttackers.get(idx);
                    if (CombatUtil.canAttack(c, defender)) {
                        combat.addAttacker(c, defender);
                    }
                }
            }
        } else {
            // Heuristic decides — but we record the decision
            // 1. Capture pre-decision state (creatures untapped, no combat)
            rl.capturePreDecisionState(possibleAttackers);

            // 2. Let heuristic make the decision (modifies combat object)
            super.declareAttackers(attacker, combat);

            // 3. Read back what the heuristic chose from the combat object
            CardCollection actualAttackers = combat.getAttackers();
            List<Integer> selectedIndices = new ArrayList<>();
            for (int i = 0; i < possibleAttackers.size(); i++) {
                if (actualAttackers.contains(possibleAttackers.get(i))) {
                    selectedIndices.add(i);
                }
            }

            // 4. Record: pre-decision state + heuristic's choice
            rl.recordHeuristicAttack(possibleAttackers, selectedIndices);
        }
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        // Build candidate list (untapped creatures that can block)
        List<Card> possibleBlockers = new ArrayList<>();
        for (Card c : defender.getCreaturesInPlay()) {
            if (!c.isTapped() && !c.hasKeyword("CARDNAME can't block.")) {
                possibleBlockers.add(c);
            }
        }
        CardCollection attackers = combat.getAttackers();
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return;

        if (rl.getConfig().getMode() == RLModelMode.GRPC && rl.isModelServerAvailable()) {
            // RL model makes the decision
            List<int[]> assignments = rl.decideBlockers(possibleBlockers, attackers);
            for (int[] pair : assignments) {
                int blockerIdx = pair[0];
                int attackerIdx = pair[1];
                if (blockerIdx >= 0 && blockerIdx < possibleBlockers.size()
                        && attackerIdx >= 0 && attackerIdx < attackers.size()) {
                    Card blocker = possibleBlockers.get(blockerIdx);
                    Card att = attackers.get(attackerIdx);
                    if (CombatUtil.canBlock(att, blocker, combat)) {
                        combat.addBlocker(att, blocker);
                    }
                }
            }
        } else {
            // Heuristic decides — record with pre-decision state
            rl.capturePreDecisionState(possibleBlockers);

            super.declareBlockers(defender, combat);

            // Read back which creatures blocked
            List<Integer> selectedIndices = new ArrayList<>();
            for (int i = 0; i < possibleBlockers.size(); i++) {
                Card blocker = possibleBlockers.get(i);
                // Check if this creature is blocking anything
                for (Card att : attackers) {
                    if (combat.getBlockers(att).contains(blocker)) {
                        selectedIndices.add(i);
                        break;
                    }
                }
            }

            rl.recordHeuristicBlock(possibleBlockers, selectedIndices);
        }
    }
}
