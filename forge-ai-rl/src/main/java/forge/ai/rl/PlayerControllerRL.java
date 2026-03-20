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
 * Only declareAttackers and declareBlockers are overridden to use the RL model.
 * When the model server is unavailable, combat also falls back to heuristic.
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

    // ===== COMBAT — RL model decisions =====

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        if (rl.getConfig().getMode() != RLModelMode.GRPC || !rl.isModelServerAvailable()) {
            super.declareAttackers(attacker, combat);
            return;
        }

        GameEntity defender = attacker.getWeakestOpponent();
        if (defender == null) return;

        CardCollection possibleAttackers = new CardCollection();
        for (Card c : attacker.getCreaturesInPlay()) {
            if (CombatUtil.canAttack(c, defender)) {
                possibleAttackers.add(c);
            }
        }

        if (possibleAttackers.isEmpty()) return;

        List<Integer> attackerIndices = rl.decideAttackers(possibleAttackers);

        for (int idx : attackerIndices) {
            if (idx >= 0 && idx < possibleAttackers.size()) {
                Card c = possibleAttackers.get(idx);
                if (CombatUtil.canAttack(c, defender)) {
                    combat.addAttacker(c, defender);
                }
            }
        }
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        if (rl.getConfig().getMode() != RLModelMode.GRPC || !rl.isModelServerAvailable()) {
            super.declareBlockers(defender, combat);
            return;
        }

        List<Card> possibleBlockers = new ArrayList<>();
        for (Card c : defender.getCreaturesInPlay()) {
            if (!c.isTapped() && !c.hasKeyword("CARDNAME can't block.")) {
                possibleBlockers.add(c);
            }
        }

        CardCollection attackers = combat.getAttackers();
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return;

        List<int[]> assignments = rl.decideBlockers(possibleBlockers, attackers);

        for (int[] pair : assignments) {
            int blockerIdx = pair[0];
            int attackerIdx = pair[1];
            if (blockerIdx >= 0 && blockerIdx < possibleBlockers.size()
                    && attackerIdx >= 0 && attackerIdx < attackers.size()) {
                Card blocker = possibleBlockers.get(blockerIdx);
                Card attacker = attackers.get(attackerIdx);
                if (CombatUtil.canBlock(attacker, blocker, combat)) {
                    combat.addBlocker(attacker, blocker);
                }
            }
        }
    }
}
