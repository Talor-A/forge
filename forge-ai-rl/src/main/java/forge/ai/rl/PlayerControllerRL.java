package forge.ai.rl;

import forge.LobbyPlayer;
import forge.game.*;
import forge.game.card.*;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.*;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import org.tinylog.Logger;

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

    // Diagnostic counters (per game)
    private int priorityModelAsked = 0;
    private int priorityModelPass = 0;
    private int priorityModelPlay = 0;
    private int priorityTargetingRejected = 0;
    private int priorityHeuristicBypass = 0;

    public PlayerControllerRL(Game game, Player p, LobbyPlayer lp, RLConfig config) {
        super(game, p, lp instanceof forge.ai.LobbyPlayerAi
                ? (forge.ai.LobbyPlayerAi) lp
                : createFallbackLobby(lp.getName()));
        this.rl = new RLController(config);
        this.rl.setPlayer(p);
    }

    public void logDiagnostics() {
        if (priorityModelAsked > 0 || priorityHeuristicBypass > 0) {
            Logger.info("RL_DIAG: priority: asked={} play={} pass={} targeting_rejected={} heuristic_bypass={}",
                    priorityModelAsked, priorityModelPlay, priorityModelPass,
                    priorityTargetingRejected, priorityHeuristicBypass);
        }
    }

    public void resetDiagnostics() {
        priorityModelAsked = 0;
        priorityModelPass = 0;
        priorityModelPlay = 0;
        priorityTargetingRejected = 0;
        priorityHeuristicBypass = 0;
    }

    private static forge.ai.LobbyPlayerAi createFallbackLobby(String name) {
        forge.ai.LobbyPlayerAi lp = new forge.ai.LobbyPlayerAi(name, null);
        lp.setAiProfile("Default");
        return lp;
    }

    public RLController getRLController() {
        return rl;
    }

    // ===== PRIORITY — which spell/ability to play =====

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        if (rl.getConfig().getMode() == RLModelMode.GRPC && rl.isModelServerAvailable()) {
            // Let the heuristic build the candidate lists (lands, filtering, etc.)
            // then use the RL model to pick from the mechanically-legal set
            List<SpellAbility> heuristicResult = super.chooseSpellAbilityToPlay();

            // Get all mechanically legal spells (broad candidate set)
            List<SpellAbility> candidates = getAi().getLastPlayableSpellAbilities();
            if (candidates == null || candidates.isEmpty()) {
                priorityHeuristicBypass++;
                return heuristicResult; // land play or early-return — use heuristic
            }

            priorityModelAsked++;
            int idx = rl.decidePriorityAction(candidates);
            if (idx < 0 || idx >= candidates.size()) {
                priorityModelPass++;
                return null; // model chose pass — null signals "pass priority"
            }

            // RL picked a spell — let heuristic set up targeting and evaluate
            SpellAbility chosen = candidates.get(idx);
            forge.ai.AiPlayDecision reason = getAi().canPlayForRL(chosen);

            if (reason.willingToPlay()) {
                // Heuristic agrees — targets set, play it
                priorityModelPlay++;
                List<SpellAbility> rlResult = new ArrayList<>();
                rlResult.add(chosen);
                return rlResult;
            }

            // Strategic vetoes only — heuristic set up targets but decided
            // not to play. We override IF targets are still valid.
            boolean isStrategicVeto = (reason == forge.ai.AiPlayDecision.CantPlayAi
                    || reason == forge.ai.AiPlayDecision.BadEtbEffects
                    || reason == forge.ai.AiPlayDecision.CurseEffects);

            if (isStrategicVeto) {
                // Only override if spell doesn't need targeting OR targets survived
                if (!chosen.usesTargeting() || chosen.isTargetNumberValid()) {
                    Logger.info("RL_OVERRIDE: {} ({}) heuristic_said={}",
                            chosen.getHostCard().getName(),
                            chosen.getApi() != null ? chosen.getApi().name() : "null",
                            reason.name());
                    priorityModelPlay++;
                    List<SpellAbility> rlResult = new ArrayList<>();
                    rlResult.add(chosen);
                    return rlResult;
                }
                // Strategic veto AND targets cleared — can't safely play
                priorityTargetingRejected++;
                Logger.info("RL_SPELL_REJECTED: {} ({}) reason={}_no_targets",
                        chosen.getHostCard().getName(),
                        chosen.getApi() != null ? chosen.getApi().name() : "null",
                        reason.name());
                return null;
            }

            // Mechanical failure — genuinely can't play
            priorityTargetingRejected++;
            Logger.info("RL_SPELL_REJECTED: {} ({}) reason={}",
                    chosen.getHostCard().getName(),
                    chosen.getApi() != null ? chosen.getApi().name() : "null",
                    reason.name());
            return null;
        } else {
            // Heuristic decides — record the decision
            List<SpellAbility> result = super.chooseSpellAbilityToPlay();

            // Get all mechanically legal spells as candidates
            List<SpellAbility> candidates = getAi().getLastPlayableSpellAbilities();
            if (candidates == null || candidates.isEmpty()) {
                return result; // land play or early-return path — nothing to record
            }

            // Determine what the heuristic chose
            SpellAbility chosenSa = null;
            if (result != null && !result.isEmpty()) {
                chosenSa = result.get(0);
                if (chosenSa != null && chosenSa.isLandAbility()) {
                    return result; // don't record land plays
                }
            }

            rl.recordHeuristicPriority(candidates, chosenSa);
            return result;
        }
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
            // Capture state BEFORE the heuristic modifies combat
            rl.capturePreDecisionState(possibleBlockers);

            super.declareBlockers(defender, combat);

            // Read back the full assignment: which blocker blocks which attacker
            // Record as (blocker, attacker) pairs matching inference format
            rl.recordHeuristicBlockAssignment(
                    possibleBlockers, attackers, combat);
        }
    }
}
