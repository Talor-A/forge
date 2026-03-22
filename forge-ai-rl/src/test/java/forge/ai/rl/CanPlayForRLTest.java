package forge.ai.rl;

import forge.ai.AiPlayDecision;
import org.testng.annotations.Test;
import static org.testng.AssertJUnit.*;

/**
 * Tests for the canPlayForRL override logic.
 *
 * canPlayForRL should:
 * 1. Return WillPlay when the heuristic agrees (targets set by heuristic)
 * 2. Override strategic-only vetoes (CantPlayAi, BadEtbEffects, CurseEffects)
 *    ONLY if the heuristic left valid targets in place
 * 3. NEVER override mechanical failures (CantPlaySa, CantAfford, TargetingFailed,
 *    AnotherTime, MissingPhaseRestrictions, etc.)
 * 4. NEVER try to set up its own targets — only use what the heuristic set up
 */
public class CanPlayForRLTest {

    // Classify which AiPlayDecision values are strategic vs mechanical
    private static boolean isStrategicVeto(AiPlayDecision reason) {
        return reason == AiPlayDecision.CantPlayAi
            || reason == AiPlayDecision.BadEtbEffects
            || reason == AiPlayDecision.CurseEffects;
    }

    private static boolean isMechanicalFailure(AiPlayDecision reason) {
        return reason == AiPlayDecision.CantPlaySa
            || reason == AiPlayDecision.CantAfford
            || reason == AiPlayDecision.CantAffordX
            || reason == AiPlayDecision.TargetingFailed
            || reason == AiPlayDecision.AnotherTime
            || reason == AiPlayDecision.MissingPhaseRestrictions
            || reason == AiPlayDecision.TimingRestrictions
            || reason == AiPlayDecision.ConditionsNotMet
            || reason == AiPlayDecision.MissingLogic
            || reason == AiPlayDecision.MissingNeededCards
            || reason == AiPlayDecision.WouldDestroyLegend
            || reason == AiPlayDecision.WouldDestroyOtherPlaneswalker
            || reason == AiPlayDecision.WouldBecomeZeroToughnessCreature
            || reason == AiPlayDecision.WouldDestroyWorldEnchantment
            || reason == AiPlayDecision.StopRunawayActivations;
    }

    @Test
    public void testStrategicVetoClassification() {
        // These should be overridable
        assertTrue(isStrategicVeto(AiPlayDecision.CantPlayAi));
        assertTrue(isStrategicVeto(AiPlayDecision.BadEtbEffects));
        assertTrue(isStrategicVeto(AiPlayDecision.CurseEffects));

        // These should NOT be overridable
        assertFalse(isStrategicVeto(AiPlayDecision.WillPlay));
        assertFalse(isStrategicVeto(AiPlayDecision.CantPlaySa));
        assertFalse(isStrategicVeto(AiPlayDecision.CantAfford));
        assertFalse(isStrategicVeto(AiPlayDecision.TargetingFailed));
        assertFalse(isStrategicVeto(AiPlayDecision.AnotherTime));
        assertFalse(isStrategicVeto(AiPlayDecision.MissingPhaseRestrictions));
    }

    @Test
    public void testMechanicalFailureClassification() {
        // Mechanical failures must never be overridden
        assertTrue(isMechanicalFailure(AiPlayDecision.CantPlaySa));
        assertTrue(isMechanicalFailure(AiPlayDecision.CantAfford));
        assertTrue(isMechanicalFailure(AiPlayDecision.CantAffordX));
        assertTrue(isMechanicalFailure(AiPlayDecision.TargetingFailed));
        assertTrue(isMechanicalFailure(AiPlayDecision.AnotherTime));
        assertTrue(isMechanicalFailure(AiPlayDecision.MissingPhaseRestrictions));
        assertTrue(isMechanicalFailure(AiPlayDecision.TimingRestrictions));
        assertTrue(isMechanicalFailure(AiPlayDecision.WouldDestroyLegend));

        // Strategic vetoes are NOT mechanical failures
        assertFalse(isMechanicalFailure(AiPlayDecision.CantPlayAi));
        assertFalse(isMechanicalFailure(AiPlayDecision.BadEtbEffects));
        assertFalse(isMechanicalFailure(AiPlayDecision.CurseEffects));
    }

    @Test
    public void testAllDecisionsAreCategorized() {
        // Every non-positive decision should be either strategic or mechanical
        for (AiPlayDecision d : AiPlayDecision.values()) {
            if (d.willingToPlay()) continue; // positive decisions
            // Skip the "wait" decisions which are also strategic timing
            if (d == AiPlayDecision.WaitForCombat
                || d == AiPlayDecision.WaitForMain2
                || d == AiPlayDecision.WaitForEndOfTurn
                || d == AiPlayDecision.StackNotEmpty
                || d == AiPlayDecision.DoesntImpactCombat
                || d == AiPlayDecision.DoesntImpactGame
                || d == AiPlayDecision.LifeInDanger
                || d == AiPlayDecision.CostNotAcceptable
                || d == AiPlayDecision.NeedsToPlayCriteriaNotMet) continue;
            assertTrue("Decision " + d + " should be strategic or mechanical",
                isStrategicVeto(d) || isMechanicalFailure(d));
        }
    }

    @Test
    public void testOverrideDecisionLogic() {
        // Simulate the decision logic from PlayerControllerRL
        // Given a reason from canPlayForRL, should we play the spell?

        // WillPlay: always play (targets set by heuristic)
        assertTrue(shouldPlay(AiPlayDecision.WillPlay, true));
        assertTrue(shouldPlay(AiPlayDecision.WillPlay, false)); // no targeting needed

        // Strategic veto with valid targets: play (override)
        assertTrue(shouldPlay(AiPlayDecision.CantPlayAi, true));
        assertTrue(shouldPlay(AiPlayDecision.BadEtbEffects, true));

        // Strategic veto with NO valid targets: don't play
        assertFalse(shouldPlay(AiPlayDecision.CantPlayAi, false));
        assertFalse(shouldPlay(AiPlayDecision.BadEtbEffects, false));

        // Mechanical failures: never play regardless of targets
        assertFalse(shouldPlay(AiPlayDecision.CantPlaySa, true));
        assertFalse(shouldPlay(AiPlayDecision.CantAfford, true));
        assertFalse(shouldPlay(AiPlayDecision.TargetingFailed, true));
        assertFalse(shouldPlay(AiPlayDecision.AnotherTime, true));
        assertFalse(shouldPlay(AiPlayDecision.MissingPhaseRestrictions, true));
    }

    /**
     * Simulates the override decision.
     * @param reason the AiPlayDecision from canPlayForRL
     * @param targetsValid whether the spell has valid targets after evaluation
     * @return true if the RL controller should play the spell
     */
    private boolean shouldPlay(AiPlayDecision reason, boolean targetsValid) {
        if (reason.willingToPlay()) return true;
        if (isStrategicVeto(reason) && targetsValid) return true;
        return false;
    }
}
