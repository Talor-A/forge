package forge.ai.rl.features;

import forge.game.card.Card;
import forge.game.combat.CombatUtil;
import forge.game.keyword.Keyword;

import java.util.ArrayList;
import java.util.List;

/**
 * Static utility class for computing combat-related features for the RL model.
 * Fills per-card features [208-231] and global combat features [76-95].
 */
public class CombatMath {

    /**
     * Inject combat math features into a battlefield creature's feature vector at indices [208-231].
     *
     * @param features      the 256-float feature array for this card
     * @param card          the creature whose features we're computing
     * @param oppCreatures  all opponent creatures on the battlefield
     * @param myCreatures   all friendly creatures on the battlefield
     */
    public static void injectPerCardFeatures(float[] features, Card card, List<Card> oppCreatures, List<Card> myCreatures) {
        if (!card.isCreature()) return;

        // Collect untapped opponent creatures
        List<Card> oppUntapped = new ArrayList<>();
        for (Card c : oppCreatures) {
            if (!c.isTapped()) {
                oppUntapped.add(c);
            }
        }

        int power = card.getNetPower();
        int toughness = card.getNetToughness();
        int damage = card.getDamage();

        // [208] can_attack
        boolean canAttack = !card.isTapped() && !card.hasSickness()
                && !card.hasKeyword(Keyword.DEFENDER) && card.isCreature();
        features[208] = canAttack ? 1f : 0f;

        // [209] is_evasive
        boolean evasive = isEvasive(card, oppUntapped);
        features[209] = evasive ? 1f : 0f;

        // [210] frac_can_block_this
        if (!oppUntapped.isEmpty()) {
            int canBlockCount = 0;
            for (Card opp : oppUntapped) {
                if (CombatUtil.canBlock(card, opp)) {
                    canBlockCount++;
                }
            }
            features[210] = (float) canBlockCount / oppUntapped.size();
        }

        // [211] frac_this_kills: fraction of opp creatures this can kill
        if (!oppCreatures.isEmpty()) {
            int killCount = 0;
            for (Card opp : oppCreatures) {
                if (canKillInCombat(card, opp)) {
                    killCount++;
                }
            }
            features[211] = (float) killCount / oppCreatures.size();
        }

        // [212] frac_kills_this: fraction of opp creatures that can kill this
        if (!oppCreatures.isEmpty()) {
            int killsThisCount = 0;
            for (Card opp : oppCreatures) {
                if (canKillInCombat(opp, card)) {
                    killsThisCount++;
                }
            }
            features[212] = (float) killsThisCount / oppCreatures.size();
        }

        // [213] lethal_damage_remaining: (toughness - damage) / toughness, clamp 0-1
        if (toughness > 0) {
            features[213] = normalize((double)(toughness - damage) / toughness, 0, 1);
        }

        // [214] power_vs_avg_toughness
        if (!oppCreatures.isEmpty()) {
            double avgToughness = 0;
            for (Card opp : oppCreatures) {
                avgToughness += opp.getNetToughness();
            }
            avgToughness /= oppCreatures.size();
            if (avgToughness > 0) {
                features[214] = normalize((double) power / avgToughness, 0, 1);
            }
        }

        // [215] toughness_vs_avg_power
        if (!oppCreatures.isEmpty()) {
            double avgPower = 0;
            for (Card opp : oppCreatures) {
                avgPower += opp.getNetPower();
            }
            avgPower /= oppCreatures.size();
            if (avgPower > 0) {
                features[215] = normalize((double) toughness / avgPower, 0, 1);
            }
        }

        // [216] has_first_strike_advantage
        boolean hasFS = card.hasKeyword(Keyword.FIRST_STRIKE) || card.hasKeyword(Keyword.DOUBLE_STRIKE);
        boolean oppHasFS = false;
        for (Card opp : oppUntapped) {
            if (CombatUtil.canBlock(card, opp)) {
                if (opp.hasKeyword(Keyword.FIRST_STRIKE) || opp.hasKeyword(Keyword.DOUBLE_STRIKE)) {
                    oppHasFS = true;
                    break;
                }
            }
        }
        features[216] = (hasFS && !oppHasFS) ? 1f : 0f;

        // [217-220] keyword checks
        features[217] = card.hasKeyword(Keyword.DEATHTOUCH) ? 1f : 0f;
        features[218] = card.hasKeyword(Keyword.INDESTRUCTIBLE) ? 1f : 0f;
        features[219] = card.hasKeyword(Keyword.LIFELINK) ? 1f : 0f;
        features[220] = card.hasKeyword(Keyword.TRAMPLE) ? 1f : 0f;

        // [221] can_trade_up: can kill opp creature with higher CMC
        boolean canTradeUp = false;
        for (Card opp : oppCreatures) {
            if (canKillInCombat(card, opp) && canKillInCombat(opp, card) && opp.getCMC() > card.getCMC()) {
                canTradeUp = true;
                break;
            }
        }
        features[221] = canTradeUp ? 1f : 0f;

        // [222] can_trade_even: can mutual-kill with same-CMC opp creature
        boolean canTradeEven = false;
        for (Card opp : oppCreatures) {
            if (canKillInCombat(card, opp) && canKillInCombat(opp, card) && opp.getCMC() == card.getCMC()) {
                canTradeEven = true;
                break;
            }
        }
        features[222] = canTradeEven ? 1f : 0f;

        // [223] is_biggest_creature: highest power on both boards
        boolean isBiggest = true;
        for (Card c : myCreatures) {
            if (c != card && c.getNetPower() > power) {
                isBiggest = false;
                break;
            }
        }
        if (isBiggest) {
            for (Card c : oppCreatures) {
                if (c.getNetPower() > power) {
                    isBiggest = false;
                    break;
                }
            }
        }
        features[223] = isBiggest ? 1f : 0f;

        // [224] power_rank_my_board: rank / (count-1), 1.0=highest
        if (myCreatures.size() > 1) {
            int rank = 0;
            for (Card c : myCreatures) {
                if (c != card && c.getNetPower() < power) {
                    rank++;
                }
            }
            features[224] = (float) rank / (myCreatures.size() - 1);
        } else {
            features[224] = 1f;
        }

        // [225] toughness_rank_my_board: rank / (count-1), 1.0=highest
        if (myCreatures.size() > 1) {
            int rank = 0;
            for (Card c : myCreatures) {
                if (c != card && c.getNetToughness() < toughness) {
                    rank++;
                }
            }
            features[225] = (float) rank / (myCreatures.size() - 1);
        } else {
            features[225] = 1f;
        }

        // [226] safe_attacker: evasive OR no single opp can profitably block (kill or trade)
        boolean safeAttacker = evasive;
        if (!safeAttacker) {
            boolean anyProfitableBlock = false;
            for (Card opp : oppUntapped) {
                if (CombatUtil.canBlock(card, opp) && canKillInCombat(opp, card)) {
                    anyProfitableBlock = true;
                    break;
                }
            }
            safeAttacker = !anyProfitableBlock;
        }
        features[226] = safeAttacker ? 1f : 0f;

        // [227] must_be_double_blocked: MENACE or power > max single blocker toughness
        boolean mustDoubleBlock = card.hasKeyword(Keyword.MENACE);
        if (!mustDoubleBlock) {
            int maxBlockerToughness = 0;
            for (Card opp : oppUntapped) {
                if (CombatUtil.canBlock(card, opp) && opp.getNetToughness() > maxBlockerToughness) {
                    maxBlockerToughness = opp.getNetToughness();
                }
            }
            if (maxBlockerToughness > 0 && power > maxBlockerToughness) {
                mustDoubleBlock = true;
            }
        }
        features[227] = mustDoubleBlock ? 1f : 0f;

        // [228] best_blocker_power: max power of opp creature that can block this, norm/20
        int bestBlockerPower = 0;
        int bestBlockerToughness = 0;
        for (Card opp : oppUntapped) {
            if (CombatUtil.canBlock(card, opp)) {
                if (opp.getNetPower() > bestBlockerPower) {
                    bestBlockerPower = opp.getNetPower();
                }
                if (opp.getNetToughness() > bestBlockerToughness) {
                    bestBlockerToughness = opp.getNetToughness();
                }
            }
        }
        features[228] = normalize(bestBlockerPower, 0, 20);

        // [229] best_blocker_toughness: norm/20
        features[229] = normalize(bestBlockerToughness, 0, 20);

        // [230] n_profitable_blocks: if untapped, how many opp attackers this can profitably block, norm/10
        if (!card.isTapped()) {
            int profitableBlocks = 0;
            for (Card opp : oppCreatures) {
                if (CombatUtil.canBlock(opp, card) && canKillInCombat(card, opp)) {
                    profitableBlocks++;
                }
            }
            features[230] = normalize(profitableBlocks, 0, 10);
        }

        // [231] power_surplus: (power - best_blocker_toughness) / 20, clamp 0-1
        features[231] = normalize((double)(power - bestBlockerToughness) / 20.0, 0, 1);
    }

    /**
     * Compute global combat features and fill indices [76-95] of the global feature array.
     */
    public static void computeGlobalCombatFeatures(float[] global, List<Card> myCreatures, List<Card> oppCreatures, int myLife, int oppLife) {
        // Collect untapped opponent creatures for evasion checks
        List<Card> oppUntapped = new ArrayList<>();
        for (Card c : oppCreatures) {
            if (!c.isTapped()) {
                oppUntapped.add(c);
            }
        }
        List<Card> myUntapped = new ArrayList<>();
        for (Card c : myCreatures) {
            if (!c.isTapped()) {
                myUntapped.add(c);
            }
        }

        int myAttackablePower = 0;
        int myEvasivePower = 0;
        int mySafeAttackPower = 0;
        int myTotalPower = 0;
        int myTotalToughness = 0;
        int myFirstStrikers = 0;
        int myDeathtouchers = 0;

        for (Card c : myCreatures) {
            int power = c.getNetPower();
            myTotalPower += power;
            myTotalToughness += c.getNetToughness();

            boolean canAttack = !c.isTapped() && !c.hasSickness()
                    && !c.hasKeyword(Keyword.DEFENDER) && c.isCreature();
            if (canAttack) {
                myAttackablePower += power;
                if (isEvasive(c, oppUntapped)) {
                    myEvasivePower += power;
                }
                // Safe attacker: evasive or no opp can profitably block
                boolean safe = isEvasive(c, oppUntapped);
                if (!safe) {
                    boolean anyProfitable = false;
                    for (Card opp : oppUntapped) {
                        if (CombatUtil.canBlock(c, opp) && canKillInCombat(opp, c)) {
                            anyProfitable = true;
                            break;
                        }
                    }
                    safe = !anyProfitable;
                }
                if (safe) {
                    mySafeAttackPower += power;
                }
            }

            if (c.hasKeyword(Keyword.FIRST_STRIKE) || c.hasKeyword(Keyword.DOUBLE_STRIKE)) {
                myFirstStrikers++;
            }
            if (c.hasKeyword(Keyword.DEATHTOUCH)) {
                myDeathtouchers++;
            }
        }

        int oppAttackablePower = 0;
        int oppEvasivePower = 0;
        int oppSafeAttackPower = 0;
        int oppTotalPower = 0;
        int oppTotalToughness = 0;
        int oppFirstStrikers = 0;
        int oppDeathtouchers = 0;

        for (Card c : oppCreatures) {
            int power = c.getNetPower();
            oppTotalPower += power;
            oppTotalToughness += c.getNetToughness();

            boolean canAttack = !c.isTapped() && !c.hasSickness()
                    && !c.hasKeyword(Keyword.DEFENDER) && c.isCreature();
            if (canAttack) {
                oppAttackablePower += power;
                if (isEvasive(c, myUntapped)) {
                    oppEvasivePower += power;
                }
                boolean safe = isEvasive(c, myUntapped);
                if (!safe) {
                    boolean anyProfitable = false;
                    for (Card my : myUntapped) {
                        if (CombatUtil.canBlock(c, my) && canKillInCombat(my, c)) {
                            anyProfitable = true;
                            break;
                        }
                    }
                    safe = !anyProfitable;
                }
                if (safe) {
                    oppSafeAttackPower += power;
                }
            }

            if (c.hasKeyword(Keyword.FIRST_STRIKE) || c.hasKeyword(Keyword.DOUBLE_STRIKE)) {
                oppFirstStrikers++;
            }
            if (c.hasKeyword(Keyword.DEATHTOUCH)) {
                oppDeathtouchers++;
            }
        }

        // [76-79] attackable/evasive power
        global[76] = normalize(myAttackablePower, 0, 60);
        global[77] = normalize(myEvasivePower, 0, 40);
        global[78] = normalize(oppAttackablePower, 0, 60);
        global[79] = normalize(oppEvasivePower, 0, 40);

        // [80-83] lethal checks
        global[80] = myAttackablePower >= oppLife ? 1f : 0f;
        global[81] = oppAttackablePower >= myLife ? 1f : 0f;
        global[82] = myEvasivePower >= oppLife ? 1f : 0f;
        global[83] = oppEvasivePower >= myLife ? 1f : 0f;

        // [84] my_safe_attack_power
        global[84] = normalize(mySafeAttackPower, 0, 40);

        // [85-87] board advantages
        global[85] = normalize(myTotalPower - oppTotalPower + 60, 0, 120);
        global[86] = normalize(myTotalToughness - oppTotalToughness + 60, 0, 120);
        global[87] = normalize(myCreatures.size() - oppCreatures.size() + 20, 0, 40);

        // [88-91] first strikers and deathtouchers
        global[88] = normalize(myFirstStrikers, 0, 10);
        global[89] = normalize(oppFirstStrikers, 0, 10);
        global[90] = normalize(myDeathtouchers, 0, 10);
        global[91] = normalize(oppDeathtouchers, 0, 10);

        // [92] alpha_strike_kills: estimated opp kills on alpha strike
        // Simple estimate: for each attacker, if it can kill any opp creature, count it
        int estimatedKills = 0;
        // More practically: count how many opp creatures would die to attackable creatures
        // Simple model: sort attackers by power, assign to blockers
        // For simplicity, count opp creatures whose toughness <= some attacker's power
        for (Card opp : oppCreatures) {
            for (Card my : myCreatures) {
                boolean canAttk = !my.isTapped() && !my.hasSickness()
                        && !my.hasKeyword(Keyword.DEFENDER);
                if (canAttk && canKillInCombat(my, opp)) {
                    estimatedKills++;
                    break;
                }
            }
        }
        global[92] = normalize(estimatedKills, 0, 20);

        // [93] turns_to_lethal
        int safeP = Math.max(mySafeAttackPower, 1);
        int turnsToLethal = Math.min((int) Math.ceil((double) oppLife / safeP), 10);
        global[93] = normalize(turnsToLethal, 0, 10);

        // [94] opp_turns_to_lethal
        int oppEP = Math.max(oppEvasivePower, 1);
        int oppTurnsToLethal = Math.min((int) Math.ceil((double) myLife / oppEP), 10);
        global[94] = normalize(oppTurnsToLethal, 0, 10);

        // [95] combat_dominance
        double myStrength = mySafeAttackPower + myEvasivePower;
        double oppStrength = oppSafeAttackPower + oppEvasivePower + 1;
        global[95] = normalize(myStrength / oppStrength, 0, 1);
    }

    /**
     * Check if an attacker is evasive — no untapped opponent creature can block it.
     */
    static boolean isEvasive(Card attacker, List<Card> oppUntapped) {
        if (oppUntapped.isEmpty()) return true;
        for (Card blocker : oppUntapped) {
            if (CombatUtil.canBlock(attacker, blocker)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if card 'a' can kill card 'b' in combat (a's power >= b's toughness, or deathtouch).
     */
    static boolean canKillInCombat(Card a, Card b) {
        if (a.hasKeyword(Keyword.DEATHTOUCH) && a.getNetPower() > 0) {
            return true;
        }
        return a.getNetPower() >= b.getNetToughness();
    }

    /**
     * Enrich a list of candidate card features with combat math.
     * Call after CardFeatures.encode() for each candidate.
     *
     * @param candidateFeatures  list of 256-float feature arrays
     * @param candidates         the corresponding Card objects
     * @param perspectivePlayer  the player making the decision
     */
    public static void enrichCandidates(List<float[]> candidateFeatures,
                                         List<Card> candidates,
                                         forge.game.player.Player perspectivePlayer) {
        if (perspectivePlayer == null || candidates.isEmpty()) return;

        // Collect creatures on both sides
        List<Card> myCreatures = new ArrayList<>();
        List<Card> oppCreatures = new ArrayList<>();
        for (Card c : perspectivePlayer.getCardsIn(forge.game.zone.ZoneType.Battlefield)) {
            if (c.isCreature()) myCreatures.add(c);
        }
        forge.game.player.Player opp = perspectivePlayer.getWeakestOpponent();
        if (opp != null) {
            for (Card c : opp.getCardsIn(forge.game.zone.ZoneType.Battlefield)) {
                if (c.isCreature()) oppCreatures.add(c);
            }
        }

        for (int i = 0; i < candidates.size(); i++) {
            Card card = candidates.get(i);
            float[] features = candidateFeatures.get(i);
            if (card.isCreature() && features.length > 231) {
                boolean isMine = card.getController() == perspectivePlayer;
                if (isMine) {
                    injectPerCardFeatures(features, card, oppCreatures, myCreatures);
                } else {
                    injectPerCardFeatures(features, card, myCreatures, oppCreatures);
                }
            }
        }
    }

    /**
     * Normalize a value to [0, 1] range.
     */
    static float normalize(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }
}
