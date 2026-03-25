package forge.ai.rl.features;

import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.keyword.Keyword;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

/**
 * Extracts a 256-dimension feature vector from a Card object.
 * Captures the card's current in-game state including what its abilities DO.
 *
 * Feature vector layout (256 floats):
 *
 * === BASIC CARD INFO [0-68] ===
 * [0-6]    : card type flags (creature, instant, sorcery, enchantment, artifact, planeswalker, land)
 * [7-12]   : color identity (W, U, B, R, G, colorless)
 * [13]     : converted mana cost (normalized)
 * [14-15]  : power/toughness (normalized)
 * [16]     : loyalty (normalized)
 * [17-21]  : state flags (tapped, summoning sick, attacking, blocking, face down)
 * [22-26]  : counters (+1/+1, -1/-1, loyalty, charge, other)
 * [27]     : number of attachments (normalized)
 * [28]     : damage marked (normalized)
 * [29-58]  : keyword flags (30 common keywords — set 1)
 * [59-68]  : zone encoding (one-hot)
 *
 * === PRIMARY ABILITY [69-108] ===
 * [69-98]  : ApiType one-hot for primary spell/ability (30 types)
 * [99-102] : ability summary (has_activated, has_triggered, has_mana_ability, n_abilities)
 * [103-106]: effect magnitude (est_damage, est_draw, est_life, est_tokens)
 * [107-108]: targeting (requires_target, targets_creatures)
 *
 * === SECOND ABILITY [109-138] ===
 * [109-138]: ApiType one-hot for second spell/ability (30 types)
 *
 * === EXTENDED KEYWORDS [139-168] ===
 * [139-168]: keyword flags (30 more keywords — set 2)
 *
 * === MANA + SPEED + TRIGGERS [169-199] ===
 * [169-173]: mana production (produces W, U, B, R, G)
 * [174-177]: spell speed (is_instant_speed, has_flash, is_modal, has_kicker)
 * [178-181]: trigger summary (has_etb, has_death, has_combat, has_upkeep)
 * [182-185]: cost info (mana_cost_W, mana_cost_U, mana_cost_B, mana_cost_R)
 * [186-189]: cost info (mana_cost_G, mana_cost_generic, mana_cost_total, has_X)
 * [190-191]: ownership flags (set by encode(Card, Player))
 *
 * === TRIGGER DETAILS [192-199] ===
 * [192-193]: ETB trigger (api_type_code, magnitude)
 * [194-195]: death trigger (api_type_code, magnitude)
 * [196-197]: combat trigger (api_type_code, magnitude)
 * [198-199]: other trigger (api_type_code, magnitude)
 *
 * === PUMP MAGNITUDE [200-201] ===
 * [200]    : pump attack boost (NumAtt, normalized)
 * [201]    : pump defense boost (NumDef, normalized)
 *
 * === AURA/EQUIPMENT HOST [202-207] ===
 * [202]    : is_attached flag
 * [203]    : host power (normalized)
 * [204]    : host toughness (normalized)
 * [205]    : host is creature
 * [206]    : host controlled by perspective player
 * [207]    : host CMC (normalized)
 *
 * === COMBAT MATH [208-231] (battlefield creatures only) ===
 * [208]    : can_attack (!tapped && !sickness && !DEFENDER)
 * [209]    : is_evasive (no opp untapped creature can block)
 * [210]    : frac_can_block_this (fraction of opp untapped that can block)
 * [211]    : frac_this_kills (fraction of opp creatures this kills in combat)
 * [212]    : frac_kills_this (fraction of opp creatures that kill this)
 * [213]    : lethal_damage_remaining ((toughness - damage) / toughness)
 * [214]    : power_vs_avg_toughness (power / avg opp toughness)
 * [215]    : toughness_vs_avg_power (toughness / avg opp power)
 * [216]    : has_first_strike_advantage (FS/DS with no opp blocker having it)
 * [217]    : has_deathtouch
 * [218]    : has_indestructible
 * [219]    : has_lifelink
 * [220]    : has_trample
 * [221]    : can_trade_up (kill higher-CMC opp creature)
 * [222]    : can_trade_even (mutual kill with same-CMC opp creature)
 * [223]    : is_biggest_creature (highest power on both boards)
 * [224]    : power_rank_my_board (rank / (count-1), 1.0=highest)
 * [225]    : toughness_rank_my_board (rank / (count-1), 1.0=highest)
 * [226]    : safe_attacker (evasive or no profitable block)
 * [227]    : must_be_double_blocked (MENACE or power > max blocker toughness)
 * [228]    : best_blocker_power (norm/20)
 * [229]    : best_blocker_toughness (norm/20)
 * [230]    : n_profitable_blocks (norm/10)
 * [231]    : power_surplus ((power - best_blocker_toughness) / 20)
 * [232]    : needs_gang_block (toughness > every blocker's power, can't be killed 1v1)
 *
 * === RESERVED + HASH [233-255] ===
 * [233-251]: reserved for future
 * [252-255]: card identity hash (4 bytes, normalized)
 */
public class CardFeatures {

    public static final int FEATURE_SIZE = 256;

    // Common keywords — set 1 (30)
    private static final Keyword[] KEYWORDS_SET1 = {
        Keyword.FLYING, Keyword.FIRST_STRIKE, Keyword.DOUBLE_STRIKE,
        Keyword.TRAMPLE, Keyword.HASTE, Keyword.VIGILANCE,
        Keyword.DEATHTOUCH, Keyword.LIFELINK, Keyword.REACH,
        Keyword.MENACE, Keyword.HEXPROOF, Keyword.SHROUD,
        Keyword.INDESTRUCTIBLE, Keyword.FLASH, Keyword.DEFENDER,
        Keyword.FEAR, Keyword.WARD, Keyword.PROWESS,
        Keyword.WITHER, Keyword.INFECT, Keyword.PROTECTION,
        Keyword.SHADOW, Keyword.UNDYING, Keyword.PERSIST,
        Keyword.CONVOKE, Keyword.DELVE, Keyword.CASCADE,
        Keyword.EQUIP, Keyword.ENCHANT, Keyword.FLANKING
    };

    // Extended keywords — set 2 (30)
    private static final Keyword[] KEYWORDS_SET2 = {
        Keyword.HORSEMANSHIP, Keyword.INTIMIDATE, Keyword.SKULK,
        Keyword.ANNIHILATOR, Keyword.ABSORB, Keyword.BUSHIDO,
        Keyword.EXALTED, Keyword.BATTLE_CRY, Keyword.MODULAR,
        Keyword.TOXIC, Keyword.AFFLICT, Keyword.PHASING,
        Keyword.CUMULATIVE_UPKEEP, Keyword.ECHO, Keyword.FADING,
        Keyword.VANISHING, Keyword.STORM, Keyword.AFFINITY,
        Keyword.CHANGELING, Keyword.DEVOID, Keyword.EMERGE,
        Keyword.IMPROVISE, Keyword.SPECTACLE, Keyword.RIOT,
        Keyword.COMPANION, Keyword.FORETELL, Keyword.ENTWINE,
        Keyword.DISTURB, Keyword.DAYBOUND, Keyword.NIGHTBOUND
    };

    // Top 30 ApiTypes — same as ActionEncoder
    private static final ApiType[] TOP_API_TYPES = {
        ApiType.DealDamage, ApiType.Draw, ApiType.Counter, ApiType.ChangeZone,
        ApiType.Pump, ApiType.PumpAll, ApiType.Destroy, ApiType.DestroyAll,
        ApiType.Sacrifice, ApiType.Discard, ApiType.GainLife, ApiType.LoseLife,
        ApiType.Token, ApiType.Animate, ApiType.Attach, ApiType.Tap,
        ApiType.Untap, ApiType.Mill, ApiType.Regenerate, ApiType.Protection,
        ApiType.Fight, ApiType.Charm, ApiType.Scry, ApiType.Explore,
        ApiType.AddOrRemoveCounter, ApiType.ManaReflected, ApiType.Mana,
        ApiType.ChangeTargets, ApiType.Fog, ApiType.RearrangeTopOfLibrary
    };

    public static float[] encode(Card card) {
        return encode(card, null);
    }

    /**
     * Encode a card with ownership context and automatic combat math enrichment.
     * When perspective is non-null and the card is a creature, combat features [208-231]
     * are automatically computed — no separate enrichment step needed.
     * @param perspective the player whose perspective we're encoding from (null = no ownership/combat info)
     */
    public static float[] encode(Card card, forge.game.player.Player perspective) {
        float[] features = new float[FEATURE_SIZE];
        if (card == null) return features;
        try {
            encodeImpl(card, features, perspective);
            // Ownership flag at index 190
            if (perspective != null) {
                features[190] = (card.getController() == perspective) ? 1f : 0f;
                features[191] = (card.getController() != perspective) ? 1f : 0f;
                // Combat math features [208-231] — automatically injected for creatures
                CombatMath.injectPerCardFeatures(features, card, perspective);
            }
            return features;
        } catch (Exception e) {
            return features;
        }
    }

    private static float[] encodeImpl(Card card, float[] features, forge.game.player.Player perspective) {

        // === BASIC CARD INFO [0-68] ===

        int idx = 0;

        // Card types [0-6]
        features[idx++] = card.isCreature() ? 1f : 0f;
        features[idx++] = card.isInstant() ? 1f : 0f;
        features[idx++] = card.isSorcery() ? 1f : 0f;
        features[idx++] = card.isEnchantment() ? 1f : 0f;
        features[idx++] = card.isArtifact() ? 1f : 0f;
        features[idx++] = card.isPlaneswalker() ? 1f : 0f;
        features[idx++] = card.isLand() ? 1f : 0f;

        // Color identity [7-12]
        features[idx++] = card.getColor().hasWhite() ? 1f : 0f;
        features[idx++] = card.getColor().hasBlue() ? 1f : 0f;
        features[idx++] = card.getColor().hasBlack() ? 1f : 0f;
        features[idx++] = card.getColor().hasRed() ? 1f : 0f;
        features[idx++] = card.getColor().hasGreen() ? 1f : 0f;
        features[idx++] = card.getColor().isColorless() ? 1f : 0f;

        // CMC [13]
        features[idx++] = normalizeValue(card.getCMC(), 0, 16);

        // Power/Toughness [14-15]
        if (card.isCreature()) {
            features[idx++] = normalizeValue(card.getNetPower(), -5, 20);
            features[idx++] = normalizeValue(card.getNetToughness(), -5, 20);
        } else {
            idx += 2;
        }

        // Loyalty [16]
        if (card.isPlaneswalker()) {
            features[idx++] = normalizeValue(card.getCurrentLoyalty(), 0, 10);
        } else {
            idx++;
        }

        // In-game state [17-21]
        features[idx++] = card.isTapped() ? 1f : 0f;
        features[idx++] = card.hasSickness() ? 1f : 0f;
        boolean attacking = false;
        try { attacking = card.isAttacking(); } catch (Exception ignored) {}
        features[idx++] = attacking ? 1f : 0f;
        features[idx++] = 0f; // blocking — populated externally
        features[idx++] = card.isFaceDown() ? 1f : 0f;

        // Counters [22-26]
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.P1P1), 0, 20);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.M1M1), 0, 10);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.LOYALTY), 0, 10);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.CHARGE), 0, 10);
        int otherCounters = 0;
        for (var entry : card.getCounters().entrySet()) {
            CounterEnumType type = entry.getKey() instanceof CounterEnumType ? (CounterEnumType) entry.getKey() : null;
            if (type != null && type != CounterEnumType.P1P1 && type != CounterEnumType.M1M1
                    && type != CounterEnumType.LOYALTY && type != CounterEnumType.CHARGE) {
                otherCounters += entry.getValue();
            }
        }
        features[idx++] = normalizeValue(otherCounters, 0, 10);

        // Attachments [27]
        features[idx++] = normalizeValue(card.getAttachedCards().size(), 0, 5);

        // Damage marked [28]
        features[idx++] = normalizeValue(card.getDamage(), 0, 20);

        // Keywords set 1 [29-58]
        for (Keyword kw : KEYWORDS_SET1) {
            features[idx++] = card.hasKeyword(kw) ? 1f : 0f;
        }

        // Zone encoding [59-68]
        ZoneType zone = card.getZone() != null ? card.getZone().getZoneType() : null;
        ZoneType[] zones = {ZoneType.Battlefield, ZoneType.Hand, ZoneType.Library,
                ZoneType.Graveyard, ZoneType.Exile, ZoneType.Stack,
                ZoneType.Command, ZoneType.Sideboard, ZoneType.Ante, ZoneType.PlanarDeck};
        for (ZoneType z : zones) {
            features[idx++] = (zone == z) ? 1f : 0f;
        }
        // idx = 69

        // === PRIMARY ABILITY [69-108] ===

        // Find primary non-mana spell ability
        SpellAbility primarySA = null;
        SpellAbility secondSA = null;
        int saCount = 0;
        boolean hasActivated = false;
        boolean hasTriggered = false;
        boolean hasManaAbility = !card.getManaAbilities().isEmpty();

        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.isManaAbility()) continue;
            saCount++;
            if (sa.isActivatedAbility()) hasActivated = true;
            if (sa.isTrigger()) hasTriggered = true;
            if (primarySA == null) {
                primarySA = sa;
            } else if (secondSA == null) {
                secondSA = sa;
            }
        }

        // Primary ApiType one-hot [69-98]
        if (primarySA != null && primarySA.getApi() != null) {
            ApiType api = primarySA.getApi();
            for (ApiType at : TOP_API_TYPES) {
                features[idx++] = (api == at) ? 1f : 0f;
            }
        } else {
            idx += 30;
        }
        // idx = 99

        // Ability summary [99-102]
        features[idx++] = hasActivated ? 1f : 0f;
        features[idx++] = hasTriggered ? 1f : 0f;
        features[idx++] = hasManaAbility ? 1f : 0f;
        features[idx++] = normalizeValue(saCount, 0, 10);
        // idx = 103

        // Effect magnitude from primary ability [103-106]
        if (primarySA != null) {
            // Estimated damage
            if (primarySA.hasParam("NumDmg")) {
                try {
                    features[idx] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("NumDmg")), 0, 20);
                } catch (NumberFormatException e) {
                    features[idx] = 0.3f;
                }
            }
            idx++;
            // Estimated cards drawn
            if (primarySA.hasParam("NumCards")) {
                try {
                    features[idx] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("NumCards")), 0, 10);
                } catch (NumberFormatException e) {
                    features[idx] = 0.2f;
                }
            }
            idx++;
            // Life gain
            if (primarySA.hasParam("LifeAmount")) {
                try {
                    features[idx] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("LifeAmount")), 0, 20);
                } catch (NumberFormatException e) {
                    features[idx] = 0.2f;
                }
            }
            idx++;
            // Token count
            if (primarySA.hasParam("TokenAmount")) {
                try {
                    features[idx] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("TokenAmount")), 0, 5);
                } catch (NumberFormatException e) {
                    features[idx] = 0.2f;
                }
            }
            idx++;
        } else {
            idx += 4;
        }
        // idx = 107

        // Targeting [107-108]
        if (primarySA != null && primarySA.usesTargeting()) {
            features[idx++] = 1f;
            String validTgts = primarySA.getTargetRestrictions() != null
                    && primarySA.getTargetRestrictions().getValidTgts() != null
                    ? String.join(" ", primarySA.getTargetRestrictions().getValidTgts()) : "";
            features[idx++] = validTgts.contains("Creature") ? 1f : 0f;
        } else {
            idx += 2;
        }
        // idx = 109

        // === SECOND ABILITY [109-138] ===

        if (secondSA != null && secondSA.getApi() != null) {
            ApiType api = secondSA.getApi();
            for (ApiType at : TOP_API_TYPES) {
                features[idx++] = (api == at) ? 1f : 0f;
            }
        } else {
            idx += 30;
        }
        // idx = 139

        // === EXTENDED KEYWORDS [139-168] ===

        for (Keyword kw : KEYWORDS_SET2) {
            features[idx++] = card.hasKeyword(kw) ? 1f : 0f;
        }
        // idx = 169

        // === MANA + SPEED + TRIGGERS [169-199] ===

        // Mana production [169-173]
        for (SpellAbility mana : card.getManaAbilities()) {
            String produced = mana.getParam("Produced");
            if (produced != null) {
                if (produced.contains("W")) features[169] = 1f;
                if (produced.contains("U")) features[170] = 1f;
                if (produced.contains("B")) features[171] = 1f;
                if (produced.contains("R")) features[172] = 1f;
                if (produced.contains("G")) features[173] = 1f;
            }
        }
        idx = 174;

        // Spell speed [174-177]
        features[idx++] = card.isInstant() || card.hasKeyword(Keyword.FLASH) ? 1f : 0f;
        features[idx++] = card.hasKeyword(Keyword.FLASH) ? 1f : 0f;
        // Is modal (charm-like)
        boolean isModal = false;
        if (primarySA != null && primarySA.getApi() == ApiType.Charm) isModal = true;
        features[idx++] = isModal ? 1f : 0f;
        // Has kicker
        features[idx++] = card.hasKeyword(Keyword.KICKER) ? 1f : 0f;
        // idx = 178

        // Trigger summary [178-181] + trigger details [192-199]
        boolean hasETB = false, hasDeath = false, hasCombat = false, hasUpkeep = false;
        for (Trigger t : card.getTriggers()) {
            TriggerType mode = t.getMode();
            boolean isETB = false, isDeath = false, isCombat = false;
            if (mode == TriggerType.ChangesZone) {
                // ETB or LTB depending on destination
                String dest = t.getParam("Destination");
                if ("Battlefield".equals(dest)) { hasETB = true; isETB = true; }
                String origin = t.getParam("Origin");
                if ("Battlefield".equals(origin)) { hasDeath = true; isDeath = true; }
            }
            if (mode == TriggerType.Attacks || mode == TriggerType.AttackersDeclared
                    || mode == TriggerType.Blocks || mode == TriggerType.DamageDone) {
                hasCombat = true;
                isCombat = true;
            }
            if (mode == TriggerType.Phase) {
                String phase = t.getParam("Phase");
                if ("Upkeep".equals(phase)) hasUpkeep = true;
            }

            // Extract trigger ability details [192-199]
            SpellAbility trigSA = t.getOverridingAbility();
            if (trigSA != null && trigSA.getApi() != null) {
                ApiType trigApi = trigSA.getApi();
                float apiCode = (trigApi == ApiType.DealDamage) ? 0.2f
                        : (trigApi == ApiType.Draw) ? 0.4f
                        : (trigApi == ApiType.Pump) ? 0.6f
                        : (trigApi == ApiType.PumpAll) ? 0.8f
                        : 1.0f;
                float magnitude = 0f;
                if (trigSA.hasParam("NumDmg")) {
                    try { magnitude = normalizeValue(Integer.parseInt(trigSA.getParam("NumDmg")), 0, 20); }
                    catch (NumberFormatException e) { magnitude = 0.3f; }
                } else if (trigSA.hasParam("NumCards")) {
                    try { magnitude = normalizeValue(Integer.parseInt(trigSA.getParam("NumCards")), 0, 20); }
                    catch (NumberFormatException e) { magnitude = 0.3f; }
                } else if (trigSA.hasParam("NumAtt")) {
                    try { magnitude = normalizeValue(Integer.parseInt(trigSA.getParam("NumAtt")), 0, 20); }
                    catch (NumberFormatException e) { magnitude = 0.3f; }
                }

                int slotBase;
                if (isETB) { slotBase = 192; }
                else if (isDeath) { slotBase = 194; }
                else if (isCombat) { slotBase = 196; }
                else { slotBase = 198; }
                features[slotBase] = apiCode;
                features[slotBase + 1] = magnitude;
            }
        }
        features[idx++] = hasETB ? 1f : 0f;
        features[idx++] = hasDeath ? 1f : 0f;
        features[idx++] = hasCombat ? 1f : 0f;
        features[idx++] = hasUpkeep ? 1f : 0f;
        // idx = 182

        // Mana cost breakdown [182-189]
        if (card.getManaCost() != null) {
            features[idx++] = normalizeValue(card.getManaCost().getShardCount(forge.card.mana.ManaCostShard.WHITE), 0, 5);
            features[idx++] = normalizeValue(card.getManaCost().getShardCount(forge.card.mana.ManaCostShard.BLUE), 0, 5);
            features[idx++] = normalizeValue(card.getManaCost().getShardCount(forge.card.mana.ManaCostShard.BLACK), 0, 5);
            features[idx++] = normalizeValue(card.getManaCost().getShardCount(forge.card.mana.ManaCostShard.RED), 0, 5);
            features[idx++] = normalizeValue(card.getManaCost().getShardCount(forge.card.mana.ManaCostShard.GREEN), 0, 5);
            features[idx++] = normalizeValue(card.getManaCost().getGenericCost(), 0, 10);
            features[idx++] = normalizeValue(card.getCMC(), 0, 16);
            features[idx++] = card.getManaCost().countX() > 0 ? 1f : 0f;
        } else {
            idx += 8;
        }
        // idx = 190

        // [190-191] ownership flags set by encode(Card, Player)
        // [192-199] trigger details set in trigger loop above

        // Pump magnitude [200-201]
        if (primarySA != null && primarySA.getApi() == ApiType.Pump) {
            if (primarySA.hasParam("NumAtt")) {
                try {
                    features[200] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("NumAtt")), 0, 20);
                } catch (NumberFormatException e) {
                    features[200] = 0.3f;
                }
            }
            if (primarySA.hasParam("NumDef")) {
                try {
                    features[201] = normalizeValue(
                            Integer.parseInt(primarySA.getParam("NumDef")), 0, 20);
                } catch (NumberFormatException e) {
                    features[201] = 0.3f;
                }
            }
        }

        // Aura/Equipment host info [202-207]
        if (card.getEntityAttachedTo() instanceof Card) {
            Card host = (Card) card.getEntityAttachedTo();
            features[202] = 1f; // is_attached
            features[203] = normalizeValue(host.getNetPower(), -5, 20);
            features[204] = normalizeValue(host.getNetToughness(), -5, 20);
            features[205] = host.isCreature() ? 1f : 0f;
            if (perspective != null) {
                features[206] = (host.getController() == perspective) ? 1f : 0f;
            }
            features[207] = normalizeValue(host.getCMC(), 0, 16);
        }

        // Reserved [208-251]

        // === CARD HASH [252-255] ===
        if (card.getPaperCard() != null) {
            int hash = card.getPaperCard().getName().hashCode();
            features[252] = normalizeValue((hash & 0xFF), 0, 256);
            features[253] = normalizeValue(((hash >> 8) & 0xFF), 0, 256);
            features[254] = normalizeValue(((hash >> 16) & 0xFF), 0, 256);
            features[255] = normalizeValue(((hash >> 24) & 0xFF), 0, 256);
        }

        return features;
    }

    private static float normalizeValue(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }
}
