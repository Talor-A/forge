package forge.ai.rl.features;

import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.game.GameEntity;

/**
 * Encodes available actions (SpellAbilities) as feature vectors for the RL model.
 * Used by the priority action decision head to choose which spell/ability to play.
 */
public class ActionEncoder {

    public static final int ACTION_FEATURE_SIZE = 64;

    /**
     * Encode a SpellAbility as a feature vector.
     *
     * Layout (64 floats):
     * [0-6]   : source card type flags
     * [7-12]  : source card color flags
     * [13]    : mana cost (CMC, normalized)
     * [14]    : is a spell (not ability)
     * [15]    : is activated ability
     * [16]    : is triggered ability
     * [17]    : is mana ability
     * [18-47] : ApiType one-hot (top 30 most common)
     * [48]    : requires target
     * [49]    : number of targets required (normalized)
     * [50]    : can target creatures
     * [51]    : can target players
     * [52]    : source card power (if creature, normalized)
     * [53]    : source card toughness (if creature, normalized)
     * [54]    : estimated damage (if damage spell, normalized)
     * [55]    : estimated cards drawn (if draw spell, normalized)
     * [56]    : can target own creature (polarity)
     * [57]    : can target opp creature (polarity)
     * [58]    : can target players
     * [59]    : can target players (duplicate for symmetry)
     * [60]    : creature_would_be_biggest (power > all creatures on board)
     * [61]    : creature_would_survive (toughness > opp max creature power)
     * [62]    : spell_is_combat_trick (Pump/PumpAll/Protection/Regenerate AND instant)
     * [63]    : removal_kills_biggest (DealDamage/Destroy kills opp biggest creature)
     */
    public static float[] encode(SpellAbility sa) {
        float[] features = new float[ACTION_FEATURE_SIZE];
        if (sa == null) return features;

        Card source = sa.getHostCard();
        int idx = 0;

        // Source card type [0-6]
        if (source != null) {
            features[idx++] = source.isCreature() ? 1f : 0f;
            features[idx++] = source.isInstant() ? 1f : 0f;
            features[idx++] = source.isSorcery() ? 1f : 0f;
            features[idx++] = source.isEnchantment() ? 1f : 0f;
            features[idx++] = source.isArtifact() ? 1f : 0f;
            features[idx++] = source.isPlaneswalker() ? 1f : 0f;
            features[idx++] = source.isLand() ? 1f : 0f;

            // Color [7-12]
            features[idx++] = source.getColor().hasWhite() ? 1f : 0f;
            features[idx++] = source.getColor().hasBlue() ? 1f : 0f;
            features[idx++] = source.getColor().hasBlack() ? 1f : 0f;
            features[idx++] = source.getColor().hasRed() ? 1f : 0f;
            features[idx++] = source.getColor().hasGreen() ? 1f : 0f;
            features[idx++] = source.getColor().isColorless() ? 1f : 0f;
        } else {
            idx += 13;
        }

        // CMC [13]
        features[idx++] = normalizeValue(sa.getPayCosts() != null ? sa.getPayCosts().getTotalMana().getCMC() : 0, 0, 16);

        // Spell vs ability type [14-17]
        features[idx++] = sa.isSpell() ? 1f : 0f;
        features[idx++] = sa.isActivatedAbility() ? 1f : 0f;
        features[idx++] = sa.isTrigger() ? 1f : 0f;
        features[idx++] = sa.isManaAbility() ? 1f : 0f;

        // ApiType one-hot [18-47] (top 30 most common types)
        ApiType[] topTypes = {
            ApiType.DealDamage, ApiType.Draw, ApiType.Counter, ApiType.ChangeZone,
            ApiType.Pump, ApiType.PumpAll, ApiType.Destroy, ApiType.DestroyAll,
            ApiType.Sacrifice, ApiType.Discard, ApiType.GainLife, ApiType.LoseLife,
            ApiType.Token, ApiType.Animate, ApiType.Attach, ApiType.Tap,
            ApiType.Untap, ApiType.Mill, ApiType.Regenerate, ApiType.Protection,
            ApiType.Fight, ApiType.Charm, ApiType.Scry, ApiType.Explore,
            ApiType.AddOrRemoveCounter, ApiType.ManaReflected, ApiType.Mana,
            ApiType.ChangeTargets, ApiType.Fog, ApiType.RearrangeTopOfLibrary
        };
        ApiType saType = sa.getApi();
        for (ApiType at : topTypes) {
            features[idx++] = (saType == at) ? 1f : 0f;
        }

        // Targeting info [48-51]
        boolean requiresTarget = sa.usesTargeting();
        features[idx++] = requiresTarget ? 1f : 0f;
        if (requiresTarget && sa.getTargetRestrictions() != null) {
            features[idx++] = normalizeValue(sa.getTargetRestrictions().getMinTargets(sa.getHostCard(), sa), 0, 5);
            String validTgts = sa.getTargetRestrictions().getValidTgts() != null ?
                    String.join(" ", sa.getTargetRestrictions().getValidTgts()) : "";
            // Fix for auras: enchant keyword sets target type differently
            if (source != null && source.isAura()) {
                boolean enchantCreature = source.hasKeyword("Enchant creature")
                        || source.hasKeyword("Enchant permanent");
                features[idx++] = enchantCreature ? 1f : 0f;
                features[idx++] = source.hasKeyword("Enchant player") ? 1f : 0f;
            } else {
                features[idx++] = validTgts.contains("Creature") ? 1f : 0f;
                features[idx++] = validTgts.contains("Player") ? 1f : 0f;
            }
        } else {
            idx += 3;
        }

        // Source card combat stats [52-53]
        if (source != null && source.isCreature()) {
            features[idx++] = normalizeValue(source.getNetPower(), -5, 20);
            features[idx++] = normalizeValue(source.getNetToughness(), -5, 20);
        } else {
            idx += 2;
        }

        // Estimated damage/draw from ability parameters [54-55]
        if (sa.hasParam("NumDmg")) {
            try {
                features[idx++] = normalizeValue(Integer.parseInt(sa.getParam("NumDmg")), 0, 20);
            } catch (NumberFormatException e) {
                features[idx++] = 0.3f; // variable damage, use moderate value
            }
        } else {
            idx++;
        }

        // Target polarity [56-59]
        if (requiresTarget && sa.getTargetRestrictions() != null) {
            String validTgts = sa.getTargetRestrictions().getValidTgts() != null ?
                    String.join(" ", sa.getTargetRestrictions().getValidTgts()) : "";
            boolean targetsCreatures = validTgts.contains("Creature");
            boolean hasOwnRestriction = validTgts.contains(".YouCtrl");
            boolean hasOppRestriction = validTgts.contains(".OppCtrl") || validTgts.contains(".YouDontCtrl");
            // If no controller restriction, spell can target either side
            features[56] = (targetsCreatures && !hasOppRestriction) ? 1f : 0f;  // can target own creature
            features[57] = (targetsCreatures && !hasOwnRestriction) ? 1f : 0f;  // can target opp creature
            boolean targetsPlayers = validTgts.contains("Player");
            features[58] = targetsPlayers ? 1f : 0f;  // can target players (either)
            features[59] = targetsPlayers ? 1f : 0f;  // same for now
        }

        if (sa.hasParam("NumCards")) {
            try {
                features[idx++] = normalizeValue(Integer.parseInt(sa.getParam("NumCards")), 0, 10);
            } catch (NumberFormatException e) {
                features[idx++] = 0.2f; // variable draw
            }
        }

        // Combat-relevant action features [60-63]
        if (source != null) {
            try {
                // [60] creature_would_be_biggest: if source isCreature, power > all creatures on board
                if (source.isCreature()) {
                    int power = source.getNetPower();
                    boolean biggest = true;
                    Player controller = source.getController();
                    if (controller != null && controller.getGame() != null) {
                        for (Player p : controller.getGame().getPlayers()) {
                            for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
                                if (c.isCreature() && c != source && c.getNetPower() >= power) {
                                    biggest = false;
                                    break;
                                }
                            }
                            if (!biggest) break;
                        }
                    }
                    features[60] = biggest ? 1f : 0f;
                }

                // [61] creature_would_survive: if source isCreature, toughness > opp max creature power
                if (source.isCreature()) {
                    int toughness = source.getNetToughness();
                    Player controller = source.getController();
                    Player opp = controller != null ? controller.getWeakestOpponent() : null;
                    if (opp != null) {
                        int oppMaxPower = 0;
                        for (Card c : opp.getCardsIn(ZoneType.Battlefield)) {
                            if (c.isCreature() && c.getNetPower() > oppMaxPower) {
                                oppMaxPower = c.getNetPower();
                            }
                        }
                        features[61] = toughness > oppMaxPower ? 1f : 0f;
                    }
                }

                // [62] spell_is_combat_trick: Pump/PumpAll/Protection/Regenerate AND isInstant
                if (saType != null && (saType == ApiType.Pump || saType == ApiType.PumpAll
                        || saType == ApiType.Protection || saType == ApiType.Regenerate)) {
                    if (source.isInstant() || source.hasKeyword(Keyword.FLASH)) {
                        features[62] = 1f;
                    }
                }

                // [63] removal_kills_biggest: if DealDamage/Destroy, estimated damage >= opp biggest creature toughness
                if (saType != null && (saType == ApiType.DealDamage || saType == ApiType.Destroy)) {
                    Player controller = source.getController();
                    Player opp = controller != null ? controller.getWeakestOpponent() : null;
                    if (opp != null) {
                        int oppMaxToughness = 0;
                        for (Card c : opp.getCardsIn(ZoneType.Battlefield)) {
                            if (c.isCreature() && c.getNetToughness() > oppMaxToughness) {
                                oppMaxToughness = c.getNetToughness();
                            }
                        }
                        if (saType == ApiType.Destroy) {
                            // Destroy always kills (unless indestructible, but we estimate)
                            features[63] = oppMaxToughness > 0 ? 1f : 0f;
                        } else if (sa.hasParam("NumDmg")) {
                            try {
                                int dmg = Integer.parseInt(sa.getParam("NumDmg"));
                                features[63] = dmg >= oppMaxToughness && oppMaxToughness > 0 ? 1f : 0f;
                            } catch (NumberFormatException e) {
                                features[63] = 0.5f; // variable damage, uncertain
                            }
                        }
                    }
                }
            } catch (Exception ignored) {
                // Guard against NPEs from game state access during unusual phases
            }
        }

        return features;
    }

    /**
     * Encode a game entity (card or player) as a target candidate feature vector.
     */
    public static float[] encodeTarget(GameEntity entity) {
        float[] features = new float[ACTION_FEATURE_SIZE];
        if (entity instanceof Card) {
            // Reuse card features for the first portion
            float[] cardFeats = CardFeatures.encode((Card) entity);
            System.arraycopy(cardFeats, 0, features, 0, Math.min(cardFeats.length, features.length));
        } else if (entity instanceof Player) {
            Player p = (Player) entity;
            features[0] = normalizeValue(p.getLife(), -10, 40);
            features[1] = normalizeValue(p.getPoisonCounters(), 0, 10);
            features[2] = normalizeValue(p.getCardsIn(forge.game.zone.ZoneType.Hand).size(), 0, 15);
            features[3] = normalizeValue(p.getCreaturesInPlay().size(), 0, 20);
            features[4] = 1f; // flag: this is a player, not a card
        }
        return features;
    }

    /**
     * Encode the "pass priority" action as a special feature vector.
     */
    public static float[] encodePassAction() {
        float[] features = new float[ACTION_FEATURE_SIZE];
        features[ACTION_FEATURE_SIZE - 1] = 1f; // flag: this is the pass action
        return features;
    }

    private static float normalizeValue(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }
}
