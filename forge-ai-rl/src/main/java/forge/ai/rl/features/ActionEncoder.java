package forge.ai.rl.features;

import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
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
     * [56-63] : reserved
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
            ApiType.ChangeTargets, ApiType.Fog, ApiType.ChangeZone
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
            features[idx++] = validTgts.contains("Creature") ? 1f : 0f;
            features[idx++] = validTgts.contains("Player") ? 1f : 0f;
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

        if (sa.hasParam("NumCards")) {
            try {
                features[idx++] = normalizeValue(Integer.parseInt(sa.getParam("NumCards")), 0, 10);
            } catch (NumberFormatException e) {
                features[idx++] = 0.2f; // variable draw
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
