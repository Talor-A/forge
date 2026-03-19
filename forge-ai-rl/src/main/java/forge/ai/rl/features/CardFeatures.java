package forge.ai.rl.features;

import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.keyword.Keyword;
import forge.game.zone.ZoneType;

/**
 * Extracts a fixed-size feature vector from a Card object.
 * This captures the card's current in-game state, not just its printed attributes.
 *
 * Feature vector layout (128 floats):
 * [0-4]   : card type flags (creature, instant, sorcery, enchantment, artifact, planeswalker, land)
 * [5-10]  : color identity (W, U, B, R, G, colorless)
 * [11]    : converted mana cost (normalized)
 * [12]    : power (normalized, 0 if not creature)
 * [13]    : toughness (normalized, 0 if not creature)
 * [14]    : loyalty (normalized, 0 if not planeswalker)
 * [15]    : tapped (0 or 1)
 * [16]    : summoning sick (0 or 1)
 * [17]    : attacking (0 or 1)
 * [18]    : blocking (0 or 1)
 * [19]    : face down (0 or 1)
 * [20-24] : +1/+1 counters, -1/-1 counters, loyalty counters, charge counters, other counters (normalized)
 * [25]    : number of attachments (normalized)
 * [26]    : damage marked (normalized)
 * [27-56] : keyword flags (30 common keywords)
 * [57-66] : zone encoding (one-hot)
 * [67-96] : ability type flags (up to 30 ApiTypes present on card)
 * [97-127]: reserved for learned card embedding lookup index and padding
 */
public class CardFeatures {

    public static final int FEATURE_SIZE = 128;

    // Common keywords to encode as binary features
    private static final Keyword[] TRACKED_KEYWORDS = {
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

    /**
     * Extract feature vector from a card in play or in hand.
     */
    public static float[] encode(Card card) {
        float[] features = new float[FEATURE_SIZE];
        if (card == null) return features;

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
        // isAttacking/isBlocking can NPE if game.getCombat()
        // is null (post-game). Guard with try-catch.
        try {
            features[idx++] = card.isAttacking() ? 1f : 0f;
        } catch (NullPointerException e) {
            features[idx++] = 0f;
        }
        features[idx++] = 0f; // blocking — needs combat context
        features[idx++] = card.isFaceDown() ? 1f : 0f;

        // Counters [22-26]
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.P1P1), 0, 20);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.M1M1), 0, 10);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.LOYALTY), 0, 10);
        features[idx++] = normalizeValue(card.getCounters(CounterEnumType.CHARGE), 0, 10);
        // Sum of all other counter types
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

        // Keywords [29-58]
        for (Keyword kw : TRACKED_KEYWORDS) {
            features[idx++] = card.hasKeyword(kw) ? 1f : 0f;
        }

        // Zone encoding [59-68] (one-hot)
        ZoneType zone = card.getZone() != null ? card.getZone().getZoneType() : null;
        ZoneType[] zones = {ZoneType.Battlefield, ZoneType.Hand, ZoneType.Library,
                ZoneType.Graveyard, ZoneType.Exile, ZoneType.Stack,
                ZoneType.Command, ZoneType.Sideboard, ZoneType.Ante, ZoneType.PlanarDeck};
        for (ZoneType z : zones) {
            features[idx++] = (zone == z) ? 1f : 0f;
        }

        // Remaining indices [69-127] reserved for ability type encoding and card embedding ID
        // Ability types will be populated by ActionEncoder which has access to SpellAbility info
        // The last 4 floats store a card name hash for the learned embedding lookup
        if (card.getPaperCard() != null) {
            int hash = card.getPaperCard().getName().hashCode();
            // Encode hash as normalized floats (never NaN/Inf)
            features[FEATURE_SIZE - 4] = normalizeValue(
                    (hash & 0xFF), 0, 256);
            features[FEATURE_SIZE - 3] = normalizeValue(
                    ((hash >> 8) & 0xFF), 0, 256);
            features[FEATURE_SIZE - 2] = normalizeValue(
                    ((hash >> 16) & 0xFF), 0, 256);
            features[FEATURE_SIZE - 1] = normalizeValue(
                    ((hash >> 24) & 0xFF), 0, 256);
        }

        return features;
    }

    /**
     * Normalize a value to [0, 1] range given expected min/max.
     * Clamps to [0, 1] if value is outside range.
     */
    private static float normalizeValue(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }
}
