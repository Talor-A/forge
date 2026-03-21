package forge.ai.rl.features;

import forge.ai.rl.RLConfig;
import forge.card.mana.ManaAtom;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.mana.ManaPool;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.zone.MagicStack;
import forge.game.zone.ZoneType;

/**
 * Converts the full game state into a GameStateFeatures object suitable for neural network input.
 * This is the bridge between Forge's rich game objects and the flat tensor representation.
 */
public class GameStateEncoder {
    private final RLConfig config;

    public GameStateEncoder(RLConfig config) {
        this.config = config;
    }

    /**
     * Encode the current game state from the perspective of the given player.
     */
    public GameStateFeatures encode(Game game, Player player) {
        try {
            return encodeImpl(game, player);
        } catch (Exception e) {
            // Return zeroed features rather than crashing — the game state
            // may have nulls during mulligan, between phases, etc.
            org.tinylog.Logger.warn("GameStateEncoder.encode failed: {}", e.getMessage());
            return GameStateFeatures.empty();
        }
    }

    private GameStateFeatures encodeImpl(Game game, Player player) {
        Player opponent = player.getWeakestOpponent();

        float[] globalFeatures = encodeGlobalState(game, player, opponent);

        // Encode each zone
        ZoneEncoding myBoard = encodeZone(player.getCardsIn(ZoneType.Battlefield), config.getMaxBoardCreatures());
        ZoneEncoding oppBoard = encodeZone(
                opponent != null ? opponent.getCardsIn(ZoneType.Battlefield) : new CardCollection(),
                config.getMaxBoardCreatures());
        ZoneEncoding myHand = encodeZone(player.getCardsIn(ZoneType.Hand), config.getMaxHandCards());
        ZoneEncoding myGraveyard = encodeZone(player.getCardsIn(ZoneType.Graveyard), config.getMaxGraveyardCards());
        ZoneEncoding oppGraveyard = encodeZone(
                opponent != null ? opponent.getCardsIn(ZoneType.Graveyard) : new CardCollection(),
                config.getMaxGraveyardCards());
        ZoneEncoding stack = encodeStack(game.getStack());

        return new GameStateFeatures(
                globalFeatures,
                myBoard.features, myBoard.mask,
                oppBoard.features, oppBoard.mask,
                myHand.features, myHand.mask,
                myGraveyard.features, myGraveyard.mask,
                oppGraveyard.features, oppGraveyard.mask,
                stack.features, stack.mask
        );
    }

    /** Global feature dimension. */
    public static final int GLOBAL_FEATURE_SIZE = 96;

    /**
     * Encode global (non-per-card) game state features.
     *
     * Layout (96 floats):
     * [0]     : my life total (normalized)
     * [1]     : opponent life total (normalized)
     * [2]     : my poison counters (normalized)
     * [3]     : opponent poison counters (normalized)
     * [4]     : turn number (normalized)
     * [5]     : am I the active player? (0 or 1)
     * [6-18]  : current phase (one-hot, 13 phases)
     * [19]    : my cards in hand count (normalized)
     * [20]    : opponent cards in hand count (normalized)
     * [21]    : my cards in library count (normalized)
     * [22]    : opponent cards in library count (normalized)
     * [23]    : my creature count on board (normalized)
     * [24]    : opponent creature count on board (normalized)
     * [25]    : my total power on board (normalized)
     * [26]    : opponent total power on board (normalized)
     * [27]    : my total toughness on board (normalized)
     * [28]    : opponent total toughness on board (normalized)
     * [29]    : my lands untapped count (normalized)
     * [30]    : my lands tapped count (normalized)
     * [31]    : opponent lands count (normalized)
     * [32]    : stack size (normalized)
     * [33]    : is my main phase 1? (0 or 1)
     * [34]    : is my main phase 2? (0 or 1)
     * [35]    : is combat? (0 or 1)
     * [36-41] : available mana in pool by color (W, U, B, R, G, colorless)
     * [42-47] : producible mana from untapped permanents (W, U, B, R, G, colorless)
     * [48]    : total available mana (sum of pool)
     * [49]    : spells cast this turn (normalized)
     * [50]    : lands played this turn (0 or 1)
     * [51]    : opponent lands untapped (normalized)
     * [52]    : my nonland permanents count (normalized)
     * [53]    : opponent nonland permanents count (normalized)
     * [54-63] : reserved
     * [64-69] : my color devotion (W, U, B, R, G, colorless)
     * [70]    : castable cards in hand (normalized)
     * [71]    : reserved
     * [72]    : my enchantment count (normalized)
     * [73]    : my artifact count (normalized)
     * [74]    : opponent enchantment count (normalized)
     * [75]    : opponent artifact count (normalized)
     * [76-95] : reserved
     */
    private float[] encodeGlobalState(Game game, Player me, Player opp) {
        float[] features = new float[GLOBAL_FEATURE_SIZE];
        PhaseHandler ph = game.getPhaseHandler();

        int idx = 0;
        features[idx++] = normalize(me.getLife(), -10, 40);
        features[idx++] = normalize(opp != null ? opp.getLife() : 20, -10, 40);
        features[idx++] = normalize(me.getPoisonCounters(), 0, 10);
        features[idx++] = normalize(opp != null ? opp.getPoisonCounters() : 0, 0, 10);
        features[idx++] = normalize(ph.getTurn(), 0, 30);
        features[idx++] = ph.getPlayerTurn() == me ? 1f : 0f;

        // Phase one-hot [6-18]
        PhaseType[] phases = PhaseType.values();
        PhaseType currentPhase = ph.getPhase();
        for (PhaseType pt : phases) {
            if (idx >= 19) break; // safety
            features[idx++] = (pt == currentPhase) ? 1f : 0f;
        }
        idx = 19; // ensure alignment

        // Hand sizes
        features[idx++] = normalize(me.getCardsIn(ZoneType.Hand).size(), 0, 15);
        features[idx++] = normalize(opp != null ? opp.getCardsIn(ZoneType.Hand).size() : 0, 0, 15);

        // Library sizes
        features[idx++] = normalize(me.getCardsIn(ZoneType.Library).size(), 0, 60);
        features[idx++] = normalize(opp != null ? opp.getCardsIn(ZoneType.Library).size() : 0, 0, 60);

        // Board statistics — collect in a single pass per player
        int myCreatures = 0, oppCreatures = 0;
        int myPower = 0, oppPower = 0;
        int myToughness = 0, oppToughness = 0;
        int myLandsUntapped = 0, myLandsTapped = 0;
        int oppLands = 0, oppLandsUntapped = 0;
        int myNonlands = 0, oppNonlands = 0;
        int myEnchantments = 0, myArtifacts = 0;
        int oppEnchantments = 0, oppArtifacts = 0;
        // Devotion: count colored mana symbols on permanents
        int[] myDevotion = new int[6]; // WUBRGC
        // Producible mana from untapped permanents
        boolean[] myProducible = new boolean[6]; // WUBRGC

        for (Card c : me.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) {
                myCreatures++;
                myPower += c.getNetPower();
                myToughness += c.getNetToughness();
            }
            if (c.isLand()) {
                if (c.isTapped()) myLandsTapped++;
                else myLandsUntapped++;
            } else {
                myNonlands++;
            }
            if (c.isEnchantment()) myEnchantments++;
            if (c.isArtifact()) myArtifacts++;

            // Devotion: count colored mana symbols in mana cost
            if (c.getManaCost() != null) {
                int[] shards = c.getManaCost().getColorShardCounts();
                for (int i = 0; i < Math.min(shards.length, 6); i++) {
                    myDevotion[i] += shards[i];
                }
            }

            // Producible mana from untapped permanents with mana abilities
            if (!c.isTapped()) {
                for (SpellAbility ma : c.getManaAbilities()) {
                    String produced = ma.getManaPart() != null
                            ? ma.getManaPart().getOrigProduced() : "";
                    if (produced.contains("W")) myProducible[0] = true;
                    if (produced.contains("U")) myProducible[1] = true;
                    if (produced.contains("B")) myProducible[2] = true;
                    if (produced.contains("R")) myProducible[3] = true;
                    if (produced.contains("G")) myProducible[4] = true;
                    if (produced.contains("C")) myProducible[5] = true;
                    // "Any" can produce any color
                    if (produced.contains("Any")) {
                        for (int i = 0; i < 5; i++) myProducible[i] = true;
                    }
                }
            }
        }
        if (opp != null) {
            for (Card c : opp.getCardsIn(ZoneType.Battlefield)) {
                if (c.isCreature()) {
                    oppCreatures++;
                    oppPower += c.getNetPower();
                    oppToughness += c.getNetToughness();
                }
                if (c.isLand()) {
                    oppLands++;
                    if (!c.isTapped()) oppLandsUntapped++;
                } else {
                    oppNonlands++;
                }
                if (c.isEnchantment()) oppEnchantments++;
                if (c.isArtifact()) oppArtifacts++;
            }
        }

        features[idx++] = normalize(myCreatures, 0, 20);    // [23]
        features[idx++] = normalize(oppCreatures, 0, 20);    // [24]
        features[idx++] = normalize(myPower, 0, 60);         // [25]
        features[idx++] = normalize(oppPower, 0, 60);        // [26]
        features[idx++] = normalize(myToughness, 0, 60);     // [27]
        features[idx++] = normalize(oppToughness, 0, 60);    // [28]

        features[idx++] = normalize(myLandsUntapped, 0, 15); // [29]
        features[idx++] = normalize(myLandsTapped, 0, 15);   // [30]
        features[idx++] = normalize(oppLands, 0, 15);        // [31]

        // Stack
        features[idx++] = normalize(game.getStack().size(), 0, 10); // [32]

        // Phase convenience flags (currentPhase is null during mulligan)
        features[idx++] = (currentPhase == PhaseType.MAIN1 && ph.getPlayerTurn() == me) ? 1f : 0f;  // [33]
        features[idx++] = (currentPhase == PhaseType.MAIN2 && ph.getPlayerTurn() == me) ? 1f : 0f;  // [34]
        features[idx++] = (currentPhase != null && currentPhase.isAfter(PhaseType.MAIN1) && currentPhase.isBefore(PhaseType.MAIN2)) ? 1f : 0f; // [35]

        // === NEW FEATURES [36-75] ===

        // [36-41] Available mana in pool by color (W, U, B, R, G, colorless)
        ManaPool pool = me.getManaPool();
        byte[] manaTypes = ManaAtom.MANATYPES; // {W, U, B, R, G, C}
        int totalPoolMana = 0;
        for (int i = 0; i < 6; i++) {
            int amount = pool.getAmountOfColor(manaTypes[i]);
            features[36 + i] = normalize(amount, 0, 10);
            totalPoolMana += amount;
        }

        // [42-47] Producible mana from untapped permanents (binary flags)
        for (int i = 0; i < 6; i++) {
            features[42 + i] = myProducible[i] ? 1f : 0f;
        }

        // [48] Total available mana (pool + untapped lands as rough estimate)
        features[48] = normalize(totalPoolMana + myLandsUntapped, 0, 15);

        // [49] Spells cast this turn
        features[49] = normalize(me.getSpellsCastThisTurn(), 0, 10);

        // [50] Lands played this turn
        features[50] = normalize(me.getLandsPlayedThisTurn(), 0, 2);

        // [51] Opponent lands untapped (threat of instant-speed response)
        features[51] = normalize(oppLandsUntapped, 0, 15);

        // [52-53] Nonland permanent counts
        features[52] = normalize(myNonlands, 0, 30);
        features[53] = normalize(oppNonlands, 0, 30);

        // [54-63] reserved

        // [64-69] Color devotion (WUBRGC)
        for (int i = 0; i < 6; i++) {
            features[64 + i] = normalize(myDevotion[i], 0, 15);
        }

        // [70] Castable cards in hand (can afford mana cost from untapped lands)
        int castable = 0;
        int availableMana = myLandsUntapped + totalPoolMana;
        for (Card c : me.getCardsIn(ZoneType.Hand)) {
            if (c.getManaCost() != null && c.getManaCost().getCMC() <= availableMana) {
                castable++;
            }
        }
        features[70] = normalize(castable, 0, 10);

        // [71] reserved

        // [72-75] Artifact/enchantment counts
        features[72] = normalize(myEnchantments, 0, 10);
        features[73] = normalize(myArtifacts, 0, 10);
        features[74] = normalize(oppEnchantments, 0, 10);
        features[75] = normalize(oppArtifacts, 0, 10);

        // [76-95] reserved

        return features;
    }

    private static class ZoneEncoding {
        float[][] features;
        boolean[] mask;
    }

    private ZoneEncoding encodeZone(CardCollectionView cards, int maxCards) {
        ZoneEncoding encoding = new ZoneEncoding();
        encoding.features = new float[maxCards][CardFeatures.FEATURE_SIZE];
        encoding.mask = new boolean[maxCards];

        int count = Math.min(cards.size(), maxCards);
        for (int i = 0; i < count; i++) {
            encoding.features[i] = CardFeatures.encode(cards.get(i));
            encoding.mask[i] = true;
        }
        return encoding;
    }

    private ZoneEncoding encodeStack(MagicStack stack) {
        int maxEntries = config.getMaxStackEntries();
        ZoneEncoding encoding = new ZoneEncoding();
        encoding.features = new float[maxEntries][CardFeatures.FEATURE_SIZE];
        encoding.mask = new boolean[maxEntries];

        int idx = 0;
        for (SpellAbilityStackInstance si : stack) {
            if (idx >= maxEntries) break;
            if (si.getSourceCard() != null) {
                encoding.features[idx] = CardFeatures.encode(si.getSourceCard());
                encoding.mask[idx] = true;
            }
            idx++;
        }
        return encoding;
    }

    private static float normalize(double value, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (value - min) / (max - min)));
    }
}
