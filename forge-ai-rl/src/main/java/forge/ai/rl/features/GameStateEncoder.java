package forge.ai.rl.features;

import forge.ai.rl.RLConfig;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
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

    /**
     * Encode global (non-per-card) game state features.
     *
     * Layout (64 floats):
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
     * [36-63] : reserved for expansion
     */
    private float[] encodeGlobalState(Game game, Player me, Player opp) {
        float[] features = new float[64];
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

        // Creature counts
        int myCreatures = 0, oppCreatures = 0;
        int myPower = 0, oppPower = 0;
        int myToughness = 0, oppToughness = 0;
        for (Card c : me.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) {
                myCreatures++;
                myPower += c.getNetPower();
                myToughness += c.getNetToughness();
            }
        }
        if (opp != null) {
            for (Card c : opp.getCardsIn(ZoneType.Battlefield)) {
                if (c.isCreature()) {
                    oppCreatures++;
                    oppPower += c.getNetPower();
                    oppToughness += c.getNetToughness();
                }
            }
        }

        features[idx++] = normalize(myCreatures, 0, 20);
        features[idx++] = normalize(oppCreatures, 0, 20);
        features[idx++] = normalize(myPower, 0, 60);
        features[idx++] = normalize(oppPower, 0, 60);
        features[idx++] = normalize(myToughness, 0, 60);
        features[idx++] = normalize(oppToughness, 0, 60);

        // Lands
        int myLandsUntapped = 0, myLandsTapped = 0, oppLands = 0;
        for (Card c : me.getCardsIn(ZoneType.Battlefield)) {
            if (c.isLand()) {
                if (c.isTapped()) myLandsTapped++;
                else myLandsUntapped++;
            }
        }
        if (opp != null) {
            for (Card c : opp.getCardsIn(ZoneType.Battlefield)) {
                if (c.isLand()) oppLands++;
            }
        }
        features[idx++] = normalize(myLandsUntapped, 0, 15);
        features[idx++] = normalize(myLandsTapped, 0, 15);
        features[idx++] = normalize(oppLands, 0, 15);

        // Stack
        features[idx++] = normalize(game.getStack().size(), 0, 10);

        // Phase convenience flags
        if (currentPhase != null) {
            features[idx++] = (currentPhase == PhaseType.MAIN1 && ph.getPlayerTurn() == me) ? 1f : 0f;
            features[idx++] = (currentPhase == PhaseType.MAIN2 && ph.getPlayerTurn() == me) ? 1f : 0f;
            features[idx++] = (currentPhase.isAfter(PhaseType.MAIN1) && currentPhase.isBefore(PhaseType.MAIN2)) ? 1f : 0f;
        } else {
            idx += 3;
        }

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
