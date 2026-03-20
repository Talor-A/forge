package forge.ai.rl;

import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import forge.LobbyPlayer;
import forge.card.ColorSet;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.game.*;
import forge.game.ability.effects.RollDiceEffect;
import forge.game.card.*;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.cost.*;
import forge.game.keyword.KeywordInterface;
import forge.game.mana.Mana;
import forge.game.mana.ManaConversionMatrix;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.*;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.*;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.PlayerZone;
import forge.game.zone.ZoneType;
import forge.item.PaperCard;
import forge.util.ITriggerEvent;
import forge.util.collect.FCollectionView;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import java.util.*;
import java.util.function.Predicate;

/**
 * PlayerController implementation for the Reinforcement Learning AI.
 *
 * This class bridges the Forge game engine's decision interface with the RL model.
 * For each decision point, it:
 * 1. Encodes the game state and available options as feature vectors
 * 2. Sends them to the model server (or ONNX runtime) for inference
 * 3. Interprets the model's output as a game action
 * 4. Records the decision for trajectory collection
 *
 * When the model server is unavailable or in HEURISTIC_FALLBACK mode, it delegates
 * to a fallback PlayerControllerAi instance to ensure games always complete.
 */
public class PlayerControllerRL extends PlayerController {

    private final RLController rl;
    private final forge.ai.PlayerControllerAi fallbackAi;
    private final forge.ai.AiController fallbackBrains;

    public PlayerControllerRL(Game game, Player p, LobbyPlayer lp, RLConfig config) {
        super(game, p, lp);
        this.rl = new RLController(config);
        this.rl.setPlayer(p);

        // Create fallback heuristic AI with a proper LobbyPlayerAi
        // (AiController needs AI profile from LobbyPlayerAi)
        forge.ai.LobbyPlayerAi fallbackLp =
                new forge.ai.LobbyPlayerAi(lp.getName(), null);
        fallbackLp.setAiProfile("Default");
        this.fallbackAi = new forge.ai.PlayerControllerAi(
                game, p, fallbackLp);
        this.fallbackBrains = fallbackAi.getAi();
    }

    public RLController getRLController() {
        return rl;
    }

    @Override
    public boolean isAI() {
        return true;
    }

    /**
     * Check if we should use the heuristic fallback for this decision.
     * Returns true if mode is HEURISTIC_FALLBACK, or if the model server
     * is unreachable in GRPC mode. This prevents the RL agent from making
     * broken decisions when the server is down.
     */
    /**
     * Check if we should use the heuristic fallback for this decision.
     *
     * Currently returns true for ALL non-combat decisions.
     * Only declareAttackers and declareBlockers bypass this to use
     * the RL model — all other decisions (priority, binary, target,
     * mulligan, etc.) use heuristic because the RL model isn't
     * trained for them and returns bad answers that prevent creatures
     * from being played.
     */
    private boolean shouldUseFallback() {
        return true;
    }

    // ===== PRIORITY / SPELL ABILITY DECISIONS =====

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        // Use the existing heuristic mechanism to enumerate playable spells, then choose via RL
        List<SpellAbility> fallbackChoice = fallbackAi.chooseSpellAbilityToPlay();
        if (fallbackChoice == null || fallbackChoice.isEmpty()) {
            return null;
        }

        // Always use heuristic for spell selection — RL
        // priority head not yet trained with full action
        // context. Combat decisions use RL (attack/block).
        return fallbackChoice;
    }

    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        // Delegate to fallback — this handles the mechanical execution
        return fallbackAi.playChosenSpellAbility(sa);
    }

    @Override
    public SpellAbility getAbilityToPlay(Card hostCard, List<SpellAbility> abilities, ITriggerEvent triggerEvent) {
        if (abilities.isEmpty()) return null;
        if (abilities.size() == 1) return abilities.get(0);

        if (shouldUseFallback()) {
            return fallbackAi.getAbilityToPlay(hostCard, abilities, triggerEvent);
        }

        int idx = rl.decidePriorityAction(abilities);
        if (idx < 0 || idx >= abilities.size()) return abilities.get(0);
        return abilities.get(idx);
    }

    @Override
    public void playSpellAbilityNoStack(SpellAbility effectSA, boolean mayChooseNewTargets) {
        fallbackAi.playSpellAbilityNoStack(effectSA, mayChooseNewTargets);
    }

    @Override
    public List<SpellAbility> orderSimultaneousSa(List<SpellAbility> activePlayerSAs) {
        // For now, delegate ordering to heuristic
        return fallbackAi.orderSimultaneousSa(activePlayerSAs);
    }

    @Override
    public void orderAndPlaySimultaneousSa(List<SpellAbility> activePlayerSAs) {
        fallbackAi.orderAndPlaySimultaneousSa(activePlayerSAs);
    }

    @Override
    public boolean playTrigger(Card host, WrappedAbility wrapperAbility, boolean isMandatory) {
        if (isMandatory) return true;

        if (shouldUseFallback()) {
            return fallbackAi.playTrigger(host, wrapperAbility, isMandatory);
        }

        return rl.decideBinary("play_trigger_" + (host != null ? host.getName() : "unknown"));
    }

    @Override
    public boolean playSaFromPlayEffect(SpellAbility tgtSA) {
        return fallbackAi.playSaFromPlayEffect(tgtSA);
    }

    // ===== COMBAT DECISIONS =====

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        // Attack decisions use RL model when available, heuristic otherwise
        if (rl.getConfig().getMode() != RLModelMode.GRPC || !rl.isModelServerAvailable()) {
            fallbackAi.declareAttackers(attacker, combat);
            return;
        }

        // Get list of possible attackers
        GameEntity defender = attacker.getWeakestOpponent();
        if (defender == null) return;

        CardCollection possibleAttackers = new CardCollection();
        for (Card c : attacker.getCreaturesInPlay()) {
            if (CombatUtil.canAttack(c, defender)) {
                possibleAttackers.add(c);
            }
        }

        if (possibleAttackers.isEmpty()) return;

        // Ask RL which creatures should attack
        List<Integer> attackerIndices = rl.decideAttackers(possibleAttackers);

        // Declare the selected creatures as attackers
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
        // Block decisions use RL model when available, heuristic otherwise
        if (rl.getConfig().getMode() != RLModelMode.GRPC || !rl.isModelServerAvailable()) {
            fallbackAi.declareBlockers(defender, combat);
            return;
        }

        // Get possible blockers and current attackers
        List<Card> possibleBlockers = new ArrayList<>();
        for (Card c : defender.getCreaturesInPlay()) {
            if (!c.isTapped() && !c.hasKeyword("CARDNAME can't block.")) {
                possibleBlockers.add(c);
            }
        }

        CardCollection attackers = combat.getAttackers();
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return;

        // Ask RL for blocking assignments
        List<int[]> assignments = rl.decideBlockers(possibleBlockers, attackers);

        // Apply blocking assignments
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

        // Mandatory blockers are handled by the game engine after declareBlockers
    }

    @Override
    public CardCollection orderBlockers(Card attacker, CardCollection blockers) {
        // Delegate ordering to heuristic for now
        return fallbackAi.orderBlockers(attacker, blockers);
    }

    @Override
    public CardCollection orderBlocker(Card attacker, Card blocker, CardCollection oldBlockers) {
        return fallbackAi.orderBlocker(attacker, blocker, oldBlockers);
    }

    @Override
    public CardCollection orderAttackers(Card blocker, CardCollection attackers) {
        return fallbackAi.orderAttackers(blocker, attackers);
    }

    @Override
    public Map<Card, Integer> assignCombatDamage(Card attacker, CardCollectionView blockers,
                                                   CardCollectionView remaining, int damageDealt,
                                                   GameEntity defender, boolean overrideOrder) {
        return fallbackAi.assignCombatDamage(attacker, blockers, remaining, damageDealt, defender, overrideOrder);
    }

    @Override
    public List<Card> exertAttackers(List<Card> attackers) {
        // For now, delegate to heuristic
        return fallbackAi.exertAttackers(attackers);
    }

    @Override
    public List<Card> enlistAttackers(List<Card> attackers) {
        return fallbackAi.enlistAttackers(attackers);
    }

    // ===== CARD SELECTION DECISIONS =====

    @Override
    public CardCollectionView chooseCardsForEffect(CardCollectionView sourceList, SpellAbility sa,
                                                     String title, int min, int max, boolean isOptional,
                                                     Map<String, Object> params) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseCardsForEffect(sourceList, sa, title, min, max, isOptional, params);
        }

        if (sourceList.isEmpty()) return new CardCollection();

        List<Integer> indices = rl.decideCardSelection(sourceList, isOptional ? 0 : min, max);
        CardCollection result = new CardCollection();
        for (int idx : indices) {
            if (idx >= 0 && idx < sourceList.size()) {
                result.add(sourceList.get(idx));
            }
        }

        // Ensure we meet minimum requirements
        if (result.size() < min && !isOptional) {
            for (int i = 0; result.size() < min && i < sourceList.size(); i++) {
                if (!result.contains(sourceList.get(i))) {
                    result.add(sourceList.get(i));
                }
            }
        }
        return result;
    }

    @Override
    public CardCollection chooseCardsForEffectMultiple(Map<String, CardCollection> validMap,
                                                        SpellAbility sa, String title, boolean isOptional) {
        return fallbackAi.chooseCardsForEffectMultiple(validMap, sa, title, isOptional);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends GameEntity> T chooseSingleEntityForEffect(FCollectionView<T> optionList,
                                                                  DelayedReveal delayedReveal, SpellAbility sa,
                                                                  String title, boolean isOptional,
                                                                  Player relatedPlayer, Map<String, Object> params) {
        if (optionList.isEmpty()) return null;
        if (optionList.size() == 1 && !isOptional) return optionList.getFirst();

        if (shouldUseFallback()) {
            return fallbackAi.chooseSingleEntityForEffect(optionList, delayedReveal, sa, title, isOptional, relatedPlayer, params);
        }

        // Encode entities as card features
        List<float[]> candidates = new ArrayList<>();
        for (T entity : optionList) {
            if (entity instanceof Card) {
                candidates.add(forge.ai.rl.features.CardFeatures.encode((Card) entity));
            } else if (entity instanceof Player) {
                candidates.add(forge.ai.rl.features.ActionEncoder.encodeTarget(entity));
            } else {
                candidates.add(new float[forge.ai.rl.features.CardFeatures.FEATURE_SIZE]);
            }
        }

        List<Integer> indices = rl.decideCardSelection(
                optionList instanceof CardCollectionView ? (CardCollectionView) optionList : new CardCollection(),
                isOptional ? 0 : 1, 1);

        if (indices.isEmpty()) return isOptional ? null : optionList.getFirst();
        int idx = indices.get(0);
        if (idx >= 0 && idx < optionList.size()) return optionList.get(idx);
        return optionList.getFirst();
    }

    @Override
    public <T extends GameEntity> List<T> chooseEntitiesForEffect(FCollectionView<T> optionList, int min, int max,
                                                                    DelayedReveal delayedReveal, SpellAbility sa,
                                                                    String title, Player relatedPlayer,
                                                                    Map<String, Object> params) {
        return fallbackAi.chooseEntitiesForEffect(optionList, min, max, delayedReveal, sa, title, relatedPlayer, params);
    }

    @Override
    public CardCollectionView choosePermanentsToSacrifice(SpellAbility sa, int min, int max,
                                                            CardCollectionView validTargets, String message) {
        if (shouldUseFallback()) {
            return fallbackAi.choosePermanentsToSacrifice(sa, min, max, validTargets, message);
        }

        List<Integer> indices = rl.decideCardSelection(validTargets, min, max);
        CardCollection result = new CardCollection();
        for (int idx : indices) {
            if (idx >= 0 && idx < validTargets.size()) {
                result.add(validTargets.get(idx));
            }
        }
        // Ensure minimum
        for (int i = 0; result.size() < min && i < validTargets.size(); i++) {
            if (!result.contains(validTargets.get(i))) result.add(validTargets.get(i));
        }
        return result;
    }

    @Override
    public CardCollectionView choosePermanentsToDestroy(SpellAbility sa, int min, int max,
                                                          CardCollectionView validTargets, String message) {
        if (shouldUseFallback()) {
            return fallbackAi.choosePermanentsToDestroy(sa, min, max, validTargets, message);
        }
        return choosePermanentsToSacrifice(sa, min, max, validTargets, message);
    }

    @Override
    public CardCollectionView chooseCardsToDiscardFrom(Player playerDiscard, SpellAbility sa,
                                                         CardCollection validCards, int min, int max) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseCardsToDiscardFrom(playerDiscard, sa, validCards, min, max);
        }

        List<Integer> indices = rl.decideCardSelection(validCards, min, max);
        CardCollection result = new CardCollection();
        for (int idx : indices) {
            if (idx >= 0 && idx < validCards.size()) result.add(validCards.get(idx));
        }
        for (int i = 0; result.size() < min && i < validCards.size(); i++) {
            if (!result.contains(validCards.get(i))) result.add(validCards.get(i));
        }
        return result;
    }

    @Override
    public CardCollectionView chooseCardsToDiscardUnlessType(int min, CardCollectionView hand,
                                                               String[] unlessTypes, SpellAbility sa) {
        return fallbackAi.chooseCardsToDiscardUnlessType(min, hand, unlessTypes, sa);
    }

    @Override
    public CardCollection chooseCardsToDiscardToMaximumHandSize(int numDiscard) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseCardsToDiscardToMaximumHandSize(numDiscard);
        }

        CardCollectionView hand = player.getCardsIn(ZoneType.Hand);
        List<Integer> indices = rl.decideCardSelection(hand, numDiscard, numDiscard);
        CardCollection result = new CardCollection();
        for (int idx : indices) {
            if (idx >= 0 && idx < hand.size()) result.add(hand.get(idx));
        }
        for (int i = 0; result.size() < numDiscard && i < hand.size(); i++) {
            if (!result.contains(hand.get(i))) result.add(hand.get(i));
        }
        return result;
    }

    @Override
    public CardCollectionView chooseCardsToDelve(int genericAmount, CardCollection grave) {
        return fallbackAi.chooseCardsToDelve(genericAmount, grave);
    }

    @Override
    public Map<Card, ManaCostShard> chooseCardsForConvokeOrImprovise(SpellAbility sa, ManaCost manaCost,
                                                                       CardCollectionView untappedCards,
                                                                       boolean artifacts, boolean creatures,
                                                                       Integer maxReduction) {
        return fallbackAi.chooseCardsForConvokeOrImprovise(sa, manaCost, untappedCards, artifacts, creatures, maxReduction);
    }

    @Override
    public List<Card> chooseCardsForSplice(SpellAbility sa, List<Card> cards) {
        return fallbackAi.chooseCardsForSplice(sa, cards);
    }

    @Override
    public CardCollectionView chooseCardsToRevealFromHand(int min, int max, CardCollectionView valid) {
        return fallbackAi.chooseCardsToRevealFromHand(min, max, valid);
    }

    // ===== TARGETING DECISIONS =====

    @Override
    public boolean chooseTargetsFor(SpellAbility currentAbility) {
        // Targeting is complex — delegate to heuristic for now
        // TODO: implement RL-based target selection
        return fallbackAi.chooseTargetsFor(currentAbility);
    }

    @Override
    public TargetChoices chooseNewTargetsFor(SpellAbility ability, Predicate<GameObject> filter, boolean optional) {
        return fallbackAi.chooseNewTargetsFor(ability, filter, optional);
    }

    @Override
    public Pair<SpellAbilityStackInstance, GameObject> chooseTarget(SpellAbility sa,
                                                                      List<Pair<SpellAbilityStackInstance, GameObject>> allTargets) {
        return fallbackAi.chooseTarget(sa, allTargets);
    }

    // ===== MULLIGAN DECISIONS =====

    @Override
    public boolean mulliganKeepHand(Player firstPlayer, int cardsToReturn) {
        // Cannot delegate to fallbackAi.mulliganKeepHand() because
        // ComputerUtil.scoreHand() casts player.getController() to
        // PlayerControllerAi which fails for our PlayerControllerRL.
        // Use simple heuristic: keep hands with 2-5 lands.
        if (shouldUseFallback()) {
            int lands = 0;
            for (forge.game.card.Card c : player.getCardsIn(ZoneType.Hand)) {
                if (c.isLand()) lands++;
            }
            return lands >= 2 && lands <= 5;
        }

        CardCollectionView hand = player.getCardsIn(ZoneType.Hand);
        return rl.decideMulligan(hand, cardsToReturn);
    }

    @Override
    public CardCollectionView tuckCardsViaMulligan(CardCollectionView hand, int cardsToReturn) {
        if (shouldUseFallback()) {
            return fallbackAi.tuckCardsViaMulligan(hand, cardsToReturn);
        }

        List<Integer> indices = rl.decideCardSelection(hand, cardsToReturn, cardsToReturn);
        CardCollection result = new CardCollection();
        for (int idx : indices) {
            if (idx >= 0 && idx < hand.size()) result.add(hand.get(idx));
        }
        for (int i = 0; result.size() < cardsToReturn && i < hand.size(); i++) {
            if (!result.contains(hand.get(i))) result.add(hand.get(i));
        }
        return result;
    }

    // ===== MANA / COST DECISIONS =====

    @Override
    public boolean payManaCost(ManaCost toPay, CostPartMana costPartMana, SpellAbility sa,
                                String prompt, ManaConversionMatrix matrix, boolean effect) {
        return fallbackAi.payManaCost(toPay, costPartMana, sa, prompt, matrix, effect);
    }

    @Override
    public boolean applyManaToCost(ManaCostBeingPaid toPay, SpellAbility ability,
                                    String prompt, ManaConversionMatrix matrix, boolean effect) {
        return fallbackAi.applyManaToCost(toPay, ability, prompt, matrix, effect);
    }

    @Override
    public Mana chooseManaFromPool(List<Mana> manaChoices) {
        return fallbackAi.chooseManaFromPool(manaChoices);
    }

    @Override
    public Map<Byte, Integer> specifyManaCombo(SpellAbility sa, ColorSet colorSet, int manaAmount, boolean different) {
        return fallbackAi.specifyManaCombo(sa, colorSet, manaAmount, different);
    }

    @Override
    public CardCollectionView chooseCardsForCost(CardCollectionView optionList, SpellAbility sa,
                                                   CostPartWithList cpl, int amount, boolean isOptional, String prompt) {
        return fallbackAi.chooseCardsForCost(optionList, sa, cpl, amount, isOptional, prompt);
    }

    @Override
    public CostDecisionMakerBase getCostDecisionMaker(Player p, SpellAbility ability, boolean effect, String prompt) {
        return fallbackAi.getCostDecisionMaker(p, ability, effect, prompt);
    }

    @Override
    public boolean helpPayForAssistSpell(ManaCostBeingPaid cost, SpellAbility sa, int max, int requested) {
        return fallbackAi.helpPayForAssistSpell(cost, sa, max, requested);
    }

    @Override
    public Player choosePlayerToAssistPayment(FCollectionView<Player> optionList, SpellAbility sa, String title, int max) {
        return fallbackAi.choosePlayerToAssistPayment(optionList, sa, title, max);
    }

    @Override
    public List<OptionalCostValue> chooseOptionalCosts(SpellAbility choosen, List<OptionalCostValue> optionalCostValues) {
        return fallbackAi.chooseOptionalCosts(choosen, optionalCostValues);
    }

    @Override
    public List<CostPart> orderCosts(List<CostPart> costs) {
        return fallbackAi.orderCosts(costs);
    }

    @Override
    public boolean payCostToPreventEffect(Cost cost, SpellAbility sa, boolean alreadyPaid, FCollectionView<Player> allPayers) {
        return fallbackAi.payCostToPreventEffect(cost, sa, alreadyPaid, allPayers);
    }

    @Override
    public boolean payCostDuringRoll(Cost cost, SpellAbility sa) {
        return fallbackAi.payCostDuringRoll(cost, sa);
    }

    @Override
    public boolean payCombatCost(Card card, Cost cost, SpellAbility sa, String prompt) {
        return fallbackAi.payCombatCost(card, cost, sa, prompt);
    }

    @Override
    public int chooseNumberForCostReduction(SpellAbility sa, int min, int max) {
        return fallbackAi.chooseNumberForCostReduction(sa, min, max);
    }

    @Override
    public int chooseNumberForKeywordCost(SpellAbility sa, Cost cost, KeywordInterface keyword, String prompt, int max) {
        return fallbackAi.chooseNumberForKeywordCost(sa, cost, keyword, prompt, max);
    }

    // ===== CONFIRMATION / BINARY DECISIONS =====

    @Override
    public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                   List<String> options, Card cardToShow, Map<String, Object> params) {
        if (shouldUseFallback()) {
            return fallbackAi.confirmAction(sa, mode, message, options, cardToShow, params);
        }
        return rl.decideBinary("confirm_" + mode.name());
    }

    @Override
    public boolean confirmBidAction(SpellAbility sa, PlayerActionConfirmMode mode, String string, int bid, Player winner) {
        return fallbackAi.confirmBidAction(sa, mode, string, bid, winner);
    }

    @Override
    public boolean confirmReplacementEffect(ReplacementEffect replacementEffect, SpellAbility effectSA,
                                              GameEntity affected, String question) {
        if (shouldUseFallback()) {
            return fallbackAi.confirmReplacementEffect(replacementEffect, effectSA, affected, question);
        }
        return rl.decideBinary("confirm_replacement");
    }

    @Override
    public boolean confirmStaticApplication(Card hostCard, PlayerActionConfirmMode mode, String message, String logic) {
        return fallbackAi.confirmStaticApplication(hostCard, mode, message, logic);
    }

    @Override
    public boolean confirmTrigger(WrappedAbility sa) {
        if (shouldUseFallback()) {
            return fallbackAi.confirmTrigger(sa);
        }
        return rl.decideBinary("confirm_trigger");
    }

    @Override
    public boolean confirmPayment(CostPart costPart, String string, SpellAbility sa) {
        return fallbackAi.confirmPayment(costPart, string, sa);
    }

    // ===== NUMBER / CHOICE DECISIONS =====

    @Override
    public int chooseNumber(SpellAbility sa, String title, int min, int max) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseNumber(sa, title, min, max);
        }
        return rl.decideNumber(min, max, "choose_number");
    }

    @Override
    public int chooseNumber(SpellAbility sa, String title, List<Integer> values, Player relatedPlayer) {
        return fallbackAi.chooseNumber(sa, title, values, relatedPlayer);
    }

    @Override
    public boolean chooseBinary(SpellAbility sa, String question, BinaryChoiceType kindOfChoice, Boolean defaultChoice) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseBinary(sa, question, kindOfChoice, defaultChoice);
        }
        return rl.decideBinary("binary_" + kindOfChoice.name());
    }

    @Override
    public boolean chooseFlipResult(SpellAbility sa, Player flipper, boolean call) {
        return fallbackAi.chooseFlipResult(sa, flipper, call);
    }

    @Override
    public Integer announceRequirements(SpellAbility ability, int min, int max, String announce) {
        return fallbackAi.announceRequirements(ability, min, max, announce);
    }

    // ===== TYPE / COLOR / NAME CHOICES =====

    @Override
    public byte chooseColor(String message, SpellAbility sa, ColorSet colors) {
        return fallbackAi.chooseColor(message, sa, colors);
    }

    @Override
    public byte chooseColorAllowColorless(String message, Card c, ColorSet colors) {
        return fallbackAi.chooseColorAllowColorless(message, c, colors);
    }

    @Override
    public ColorSet chooseColors(String message, SpellAbility sa, int min, int max, ColorSet options) {
        return fallbackAi.chooseColors(message, sa, min, max, options);
    }

    @Override
    public String chooseSomeType(String kindOfType, SpellAbility sa, Collection<String> validTypes, boolean isOptional) {
        return fallbackAi.chooseSomeType(kindOfType, sa, validTypes, isOptional);
    }

    @Override
    public String chooseCardName(SpellAbility sa, Predicate<ICardFace> cpp, String valid, String message) {
        return fallbackAi.chooseCardName(sa, cpp, valid, message);
    }

    @Override
    public String chooseCardName(SpellAbility sa, List<ICardFace> faces, String message) {
        return fallbackAi.chooseCardName(sa, faces, message);
    }

    @Override
    public CounterType chooseCounterType(List<CounterType> options, SpellAbility sa, String prompt,
                                           Map<String, Object> params) {
        return fallbackAi.chooseCounterType(options, sa, prompt, params);
    }

    @Override
    public String chooseKeywordForPump(List<String> options, SpellAbility sa, String prompt, Card tgtCard) {
        return fallbackAi.chooseKeywordForPump(options, sa, prompt, tgtCard);
    }

    @Override
    public String chooseProtectionType(SpellAbility sa, List<String> choices) {
        return fallbackAi.chooseProtectionType(sa, choices);
    }

    @Override
    public ICardFace chooseSingleCardFace(SpellAbility sa, String message, Predicate<ICardFace> cpp, String name) {
        return fallbackAi.chooseSingleCardFace(sa, message, cpp, name);
    }

    @Override
    public ICardFace chooseSingleCardFace(SpellAbility sa, List<ICardFace> faces, String message) {
        return fallbackAi.chooseSingleCardFace(sa, faces, message);
    }

    @Override
    public CardState chooseSingleCardState(SpellAbility sa, List<CardState> states, String message,
                                             Map<String, Object> params) {
        return fallbackAi.chooseSingleCardState(sa, states, message, params);
    }

    // ===== SPELL ABILITY CHOICE =====

    @Override
    public List<SpellAbility> chooseSpellAbilitiesForEffect(List<SpellAbility> spells, SpellAbility sa,
                                                              String title, int num, Map<String, Object> params) {
        return fallbackAi.chooseSpellAbilitiesForEffect(spells, sa, title, num, params);
    }

    @Override
    public SpellAbility chooseSingleSpellForEffect(List<SpellAbility> spells, SpellAbility sa,
                                                     String title, Map<String, Object> params) {
        return fallbackAi.chooseSingleSpellForEffect(spells, sa, title, params);
    }

    @Override
    public List<AbilitySub> chooseModeForAbility(SpellAbility sa, List<AbilitySub> possible,
                                                   int min, int num, boolean allowRepeat) {
        return fallbackAi.chooseModeForAbility(sa, possible, min, num, allowRepeat);
    }

    // ===== LIBRARY / SCRY / SURVEIL =====

    @Override
    public ImmutablePair<CardCollection, CardCollection> arrangeForScry(CardCollection topN) {
        if (shouldUseFallback()) {
            return fallbackAi.arrangeForScry(topN);
        }

        // RL decides which cards go to top vs bottom
        List<Integer> keepOnTop = rl.decideCardSelection(topN, 0, topN.size());
        CardCollection top = new CardCollection();
        CardCollection bottom = new CardCollection();
        Set<Integer> topSet = new HashSet<>(keepOnTop);
        for (int i = 0; i < topN.size(); i++) {
            if (topSet.contains(i)) {
                top.add(topN.get(i));
            } else {
                bottom.add(topN.get(i));
            }
        }
        return ImmutablePair.of(top, bottom);
    }

    @Override
    public ImmutablePair<CardCollection, CardCollection> arrangeForSurveil(CardCollection topN) {
        if (shouldUseFallback()) {
            return fallbackAi.arrangeForSurveil(topN);
        }
        // Same logic as scry for now
        return arrangeForScry(topN);
    }

    @Override
    public boolean willPutCardOnTop(Card c) {
        if (shouldUseFallback()) {
            return fallbackAi.willPutCardOnTop(c);
        }
        return rl.decideBinary("put_on_top_" + c.getName());
    }

    @Override
    public CardCollectionView orderMoveToZoneList(CardCollectionView cards, ZoneType destinationZone, SpellAbility source) {
        return fallbackAi.orderMoveToZoneList(cards, destinationZone, source);
    }

    // ===== PILE / MISC CHOICES =====

    @Override
    public boolean chooseCardsPile(SpellAbility sa, CardCollectionView pile1, CardCollectionView pile2, String faceUp) {
        if (shouldUseFallback()) {
            return fallbackAi.chooseCardsPile(sa, pile1, pile2, faceUp);
        }
        return rl.decideBinary("choose_pile");
    }

    @Override
    public Object vote(SpellAbility sa, String prompt, List<Object> options,
                        ListMultimap<Object, Player> votes, Player forPlayer, boolean optional) {
        return fallbackAi.vote(sa, prompt, options, votes, forPlayer, optional);
    }

    @Override
    public ReplacementEffect chooseSingleReplacementEffect(List<ReplacementEffect> possibleReplacers) {
        return fallbackAi.chooseSingleReplacementEffect(possibleReplacers);
    }

    @Override
    public StaticAbility chooseSingleStaticAbility(List<StaticAbility> possibleReplacers) {
        return fallbackAi.chooseSingleStaticAbility(possibleReplacers);
    }

    @Override
    public Map<GameEntity, Integer> divideShield(Card effectSource, Map<GameEntity, Integer> affected, int shieldAmount) {
        return fallbackAi.divideShield(effectSource, affected, shieldAmount);
    }

    // ===== GAME SETUP =====

    @Override
    public List<PaperCard> sideboard(Deck deck, GameType gameType, String message) {
        return fallbackAi.sideboard(deck, gameType, message);
    }

    @Override
    public List<PaperCard> chooseCardsYouWonToAddToDeck(List<PaperCard> losses) {
        return fallbackAi.chooseCardsYouWonToAddToDeck(losses);
    }

    @Override
    public Player chooseStartingPlayer(boolean isFirstGame) {
        return fallbackAi.chooseStartingPlayer(isFirstGame);
    }

    @Override
    public PlayerZone chooseStartingHand(List<PlayerZone> zones) {
        return fallbackAi.chooseStartingHand(zones);
    }

    @Override
    public List<SpellAbility> chooseSaToActivateFromOpeningHand(List<SpellAbility> usableFromOpeningHand) {
        return fallbackAi.chooseSaToActivateFromOpeningHand(usableFromOpeningHand);
    }

    // ===== ZONE CHANGE =====

    @Override
    public Card chooseSingleCardForZoneChange(ZoneType destination, List<ZoneType> origin, SpellAbility sa,
                                                CardCollection fetchList, DelayedReveal delayedReveal,
                                                String selectPrompt, boolean isOptional, Player decider) {
        return fallbackAi.chooseSingleCardForZoneChange(destination, origin, sa, fetchList, delayedReveal, selectPrompt, isOptional, decider);
    }

    @Override
    public List<Card> chooseCardsForZoneChange(ZoneType destination, List<ZoneType> origin, SpellAbility sa,
                                                 CardCollection fetchList, int min, int max,
                                                 DelayedReveal delayedReveal, String selectPrompt, Player decider) {
        return fallbackAi.chooseCardsForZoneChange(destination, origin, sa, fetchList, min, max, delayedReveal, selectPrompt, decider);
    }

    // ===== SPECIAL MECHANICS =====

    @Override
    public String chooseSector(Card assignee, String ai, List<String> sectors) {
        return fallbackAi.chooseSector(assignee, ai, sectors);
    }

    @Override
    public List<Card> chooseContraptionsToCrank(List<Card> contraptions) {
        return fallbackAi.chooseContraptionsToCrank(contraptions);
    }

    @Override
    public int chooseSprocket(Card assignee, List<Integer> sprockets) {
        return fallbackAi.chooseSprocket(assignee, sprockets);
    }

    @Override
    public PlanarDice choosePDRollToIgnore(List<PlanarDice> rolls) {
        return fallbackAi.choosePDRollToIgnore(rolls);
    }

    @Override
    public Integer chooseRollToIgnore(List<Integer> rolls) {
        return fallbackAi.chooseRollToIgnore(rolls);
    }

    @Override
    public List<Integer> chooseDiceToReroll(List<Integer> rolls) {
        return fallbackAi.chooseDiceToReroll(rolls);
    }

    @Override
    public Integer chooseRollToModify(List<Integer> rolls) {
        return fallbackAi.chooseRollToModify(rolls);
    }

    @Override
    public RollDiceEffect.DieRollResult chooseRollToSwap(List<RollDiceEffect.DieRollResult> rolls) {
        return fallbackAi.chooseRollToSwap(rolls);
    }

    @Override
    public String chooseRollSwapValue(List<String> swapChoices, Integer currentResult, int power, int toughness) {
        return fallbackAi.chooseRollSwapValue(swapChoices, currentResult, power, toughness);
    }

    // ===== UI / NOTIFICATION (no-ops for AI) =====

    @Override
    public void reveal(CardCollectionView cards, ZoneType zone, Player owner, String messagePrefix, boolean addMsgSuffix) {
        // AI doesn't need visual reveals
    }

    @Override
    public void reveal(List<CardView> cards, ZoneType zone, PlayerView owner, String messagePrefix, boolean addMsgSuffix) {
        // AI doesn't need visual reveals
    }

    @Override
    public void notifyOfValue(SpellAbility saSource, GameObject realtedTarget, String value) {
        // AI processes revealed information internally
    }

    @Override
    public void revealAnte(String message, Multimap<Player, PaperCard> removedAnteCards) {
        // No UI needed
    }

    @Override
    public void revealAISkipCards(String message, Map<Player, Map<DeckSection, List<? extends PaperCard>>> deckCards) {
        // No UI needed
    }

    @Override
    public void revealUnsupported(Map<Player, List<PaperCard>> unsupported) {
        // No UI needed
    }

    @Override
    public void resetAtEndOfTurn() {
        fallbackAi.resetAtEndOfTurn();
    }

    @Override
    public void autoPassCancel() {
        // No auto-pass for AI
    }

    @Override
    public void awaitNextInput() {
        // AI doesn't wait for input
    }

    @Override
    public void cancelAwaitNextInput() {
        // AI doesn't wait for input
    }
}
