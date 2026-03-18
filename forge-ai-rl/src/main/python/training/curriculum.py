"""
Curriculum Learning — Manages progressive complexity increase in training.

Starts with simple card pools and gradually introduces more complex mechanics.
"""

import logging
from dataclasses import dataclass
from typing import List, Set

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """A stage in the curriculum with a defined card pool and complexity level."""
    name: str
    description: str
    card_pool_tags: List[str]  # tags like "vanilla_creatures", "keywords_basic", etc.
    min_win_rate: float  # win rate threshold to advance to next stage
    min_games: int  # minimum games before advancement is considered
    complexity_level: int


# Pre-defined curriculum stages
CURRICULUM_STAGES = [
    CurriculumStage(
        name="Stage A: Vanilla Creatures",
        description="Creatures with no abilities. Learn combat math, mana curve, life management.",
        card_pool_tags=["vanilla_creatures", "basic_lands"],
        min_win_rate=0.60,
        min_games=5000,
        complexity_level=1,
    ),
    CurriculumStage(
        name="Stage B: Keywords",
        description="Add flying, trample, first strike, deathtouch, lifelink, haste, vigilance.",
        card_pool_tags=["vanilla_creatures", "keyword_creatures", "basic_lands"],
        min_win_rate=0.58,
        min_games=10000,
        complexity_level=2,
    ),
    CurriculumStage(
        name="Stage C: Removal & Combat Tricks",
        description="Add instant-speed removal, pump spells, combat tricks.",
        card_pool_tags=["vanilla_creatures", "keyword_creatures", "removal_spells",
                        "combat_tricks", "basic_lands"],
        min_win_rate=0.56,
        min_games=15000,
        complexity_level=3,
    ),
    CurriculumStage(
        name="Stage D: Card Draw & Counters",
        description="Add card draw, counterspells, stack interaction.",
        card_pool_tags=["vanilla_creatures", "keyword_creatures", "removal_spells",
                        "combat_tricks", "card_draw", "counterspells", "basic_lands"],
        min_win_rate=0.55,
        min_games=20000,
        complexity_level=4,
    ),
    CurriculumStage(
        name="Stage E: Complex Permanents",
        description="Add enchantments, artifacts, planeswalkers, activated abilities.",
        card_pool_tags=["all_creatures", "removal_spells", "combat_tricks", "card_draw",
                        "counterspells", "enchantments", "artifacts", "planeswalkers",
                        "basic_lands", "dual_lands"],
        min_win_rate=0.54,
        min_games=30000,
        complexity_level=5,
    ),
    CurriculumStage(
        name="Stage F: Full Card Pool",
        description="Full Standard/Modern card pool with all mechanics.",
        card_pool_tags=["all"],
        min_win_rate=0.52,
        min_games=50000,
        complexity_level=6,
    ),
]


class CurriculumManager:
    """Manages curriculum progression during training."""

    def __init__(self, stages: List[CurriculumStage] = None):
        self.stages = stages or CURRICULUM_STAGES
        self.current_stage_index = 0
        self.stage_games_played = 0
        self.stage_wins = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_index]

    def record_game(self, won: bool):
        """Record a game result for the current stage."""
        self.stage_games_played += 1
        if won:
            self.stage_wins += 1

    def should_advance(self) -> bool:
        """Check if we should advance to the next stage."""
        if self.current_stage_index >= len(self.stages) - 1:
            return False  # Already at max stage

        stage = self.current_stage
        if self.stage_games_played < stage.min_games:
            return False

        win_rate = self.stage_wins / max(1, self.stage_games_played)
        return win_rate >= stage.min_win_rate

    def advance(self) -> bool:
        """Advance to the next stage. Returns True if advanced."""
        if not self.should_advance():
            return False

        win_rate = self.stage_wins / max(1, self.stage_games_played)
        logger.info(f"Advancing from {self.current_stage.name} "
                    f"(win rate: {win_rate:.2%}, games: {self.stage_games_played})")

        self.current_stage_index += 1
        self.stage_games_played = 0
        self.stage_wins = 0

        logger.info(f"Now at {self.current_stage.name}: {self.current_stage.description}")
        return True

    def get_card_pool_tags(self) -> List[str]:
        """Get the card pool tags for the current stage."""
        return self.current_stage.card_pool_tags

    def get_stats(self) -> dict:
        """Get current curriculum statistics."""
        win_rate = self.stage_wins / max(1, self.stage_games_played)
        return {
            'current_stage': self.current_stage.name,
            'stage_index': self.current_stage_index,
            'games_played': self.stage_games_played,
            'win_rate': win_rate,
            'games_to_advance': max(0, self.current_stage.min_games - self.stage_games_played),
            'win_rate_needed': self.current_stage.min_win_rate,
        }
