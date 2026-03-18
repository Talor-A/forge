"""
Benchmark — Evaluate RL agent performance against heuristic AI.

Provides standardized benchmarking and Elo estimation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    total_games: int = 0
    rl_wins: int = 0
    heuristic_wins: int = 0
    draws: int = 0
    errors: int = 0
    avg_game_duration_ms: float = 0.0
    avg_game_turns: float = 0.0
    rl_win_rate: float = 0.0
    estimated_elo_diff: float = 0.0

    # Per-matchup results
    matchup_results: dict = field(default_factory=dict)

    def compute_elo_diff(self):
        """Estimate Elo difference from win rate."""
        if self.total_games == 0:
            self.estimated_elo_diff = 0
            return
        wr = self.rl_win_rate
        if wr <= 0 or wr >= 1:
            self.estimated_elo_diff = 400 * (1 if wr > 0.5 else -1)
        else:
            import math
            self.estimated_elo_diff = -400 * math.log10(1 / wr - 1)

    def __str__(self):
        return (f"Benchmark: {self.total_games} games | "
                f"RL wins: {self.rl_wins} ({self.rl_win_rate:.1%}) | "
                f"Heuristic wins: {self.heuristic_wins} | "
                f"Errors: {self.errors} | "
                f"Est. Elo diff: {self.estimated_elo_diff:+.0f} | "
                f"Avg duration: {self.avg_game_duration_ms:.0f}ms")


class Benchmarker:
    """
    Runs standardized benchmarks of the RL agent.

    Benchmark suite includes:
    - Aggro vs Aggro
    - Aggro vs Control
    - Control vs Control
    - Midrange vs Midrange
    - RL as both player 1 and player 2
    """

    def __init__(self):
        self.results_history: List[BenchmarkResult] = []

    def create_result(self, total_games: int, rl_wins: int, heuristic_wins: int,
                      errors: int = 0, avg_duration: float = 0.0) -> BenchmarkResult:
        """Create a benchmark result from game statistics."""
        result = BenchmarkResult(
            total_games=total_games,
            rl_wins=rl_wins,
            heuristic_wins=heuristic_wins,
            errors=errors,
            avg_game_duration_ms=avg_duration,
            rl_win_rate=rl_wins / max(1, rl_wins + heuristic_wins),
        )
        result.compute_elo_diff()
        self.results_history.append(result)
        return result

    def print_progress(self):
        """Print training progress over time."""
        if not self.results_history:
            logger.info("No benchmark results yet")
            return

        logger.info("=== Training Progress ===")
        for i, result in enumerate(self.results_history):
            logger.info(f"  Benchmark {i + 1}: {result}")

        # Show trend
        if len(self.results_history) >= 2:
            first_wr = self.results_history[0].rl_win_rate
            last_wr = self.results_history[-1].rl_win_rate
            logger.info(f"  Win rate trend: {first_wr:.1%} → {last_wr:.1%} "
                        f"({'improving' if last_wr > first_wr else 'declining'})")
