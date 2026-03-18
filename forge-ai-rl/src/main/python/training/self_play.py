"""
Self-Play Manager — Orchestrates self-play training with population-based training.

Manages a population of agents that play against each other and against historical
snapshots to prevent forgetting and strategy cycling.
"""

import os
import time
import logging
import random
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mtg_model import MTGModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentSnapshot:
    """A saved checkpoint of an agent at a point in training."""
    path: str
    elo: float = 1200.0
    games_played: int = 0
    creation_step: int = 0
    agent_type: str = "main"  # main, exploiter, league_exploiter


@dataclass
class EloTracker:
    """Tracks Elo ratings for a population of agents."""
    K: float = 32.0  # Elo K-factor
    ratings: dict = field(default_factory=dict)

    def update(self, winner_id: str, loser_id: str):
        """Update Elo ratings after a game."""
        if winner_id not in self.ratings:
            self.ratings[winner_id] = 1200.0
        if loser_id not in self.ratings:
            self.ratings[loser_id] = 1200.0

        expected_win = 1.0 / (1.0 + 10 ** ((self.ratings[loser_id] - self.ratings[winner_id]) / 400))
        self.ratings[winner_id] += self.K * (1.0 - expected_win)
        self.ratings[loser_id] += self.K * (0.0 - (1.0 - expected_win))

    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, 1200.0)


class SelfPlayManager:
    """
    Manages AlphaStar-style league training:
    - Main agents: train against all opponents
    - Exploiter agents: specifically target weaknesses in main agents
    - League exploiters: target the full history
    """

    def __init__(self, checkpoint_dir: str = "rl_data/league",
                 num_main_agents: int = 3, num_exploiters: int = 2,
                 snapshot_interval: int = 1000):
        self.checkpoint_dir = checkpoint_dir
        self.num_main_agents = num_main_agents
        self.num_exploiters = num_exploiters
        self.snapshot_interval = snapshot_interval

        self.snapshots: List[AgentSnapshot] = []
        self.elo_tracker = EloTracker()
        self.games_played = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_snapshot(self, model: MTGModel, agent_id: str, step: int,
                      agent_type: str = "main") -> AgentSnapshot:
        """Save a model checkpoint as a league snapshot."""
        path = os.path.join(self.checkpoint_dir, f"{agent_type}_{agent_id}_step{step}.pt")
        model.save(path)

        snapshot = AgentSnapshot(
            path=path,
            elo=self.elo_tracker.get_rating(agent_id),
            creation_step=step,
            agent_type=agent_type,
        )
        self.snapshots.append(snapshot)
        logger.info(f"Saved snapshot: {agent_id} (Elo: {snapshot.elo:.0f}, step: {step})")
        return snapshot

    def choose_opponent(self, agent_type: str = "main") -> Optional[AgentSnapshot]:
        """
        Choose an opponent based on agent type:
        - main: uniform random from all snapshots
        - exploiter: weighted toward recent main agent snapshots
        - league_exploiter: weighted toward all historical snapshots
        """
        if not self.snapshots:
            return None

        if agent_type == "main":
            return random.choice(self.snapshots)
        elif agent_type == "exploiter":
            # Prefer recent main agents
            main_snapshots = [s for s in self.snapshots if s.agent_type == "main"]
            if not main_snapshots:
                return random.choice(self.snapshots)
            # Weight by recency
            weights = [i + 1 for i in range(len(main_snapshots))]
            return random.choices(main_snapshots, weights=weights, k=1)[0]
        else:  # league_exploiter
            # Uniform over all history
            return random.choice(self.snapshots)

    def record_result(self, winner_id: str, loser_id: str):
        """Record a game result and update Elo ratings."""
        self.elo_tracker.update(winner_id, loser_id)
        self.games_played += 1

        if self.games_played % 100 == 0:
            self._log_standings()

    def _log_standings(self):
        """Log current Elo standings."""
        sorted_ratings = sorted(self.elo_tracker.ratings.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"=== League Standings (after {self.games_played} games) ===")
        for agent_id, elo in sorted_ratings[:10]:
            logger.info(f"  {agent_id}: {elo:.0f}")

    def get_league_stats(self) -> dict:
        """Get league statistics."""
        return {
            'total_games': self.games_played,
            'num_snapshots': len(self.snapshots),
            'ratings': dict(self.elo_tracker.ratings),
            'top_agent': max(self.elo_tracker.ratings.items(), key=lambda x: x[1])[0]
            if self.elo_tracker.ratings else None,
        }
