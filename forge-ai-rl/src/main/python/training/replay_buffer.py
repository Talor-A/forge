"""
Replay Buffer — Stores and samples game trajectories for training.
Supports loading trajectories from JSONL files produced by the Java TrajectoryRecorder.
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class TrajectoryStep:
    """A single decision step from a game trajectory."""
    __slots__ = ['decision_type', 'global_features', 'game_state_flat', 'candidate_features',
                 'selected_indices', 'action_probabilities', 'value_estimate',
                 'intermediate_reward', 'terminal_reward', 'used_fallback']

    def __init__(self, data: dict):
        self.decision_type = data.get('decisionType', '')
        self.global_features = np.array(data.get('globalFeatures', []), dtype=np.float32)
        self.game_state_flat = np.array(data.get('gameStateFlat', []), dtype=np.float32)
        self.candidate_features = np.array(data.get('candidateFeatures', []), dtype=np.float32)
        self.selected_indices = data.get('selectedIndices', [])
        self.action_probabilities = np.array(data.get('actionProbabilities', []), dtype=np.float32)
        self.value_estimate = data.get('valueEstimate', 0.0)
        self.intermediate_reward = data.get('intermediateReward', 0.0)
        self.terminal_reward = data.get('terminalReward', 0.0)
        self.used_fallback = data.get('usedFallback', True)


class GameTrajectory:
    """A complete game trajectory (sequence of decision steps)."""

    def __init__(self, game_id: str, won: bool, steps: List[TrajectoryStep]):
        self.game_id = game_id
        self.won = won
        self.steps = steps

    def compute_returns(self, gamma: float = 0.999) -> List[float]:
        """Compute discounted returns for each step."""
        returns = []
        G = 0.0
        for step in reversed(self.steps):
            reward = step.intermediate_reward + step.terminal_reward
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def __len__(self):
        return len(self.steps)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling game trajectories.
    Supports loading from disk and prioritized sampling.
    """

    def __init__(self, max_trajectories: int = 100000):
        self.max_trajectories = max_trajectories
        self.trajectories: List[GameTrajectory] = []
        self.win_trajectories: List[int] = []  # indices of winning trajectories
        self.loss_trajectories: List[int] = []  # indices of losing trajectories

    def load_from_directory(self, directory: str, max_files: int = None) -> int:
        """
        Load trajectories from JSONL files in a directory.
        Returns the number of trajectories loaded.
        """
        path = Path(directory)
        if not path.exists():
            return 0

        files = sorted(path.glob('traj_*.jsonl'))
        if max_files:
            files = files[:max_files]

        loaded = 0
        for filepath in files:
            try:
                traj = self._load_trajectory(filepath)
                if traj and len(traj.steps) > 0:
                    self._add_trajectory(traj)
                    loaded += 1
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return loaded

    def _load_trajectory(self, filepath: Path) -> Optional[GameTrajectory]:
        """Load a single trajectory from a JSONL file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None

            # First line is header
            header = json.loads(lines[0])
            game_id = header.get('gameId', '')
            won = header.get('won', False)

            # Remaining lines are decision records
            steps = []
            for line in lines[1:]:
                data = json.loads(line)
                steps.append(TrajectoryStep(data))

            return GameTrajectory(game_id, won, steps)

    def _add_trajectory(self, traj: GameTrajectory):
        """Add a trajectory to the buffer."""
        idx = len(self.trajectories)
        self.trajectories.append(traj)

        if traj.won:
            self.win_trajectories.append(idx)
        else:
            self.loss_trajectories.append(idx)

        # Evict oldest if over capacity
        while len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
            # Rebuild index lists
            self.win_trajectories = [i for i, t in enumerate(self.trajectories) if t.won]
            self.loss_trajectories = [i for i, t in enumerate(self.trajectories) if not t.won]

    def sample_trajectories(self, batch_size: int, balanced: bool = True) -> List[GameTrajectory]:
        """
        Sample a batch of trajectories.
        If balanced=True, try to sample equal numbers of wins and losses.
        """
        if not self.trajectories:
            return []

        if balanced and self.win_trajectories and self.loss_trajectories:
            half = batch_size // 2
            win_sample = random.choices(self.win_trajectories, k=half)
            loss_sample = random.choices(self.loss_trajectories, k=batch_size - half)
            indices = win_sample + loss_sample
            random.shuffle(indices)
            return [self.trajectories[i] for i in indices]
        else:
            return random.choices(self.trajectories, k=batch_size)

    def sample_steps(self, batch_size: int, decision_type: str = None) -> List[TrajectoryStep]:
        """Sample individual steps (for supervised learning on specific decision types)."""
        all_steps = []
        for traj in self.trajectories:
            for step in traj.steps:
                if decision_type is None or step.decision_type == decision_type:
                    all_steps.append(step)

        if not all_steps:
            return []
        return random.choices(all_steps, k=min(batch_size, len(all_steps)))

    def stats(self) -> dict:
        """Return buffer statistics."""
        total_steps = sum(len(t) for t in self.trajectories)
        return {
            'total_trajectories': len(self.trajectories),
            'total_steps': total_steps,
            'wins': len(self.win_trajectories),
            'losses': len(self.loss_trajectories),
            'win_rate': len(self.win_trajectories) / max(1, len(self.trajectories)),
        }
