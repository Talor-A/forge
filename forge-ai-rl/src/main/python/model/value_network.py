"""
Value Network — Estimates the win probability from a game state embedding.
Used as the critic in PPO and for MCTS-style evaluation.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Simple MLP that maps game state embedding → scalar win probability.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1]: -1 = certain loss, +1 = certain win
        )

    def forward(self, game_state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            game_state_embedding: (batch, input_dim)
        Returns:
            value: (batch, 1) — estimated advantage in [-1, 1]
        """
        return self.network(game_state_embedding)
