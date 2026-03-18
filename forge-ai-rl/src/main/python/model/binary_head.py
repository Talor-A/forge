"""
Binary Decision Head — Simple yes/no decisions.
Used for confirmAction, confirmTrigger, and similar binary choices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryHead(nn.Module):
    """
    Maps game state to a binary yes/no decision.
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        """Returns logit: positive = yes, negative = no."""
        return self.network(game_state).squeeze(-1)

    def decide(self, game_state: torch.Tensor) -> tuple:
        """Sample a decision. Returns (decision: bool, log_prob, entropy)."""
        logit = self.forward(game_state)
        prob = torch.sigmoid(logit)
        decision = torch.bernoulli(prob).bool()
        log_prob = F.logsigmoid(logit) * decision.float() + F.logsigmoid(-logit) * (~decision).float()
        entropy = -(prob * F.logsigmoid(logit) + (1 - prob) * F.logsigmoid(-logit))
        return decision, log_prob, entropy
