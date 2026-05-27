"""
Priority Action Head — Chooses which spell/ability to play or passes priority.

Uses cross-attention between the game state and available actions to produce
a probability distribution over actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorityHead(nn.Module):
    """
    Given a game state embedding and a set of available actions (encoded as feature vectors),
    outputs a probability distribution over actions including a "pass" option.
    """

    def __init__(self, state_dim: int = 512, action_feature_dim: int = 64,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Project actions to hidden dim
        self.action_projection = nn.Linear(action_feature_dim, hidden_dim)

        # Project game state to hidden dim for cross-attention
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Cross-attention: actions attend to game state
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Score each action
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, game_state: torch.Tensor, action_features: torch.Tensor,
                action_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            game_state: (batch, state_dim)
            action_features: (batch, max_actions, action_feature_dim)
            action_mask: (batch, max_actions) — True for valid actions

        Returns:
            action_logits: (batch, max_actions) — logits for each action (masked)
        """
        batch_size, max_actions, _ = action_features.shape

        # Project
        actions = self.action_projection(action_features)  # (batch, max_actions, hidden_dim)
        state = self.state_projection(game_state).unsqueeze(1)  # (batch, 1, hidden_dim)

        # Cross-attention: actions query the single game-state token as key/value
        attn_out, _ = self.cross_attention(actions, state, state)

        # Combine attention output with original action features
        combined = torch.cat([attn_out, actions], dim=-1)  # (batch, max_actions, hidden_dim * 2)

        # Score each action
        logits = self.score_network(combined).squeeze(-1)  # (batch, max_actions)

        # Mask invalid actions with large negative value
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits

    def get_action_probs(self, game_state: torch.Tensor, action_features: torch.Tensor,
                         action_mask: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax of logits)."""
        logits = self.forward(game_state, action_features, action_mask)
        return F.softmax(logits, dim=-1)

    def sample_action(self, game_state: torch.Tensor, action_features: torch.Tensor,
                      action_mask: torch.Tensor) -> tuple:
        """
        Sample an action from the policy distribution.
        Returns (action_index, log_probability, entropy).
        """
        logits = self.forward(game_state, action_features, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
