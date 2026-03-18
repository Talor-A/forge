"""
Target Selection Head — Pointer network that selects targets from a candidate set.
Handles both single-target and multi-target selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetHead(nn.Module):
    """
    Pointer network for selecting targets. For multi-target, uses autoregressive
    selection (pick one at a time, conditioning on previous picks).
    """

    def __init__(self, state_dim: int = 512, target_feature_dim: int = 64,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.target_projection = nn.Linear(target_feature_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Pointer attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Context update for multi-target (GRU updates the query after each selection)
        self.context_gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.scale = hidden_dim ** 0.5

    def forward(self, game_state: torch.Tensor, target_features: torch.Tensor,
                target_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for single-target selection.

        Args:
            game_state: (batch, state_dim)
            target_features: (batch, max_targets, target_feature_dim)
            target_mask: (batch, max_targets)

        Returns:
            target_logits: (batch, max_targets)
        """
        targets = self.target_projection(target_features)  # (batch, max_targets, hidden_dim)
        state = self.state_projection(game_state)  # (batch, hidden_dim)

        query = self.query_proj(state).unsqueeze(1)  # (batch, 1, hidden_dim)
        keys = self.key_proj(targets)  # (batch, max_targets, hidden_dim)

        # Pointer attention scores
        logits = (query * keys).sum(dim=-1) / self.scale  # (batch, max_targets)
        logits = logits.squeeze(1) if logits.dim() == 3 else logits
        logits = logits.masked_fill(~target_mask, float('-inf'))

        return logits

    def select_multiple(self, game_state: torch.Tensor, target_features: torch.Tensor,
                        target_mask: torch.Tensor, num_selections: int) -> list:
        """
        Autoregressive multi-target selection.

        Returns list of (action_index, log_prob) tuples.
        """
        targets = self.target_projection(target_features)
        state = self.state_projection(game_state)  # (batch, hidden_dim)
        keys = self.key_proj(targets)

        selections = []
        current_mask = target_mask.clone()
        context = state

        for _ in range(num_selections):
            query = self.query_proj(context).unsqueeze(1)
            logits = (query * keys).sum(dim=-1).squeeze(1) / self.scale
            logits = logits.masked_fill(~current_mask, float('-inf'))

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            selections.append((action, log_prob))

            # Update context with selected target
            selected_target = targets[torch.arange(targets.size(0)), action]
            context = self.context_gru(selected_target, context)

            # Remove selected from mask
            current_mask = current_mask.scatter(1, action.unsqueeze(1), False)

            # Stop if no valid targets remain
            if not current_mask.any():
                break

        return selections
