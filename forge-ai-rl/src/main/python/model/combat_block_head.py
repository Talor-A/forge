"""
Block Declaration Head — Assigns blockers to attackers.

This is an assignment problem: each blocker can block at most one attacker,
and each attacker can be blocked by multiple blockers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockHead(nn.Module):
    """
    For each (blocker, attacker) pair, outputs a probability of that assignment.
    Uses attention between blockers and attackers to make coordinated decisions.
    """

    def __init__(self, state_dim: int = 512, card_feature_dim: int = 256,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.blocker_projection = nn.Linear(card_feature_dim, hidden_dim)
        self.attacker_projection = nn.Linear(card_feature_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Cross-attention: blockers attend to attackers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Assignment scorer: for each blocker, score each attacker + "don't block"
        self.assignment_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, game_state: torch.Tensor,
                blocker_features: torch.Tensor, blocker_mask: torch.Tensor,
                attacker_features: torch.Tensor, attacker_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            game_state: (batch, state_dim)
            blocker_features: (batch, max_blockers, card_feature_dim)
            blocker_mask: (batch, max_blockers)
            attacker_features: (batch, max_attackers, card_feature_dim)
            attacker_mask: (batch, max_attackers)

        Returns:
            assignment_logits: (batch, max_blockers, max_attackers + 1)
            The +1 is for "don't block" option.
        """
        blockers = self.blocker_projection(blocker_features)
        attackers = self.attacker_projection(attacker_features)
        state = self.state_projection(game_state)

        # Cross attention: blockers attend to attackers
        attn_out, _ = self.cross_attention(
            blockers, attackers, attackers,
            key_padding_mask=~attacker_mask
        )

        max_blockers = blockers.shape[1]
        max_attackers = attackers.shape[1]
        batch_size = blockers.shape[0]

        # For each (blocker, attacker) pair, compute assignment score
        # Expand blockers and attackers for pairwise computation
        blockers_exp = attn_out.unsqueeze(2).expand(-1, -1, max_attackers, -1)
        attackers_exp = attackers.unsqueeze(1).expand(-1, max_blockers, -1, -1)
        state_exp = state.unsqueeze(1).unsqueeze(2).expand(-1, max_blockers, max_attackers, -1)

        combined = torch.cat([blockers_exp, attackers_exp, state_exp], dim=-1)
        pair_scores = self.assignment_scorer(combined).squeeze(-1)  # (batch, max_blockers, max_attackers)

        # Add "don't block" option score (learned bias)
        no_block_score = torch.zeros(batch_size, max_blockers, 1, device=pair_scores.device)
        logits = torch.cat([pair_scores, no_block_score], dim=-1)  # (batch, max_blockers, max_attackers + 1)

        # Mask invalid attackers (keep "don't block" always valid)
        extended_mask = torch.cat([attacker_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=attacker_mask.device)], dim=-1)
        logits = logits.masked_fill(~extended_mask.unsqueeze(1), float('-inf'))

        return logits

    def sample_assignments(self, game_state, blocker_features, blocker_mask,
                           attacker_features, attacker_mask):
        """
        Sample blocking assignments. For each blocker, independently choose
        which attacker to block (or don't block).

        Returns (assignments, log_probs):
            assignments: (batch, max_blockers) — index of assigned attacker, or max_attackers for "no block"
            log_probs: (batch,) — total log probability
        """
        logits = self.forward(game_state, blocker_features, blocker_mask,
                              attacker_features, attacker_mask)

        total_log_prob = torch.zeros(logits.shape[0], device=logits.device)
        assignments = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long, device=logits.device)

        for i in range(logits.shape[1]):  # for each blocker
            dist = torch.distributions.Categorical(logits=logits[:, i, :])
            action = dist.sample()
            assignments[:, i] = action
            # Only count log prob for valid blockers
            total_log_prob += dist.log_prob(action) * blocker_mask[:, i].float()

        return assignments, total_log_prob
