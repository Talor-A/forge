"""Unit tests for AttackHead (combat_attack_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_combat_attack_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_combat_attack_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.combat_attack_head import AttackHead

STATE_DIM = 64
CARD_DIM = 32
HIDDEN_DIM = 16


def _build(seed=0):
    torch.manual_seed(seed)
    return AttackHead(
        state_dim=STATE_DIM,
        card_feature_dim=CARD_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dropout=0.0,
    )


def _inputs(batch=2, n_creatures=4, seed=1):
    torch.manual_seed(seed)
    gs = torch.randn(batch, STATE_DIM)
    cf = torch.randn(batch, n_creatures, CARD_DIM)
    mask = torch.ones(batch, n_creatures, dtype=torch.bool)
    return gs, cf, mask


class TestAttackHeadOutputShape(unittest.TestCase):
    def test_forward_shape(self):
        net = _build()
        gs, cf, mask = _inputs(batch=3, n_creatures=5)
        logits = net(gs, cf, mask)
        self.assertEqual(logits.shape, (3, 5))

    def test_sample_shapes(self):
        net = _build()
        gs, cf, mask = _inputs(batch=4, n_creatures=6)
        decisions, log_probs, entropy = net.sample_attacks(gs, cf, mask)
        self.assertEqual(decisions.shape, (4, 6))
        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(entropy.shape, (4,))

    def test_decisions_are_bool(self):
        net = _build()
        gs, cf, mask = _inputs()
        decisions, _, _ = net.sample_attacks(gs, cf, mask)
        self.assertEqual(decisions.dtype, torch.bool)


class TestAttackHeadMasking(unittest.TestCase):
    def test_padded_creatures_always_not_attacking(self):
        """Decisions for padding creatures must always be False."""
        net = _build()
        gs, cf, mask = _inputs(batch=8, n_creatures=5)
        mask[:, 3:] = False  # last 2 slots are padding

        torch.manual_seed(42)
        decisions, _, _ = net.sample_attacks(gs, cf, mask)

        self.assertTrue((~decisions[:, 3:]).all(),
                        "padded creature slots sampled as attacking")

    def test_padded_creatures_zero_log_prob_contribution(self):
        """Masking out a creature must not change the total log_prob."""
        net = _build(seed=0)
        gs, cf, full_mask = _inputs(batch=4, n_creatures=4)

        partial_mask = full_mask.clone()
        partial_mask[:, -1] = False

        torch.manual_seed(99)
        _, lp_full, _ = net.sample_attacks(gs, cf, full_mask)
        torch.manual_seed(99)
        _, lp_partial, _ = net.sample_attacks(gs, cf, partial_mask)

        # Removing a creature removes its log_prob term; full sum must be <= partial
        # (non-positive terms: removing one makes total less negative i.e. larger)
        self.assertTrue((lp_partial >= lp_full - 1e-5).all(),
                        "masking a creature lowered total log_prob")

    def test_padded_logits_are_strongly_negative(self):
        """Padding positions get -100.0 logit (not -inf), which keeps gradients
        flowing but ensures near-zero attack probability."""
        net = _build()
        gs, cf, mask = _inputs(batch=2, n_creatures=4)
        mask[:, 2:] = False
        logits = net(gs, cf, mask)
        self.assertTrue((logits[:, 2:] == -100.0).all(),
                        "padded logits should be -100.0")

    def test_log_probs_non_positive(self):
        """Total log_prob must be <= 0."""
        net = _build()
        for seed in range(5):
            gs, cf, mask = _inputs(batch=4, n_creatures=5, seed=seed)
            _, log_probs, _ = net.sample_attacks(gs, cf, mask)
            self.assertTrue((log_probs <= 0).all(), f"log_prob > 0 for seed={seed}")


class TestAttackHeadGradients(unittest.TestCase):
    def test_gradients_flow_to_every_param(self):
        net = _build()
        gs, cf, mask = _inputs()
        logits = net(gs, cf, mask)
        target = torch.ones_like(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits.masked_fill(~mask, 0.0), target * mask.float()
        )
        loss.backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")


class TestAttackHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must learn to always attack with all creatures on a fixed batch.

        Note: logits are clamped to [-5, 5] so sigmoid saturates at ~0.993, not 1.0.
        BCE cannot reach 0; threshold of 0.1 accommodates this intentional cap.
        """
        net = _build(seed=0)
        net.train()
        gs, cf, mask = _inputs(batch=8, n_creatures=3, seed=2)
        target = mask.float()  # attack with everyone
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            logits = net(gs, cf, mask)
            return F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / mask.sum()

        initial = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final = loss_fn().item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.1,
                        f"attack head failed to overfit: {initial:.4f} -> {final:.4f}")

    def test_determinism(self):
        gs, cf, mask = _inputs(batch=3, n_creatures=4, seed=5)
        def run():
            net = _build(seed=42)
            net.eval()
            with torch.no_grad():
                return net(gs, cf, mask).clone()
        self.assertTrue(torch.allclose(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
