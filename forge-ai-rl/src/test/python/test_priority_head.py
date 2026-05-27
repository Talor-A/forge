"""Unit tests for PriorityHead (priority_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_priority_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_priority_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.priority_head import PriorityHead

STATE_DIM = 64
ACTION_DIM = 32
HIDDEN_DIM = 16


def _build(seed=0):
    torch.manual_seed(seed)
    return PriorityHead(
        state_dim=STATE_DIM,
        action_feature_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dropout=0.0,
    )


def _inputs(batch=2, n_actions=5, seed=1):
    torch.manual_seed(seed)
    gs = torch.randn(batch, STATE_DIM)
    af = torch.randn(batch, n_actions, ACTION_DIM)
    mask = torch.ones(batch, n_actions, dtype=torch.bool)
    return gs, af, mask


class TestPriorityHeadOutputShape(unittest.TestCase):
    def test_forward_shape(self):
        net = _build()
        gs, af, mask = _inputs(batch=3, n_actions=6)
        logits = net(gs, af, mask)
        self.assertEqual(logits.shape, (3, 6))

    def test_probs_shape_and_sum_to_one(self):
        net = _build()
        gs, af, mask = _inputs(batch=4, n_actions=5)
        probs = net.get_action_probs(gs, af, mask)
        self.assertEqual(probs.shape, (4, 5))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5),
                        "action probabilities do not sum to 1")

    def test_sample_shapes(self):
        net = _build()
        gs, af, mask = _inputs(batch=4, n_actions=5)
        action, log_prob, entropy = net.sample_action(gs, af, mask)
        self.assertEqual(action.shape, (4,))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))


class TestPriorityHeadMasking(unittest.TestCase):
    def test_invalid_actions_get_neg_inf(self):
        net = _build()
        gs, af, mask = _inputs(batch=2, n_actions=5)
        mask[:, 3:] = False
        logits = net(gs, af, mask)
        self.assertTrue((logits[:, 3:] == float('-inf')).all(),
                        "masked action slots should be -inf")

    def test_valid_logits_are_finite(self):
        net = _build()
        gs, af, mask = _inputs(batch=2, n_actions=5)
        mask[:, 3:] = False
        logits = net(gs, af, mask)
        self.assertTrue(torch.isfinite(logits[:, :3]).all(),
                        "valid action logits should be finite")

    def test_single_valid_action_always_selected(self):
        """When only one action is valid, it must be sampled with certainty."""
        net = _build()
        torch.manual_seed(0)
        gs, af, mask = _inputs(batch=8, n_actions=4)
        mask[:] = False
        mask[:, 2] = True  # only action 2 is valid

        action, log_prob, _ = net.sample_action(gs, af, mask)
        self.assertTrue((action == 2).all(), "only valid action was not selected")
        self.assertTrue(torch.allclose(log_prob, torch.zeros(8), atol=1e-5),
                        "log_prob for deterministic choice must be 0")

    def test_sampled_action_always_in_valid_range(self):
        net = _build()
        torch.manual_seed(10)
        gs, af, mask = _inputs(batch=16, n_actions=6)
        mask[:, 4:] = False
        action, _, _ = net.sample_action(gs, af, mask)
        self.assertTrue((action < 4).all(), "sampled a masked action")

    def test_entropy_non_negative(self):
        net = _build()
        gs, af, mask = _inputs(batch=8, n_actions=5)
        _, _, entropy = net.sample_action(gs, af, mask)
        self.assertTrue((entropy >= 0).all())


class TestPriorityHeadGradients(unittest.TestCase):
    def test_gradients_flow_to_every_param(self):
        net = _build()
        gs, af, mask = _inputs()
        logits = net(gs, af, mask)
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")


class TestPriorityHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must learn to always pick action 0 on a fixed input."""
        net = _build(seed=0)
        net.train()
        gs, af, mask = _inputs(batch=8, n_actions=4, seed=2)
        target = torch.zeros(8, dtype=torch.long)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            return F.cross_entropy(net(gs, af, mask), target)

        initial = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final = loss_fn().item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.05,
                        f"priority head failed to overfit: {initial:.4f} -> {final:.4f}")

    def test_determinism(self):
        gs, af, mask = _inputs(batch=3, n_actions=4, seed=5)
        def run():
            net = _build(seed=42)
            net.eval()
            with torch.no_grad():
                return net(gs, af, mask).clone()
        self.assertTrue(torch.allclose(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
