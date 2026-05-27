"""Unit tests for BinaryHead (binary_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_binary_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_binary_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.binary_head import BinaryHead

STATE_DIM = 64
HIDDEN_DIM = 32


def _build(seed=0):
    torch.manual_seed(seed)
    return BinaryHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)


def _inputs(batch=4, seed=1):
    torch.manual_seed(seed)
    return torch.randn(batch, STATE_DIM)


class TestBinaryHeadOutputShape(unittest.TestCase):
    def test_forward_shape(self):
        """forward returns one logit per sample."""
        net = _build()
        gs = _inputs(batch=4)
        logits = net(gs)
        self.assertEqual(logits.shape, (4,))

    def test_forward_scalar_unbatched(self):
        """1-sample batch gives a 1-element tensor, not a scalar."""
        net = _build()
        logits = net(_inputs(batch=1))
        self.assertEqual(logits.shape, (1,))

    def test_decide_shapes(self):
        net = _build()
        gs = _inputs(batch=6)
        decision, log_prob, entropy = net.decide(gs)
        self.assertEqual(decision.shape, (6,))
        self.assertEqual(log_prob.shape, (6,))
        self.assertEqual(entropy.shape, (6,))

    def test_decision_is_bool(self):
        net = _build()
        decision, _, _ = net.decide(_inputs(batch=4))
        self.assertEqual(decision.dtype, torch.bool)


class TestBinaryHeadLogProbs(unittest.TestCase):
    def test_log_prob_matches_true_decision(self):
        """When decision=True, log_prob == log_sigmoid(logit)."""
        net = _build()
        gs = _inputs(batch=1)
        logit = net(gs)

        prob = torch.sigmoid(logit)
        forced_yes = torch.tensor([True])
        lp_yes = F.logsigmoid(logit) * forced_yes.float() + F.logsigmoid(-logit) * (~forced_yes).float()
        self.assertAlmostEqual(lp_yes.item(), F.logsigmoid(logit).item(), places=5)

    def test_log_prob_is_non_positive(self):
        """log P(decision) <= 0 always."""
        net = _build()
        for seed in range(5):
            _, log_prob, _ = net.decide(_inputs(batch=8, seed=seed))
            self.assertTrue((log_prob <= 0).all(), f"log_prob > 0 for seed={seed}")

    def test_entropy_non_negative(self):
        """Bernoulli entropy is always >= 0."""
        net = _build()
        for seed in range(5):
            _, _, entropy = net.decide(_inputs(batch=8, seed=seed))
            self.assertTrue((entropy >= 0).all(), f"negative entropy for seed={seed}")

    def test_entropy_peaks_at_half_prob(self):
        """Maximum entropy for Bernoulli is log(2) ≈ 0.693, at prob=0.5."""
        import math
        net = _build()
        _, _, entropy = net.decide(_inputs(batch=32))
        self.assertTrue((entropy <= math.log(2) + 1e-5).all(),
                        f"entropy exceeds log(2): max={entropy.max().item():.4f}")


class TestBinaryHeadGradients(unittest.TestCase):
    def test_gradients_flow_to_every_param(self):
        net = _build()
        gs = _inputs(batch=4)
        logit = net(gs)
        target = torch.ones(4)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        loss.backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")

    def test_loss_finite_on_random_inputs(self):
        net = _build()
        for seed in range(5):
            gs = _inputs(batch=8, seed=seed)
            logit = net(gs)
            loss = F.binary_cross_entropy_with_logits(logit, torch.ones(8))
            self.assertTrue(torch.isfinite(loss), f"loss not finite for seed={seed}")


class TestBinaryHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must learn to consistently say 'yes' on a fixed input."""
        net = _build(seed=0)
        net.train()
        gs = _inputs(batch=16, seed=2)
        target = torch.ones(16)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        initial = F.binary_cross_entropy_with_logits(net(gs), target).item()
        for _ in range(200):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(gs), target).backward()
            opt.step()
        final = F.binary_cross_entropy_with_logits(net(gs), target).item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.02, f"failed to overfit: {initial:.4f} -> {final:.4f}")

    def test_determinism(self):
        gs = _inputs(batch=4)
        def run():
            net = _build(seed=7)
            net.eval()
            with torch.no_grad():
                return net(gs).clone()
        self.assertTrue(torch.allclose(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
