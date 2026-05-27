"""Unit tests for MulliganHead (mulligan_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_mulligan_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_mulligan_head

NOTE — choose_bottom_cards is NOT autoregressive: bottom_scores are computed once
from evaluate_hand and the loop only re-masks; no GRU update between selections.
The model ranks all cards simultaneously and removes them greedily. The loop
structure looks autoregressive but isn't — worth noting when debugging quality
issues with bottom selection.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.mulligan_head import MulliganHead

STATE_DIM = 64
CARD_DIM = 32
HIDDEN_DIM = 16


def _build(seed=0):
    torch.manual_seed(seed)
    return MulliganHead(
        state_dim=STATE_DIM,
        card_feature_dim=CARD_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dropout=0.0,
    )


def _inputs(batch=2, n_cards=7, seed=1):
    torch.manual_seed(seed)
    gs = torch.randn(batch, STATE_DIM)
    hf = torch.randn(batch, n_cards, CARD_DIM)
    mask = torch.ones(batch, n_cards, dtype=torch.bool)
    return gs, hf, mask


class TestMulliganHeadShape(unittest.TestCase):
    def test_forward_shape(self):
        """forward (used for training) returns one keep-logit per sample."""
        net = _build()
        gs, hf, mask = _inputs(batch=4)
        logits = net(gs, hf, mask)
        self.assertEqual(logits.shape, (4,))

    def test_evaluate_hand_shapes(self):
        net = _build()
        gs, hf, mask = _inputs(batch=3, n_cards=7)
        keep_logit, bottom_scores = net.evaluate_hand(gs, hf, mask)
        self.assertEqual(keep_logit.shape, (3, 1))
        self.assertEqual(bottom_scores.shape, (3, 7))

    def test_decide_keep_shapes(self):
        net = _build()
        gs, hf, mask = _inputs(batch=5)
        keep, log_prob = net.decide_keep(gs, hf, mask)
        self.assertEqual(keep.shape, (5,))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(keep.dtype, torch.bool)

    def test_choose_bottom_cards_shape(self):
        net = _build()
        gs, hf, mask = _inputs(batch=4, n_cards=7)
        indices, log_prob = net.choose_bottom_cards(gs, hf, mask, num_bottom=2)
        self.assertEqual(indices.shape, (4, 2))
        self.assertEqual(log_prob.shape, (4,))


class TestMulliganHeadMasking(unittest.TestCase):
    def test_masked_cards_not_in_bottom_scores(self):
        """Cards excluded by hand_mask must get -inf bottom score."""
        net = _build()
        gs, hf, mask = _inputs(batch=2, n_cards=7)
        mask[:, 5:] = False
        _, bottom_scores = net.evaluate_hand(gs, hf, mask)
        self.assertTrue((bottom_scores[:, 5:] == float('-inf')).all(),
                        "masked card slots should have -inf bottom score")

    def test_bottom_selection_no_repeats(self):
        """Same card must not be chosen twice for the bottom."""
        net = _build()
        torch.manual_seed(0)
        gs, hf, mask = _inputs(batch=16, n_cards=7)
        indices, _ = net.choose_bottom_cards(gs, hf, mask, num_bottom=3)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse((indices[:, i] == indices[:, j]).any(),
                                 f"bottom selections {i} and {j} are the same card")

    def test_bottom_indices_in_valid_range(self):
        net = _build()
        torch.manual_seed(0)
        gs, hf, mask = _inputs(batch=8, n_cards=7)
        indices, _ = net.choose_bottom_cards(gs, hf, mask, num_bottom=2)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 7).all())

    def test_num_bottom_zero_returns_correct_shape(self):
        """choose_bottom_cards with num_bottom=0 must return (batch, 0) indices."""
        net = _build()
        gs, hf, mask = _inputs(batch=4, n_cards=7)
        indices, log_prob = net.choose_bottom_cards(gs, hf, mask, num_bottom=0)
        self.assertEqual(indices.shape, (4, 0))
        self.assertEqual(log_prob.shape, (4,))
        self.assertTrue((log_prob == 0).all(), "log_prob should be 0 when num_bottom=0")


class TestMulliganHeadLogProbs(unittest.TestCase):
    def test_keep_log_prob_non_positive(self):
        net = _build()
        for seed in range(5):
            gs, hf, mask = _inputs(batch=8, seed=seed)
            _, log_prob = net.decide_keep(gs, hf, mask)
            self.assertTrue((log_prob <= 0).all(), f"log_prob > 0 for seed={seed}")

    def test_bottom_log_prob_non_positive(self):
        net = _build()
        for seed in range(5):
            gs, hf, mask = _inputs(batch=8, seed=seed)
            _, log_prob = net.choose_bottom_cards(gs, hf, mask, num_bottom=2)
            self.assertTrue((log_prob <= 0).all(), f"log_prob > 0 for seed={seed}")


class TestMulliganHeadGradients(unittest.TestCase):
    def test_gradients_flow_through_forward(self):
        """Params reachable via forward() (keep path) get finite gradients.

        forward() calls evaluate_hand() and returns only the keep_logit; the
        bottom_scorer branch is not exercised. bottom_scorer receives no gradient
        from the keep-decision training loss — a gap if bottom-card quality matters.
        """
        net = _build()
        gs, hf, mask = _inputs()
        logit = net(gs, hf, mask)
        loss = F.binary_cross_entropy_with_logits(logit, torch.ones(logit.shape[0]))
        loss.backward()

        bottom_params = {f"bottom_scorer.{n}" for n, _ in net.bottom_scorer.named_parameters()}
        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            if name in bottom_params:
                continue  # not reachable from forward()
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")

    def test_bottom_scorer_receives_gradient_via_evaluate_hand(self):
        """bottom_scorer must get gradients when loss is on bottom_scores."""
        net = _build()
        gs, hf, mask = _inputs(batch=4)
        _, bottom_scores = net.evaluate_hand(gs, hf, mask)
        target = torch.zeros(4, dtype=torch.long)
        loss = F.cross_entropy(bottom_scores, target)
        loss.backward()
        self.assertIsNotNone(net.bottom_scorer[0].weight.grad,
                             "bottom_scorer[0].weight got no gradient")
        self.assertTrue(torch.isfinite(net.bottom_scorer[0].weight.grad).all())


class TestMulliganHeadOverfit(unittest.TestCase):
    def test_overfit_keep_decision(self):
        """Model must learn to always keep on a fixed input."""
        net = _build(seed=0)
        net.train()
        gs, hf, mask = _inputs(batch=16, n_cards=7, seed=2)
        target = torch.ones(16)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            return F.binary_cross_entropy_with_logits(net(gs, hf, mask), target)

        initial = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final = loss_fn().item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.02,
                        f"mulligan head failed to overfit: {initial:.4f} -> {final:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
