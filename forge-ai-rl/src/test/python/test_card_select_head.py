"""Unit tests for CardSelectHead (card_select_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_card_select_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_card_select_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.card_select_head import CardSelectHead

STATE_DIM = 64
CARD_DIM = 32
HIDDEN_DIM = 16


def _build(seed=0):
    torch.manual_seed(seed)
    return CardSelectHead(
        state_dim=STATE_DIM,
        card_feature_dim=CARD_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dropout=0.0,
    )


def _inputs(batch=2, n_cards=6, seed=1):
    torch.manual_seed(seed)
    gs = torch.randn(batch, STATE_DIM)
    cf = torch.randn(batch, n_cards, CARD_DIM)
    mask = torch.ones(batch, n_cards, dtype=torch.bool)
    return gs, cf, mask


class TestCardSelectHeadShape(unittest.TestCase):
    def test_forward_shape(self):
        """forward returns one logit per card slot."""
        net = _build()
        gs, cf, mask = _inputs(batch=3, n_cards=6)
        logits = net(gs, cf, mask)
        self.assertEqual(logits.shape, (3, 6))

    def test_select_cards_shape(self):
        net = _build()
        gs, cf, mask = _inputs(batch=4, n_cards=8)
        indices, log_prob = net.select_cards(gs, cf, mask, num_select=3)
        self.assertEqual(indices.shape, (4, 3))
        self.assertEqual(log_prob.shape, (4,))

    def test_select_one_card_shape(self):
        net = _build()
        gs, cf, mask = _inputs(batch=4, n_cards=6)
        indices, log_prob = net.select_cards(gs, cf, mask, num_select=1)
        self.assertEqual(indices.shape, (4, 1))
        self.assertEqual(log_prob.shape, (4,))


class TestCardSelectHeadMasking(unittest.TestCase):
    def test_masked_cards_get_neg_inf(self):
        net = _build()
        gs, cf, mask = _inputs(batch=2, n_cards=6)
        mask[:, 4:] = False
        logits = net(gs, cf, mask)
        self.assertTrue((logits[:, 4:] == float('-inf')).all(),
                        "masked card slots should have -inf logit")

    def test_valid_logits_are_finite(self):
        net = _build()
        gs, cf, mask = _inputs(batch=2, n_cards=6)
        mask[:, 4:] = False
        logits = net(gs, cf, mask)
        self.assertTrue(torch.isfinite(logits[:, :4]).all())

    def test_select_cards_no_repeats(self):
        """Each selected index must be unique within its sample."""
        net = _build()
        torch.manual_seed(0)
        gs, cf, mask = _inputs(batch=16, n_cards=8)
        indices, _ = net.select_cards(gs, cf, mask, num_select=4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse((indices[:, i] == indices[:, j]).any(),
                                 f"selections {i} and {j} are the same card")

    def test_selected_indices_in_valid_range(self):
        net = _build()
        torch.manual_seed(0)
        gs, cf, mask = _inputs(batch=8, n_cards=6)
        indices, _ = net.select_cards(gs, cf, mask, num_select=3)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 6).all())

    def test_mixed_mask_counts_no_crash(self):
        """select_cards must not crash when batch elements have different numbers
        of valid candidates. The per-element has_valid check prevents all-inf
        Categorical rows for exhausted elements."""
        net = _build()
        torch.manual_seed(0)
        batch, n_cards = 2, 4
        gs = torch.randn(batch, STATE_DIM)
        cf = torch.randn(batch, n_cards, CARD_DIM)
        mask = torch.ones(batch, n_cards, dtype=torch.bool)
        mask[1, 1:] = False  # element 1 has only 1 valid card; num_select=2 exhausts it

        with torch.no_grad():
            indices, log_prob = net.select_cards(gs, cf, mask, num_select=2)

        self.assertEqual(indices.shape, (batch, 2))
        self.assertTrue(torch.isfinite(log_prob).all(), "log_prob has NaN/Inf")
        # Element 1 has only 1 valid selection; its log_prob contribution for the
        # second step is 0 (not counted), so it is strictly greater than element 0's.
        self.assertGreater(log_prob[1].item(), log_prob[0].item() - 1e-5)


class TestCardSelectHeadLogProbs(unittest.TestCase):
    def test_log_probs_non_positive(self):
        net = _build()
        for seed in range(5):
            gs, cf, mask = _inputs(batch=8, seed=seed)
            _, log_prob = net.select_cards(gs, cf, mask, num_select=2)
            self.assertTrue((log_prob <= 0).all(), f"log_prob > 0 for seed={seed}")


class TestCardSelectHeadGradients(unittest.TestCase):
    def test_gradients_flow_through_forward(self):
        """Params reachable via forward() all receive finite gradients.

        selection_gru is excluded: it is only used in select_cards(), which is a
        sampling path and provides no gradient signal via the supervised forward pass.
        This means selection_gru is never trained via cross-entropy loss — a design
        gap worth noting if bottom-card quality matters.
        """
        net = _build()
        gs, cf, mask = _inputs()
        logits = net(gs, cf, mask)
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        gru_params = {n for n, _ in net.selection_gru.named_parameters()}
        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            short = name.split(".", 1)[-1] if "." in name else name
            if short in gru_params or name.startswith("selection_gru"):
                continue  # not reachable from forward()
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")


class TestCardSelectHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must learn to always score card 0 highest on a fixed input."""
        net = _build(seed=0)
        net.train()
        gs, cf, mask = _inputs(batch=8, n_cards=5, seed=2)
        target = torch.zeros(8, dtype=torch.long)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            return F.cross_entropy(net(gs, cf, mask), target)

        initial = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final = loss_fn().item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.05,
                        f"card select head failed to overfit: {initial:.4f} -> {final:.4f}")

    def test_determinism(self):
        gs, cf, mask = _inputs(batch=3, n_cards=5, seed=5)
        def run():
            net = _build(seed=42)
            net.eval()
            with torch.no_grad():
                return net(gs, cf, mask).clone()
        self.assertTrue(torch.allclose(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
