"""Unit tests for TargetHead (target_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_target_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_target_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.target_head import TargetHead, ACTION_DIM

STATE_DIM = 64
TARGET_DIM = 32
HIDDEN_DIM = 32


def _build(seed=0):
    torch.manual_seed(seed)
    return TargetHead(
        state_dim=STATE_DIM,
        target_feature_dim=TARGET_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.0,
    )


def _inputs(batch=2, n_targets=5, seed=1, with_spell=False):
    torch.manual_seed(seed)
    gs = torch.randn(batch, STATE_DIM)
    tf = torch.randn(batch, n_targets, TARGET_DIM)
    mask = torch.ones(batch, n_targets, dtype=torch.bool)
    spell = torch.randn(batch, ACTION_DIM) if with_spell else None
    return gs, tf, mask, spell


class TestTargetHeadOutputShape(unittest.TestCase):
    def test_forward_shape_no_spell(self):
        net = _build()
        gs, tf, mask, _ = _inputs(batch=3, n_targets=6)
        logits = net(gs, tf, mask)
        self.assertEqual(logits.shape, (3, 6))

    def test_forward_shape_with_spell(self):
        net = _build()
        gs, tf, mask, spell = _inputs(batch=3, n_targets=6, with_spell=True)
        logits = net(gs, tf, mask, spell)
        self.assertEqual(logits.shape, (3, 6))

    def test_select_multiple_returns_correct_count(self):
        net = _build()
        gs, tf, mask, _ = _inputs(batch=4, n_targets=8)
        selections = net.select_multiple(gs, tf, mask, num_selections=3)
        self.assertEqual(len(selections), 3)
        for action, log_prob in selections:
            self.assertEqual(action.shape, (4,))
            self.assertEqual(log_prob.shape, (4,))


class TestTargetHeadMasking(unittest.TestCase):
    def test_invalid_targets_get_neg_inf(self):
        net = _build()
        gs, tf, mask, _ = _inputs(batch=2, n_targets=5)
        mask[:, 3:] = False
        logits = net(gs, tf, mask)
        self.assertTrue((logits[:, 3:] == float('-inf')).all(),
                        "masked target slots should be -inf")

    def test_valid_logits_are_finite(self):
        net = _build()
        gs, tf, mask, _ = _inputs(batch=2, n_targets=5)
        mask[:, 3:] = False
        logits = net(gs, tf, mask)
        self.assertTrue(torch.isfinite(logits[:, :3]).all())

    def test_select_multiple_never_repeats(self):
        """Autoregressive selection must not pick the same target twice."""
        net = _build()
        torch.manual_seed(0)
        gs, tf, mask, _ = _inputs(batch=16, n_targets=8)
        selections = net.select_multiple(gs, tf, mask, num_selections=4)
        actions = torch.stack([a for a, _ in selections], dim=1)  # (batch, 4)
        for i in range(actions.shape[1]):
            for j in range(i + 1, actions.shape[1]):
                self.assertFalse((actions[:, i] == actions[:, j]).any(),
                                 f"selections {i} and {j} collide")

    def test_single_valid_target_always_selected(self):
        net = _build()
        torch.manual_seed(0)
        gs, tf, mask, _ = _inputs(batch=8, n_targets=5)
        mask[:] = False
        mask[:, 1] = True
        logits = net(gs, tf, mask)
        probs = F.softmax(logits, dim=-1)
        self.assertTrue(torch.allclose(probs[:, 1], torch.ones(8), atol=1e-4),
                        "only valid target should have probability 1")


class TestTargetHeadSpellContext(unittest.TestCase):
    def test_spell_features_change_logits(self):
        """Providing spell context must produce different logits than without."""
        net = _build()
        net.eval()
        gs, tf, mask, spell = _inputs(batch=4, n_targets=5, seed=3, with_spell=True)
        with torch.no_grad():
            logits_plain = net(gs, tf, mask, spell_features=None)
            logits_spell = net(gs, tf, mask, spell_features=spell)
        self.assertFalse(torch.allclose(logits_plain, logits_spell),
                         "spell_features had no effect on logits")

    def test_spell_features_same_seed_same_output(self):
        """Identical inputs with spell features must give identical outputs."""
        gs, tf, mask, spell = _inputs(batch=4, n_targets=5, seed=3, with_spell=True)
        def run():
            net = _build(seed=42)
            net.eval()
            with torch.no_grad():
                return net(gs, tf, mask, spell_features=spell).clone()
        self.assertTrue(torch.allclose(run(), run()))


class TestTargetHeadGradients(unittest.TestCase):
    def test_gradients_flow_without_spell(self):
        """Params reachable from forward() without spell_features get gradients.
        spell_projection is excluded: it is only wired in when spell_features is
        not None, so no gradient flows through it on the plain forward call."""
        net = _build()
        gs, tf, mask, _ = _inputs()
        logits = net(gs, tf, mask)
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            if name.startswith("spell_projection") or name.startswith("context_gru"):
                continue  # only active on optional paths (spell or select_multiple)
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: grad has NaN/Inf")

    def test_gradients_flow_with_spell(self):
        """spell_projection parameters must receive gradients when spell_features is provided."""
        net = _build()
        gs, tf, mask, spell = _inputs(with_spell=True)
        logits = net(gs, tf, mask, spell_features=spell)
        loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long))
        loss.backward()
        self.assertIsNotNone(net.spell_projection.weight.grad,
                             "spell_projection got no gradient")
        self.assertTrue(torch.isfinite(net.spell_projection.weight.grad).all())


class TestTargetHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must learn to always pick target 0 on a fixed input."""
        net = _build(seed=0)
        net.train()
        gs, tf, mask, _ = _inputs(batch=8, n_targets=5, seed=2)
        target = torch.zeros(8, dtype=torch.long)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            return F.cross_entropy(net(gs, tf, mask), target)

        initial = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final = loss_fn().item()

        self.assertGreater(initial, 0.1, f"initial loss {initial:.4f} suspiciously low")
        self.assertLess(final, 0.05,
                        f"target head failed to overfit: {initial:.4f} -> {final:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
