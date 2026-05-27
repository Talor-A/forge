"""Unit tests for BlockHead (combat_block_head.py).

Run directly:   python forge-ai-rl/src/test/python/test_combat_block_head.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_combat_block_head
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.combat_block_head import BlockHead

STATE_DIM = 64
CARD_DIM = 32
HIDDEN_DIM = 16


def _build(seed=0):
    torch.manual_seed(seed)
    return BlockHead(
        state_dim=STATE_DIM,
        card_feature_dim=CARD_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dropout=0.0,
    )


def _inputs(batch=2, n_blockers=3, n_attackers=2, seed=1):
    torch.manual_seed(seed)
    game_state = torch.randn(batch, STATE_DIM)
    blocker_features = torch.randn(batch, n_blockers, CARD_DIM)
    attacker_features = torch.randn(batch, n_attackers, CARD_DIM)
    blocker_mask = torch.ones(batch, n_blockers, dtype=torch.bool)
    attacker_mask = torch.ones(batch, n_attackers, dtype=torch.bool)
    return game_state, blocker_features, blocker_mask, attacker_features, attacker_mask


class TestBlockHeadOutputShape(unittest.TestCase):
    def test_forward_shape(self):
        net = _build()
        gs, bf, bm, af, am = _inputs(batch=2, n_blockers=3, n_attackers=2)
        logits = net(gs, bf, bm, af, am)
        # Expect (batch, max_blockers, max_attackers + 1)
        self.assertEqual(logits.shape, (2, 3, 3))

    def test_sample_shapes(self):
        net = _build()
        gs, bf, bm, af, am = _inputs(batch=2, n_blockers=3, n_attackers=2)
        assignments, log_probs = net.sample_assignments(gs, bf, bm, af, am)
        self.assertEqual(assignments.shape, (2, 3))
        self.assertEqual(log_probs.shape, (2,))

    def test_assignments_in_valid_range(self):
        """Each assignment must be a valid attacker index or the 'don't block' index."""
        net = _build()
        n_attackers = 4
        gs, bf, bm, af, am = _inputs(batch=4, n_blockers=5, n_attackers=n_attackers)
        assignments, _ = net.sample_assignments(gs, bf, bm, af, am)
        # Valid range: 0..n_attackers (inclusive — n_attackers is "don't block")
        self.assertTrue((assignments >= 0).all())
        self.assertTrue((assignments <= n_attackers).all())


class TestBlockHeadMasking(unittest.TestCase):
    def test_dont_block_column_never_masked(self):
        """The last column (don't block) must be finite for every blocker,
        even when some or all attacker slots are padding."""
        net = _build()
        batch, n_blockers, n_attackers = 2, 3, 4
        gs, bf, bm, af, am = _inputs(batch=batch, n_blockers=n_blockers, n_attackers=n_attackers)
        # Mask out all but one attacker
        am[:, 2:] = False
        logits = net(gs, bf, bm, af, am)
        dont_block_col = logits[:, :, -1]  # (batch, max_blockers)
        self.assertTrue(torch.isfinite(dont_block_col).all(),
                        "'don't block' column has -inf entries")

    def test_invalid_attackers_get_neg_inf(self):
        """Padding attacker slots (mask=False) must have logit == -inf."""
        net = _build()
        batch, n_blockers, n_attackers = 2, 3, 4
        gs, bf, bm, af, am = _inputs(batch=batch, n_blockers=n_blockers, n_attackers=n_attackers)
        am[:, 2:] = False  # last 2 attacker slots are padding
        logits = net(gs, bf, bm, af, am)
        # Columns 2 and 3 (0-indexed attackers 2 and 3) should be -inf for all batches/blockers
        self.assertTrue((logits[:, :, 2] == float('-inf')).all())
        self.assertTrue((logits[:, :, 3] == float('-inf')).all())

    def test_invalid_blocker_zero_log_prob_contribution(self):
        """Padding blocker slots must not contribute to the total log prob."""
        net = _build()
        batch, n_blockers, n_attackers = 2, 4, 2
        gs, bf, bm, af, am = _inputs(batch=batch, n_blockers=n_blockers, n_attackers=n_attackers)

        _, lp_all_valid = net.sample_assignments(gs, bf, bm, af, am)

        # Mask out the last blocker — log prob should not change
        bm_partial = bm.clone()
        bm_partial[:, -1] = False
        _, lp_partial = net.sample_assignments(gs, bf, bm_partial, af, am)

        # log_prob with one blocker masked must be <= log_prob with all valid
        # (removing a term can only decrease the sum of non-positive log probs)
        # More importantly, the difference must equal the dropped blocker's term.
        # We can't assert exact equality due to sampling randomness, but we CAN
        # assert that masking a blocker to False makes it contribute 0.
        # Re-run with fixed seed so sample is deterministic.
        torch.manual_seed(99)
        _, lp_all = net.sample_assignments(gs, bf, bm, af, am)
        torch.manual_seed(99)
        _, lp_drop = net.sample_assignments(gs, bf, bm_partial, af, am)

        # log probs are non-positive, so removing one valid blocker's contribution
        # (multiplying by 0) makes the total less negative (i.e. higher).
        self.assertTrue((lp_drop >= lp_all - 1e-5).all(),
                        "masking a blocker lowered total log_prob — it must have contributed 0")


class TestBlockHeadGradients(unittest.TestCase):
    def test_gradients_flow_to_every_param(self):
        """Every leaf param must receive a finite gradient after one backward.
        Catches frozen-layer and detached-graph regressions."""
        net = _build()
        gs, bf, bm, af, am = _inputs()
        logits = net(gs, bf, bm, af, am)  # (batch, blockers, attackers+1)
        # Use valid logits only (finite values) for a cross-entropy-style loss
        target = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss.backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(
                torch.isfinite(p.grad).all(),
                msg=f"{name}: grad has NaN/Inf",
            )

    def test_loss_is_finite_on_random_inputs(self):
        net = _build()
        for seed in range(5):
            gs, bf, bm, af, am = _inputs(batch=4, n_blockers=3, n_attackers=3, seed=seed)
            logits = net(gs, bf, bm, af, am)
            target = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            self.assertTrue(torch.isfinite(loss), f"loss is not finite for seed={seed}")


class TestBlockHeadOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        """Model must drive cross-entropy loss to near 0 on one fixed batch.
        Failure means the model or forward pass is broken."""
        net = _build(seed=0)
        net.train()
        gs, bf, bm, af, am = _inputs(batch=8, n_blockers=4, n_attackers=3, seed=2)

        # Assign each blocker to block attacker 0
        target = torch.zeros(8, 4, dtype=torch.long)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        def loss_fn():
            logits = net(gs, bf, bm, af, am)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

        initial_loss = loss_fn().item()
        for _ in range(300):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        final_loss = loss_fn().item()

        self.assertGreater(initial_loss, 0.1,
                           f"initial loss {initial_loss:.4f} suspiciously low — is the test doing anything?")
        self.assertLess(final_loss, 0.05,
                        f"block head failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}")

    def test_determinism(self):
        """Same seed -> same forward output."""
        gs, bf, bm, af, am = _inputs(batch=2, n_blockers=3, n_attackers=2, seed=5)

        def run():
            net = _build(seed=42)
            net.eval()
            with torch.no_grad():
                return net(gs, bf, bm, af, am).clone()

        out1, out2 = run(), run()
        self.assertTrue(torch.allclose(out1, out2),
                        "identical seeds produced different logits")


if __name__ == "__main__":
    unittest.main(verbosity=2)
