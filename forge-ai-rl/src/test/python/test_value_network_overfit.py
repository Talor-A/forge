"""Overfit-one-batch test for ValueNetwork.

test that the model can drive loss to ~0 on one batch with a fixed input.

shows that the model is not silently broken, and is capable of memorizing
a tiny test set.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from model.value_network import ValueNetwork


def _build(seed=0):
    torch.manual_seed(seed)
    # dropout=0: dropout noise prevents the network from tightly fitting a
    # fixed batch, which would defeat the point of the diagnostic.
    return ValueNetwork(input_dim=128, hidden_dim=64, dropout=0.0)


class TestValueNetworkOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        net = _build(seed=0)
        torch.manual_seed(1)
        x = torch.randn(8, 128)
        # Targets in (-0.9, 0.9). ValueNetwork ends in Tanh — labeling with
        # +/-1 would set a noise floor (atanh(+/-1) = inf) and mask real bugs.
        y = torch.linspace(-0.9, 0.9, 8).unsqueeze(-1)

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        initial_loss = F.mse_loss(net(x), y).item()

        for _ in range(200):
            opt.zero_grad()
            loss = F.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        final_loss = loss.item()

        self.assertGreater(
            initial_loss,
            0.01,
            msg=f"initial loss {initial_loss:.6f} suspiciously low — "
            "is the test really doing anything?",
        )
        self.assertLess(
            final_loss,
            1e-3,
            msg=f"value net failed to overfit 8 examples in 200 steps: "
            f"{initial_loss:.4f} -> {final_loss:.4f}",
        )

    def test_gradients_flow_to_every_param(self):
        """Every leaf param must receive a finite gradient after one backward.
        Catches frozen-layer and detached-graph regressions."""
        net = _build(seed=0)
        x = torch.randn(4, 128)
        y = torch.zeros(4, 1)
        F.mse_loss(net(x), y).backward()

        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name}: requires_grad=False")
            self.assertIsNotNone(p.grad, f"{name}: no .grad after backward")
            self.assertTrue(
                torch.isfinite(p.grad).all(),
                msg=f"{name}: grad has NaN/Inf",
            )

    def test_determinism(self):
        """Same seed -> same training trajectory. Cheapest way to flag
        hidden randomness in the loop (forgotten seed, nondeterministic op)."""

        def run():
            net = _build(seed=42)
            torch.manual_seed(7)
            x = torch.randn(4, 128)
            y = torch.linspace(-0.5, 0.5, 4).unsqueeze(-1)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            for _ in range(20):
                opt.zero_grad()
                F.mse_loss(net(x), y).backward()
                opt.step()
            return F.mse_loss(net(x), y).item()

        self.assertAlmostEqual(run(), run(), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
