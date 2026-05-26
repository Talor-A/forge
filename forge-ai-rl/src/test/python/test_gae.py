"""Unit tests for GAE advantage computation in ppo_trainer._compute_gae_returns.


Run directly:   python forge-ai-rl/src/test/python/test_gae.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_gae
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from training.ppo_trainer import GAE_GAMMA, GAE_LAMBDA, _compute_gae_returns

# Pin the hyperparams these tests were computed against. If someone retunes
# GAE_GAMMA / GAE_LAMBDA, this assertion fails first and points the reader
# at the cause rather than at baffling numerical mismatches below.
EXPECTED_GAMMA = 0.99
EXPECTED_LAMBDA = 0.95


def _rec(value=0.0, ir=0.0):
    return {"valueEstimate": value, "intermediateReward": ir}


class TestGAEReturns(unittest.TestCase):
    def setUp(self):
        self.assertEqual(GAE_GAMMA, EXPECTED_GAMMA)
        self.assertEqual(GAE_LAMBDA, EXPECTED_LAMBDA)

    def test_empty_input(self):
        self.assertEqual(_compute_gae_returns([], won=True), [])
        self.assertEqual(_compute_gae_returns([], won=False), [])

    def test_three_step_hand_computed(self):
        """Pinned numerical output for a fixed 3-step trajectory.

        Inputs: values=[0.1, 0.2, 0.3], won=True (terminal=+1), shaping=0.
        With gamma=0.99, lambda=0.95, gl=0.9405:
          delta_2 = 1.0 + 0.99*0   - 0.3 = 0.7    A_2 = 0.7
          delta_1 = 0.0 + 0.99*0.3 - 0.2 = 0.097  A_1 = 0.097 + gl*0.7      = 0.75535
          delta_0 = 0.0 + 0.99*0.2 - 0.1 = 0.098  A_0 = 0.098 + gl*0.75535 ~= 0.808407
        """
        records = [_rec(value=0.1), _rec(value=0.2), _rec(value=0.3)]
        out = _compute_gae_returns(records, won=True, shaping_coeff=0.0)
        advs = [float(a) for a, _ in out]
        rets = [float(r) for _, r in out]

        for got, want in zip(advs, [0.808406675, 0.75535, 0.7]):
            self.assertAlmostEqual(got, want, places=5)
        for got, want in zip(rets, [0.908406675, 0.95535, 1.0]):
            self.assertAlmostEqual(got, want, places=5)

    def test_terminal_only_geometric_decay(self):
        """Zero values + zero intermediate rewards: advantages collapse to
        the geometric series A_{T-k} = (gamma*lambda)^k * terminal. This is
        the cleanest off-by-one canary — a stride bug shifts the exponents
        by one and the test catches it instantly.
        """
        gl = GAE_GAMMA * GAE_LAMBDA
        for won, terminal in [(True, 1.0), (False, -1.0)]:
            records = [_rec() for _ in range(4)]
            out = _compute_gae_returns(records, won=won, shaping_coeff=0.0)
            advs = [float(a) for a, _ in out]
            expected = [terminal * (gl**k) for k in (3, 2, 1, 0)]
            for got, want in zip(advs, expected):
                self.assertAlmostEqual(
                    got,
                    want,
                    places=5,
                    msg=f"won={won}: got {advs}, expected {expected}",
                )

    def test_returns_equal_advantages_plus_values(self):
        """Sanity: returns = advantages + values (the value-net training target)."""
        records = [_rec(value=v, ir=0.05) for v in (0.3, -0.1, 0.4, 0.0, 0.2)]
        for won in (True, False):
            for coeff in (0.0, 0.5):
                out = _compute_gae_returns(records, won=won, shaping_coeff=coeff)
                for (a, r), rec in zip(out, records):
                    self.assertAlmostEqual(
                        float(r),
                        float(a) + rec["valueEstimate"],
                        places=5,
                        msg=f"won={won} coeff={coeff}",
                    )

    def test_shaping_coefficient_isolates_intermediate(self):
        """Shaping scales per-step rewards; it must NOT touch the terminal.

        Concretely: GAE is linear in rewards (for fixed values), so
            advs(coeff=c) - advs(coeff=0) == GAE(rewards=c*ir, values=0)
        which depends only on intermediate rewards — never on `won`.
        """
        records = [_rec(value=0.0, ir=ir) for ir in (0.1, 0.2, 0.3)]

        def advs(won, coeff):
            return [
                float(a)
                for a, _ in _compute_gae_returns(records, won=won, shaping_coeff=coeff)
            ]

        diff_won = [c - z for c, z in zip(advs(True, 1.0), advs(True, 0.0))]
        diff_lost = [c - z for c, z in zip(advs(False, 1.0), advs(False, 0.0))]

        for a, b in zip(diff_won, diff_lost):
            self.assertAlmostEqual(
                a, b, places=5, msg="shaping bled into the terminal reward"
            )

        # And the difference matches the pure-intermediate GAE.
        # gl=0.9405, V=0:
        #   A_2 = 0.3
        #   A_1 = 0.2 + gl*0.3       = 0.48215
        #   A_0 = 0.1 + gl*0.48215  ~= 0.553462
        expected = [0.553462075, 0.48215, 0.3]
        for got, want in zip(diff_won, expected):
            self.assertAlmostEqual(got, want, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
