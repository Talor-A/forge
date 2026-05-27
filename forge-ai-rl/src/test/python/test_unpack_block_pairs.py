"""Unit tests for _unpack_block_pairs in train_decisions.py.

Run directly:   python forge-ai-rl/src/test/python/test_unpack_block_pairs.py
"""

import os
import sys
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "main", "python"))

from training.train_decisions import _unpack_block_pairs

CARD_DIM = 256


def _make_pairs(n_blockers, n_attackers, seed=0):
    """Build synthetic pair_features with the expected layout: b0a0, b0a1, ..., b1a0, ...
    Each blocker has a distinct feature vector; each attacker also distinct."""
    rng = np.random.default_rng(seed)
    blocker_vecs = rng.standard_normal((n_blockers, CARD_DIM)).astype(np.float32)
    attacker_vecs = rng.standard_normal((n_attackers, CARD_DIM)).astype(np.float32)
    pairs = []
    for b in range(n_blockers):
        for a in range(n_attackers):
            pairs.append(np.concatenate([blocker_vecs[b], attacker_vecs[a]]))
    return np.stack(pairs), blocker_vecs, attacker_vecs


class TestUnpackBlockPairs(unittest.TestCase):
    def test_empty_input(self):
        bf, af, tgt = _unpack_block_pairs(
            np.zeros((0, CARD_DIM * 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            CARD_DIM,
        )
        self.assertEqual(bf.shape, (0, CARD_DIM))
        self.assertEqual(af.shape, (0, CARD_DIM))
        self.assertEqual(tgt.shape, (0,))

    def test_inferred_shape(self):
        pairs, _, _ = _make_pairs(3, 2)
        action_mask = np.zeros(6, dtype=np.float32)
        bf, af, tgt = _unpack_block_pairs(pairs, action_mask, CARD_DIM)
        self.assertEqual(bf.shape, (3, CARD_DIM))
        self.assertEqual(af.shape, (2, CARD_DIM))
        self.assertEqual(tgt.shape, (3,))

    def test_blocker_features_match_first_pair_of_each_group(self):
        pairs, blocker_vecs, _ = _make_pairs(3, 2)
        bf, _, _ = _unpack_block_pairs(pairs, np.zeros(6, dtype=np.float32), CARD_DIM)
        for b in range(3):
            np.testing.assert_array_almost_equal(bf[b], blocker_vecs[b])

    def test_attacker_features_match_second_half_of_first_group(self):
        pairs, _, attacker_vecs = _make_pairs(3, 2)
        _, af, _ = _unpack_block_pairs(pairs, np.zeros(6, dtype=np.float32), CARD_DIM)
        for a in range(2):
            np.testing.assert_array_almost_equal(af[a], attacker_vecs[a])

    def test_selected_pair_maps_to_correct_attacker_index(self):
        pairs, _, _ = _make_pairs(3, 4)  # 12 pairs: b0a0..b0a3, b1a0..b1a3, b2a0..b2a3
        action_mask = np.zeros(12, dtype=np.float32)
        # b0 blocks a2, b1 blocks a0, b2 doesn't block
        action_mask[2] = 1.0   # b0a2
        action_mask[4] = 1.0   # b1a0
        _, _, tgt = _unpack_block_pairs(pairs, action_mask, CARD_DIM)
        self.assertEqual(tgt[0], 2)  # b0 → attacker 2
        self.assertEqual(tgt[1], 0)  # b1 → attacker 0
        self.assertEqual(tgt[2], 4)  # b2 → don't block (= n_attackers)

    def test_no_selection_gives_dont_block_for_all(self):
        pairs, _, _ = _make_pairs(2, 3)
        _, _, tgt = _unpack_block_pairs(pairs, np.zeros(6, dtype=np.float32), CARD_DIM)
        # All blockers default to "don't block" = n_attackers = 3
        self.assertTrue((tgt == 3).all(), f"expected all 3, got {tgt}")

    def test_single_blocker_single_attacker(self):
        pairs, bv, av = _make_pairs(1, 1)
        action_mask = np.array([1.0], dtype=np.float32)
        bf, af, tgt = _unpack_block_pairs(pairs, action_mask, CARD_DIM)
        self.assertEqual(bf.shape, (1, CARD_DIM))
        self.assertEqual(af.shape, (1, CARD_DIM))
        self.assertEqual(tgt[0], 0)  # only attacker, selected


class TestBatchRemap(unittest.TestCase):
    """The 'don't block' index from _unpack_block_pairs is n_attackers (per-sample),
    but _block_batch must remap it to max_at (batch-wide) before passing to the loss.
    These tests verify the invariant holds after that remap."""

    def _simulate_batch_remap(self, samples):
        """Reproduce the remap logic from _block_batch so we can assert on it."""
        unpacked = [
            _unpack_block_pairs(s['pairs'], s['mask'], CARD_DIM)
            for s in samples
        ]
        max_bl = max(len(bf) for bf, _, _ in unpacked)
        max_at = max(len(af) for _, af, _ in unpacked)

        bs = len(samples)
        blocker_mask = torch.zeros(bs, max(max_bl, 1), dtype=torch.bool)
        targets = torch.full((bs, max(max_bl, 1)), max_at, dtype=torch.long)

        for i, (bf, af, tgt) in enumerate(unpacked):
            nb, na = len(bf), len(af)
            if nb > 0:
                blocker_mask[i, :nb] = True
                tgt_t = torch.from_numpy(tgt).long()
                tgt_t[tgt_t == na] = max_at
                targets[i, :nb] = tgt_t

        return targets, blocker_mask, max_at

    def test_dont_block_always_at_max_at(self):
        """After remap, every 'don't block' target must equal max_at, not a per-sample n_a."""
        pairs_2a, _, _ = _make_pairs(2, 2, seed=0)  # 2 blockers, 2 attackers
        pairs_3a, _, _ = _make_pairs(1, 3, seed=1)  # 1 blocker, 3 attackers

        samples = [
            {'pairs': pairs_2a, 'mask': np.zeros(4, dtype=np.float32)},  # all don't block
            {'pairs': pairs_3a, 'mask': np.zeros(3, dtype=np.float32)},  # all don't block
        ]

        targets, blocker_mask, max_at = self._simulate_batch_remap(samples)
        self.assertEqual(max_at, 3)

        # Every valid blocker that chose "don't block" must have target == max_at (3), not 2
        valid_targets = targets[blocker_mask]
        self.assertTrue((valid_targets == max_at).all(),
                        f"'don't block' targets not all == {max_at}: {valid_targets.tolist()}")

    def test_targets_always_in_range(self):
        """All targets (valid or padding) must be in [0, max_at]."""
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n_b1, n_a1 = rng.integers(1, 4), rng.integers(1, 4)
            n_b2, n_a2 = rng.integers(1, 4), rng.integers(1, 4)
            pairs1, _, _ = _make_pairs(n_b1, n_a1, seed=seed)
            pairs2, _, _ = _make_pairs(n_b2, n_a2, seed=seed + 100)
            mask1 = (rng.random(n_b1 * n_a1) > 0.7).astype(np.float32)
            mask2 = (rng.random(n_b2 * n_a2) > 0.7).astype(np.float32)
            samples = [
                {'pairs': pairs1, 'mask': mask1},
                {'pairs': pairs2, 'mask': mask2},
            ]
            targets, _, max_at = self._simulate_batch_remap(samples)
            self.assertTrue((targets >= 0).all() and (targets <= max_at).all(),
                            f"seed={seed}: target out of [0, {max_at}]: {targets}")

    def test_selected_attacker_unchanged_by_remap(self):
        """Remap must not corrupt targets that point to a real attacker."""
        pairs, _, _ = _make_pairs(2, 2, seed=7)
        mask = np.zeros(4, dtype=np.float32)
        mask[1] = 1.0  # b0 blocks a1

        samples = [{'pairs': pairs, 'mask': mask}]
        targets, blocker_mask, max_at = self._simulate_batch_remap(samples)

        self.assertEqual(targets[0, 0].item(), 1, "b0 should still target attacker 1 after remap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
