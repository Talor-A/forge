"""End-to-end roundtrip: synthetic JSONL → preprocess_trajectories.py
→ Mmap*Dataset.

After deleting chunked_loader.py, the only path from JSONL into
training is preprocess_trajectories → mmap arrays → Mmap*Dataset.
Both halves had zero tests; this is the smoke check.

The test writes 2 synthetic trajectory files into a temp dir, runs
the preprocessor as a subprocess (mirrors how smoke.sh / scripts/
invoke it), then loads via MmapValueDataset and MmapAttackDataset
and asserts shapes + that the value target threads through.

Run directly:    python forge-ai-rl/src/test/python/test_preprocess_mmap_roundtrip.py
Or via unittest: python -m unittest forge-ai-rl.src.test.python.test_preprocess_mmap_roundtrip
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.join(_HERE, "..", "..", "main", "python")
sys.path.insert(0, _PY_DIR)

import numpy as np

from training.mmap_dataset import (
    CARD_DIM,
    GAME_STATE_DIM,
    GLOBAL_DIM,
    ZONES_CONFIG,
    MmapAttackDataset,
    MmapValueDataset,
)


def _full_state(seed, fill_zone=None):
    """Build a full GAME_STATE_DIM flat state + GLOBAL_DIM globals.

    If fill_zone is given, the first slot of that zone is non-zero
    so its mask should come back True.
    """
    rng = np.random.default_rng(seed)
    flat = np.zeros(GAME_STATE_DIM, dtype=np.float32)
    gf = rng.uniform(-1.0, 1.0, GLOBAL_DIM).astype(np.float32)
    if fill_zone is not None:
        offset = GLOBAL_DIM
        for name, count in ZONES_CONFIG:
            if name == fill_zone:
                flat[offset:offset + CARD_DIM] = 0.5
                break
            offset += count * CARD_DIM
    return flat.tolist(), gf.tolist()


def _write_traj(path, won, decisions):
    with open(path, "w") as f:
        f.write(json.dumps({
            "gameId": "test",
            "won": won,
            "totalDecisions": len(decisions),
            "durationMs": 1000,
        }) + "\n")
        for d in decisions:
            f.write(json.dumps(d) + "\n")


class TestPreprocessRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="preprocess-rt-")
        cls.traj_dir = os.path.join(cls.work, "trajectories")
        cls.preproc_dir = os.path.join(cls.work, "preprocessed")
        os.makedirs(cls.traj_dir)

        # Two games × two players = 4 files. Each game has one
        # PRIORITY snapshot (becomes a value sample) and one
        # DECLARE_ATTACKERS decision (becomes an attack sample).
        for game_id, players in [
            (1773000000001, [("P1_W", True), ("P2_L", False)]),
            (1773000000002, [("P1_L", False), ("P2_W", True)]),
        ]:
            for tag, won in players:
                flat, gf = _full_state(
                    seed=game_id, fill_zone="my_board")
                attackers = np.zeros(
                    (3, CARD_DIM), dtype=np.float32).tolist()
                attackers[0] = [0.3] * CARD_DIM  # nonzero so it's kept
                decisions = [
                    {
                        "turnIndex": 1,
                        "decisionType": "PRIORITY_ACTION",
                        "contextInfo": "snapshot",
                        "globalFeatures": gf,
                        "gameStateFlat": flat,
                        "candidateFeatures": [[0.1] * 64,
                                              [0.2] * 64],
                        "candidateCount": 2,
                        "selectedIndices": [0],
                        "actionProbabilities": [1.0, 0.0],
                        "valueEstimate": 0.0,
                        "intermediateReward": 0.0,
                    },
                    {
                        "turnIndex": 2,
                        "decisionType": "DECLARE_ATTACKERS",
                        "contextInfo": "attack_1_of_1",
                        "globalFeatures": gf,
                        "gameStateFlat": flat,
                        "candidateFeatures": attackers,
                        "candidateCount": 3,
                        "selectedIndices": [0],
                        "actionProbabilities": [],
                        "valueEstimate": 0.0,
                        "intermediateReward": 0.0,
                    },
                ]
                name = (f"traj_P1_vs_P2_{tag}_"
                        f"{game_id}.jsonl")
                _write_traj(
                    os.path.join(cls.traj_dir, name),
                    won=won, decisions=decisions)

        # Run the real preprocessor as a subprocess.
        result = subprocess.run(
            [sys.executable, "-m",
             "training.preprocess_trajectories",
             "--data-dir", cls.traj_dir,
             "--output-dir", cls.preproc_dir],
            cwd=_PY_DIR,
            env={**os.environ, "PYTHONPATH": _PY_DIR},
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"preprocess failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_metadata_written(self):
        meta_path = os.path.join(self.preproc_dir,
                                 "metadata.json")
        self.assertTrue(os.path.exists(meta_path),
                        "preprocess did not write metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        # The preprocessor's exact metadata schema isn't pinned
        # here — we just need it to be readable JSON.
        self.assertIsInstance(meta, dict)

    def test_shared_arrays_present(self):
        for fname in ("game_state.npy", "global_features.npy",
                      "outcome.npy", "file_id.npy"):
            path = os.path.join(self.preproc_dir, "shared",
                                fname)
            self.assertTrue(os.path.exists(path),
                            f"missing shared/{fname}")

    def test_value_dataset_loads_with_correct_shapes(self):
        ds = MmapValueDataset(self.preproc_dir, train=True,
                              val_fraction=0.25)
        self.assertGreater(
            len(ds), 0,
            "MmapValueDataset is empty — preprocess didn't "
            "produce any shared snapshots")
        s = ds[0]
        self.assertEqual(tuple(s["global_features"].shape),
                         (GLOBAL_DIM,))
        self.assertEqual(tuple(s["my_board"].shape),
                         (40, CARD_DIM))
        # fill_zone='my_board' slot 0 → mask[0] True, others False.
        mask = s["my_board_mask"]
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))
        self.assertIn(float(s["value_target"]), (-1.0, 1.0,
                                                  0.0))

    def test_train_val_split_disjoint_by_file_id(self):
        train = MmapValueDataset(self.preproc_dir, train=True,
                                 val_fraction=0.5,
                                 shared=None)
        val = MmapValueDataset(self.preproc_dir, train=False,
                               val_fraction=0.5,
                               shared=train.shared)
        train_files = {int(train.shared.file_id[i])
                       for i in train.indices}
        val_files = {int(val.shared.file_id[i])
                     for i in val.indices}
        self.assertEqual(
            train_files & val_files, set(),
            "train and val share files (data leak)")

    def test_attack_dataset_loads(self):
        ds = MmapAttackDataset(self.preproc_dir, train=True,
                               val_fraction=0.25)
        self.assertGreater(
            len(ds), 0,
            "MmapAttackDataset is empty — preprocess "
            "didn't process DECLARE_ATTACKERS")
        s = ds[0]
        self.assertEqual(s["creature_features"].shape[1],
                         CARD_DIM)
        self.assertEqual(s["action_mask"].shape[0],
                         s["n_creatures"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
