"""
Lazy trajectory dataset — reads samples from disk on demand.

Instead of loading all 153K+ samples into RAM (~12GB), this indexes
the JSONL files and reads individual samples when requested.
Memory usage: O(num_files) for the index, not O(num_samples).

Usage:
    dataset = LazyTrajectoryDataset(data_dir, decision_types=['all'])
    sample = dataset[42]  # reads from disk on demand
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional


class LazyTrajectoryDataset(torch.utils.data.Dataset):
    """Lazily loads trajectory samples from JSONL files.

    Indexes files on init (fast, ~1s for 2000 files), then
    reads individual records from disk on __getitem__.
    """

    def __init__(self, data_dir: str,
                 decision_types: Optional[List[str]] = None,
                 max_files: Optional[int] = None):
        """
        Args:
            data_dir: path to directory with traj_*.jsonl files
            decision_types: list of types to include, e.g.
                ['DECLARE_ATTACKERS', 'PRIORITY_ACTION']
                or None/'all' for all records
            max_files: limit number of files to index
        """
        self.data_dir = data_dir
        self.decision_types = decision_types

        # Index: list of (filepath, line_number, won)
        self.index = []
        self._build_index(max_files)

    def _build_index(self, max_files):
        """Scan files and record (file, line_idx, won) for
        each decision record."""
        path = Path(self.data_dir)
        files = sorted(path.glob('traj_*.jsonl'))
        if max_files:
            files = files[:max_files]

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                if len(lines) < 2:
                    continue
                header = json.loads(lines[0])
                won = header.get('won', False)

                for line_idx in range(1, len(lines)):
                    # Quick check decision type without
                    # full parse if filtering
                    if self.decision_types:
                        line = lines[line_idx]
                        # Fast substring check
                        skip = True
                        for dt in self.decision_types:
                            if dt in line:
                                skip = False
                                break
                        if skip:
                            continue

                    self.index.append(
                        (str(filepath), line_idx, won))
            except Exception:
                pass

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filepath, line_idx, won = self.index[idx]

        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    rec = json.loads(line)
                    break
            else:
                raise IndexError(
                    f"Line {line_idx} not found in {filepath}")

        return self._parse_record(rec, won)

    def _parse_record(self, rec, won):
        """Parse a single JSONL record into tensors."""
        dt = rec.get('decisionType', '')

        gf = np.array(
            rec.get('globalFeatures', []),
            dtype=np.float32)
        np.clip(gf, -10, 10, out=gf)
        gf = np.nan_to_num(gf)
        g = np.zeros(64, dtype=np.float32)
        gl = min(len(gf), 64)
        if gl > 0:
            g[:gl] = gf[:gl]

        flat = np.array(
            rec.get('gameStateFlat', []),
            dtype=np.float32)
        np.clip(flat, -10, 10, out=flat)
        flat = np.nan_to_num(flat)

        outcome = 1.0 if won else -1.0

        return {
            'decision_type': dt,
            'global_features': g,
            'game_state_flat': flat,
            'outcome': outcome,
            'won': won,
            'raw': rec,  # for head-specific parsing
        }


class ValueDataset(torch.utils.data.Dataset):
    """Lazy dataset for value network training.
    Parses game state into zone tensors on demand."""

    ZONES = [
        ('my_board', 40), ('opp_board', 40),
        ('hand', 15), ('my_gy', 20),
        ('opp_gy', 20), ('stack', 10),
    ]

    def __init__(self, data_dir: str,
                 max_files: Optional[int] = None):
        self.base = LazyTrajectoryDataset(
            data_dir, decision_types=None,
            max_files=max_files)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        raw = self.base[idx]
        g = raw['global_features']
        flat = raw['game_state_flat']

        card_dim = 256
        zones = {}
        masks = {}
        offset = 96
        for name, count in self.ZONES:
            zs = count * card_dim
            zd = np.zeros((count, card_dim),
                          dtype=np.float32)
            zm = np.zeros(count, dtype=np.bool_)
            if offset + zs <= len(flat):
                chunk = flat[offset:offset + zs].reshape(
                    count, card_dim)
                for j in range(count):
                    if np.any(chunk[j] != 0):
                        zd[j] = chunk[j]
                        zm[j] = True
            offset += zs
            zones[name] = torch.from_numpy(zd)
            masks[name + '_mask'] = torch.from_numpy(zm)

        return {
            'global_features': torch.from_numpy(g),
            'value_target': torch.tensor(
                raw['outcome'], dtype=torch.float32),
            **zones,
            **masks,
        }


class PriorityDataset(torch.utils.data.Dataset):
    """Lazy dataset for priority head training."""

    def __init__(self, data_dir: str,
                 max_files: Optional[int] = None):
        self.base = LazyTrajectoryDataset(
            data_dir,
            decision_types=['PRIORITY_ACTION'],
            max_files=max_files)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        raw = self.base[idx]
        rec = raw['raw']
        cand = rec.get('candidateFeatures', [])
        sel = rec.get('selectedIndices', [])

        n = len(cand)
        actions = np.zeros((n, 64), dtype=np.float32)
        for j, cf in enumerate(cand):
            cl = min(len(cf), 64)
            actions[j, :cl] = np.array(
                cf[:cl], dtype=np.float32)
        np.clip(actions, -10, 10, out=actions)
        actions = np.nan_to_num(actions)

        selected_idx = sel[0] if sel else n - 1
        if selected_idx >= n:
            selected_idx = n - 1

        return {
            'global_features': raw['global_features'],
            'game_state_flat': raw['game_state_flat'],
            'action_features': actions,
            'selected_idx': selected_idx,
            'n_actions': n,
            'won': raw['won'],
        }


class AttackDataset(torch.utils.data.Dataset):
    """Lazy dataset for attack head training."""

    def __init__(self, data_dir: str,
                 max_files: Optional[int] = None):
        self.base = LazyTrajectoryDataset(
            data_dir,
            decision_types=['DECLARE_ATTACKERS'],
            max_files=max_files)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        raw = self.base[idx]
        rec = raw['raw']
        cand = rec.get('candidateFeatures', [])
        sel = rec.get('selectedIndices', [])

        n = len(cand)
        creatures = np.zeros((n, 128), dtype=np.float32)
        for j, cf in enumerate(cand):
            cl = min(len(cf), 128)
            creatures[j, :cl] = np.array(
                cf[:cl], dtype=np.float32)
        np.clip(creatures, -10, 10, out=creatures)
        creatures = np.nan_to_num(creatures)

        mask = np.zeros(n, dtype=np.float32)
        for i in sel:
            if 0 <= i < n:
                mask[i] = 1.0

        return {
            'global_features': raw['global_features'],
            'game_state_flat': raw['game_state_flat'],
            'creature_features': creatures,
            'action_mask': mask,
            'n_creatures': n,
            'won': raw['won'],
        }


class BlockDataset(torch.utils.data.Dataset):
    """Lazy dataset for block head training.
    Expects 256-dim pair features."""

    def __init__(self, data_dir: str,
                 max_files: Optional[int] = None):
        self.base = LazyTrajectoryDataset(
            data_dir,
            decision_types=['DECLARE_BLOCKERS'],
            max_files=max_files)
        # Filter to only 256-dim pair format
        self.valid_indices = []
        for i in range(len(self.base)):
            filepath, line_idx, won = self.base.index[i]
            # Quick check: read just the line
            try:
                with open(filepath) as f:
                    for li, line in enumerate(f):
                        if li == line_idx:
                            rec = json.loads(line)
                            cand = rec.get(
                                'candidateFeatures', [])
                            if cand and len(cand[0]) > 200:
                                self.valid_indices.append(i)
                            break
            except Exception:
                pass

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        raw = self.base[self.valid_indices[idx]]
        rec = raw['raw']
        cand = rec.get('candidateFeatures', [])
        sel = rec.get('selectedIndices', [])

        n = len(cand)
        pairs = np.zeros((n, 256), dtype=np.float32)
        for j, cf in enumerate(cand):
            cl = min(len(cf), 256)
            pairs[j, :cl] = np.array(
                cf[:cl], dtype=np.float32)
        np.clip(pairs, -10, 10, out=pairs)
        pairs = np.nan_to_num(pairs)

        mask = np.zeros(n, dtype=np.float32)
        for i in sel:
            if 0 <= i < n:
                mask[i] = 1.0

        return {
            'global_features': raw['global_features'],
            'game_state_flat': raw['game_state_flat'],
            'pair_features': pairs,
            'action_mask': mask,
            'n_pairs': n,
            'won': raw['won'],
        }


def count_samples(data_dir, max_files=None):
    """Quick count of samples by type without loading data."""
    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]

    counts = {
        'DECLARE_ATTACKERS': 0,
        'DECLARE_BLOCKERS': 0,
        'PRIORITY_ACTION': 0,
        'total': 0,
        'files': len(files),
    }

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  # skip header
                    counts['total'] += 1
                    for dt in ['DECLARE_ATTACKERS',
                               'DECLARE_BLOCKERS',
                               'PRIORITY_ACTION']:
                        if dt in line:
                            counts[dt] += 1
                            break
        except Exception:
            pass

    return counts
