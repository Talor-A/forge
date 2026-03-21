"""
Memory-mapped datasets for training.

Loads preprocessed numpy arrays with mmap_mode='r', keeping RAM
usage at O(batch_size) regardless of dataset size. Compatible
with PyTorch DataLoader (num_workers=0).

Usage:
    ds = MmapValueDataset('preprocessed/', train=True)
    loader = DataLoader(ds, batch_size=256, shuffle=True,
                        num_workers=0, collate_fn=lambda x: x)
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Set


def _load_metadata(preprocessed_dir):
    """Load metadata.json from preprocessed directory."""
    path = os.path.join(preprocessed_dir, 'metadata.json')
    with open(path) as f:
        return json.load(f)


def _split_file_ids(file_ids, val_fraction=0.1, seed=42):
    """Split file IDs into train/val sets."""
    unique_ids = sorted(set(file_ids.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)
    n_val = max(1, int(len(unique_ids) * val_fraction))
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    return train_ids, val_ids


def parse_game_state(flat, global_feats):
    """Parse flat game state into zone tensors.
    Same logic as train_decisions_ui.parse_game_state."""
    card_dim = 128
    zones = [('my_board', 30), ('opp_board', 30),
             ('hand', 15), ('my_gy', 40),
             ('opp_gy', 40), ('stack', 10)]
    g = np.zeros(64, dtype=np.float32)
    gl = min(len(global_feats), 64)
    if gl > 0:
        g[:gl] = global_feats[:gl]
    zdata = {}
    zmask = {}
    offset = 64
    for name, count in zones:
        zs = count * card_dim
        zd = np.zeros((count, card_dim), dtype=np.float32)
        zm = np.zeros(count, dtype=np.bool_)
        if offset + zs <= len(flat):
            raw = flat[offset:offset + zs].reshape(
                count, card_dim)
            for j in range(count):
                if np.any(raw[j] != 0):
                    zd[j] = raw[j]
                    zm[j] = True
        offset += zs
        zdata[name] = zd
        zmask[name + '_mask'] = zm
    return g, zdata, zmask


class MmapValueDataset(Dataset):
    """Memory-mapped dataset for value network training.

    Returns dicts compatible with SimpleDataset format:
    {global_features, zones, masks, value_target}
    """

    def __init__(self, preprocessed_dir: str,
                 train: bool = True,
                 val_fraction: float = 0.1):
        vdir = os.path.join(preprocessed_dir, 'value')
        self.game_state = np.load(
            os.path.join(vdir, 'game_state.npy'),
            mmap_mode='r')
        self.global_features = np.load(
            os.path.join(vdir, 'global_features.npy'),
            mmap_mode='r')
        self.outcome = np.load(
            os.path.join(vdir, 'outcome.npy'),
            mmap_mode='r')
        self.file_id = np.load(
            os.path.join(vdir, 'file_id.npy'),
            mmap_mode='r')

        # Train/val split by file ID
        train_ids, val_ids = _split_file_ids(
            self.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.file_id))
            if self.file_id[i] in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        flat = np.array(self.game_state[i])  # copy from mmap
        gf = np.array(self.global_features[i])

        g, zones, masks = parse_game_state(flat, gf)

        return {
            'global_features': torch.from_numpy(g),
            'my_board': torch.from_numpy(
                zones['my_board']),
            'my_board_mask': torch.from_numpy(
                masks['my_board_mask']),
            'opp_board': torch.from_numpy(
                zones['opp_board']),
            'opp_board_mask': torch.from_numpy(
                masks['opp_board_mask']),
            'hand': torch.from_numpy(zones['hand']),
            'hand_mask': torch.from_numpy(
                masks['hand_mask']),
            'my_gy': torch.from_numpy(zones['my_gy']),
            'my_gy_mask': torch.from_numpy(
                masks['my_gy_mask']),
            'opp_gy': torch.from_numpy(zones['opp_gy']),
            'opp_gy_mask': torch.from_numpy(
                masks['opp_gy_mask']),
            'stack': torch.from_numpy(zones['stack']),
            'stack_mask': torch.from_numpy(
                masks['stack_mask']),
            'value_target': torch.tensor(
                float(self.outcome[i]),
                dtype=torch.float32),
        }


class MmapPriorityDataset(Dataset):
    """Memory-mapped dataset for priority head training.

    Returns dicts compatible with make_priority_batch:
    {global_features, game_state_flat, action_features,
     selected_idx, n_actions, won}
    """

    def __init__(self, preprocessed_dir: str,
                 train: bool = True,
                 val_fraction: float = 0.1):
        pdir = os.path.join(preprocessed_dir, 'priority')
        self.game_state = np.load(
            os.path.join(pdir, 'game_state.npy'),
            mmap_mode='r')
        self.global_features = np.load(
            os.path.join(pdir, 'global_features.npy'),
            mmap_mode='r')
        self.candidates = np.load(
            os.path.join(pdir, 'candidates.npy'),
            mmap_mode='r')
        self.candidate_mask = np.load(
            os.path.join(pdir, 'candidate_mask.npy'),
            mmap_mode='r')
        self.selected_idx = np.load(
            os.path.join(pdir, 'selected_idx.npy'),
            mmap_mode='r')
        self.outcome = np.load(
            os.path.join(pdir, 'outcome.npy'),
            mmap_mode='r')
        self.file_id = np.load(
            os.path.join(pdir, 'file_id.npy'),
            mmap_mode='r')

        train_ids, val_ids = _split_file_ids(
            self.file_id, val_fraction)
        target_ids = train_ids if train else val_ids
        self.indices = np.array([
            i for i in range(len(self.file_id))
            if self.file_id[i] in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        mask = np.array(self.candidate_mask[i])
        n_actions = int(mask.sum())

        return {
            'global_features': np.array(
                self.global_features[i]),
            'game_state_flat': np.array(
                self.game_state[i]),
            'action_features': np.array(
                self.candidates[i, :n_actions]),
            'selected_idx': int(self.selected_idx[i]),
            'n_actions': n_actions,
            'won': float(self.outcome[i] > 0),
        }


class MmapAttackDataset(Dataset):
    """Memory-mapped dataset for attack head training."""

    def __init__(self, preprocessed_dir: str,
                 train: bool = True,
                 val_fraction: float = 0.1):
        adir = os.path.join(preprocessed_dir, 'attack')
        self.game_state = np.load(
            os.path.join(adir, 'game_state.npy'),
            mmap_mode='r')
        self.global_features = np.load(
            os.path.join(adir, 'global_features.npy'),
            mmap_mode='r')
        self.creatures = np.load(
            os.path.join(adir, 'creatures.npy'),
            mmap_mode='r')
        self.creature_mask = np.load(
            os.path.join(adir, 'creature_mask.npy'),
            mmap_mode='r')
        self.action_mask = np.load(
            os.path.join(adir, 'action_mask.npy'),
            mmap_mode='r')
        self.outcome = np.load(
            os.path.join(adir, 'outcome.npy'),
            mmap_mode='r')
        self.file_id = np.load(
            os.path.join(adir, 'file_id.npy'),
            mmap_mode='r')

        train_ids, val_ids = _split_file_ids(
            self.file_id, val_fraction)
        target_ids = train_ids if train else val_ids
        self.indices = np.array([
            i for i in range(len(self.file_id))
            if self.file_id[i] in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        cmask = np.array(self.creature_mask[i])
        n = int(cmask.sum())

        return {
            'global_features': np.array(
                self.global_features[i]),
            'game_state_flat': np.array(
                self.game_state[i]),
            'creature_features': np.array(
                self.creatures[i, :n]),
            'action_mask': np.array(
                self.action_mask[i, :n]),
            'n_creatures': n,
            'won': float(self.outcome[i] > 0),
        }


class MmapBlockDataset(Dataset):
    """Memory-mapped dataset for block head training."""

    def __init__(self, preprocessed_dir: str,
                 train: bool = True,
                 val_fraction: float = 0.1):
        bdir = os.path.join(preprocessed_dir, 'block')
        self.game_state = np.load(
            os.path.join(bdir, 'game_state.npy'),
            mmap_mode='r')
        self.global_features = np.load(
            os.path.join(bdir, 'global_features.npy'),
            mmap_mode='r')
        self.pairs = np.load(
            os.path.join(bdir, 'pairs.npy'),
            mmap_mode='r')
        self.pair_mask = np.load(
            os.path.join(bdir, 'pair_mask.npy'),
            mmap_mode='r')
        self.action_mask = np.load(
            os.path.join(bdir, 'action_mask.npy'),
            mmap_mode='r')
        self.outcome = np.load(
            os.path.join(bdir, 'outcome.npy'),
            mmap_mode='r')
        self.file_id = np.load(
            os.path.join(bdir, 'file_id.npy'),
            mmap_mode='r')

        train_ids, val_ids = _split_file_ids(
            self.file_id, val_fraction)
        target_ids = train_ids if train else val_ids
        self.indices = np.array([
            i for i in range(len(self.file_id))
            if self.file_id[i] in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        pmask = np.array(self.pair_mask[i])
        n = int(pmask.sum())

        return {
            'global_features': np.array(
                self.global_features[i]),
            'game_state_flat': np.array(
                self.game_state[i]),
            'pair_features': np.array(
                self.pairs[i, :n]),
            'action_mask': np.array(
                self.action_mask[i, :n]),
            'n_pairs': n,
            'won': float(self.outcome[i] > 0),
        }
