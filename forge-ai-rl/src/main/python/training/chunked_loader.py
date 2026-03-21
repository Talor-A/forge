"""
Chunked data loader — loads trajectory files in chunks to bound memory.

Instead of loading all 2000 files at once (~12GB), loads N files at a
time, yields samples, then moves to the next chunk. This keeps memory
usage at O(chunk_size * samples_per_file) instead of O(total_samples).

Compatible with existing training code that expects lists of dicts.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator

from training.mmap_dataset import (
    parse_game_state, CARD_DIM, GLOBAL_DIM,
    ZONES_CONFIG as _ZONES_CONFIG, GAME_STATE_DIM)


def chunked_file_loader(data_dir: str,
                        chunk_size: int = 200,
                        max_files: Optional[int] = None,
                        shuffle_files: bool = True):
    """Yield chunks of trajectory files.

    Args:
        data_dir: path to traj_*.jsonl files
        chunk_size: files per chunk
        max_files: total file limit
        shuffle_files: randomize file order

    Yields:
        list of file paths for each chunk
    """
    import random
    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    if shuffle_files:
        random.shuffle(files)

    for i in range(0, len(files), chunk_size):
        yield files[i:i + chunk_size]


ZONES_CONFIG = _ZONES_CONFIG


def load_value_samples(files: List[Path]) -> list:
    """Load value training samples from a chunk of files.

    Returns samples in the format expected by SimpleDataset:
    {global_features, zones{...}, masks{...}, value_target}
    """
    card_dim = CARD_DIM
    samples = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            for line in lines[1:]:
                rec = json.loads(line)
                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                if len(flat) == 0:
                    continue
                np.clip(gf, -10, 10, out=gf)
                np.clip(flat, -10, 10, out=flat)
                gf = np.nan_to_num(gf)
                flat = np.nan_to_num(flat)

                g = np.zeros(GLOBAL_DIM, dtype=np.float32)
                gl = min(len(gf), GLOBAL_DIM)
                if gl > 0:
                    g[:gl] = gf[:gl]

                # Parse zones from flat state
                zones = {}
                masks = {}
                offset = GLOBAL_DIM
                for name, count in ZONES_CONFIG:
                    zs = count * card_dim
                    zd = np.zeros((count, card_dim),
                                  dtype=np.float32)
                    zm = np.zeros(count, dtype=np.bool_)
                    if offset + zs <= len(flat):
                        raw = flat[offset:offset + zs
                                   ].reshape(count, card_dim)
                        for j in range(count):
                            if np.any(raw[j] != 0):
                                zd[j] = raw[j]
                                zm[j] = True
                    offset += zs
                    zones[name] = zd
                    masks[name + '_mask'] = zm

                samples.append({
                    'global_features': g,
                    'zones': zones,
                    'masks': masks,
                    'value_target':
                        1.0 if won else -1.0,
                })
        except Exception:
            pass
    return samples


def load_decision_samples(files: List[Path],
                          heads: Optional[List[str]] = None):
    """Load decision training samples from a chunk of files.

    Returns:
        (attack_samples, block_samples, priority_samples)
    """
    attack_samples = []
    block_samples = []
    priority_samples = []

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get('decisionType', '')

                # Skip types we don't need
                if heads:
                    if dt == 'PRIORITY_ACTION' and \
                            'priority' not in heads:
                        continue
                    if dt == 'DECLARE_ATTACKERS' and \
                            'attack' not in heads:
                        continue
                    if dt == 'DECLARE_BLOCKERS' and \
                            'block' not in heads:
                        continue

                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])
                if len(cand) < 1:
                    continue

                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(gf, -10, 10, out=gf)
                gf = np.nan_to_num(gf)
                g = np.zeros(GLOBAL_DIM, dtype=np.float32)
                gl = min(len(gf), GLOBAL_DIM)
                if gl > 0:
                    g[:gl] = gf[:gl]

                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                if dt == 'PRIORITY_ACTION':
                    n = len(cand)
                    actions = np.zeros(
                        (n, 64), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 64)
                        actions[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(actions, -10, 10, out=actions)
                    actions = np.nan_to_num(actions)

                    selected_idx = sel[0] if sel else n - 1
                    if selected_idx >= n:
                        selected_idx = n - 1

                    priority_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'action_features': actions,
                        'selected_idx': selected_idx,
                        'n_actions': n,
                        'won': 1.0 if won else 0.0,
                    })

                elif dt == 'DECLARE_ATTACKERS':
                    n = len(cand)
                    creatures = np.zeros(
                        (n, CARD_DIM), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), CARD_DIM)
                        creatures[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(creatures, -10, 10,
                            out=creatures)
                    creatures = np.nan_to_num(creatures)

                    mask = np.zeros(n, dtype=np.float32)
                    for idx in sel:
                        if 0 <= idx < n:
                            mask[idx] = 1.0

                    attack_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'creature_features': creatures,
                        'action_mask': mask,
                        'n_creatures': n,
                        'won': 1.0 if won else 0.0,
                    })

                elif dt == 'DECLARE_BLOCKERS':
                    n = len(cand)
                    feat_dim = len(cand[0]) if cand else 0
                    if feat_dim < 200:
                        continue  # skip old format

                    pairs = np.zeros(
                        (n, 256), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 256)
                        pairs[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(pairs, -10, 10, out=pairs)
                    pairs = np.nan_to_num(pairs)

                    mask = np.zeros(n, dtype=np.float32)
                    for idx in sel:
                        if 0 <= idx < n:
                            mask[idx] = 1.0

                    block_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'pair_features': pairs,
                        'action_mask': mask,
                        'n_pairs': n,
                        'won': 1.0 if won else 0.0,
                    })
        except Exception:
            pass

    return attack_samples, block_samples, priority_samples
