#!/usr/bin/env python3
"""
Preprocess trajectory JSONL files into memory-mapped numpy arrays.

Structure:
    preprocessed/
    ├── metadata.json
    ├── shared/
    │   ├── game_state.npy      (N_total × 37216, float32)
    │   ├── global_features.npy (N_total × 96, float32)
    │   ├── outcome.npy         (N_total, float32)
    │   └── file_id.npy         (N_total, int32)
    ├── priority/
    │   ├── gs_index.npy        (N_pri, int64) → row in shared
    │   ├── candidates.npy      (N_pri × MAX × 64)
    │   ├── candidate_mask.npy  (N_pri × MAX, bool)
    │   ├── selected_idx.npy    (N_pri, int32)
    │   └── action_probs.npy    (N_pri × MAX, float32)
    ├── attack/
    │   ├── gs_index.npy        → row in shared
    │   ├── creatures.npy       (N_atk × MAX × 256)
    │   ├── creature_mask.npy
    │   ├── action_mask.npy
    │   └── action_probs.npy
    └── block/
        ├── gs_index.npy        → row in shared
        ├── pairs.npy           (N_blk × MAX × 512)
        ├── pair_mask.npy
        ├── action_mask.npy
        └── action_probs.npy

Game state is stored ONCE in shared/. Each decision type stores
only an index into shared/ plus its type-specific candidate data.
Disk usage: ~13GB instead of ~25GB.

Usage:
    python training/preprocess_trajectories.py \
        --data-dir /path/to/trajectories \
        --output-dir /path/to/preprocessed
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Canonical dimension constants
GLOBAL_DIM = 96
CARD_DIM = 256
GAME_STATE_DIM = 37216  # 96 + 145*256


def scan_files(files):
    """Pass 1: Count samples, find max candidate sizes."""
    counts = {'total': 0, 'priority': 0,
              'attack': 0, 'block': 0}
    max_cand = {'priority': 0, 'attack': 0, 'block': 0}

    print(f"  Pass 1: Scanning {len(files)} files...",
          flush=True)

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    {fi+1}/{len(files)}", flush=True)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            for line in lines[1:]:
                counts['total'] += 1
                rec = json.loads(line)
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                nc = len(cand)

                if dt == 'PRIORITY_ACTION':
                    counts['priority'] += 1
                    max_cand['priority'] = max(
                        max_cand['priority'], nc)
                elif dt == 'DECLARE_ATTACKERS':
                    counts['attack'] += 1
                    max_cand['attack'] = max(
                        max_cand['attack'], nc)
                elif dt == 'DECLARE_BLOCKERS':
                    if cand and len(cand[0]) > CARD_DIM + 10:
                        counts['block'] += 1
                        max_cand['block'] = max(
                            max_cand['block'], nc)
        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    print(f"  Counts: {counts}", flush=True)
    print(f"  Max candidates: {max_cand}", flush=True)
    return counts, max_cand


GAMMA = 0.95  # discount factor for value returns

def sanitize(arr):
    """Clip and replace NaN — matches training code."""
    np.clip(arr, -10, 10, out=arr)
    np.nan_to_num(arr, copy=False)
    return arr


def preprocess(files, output_dir, counts, max_cand):
    """Pass 2: Create and fill memory-mapped arrays."""
    nt = counts['total']
    np_ = counts['priority']
    na = counts['attack']
    nb = counts['block']

    # Create directories
    for d in ['shared', 'priority', 'attack', 'block']:
        os.makedirs(os.path.join(output_dir, d),
                    exist_ok=True)

    # Shared arrays (one row per decision record)
    print(f"  Creating shared arrays ({nt} records)...",
          flush=True)
    sh = os.path.join(output_dir, 'shared')
    gs = np.lib.format.open_memmap(
        os.path.join(sh, 'game_state.npy'),
        mode='w+', dtype=np.float32,
        shape=(nt, GAME_STATE_DIM))
    gf = np.lib.format.open_memmap(
        os.path.join(sh, 'global_features.npy'),
        mode='w+', dtype=np.float32,
        shape=(nt, GLOBAL_DIM))
    outcome = np.lib.format.open_memmap(
        os.path.join(sh, 'outcome.npy'),
        mode='w+', dtype=np.float32, shape=(nt,))
    file_id = np.lib.format.open_memmap(
        os.path.join(sh, 'file_id.npy'),
        mode='w+', dtype=np.int32, shape=(nt,))

    # Priority arrays
    print(f"  Creating priority arrays ({np_} records, "
          f"max_cand={max_cand['priority']})...",
          flush=True)
    pd = os.path.join(output_dir, 'priority')
    mp = max_cand['priority']
    p_gs_idx = np.lib.format.open_memmap(
        os.path.join(pd, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(np_,))
    p_cand = np.lib.format.open_memmap(
        os.path.join(pd, 'candidates.npy'),
        mode='w+', dtype=np.float32,
        shape=(np_, mp, 64))
    p_cmask = np.lib.format.open_memmap(
        os.path.join(pd, 'candidate_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(np_, mp))
    p_sel = np.lib.format.open_memmap(
        os.path.join(pd, 'selected_idx.npy'),
        mode='w+', dtype=np.int32, shape=(np_,))
    p_aprobs = np.lib.format.open_memmap(
        os.path.join(pd, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(np_, mp))

    # Attack arrays
    print(f"  Creating attack arrays ({na} records, "
          f"max_cand={max_cand['attack']})...",
          flush=True)
    ad = os.path.join(output_dir, 'attack')
    ma = max_cand['attack']
    a_gs_idx = np.lib.format.open_memmap(
        os.path.join(ad, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(na,))
    a_crt = np.lib.format.open_memmap(
        os.path.join(ad, 'creatures.npy'),
        mode='w+', dtype=np.float32,
        shape=(na, ma, CARD_DIM))
    a_cmask = np.lib.format.open_memmap(
        os.path.join(ad, 'creature_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(na, ma))
    a_amask = np.lib.format.open_memmap(
        os.path.join(ad, 'action_mask.npy'),
        mode='w+', dtype=np.float32, shape=(na, ma))
    a_aprobs = np.lib.format.open_memmap(
        os.path.join(ad, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(na, ma))

    # Block arrays
    print(f"  Creating block arrays ({nb} records, "
          f"max_cand={max_cand['block']})...",
          flush=True)
    bd = os.path.join(output_dir, 'block')
    mb = max_cand['block']
    b_gs_idx = np.lib.format.open_memmap(
        os.path.join(bd, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(nb,))
    b_pairs = np.lib.format.open_memmap(
        os.path.join(bd, 'pairs.npy'),
        mode='w+', dtype=np.float32,
        shape=(nb, mb, CARD_DIM * 2))
    b_pmask = np.lib.format.open_memmap(
        os.path.join(bd, 'pair_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(nb, mb))
    b_amask = np.lib.format.open_memmap(
        os.path.join(bd, 'action_mask.npy'),
        mode='w+', dtype=np.float32, shape=(nb, mb))
    b_aprobs = np.lib.format.open_memmap(
        os.path.join(bd, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(nb, mb))

    # Pass 2: Fill arrays
    print(f"\n  Pass 2: Writing {len(files)} files...",
          flush=True)
    si = 0  # shared index
    pi = 0  # priority index
    ai = 0  # attack index
    bi = 0  # block index

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    {fi+1}/{len(files)} "
                  f"(s={si} p={pi} a={ai} b={bi})",
                  flush=True)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            # Parse all records first to compute
            # discounted returns backward
            records = []
            for line in lines[1:]:
                records.append(json.loads(line))

            # Compute discounted returns:
            # G_t = r_t + γ*G_{t+1}
            # where r_t = intermediateReward (shaping)
            # and terminal reward on last step
            n_recs = len(records)
            returns = np.zeros(n_recs, dtype=np.float32)
            G = 0.0
            for t in range(n_recs - 1, -1, -1):
                r = records[t].get(
                    'intermediateReward', 0.0)
                tr = records[t].get(
                    'terminalReward', 0.0)
                G = r + tr + GAMMA * G
                returns[t] = G

            for rec_idx, rec in enumerate(records):
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])
                aprobs = rec.get(
                    'actionProbabilities', [])

                # Parse game state
                flat_raw = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                if len(flat_raw) == 0:
                    continue
                flat = np.zeros(GAME_STATE_DIM,
                                dtype=np.float32)
                fl = min(len(flat_raw), GAME_STATE_DIM)
                flat[:fl] = flat_raw[:fl]
                sanitize(flat)

                gf_raw = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                g = np.zeros(GLOBAL_DIM, dtype=np.float32)
                gl = min(len(gf_raw), GLOBAL_DIM)
                if gl > 0:
                    g[:gl] = gf_raw[:gl]
                sanitize(g)

                # Write shared — discounted return, not
                # binary outcome
                gs[si] = flat
                gf[si] = g
                outcome[si] = returns[rec_idx]
                file_id[si] = fi
                shared_row = si
                si += 1

                nc = len(cand)

                if dt == 'PRIORITY_ACTION':
                    p_gs_idx[pi] = shared_row
                    for j in range(min(nc, mp)):
                        cf = cand[j]
                        cl = min(len(cf), 64)
                        p_cand[pi, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        p_cmask[pi, j] = True
                    si_val = sel[0] if sel else nc - 1
                    if si_val >= mp:
                        si_val = mp - 1
                    p_sel[pi] = si_val
                    for j in range(min(len(aprobs), mp)):
                        p_aprobs[pi, j] = aprobs[j]
                    pi += 1

                elif dt == 'DECLARE_ATTACKERS':
                    a_gs_idx[ai] = shared_row
                    for j in range(min(nc, ma)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM)
                        a_crt[ai, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        a_cmask[ai, j] = True
                    for s in sel:
                        if 0 <= s < ma:
                            a_amask[ai, s] = 1.0
                    for j in range(min(len(aprobs), ma)):
                        a_aprobs[ai, j] = aprobs[j]
                    ai += 1

                elif dt == 'DECLARE_BLOCKERS':
                    if not cand or len(cand[0]) <= CARD_DIM + 10:
                        continue
                    b_gs_idx[bi] = shared_row
                    for j in range(min(nc, mb)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM * 2)
                        b_pairs[bi, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        b_pmask[bi, j] = True
                    for s in sel:
                        if 0 <= s < mb:
                            b_amask[bi, s] = 1.0
                    for j in range(min(len(aprobs), mb)):
                        b_aprobs[bi, j] = aprobs[j]
                    bi += 1

        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    # Flush
    for arr in [gs, gf, outcome, file_id,
                p_gs_idx, p_cand, p_cmask, p_sel,
                p_aprobs,
                a_gs_idx, a_crt, a_cmask, a_amask,
                a_aprobs,
                b_gs_idx, b_pairs, b_pmask, b_amask,
                b_aprobs]:
        arr.flush()

    return {'total': si, 'priority': pi,
            'attack': ai, 'block': bi}


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectories to mmap')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--output-dir',
        default='../../rl_data/preprocessed')
    args = parser.parse_args()

    t0 = time.time()
    print("=== Trajectory Preprocessing ===", flush=True)

    path = Path(args.data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    print(f"  Found {len(files)} trajectory files",
          flush=True)
    if not files:
        print("  ERROR: No files found", flush=True)
        sys.exit(1)

    # Pass 1
    counts, max_cand = scan_files(files)

    # Create output
    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 2
    final = preprocess(files, args.output_dir,
                       counts, max_cand)

    # Metadata
    metadata = {
        'source_dir': str(args.data_dir),
        'n_files': len(files),
        'counts': counts,
        'final_counts': final,
        'max_candidates': max_cand,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'game_state_dim': GAME_STATE_DIM,
        'global_feature_dim': GLOBAL_DIM,
        'card_feature_dim': CARD_DIM,
        'shared_game_state': True,
        'discount_gamma': GAMMA,
        'value_targets': 'discounted_returns',
    }
    with open(os.path.join(args.output_dir,
                           'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Preprocessing Complete ===", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Shared: {final['total']} records",
          flush=True)
    print(f"  Priority: {final['priority']}", flush=True)
    print(f"  Attack: {final['attack']}", flush=True)
    print(f"  Block: {final['block']}", flush=True)

    # Disk usage
    total_bytes = 0
    for root, dirs, fnames in os.walk(args.output_dir):
        for fn in fnames:
            total_bytes += os.path.getsize(
                os.path.join(root, fn))
    print(f"  Disk: {total_bytes/1024**3:.1f} GB",
          flush=True)


if __name__ == '__main__':
    main()
