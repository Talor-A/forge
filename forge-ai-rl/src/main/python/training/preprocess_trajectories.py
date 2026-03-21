#!/usr/bin/env python3
"""
Preprocess trajectory JSONL files into memory-mapped numpy arrays.

Converts raw JSONL trajectory files into a binary format that can be
loaded with near-zero RAM overhead via numpy mmap. Run once after data
collection, before training.

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


def scan_files(files):
    """Pass 1: Count samples per type, find max candidate
    sizes, assign file IDs."""
    counts = {
        'value': 0,
        'priority': 0,
        'attack': 0,
        'block': 0,
    }
    max_cand = {
        'priority': 0,
        'attack': 0,
        'block': 0,
    }
    file_map = {}  # sample_idx -> file_idx

    print(f"  Pass 1: Scanning {len(files)} files...",
          flush=True)

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    Scanned {fi+1}/{len(files)} files",
                  flush=True)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                n_cand = len(cand)

                # Every record is a value sample
                counts['value'] += 1

                if dt == 'PRIORITY_ACTION':
                    counts['priority'] += 1
                    if n_cand > max_cand['priority']:
                        max_cand['priority'] = n_cand
                elif dt == 'DECLARE_ATTACKERS':
                    counts['attack'] += 1
                    if n_cand > max_cand['attack']:
                        max_cand['attack'] = n_cand
                elif dt == 'DECLARE_BLOCKERS':
                    # Only count 256-dim pair format
                    if cand and len(cand[0]) > 200:
                        counts['block'] += 1
                        if n_cand > max_cand['block']:
                            max_cand['block'] = n_cand
        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    print(f"  Counts: {counts}", flush=True)
    print(f"  Max candidates: {max_cand}", flush=True)
    return counts, max_cand


def create_arrays(output_dir, counts, max_cand):
    """Create memory-mapped arrays in write mode."""
    arrays = {}

    # Value arrays
    vdir = os.path.join(output_dir, 'value')
    os.makedirs(vdir, exist_ok=True)
    n = counts['value']
    if n > 0:
        arrays['value'] = {
            'game_state': np.lib.format.open_memmap(
                os.path.join(vdir, 'game_state.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 21184)),
            'global_features': np.lib.format.open_memmap(
                os.path.join(vdir, 'global_features.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 64)),
            'outcome': np.lib.format.open_memmap(
                os.path.join(vdir, 'outcome.npy'),
                mode='w+', dtype=np.float32,
                shape=(n,)),
            'file_id': np.lib.format.open_memmap(
                os.path.join(vdir, 'file_id.npy'),
                mode='w+', dtype=np.int32,
                shape=(n,)),
        }

    # Priority arrays
    pdir = os.path.join(output_dir, 'priority')
    os.makedirs(pdir, exist_ok=True)
    n = counts['priority']
    mc = max_cand['priority']
    if n > 0 and mc > 0:
        arrays['priority'] = {
            'game_state': np.lib.format.open_memmap(
                os.path.join(pdir, 'game_state.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 21184)),
            'global_features': np.lib.format.open_memmap(
                os.path.join(pdir, 'global_features.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 64)),
            'candidates': np.lib.format.open_memmap(
                os.path.join(pdir, 'candidates.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc, 64)),
            'candidate_mask': np.lib.format.open_memmap(
                os.path.join(pdir, 'candidate_mask.npy'),
                mode='w+', dtype=np.bool_,
                shape=(n, mc)),
            'selected_idx': np.lib.format.open_memmap(
                os.path.join(pdir, 'selected_idx.npy'),
                mode='w+', dtype=np.int32,
                shape=(n,)),
            'outcome': np.lib.format.open_memmap(
                os.path.join(pdir, 'outcome.npy'),
                mode='w+', dtype=np.float32,
                shape=(n,)),
            'action_probs': np.lib.format.open_memmap(
                os.path.join(pdir, 'action_probs.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc)),
            'file_id': np.lib.format.open_memmap(
                os.path.join(pdir, 'file_id.npy'),
                mode='w+', dtype=np.int32,
                shape=(n,)),
        }

    # Attack arrays
    adir = os.path.join(output_dir, 'attack')
    os.makedirs(adir, exist_ok=True)
    n = counts['attack']
    mc = max_cand['attack']
    if n > 0 and mc > 0:
        arrays['attack'] = {
            'game_state': np.lib.format.open_memmap(
                os.path.join(adir, 'game_state.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 21184)),
            'global_features': np.lib.format.open_memmap(
                os.path.join(adir, 'global_features.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 64)),
            'creatures': np.lib.format.open_memmap(
                os.path.join(adir, 'creatures.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc, 128)),
            'creature_mask': np.lib.format.open_memmap(
                os.path.join(adir, 'creature_mask.npy'),
                mode='w+', dtype=np.bool_,
                shape=(n, mc)),
            'action_mask': np.lib.format.open_memmap(
                os.path.join(adir, 'action_mask.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc)),
            'outcome': np.lib.format.open_memmap(
                os.path.join(adir, 'outcome.npy'),
                mode='w+', dtype=np.float32,
                shape=(n,)),
            'action_probs': np.lib.format.open_memmap(
                os.path.join(adir, 'action_probs.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc)),
            'file_id': np.lib.format.open_memmap(
                os.path.join(adir, 'file_id.npy'),
                mode='w+', dtype=np.int32,
                shape=(n,)),
        }

    # Block arrays (256-dim pairs)
    bdir = os.path.join(output_dir, 'block')
    os.makedirs(bdir, exist_ok=True)
    n = counts['block']
    mc = max_cand['block']
    if n > 0 and mc > 0:
        arrays['block'] = {
            'game_state': np.lib.format.open_memmap(
                os.path.join(bdir, 'game_state.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 21184)),
            'global_features': np.lib.format.open_memmap(
                os.path.join(bdir, 'global_features.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, 64)),
            'pairs': np.lib.format.open_memmap(
                os.path.join(bdir, 'pairs.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc, 256)),
            'pair_mask': np.lib.format.open_memmap(
                os.path.join(bdir, 'pair_mask.npy'),
                mode='w+', dtype=np.bool_,
                shape=(n, mc)),
            'action_mask': np.lib.format.open_memmap(
                os.path.join(bdir, 'action_mask.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc)),
            'outcome': np.lib.format.open_memmap(
                os.path.join(bdir, 'outcome.npy'),
                mode='w+', dtype=np.float32,
                shape=(n,)),
            'action_probs': np.lib.format.open_memmap(
                os.path.join(bdir, 'action_probs.npy'),
                mode='w+', dtype=np.float32,
                shape=(n, mc)),
            'file_id': np.lib.format.open_memmap(
                os.path.join(bdir, 'file_id.npy'),
                mode='w+', dtype=np.int32,
                shape=(n,)),
        }

    return arrays


def sanitize(arr):
    """Clip and replace NaN — matches training code."""
    np.clip(arr, -10, 10, out=arr)
    np.nan_to_num(arr, copy=False)
    return arr


def fill_arrays(files, arrays, counts, max_cand):
    """Pass 2: Stream through JSONL files, fill arrays."""
    idx = {k: 0 for k in counts}

    print(f"  Pass 2: Writing {len(files)} files...",
          flush=True)

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    Processed {fi+1}/{len(files)} files"
                  f" (v={idx['value']}, p={idx['priority']},"
                  f" a={idx['attack']}, b={idx['block']})",
                  flush=True)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue

            header = json.loads(lines[0])
            won = header.get('won', False)
            outcome = 1.0 if won else -1.0

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])
                aprobs = rec.get(
                    'actionProbabilities', [])

                # Game state (shared across all types)
                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                if len(flat) == 0:
                    continue
                # Pad/truncate to 21184
                gs = np.zeros(21184, dtype=np.float32)
                gl = min(len(flat), 21184)
                gs[:gl] = flat[:gl]
                sanitize(gs)

                # Global features
                gf_raw = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                gf = np.zeros(64, dtype=np.float32)
                gfl = min(len(gf_raw), 64)
                if gfl > 0:
                    gf[:gfl] = gf_raw[:gfl]
                sanitize(gf)

                # Value sample (every record)
                vi = idx['value']
                if 'value' in arrays:
                    arrays['value']['game_state'][vi] = gs
                    arrays['value']['global_features'][vi] = gf
                    arrays['value']['outcome'][vi] = outcome
                    arrays['value']['file_id'][vi] = fi
                    idx['value'] += 1

                n_cand = len(cand)

                if dt == 'PRIORITY_ACTION' and \
                        'priority' in arrays:
                    pi = idx['priority']
                    mc = max_cand['priority']
                    arrays['priority']['game_state'][pi] = gs
                    arrays['priority']['global_features'][pi] = gf
                    arrays['priority']['outcome'][pi] = outcome
                    arrays['priority']['file_id'][pi] = fi

                    # Candidates (64-dim)
                    for j in range(min(n_cand, mc)):
                        cf = cand[j]
                        cl = min(len(cf), 64)
                        arrays['priority']['candidates'][
                            pi, j, :cl] = sanitize(
                                np.array(cf[:cl],
                                         dtype=np.float32))
                        arrays['priority']['candidate_mask'][
                            pi, j] = True

                    # Selected index
                    si = sel[0] if sel else n_cand - 1
                    if si >= mc:
                        si = mc - 1
                    arrays['priority']['selected_idx'][pi] = si

                    # Action probs
                    for j in range(min(len(aprobs), mc)):
                        arrays['priority']['action_probs'][
                            pi, j] = aprobs[j]

                    idx['priority'] += 1

                elif dt == 'DECLARE_ATTACKERS' and \
                        'attack' in arrays:
                    ai = idx['attack']
                    mc = max_cand['attack']
                    arrays['attack']['game_state'][ai] = gs
                    arrays['attack']['global_features'][ai] = gf
                    arrays['attack']['outcome'][ai] = outcome
                    arrays['attack']['file_id'][ai] = fi

                    # Creatures (128-dim)
                    for j in range(min(n_cand, mc)):
                        cf = cand[j]
                        cl = min(len(cf), 128)
                        arrays['attack']['creatures'][
                            ai, j, :cl] = sanitize(
                                np.array(cf[:cl],
                                         dtype=np.float32))
                        arrays['attack']['creature_mask'][
                            ai, j] = True

                    # Action mask (multi-select)
                    for si in sel:
                        if 0 <= si < mc:
                            arrays['attack']['action_mask'][
                                ai, si] = 1.0

                    # Action probs
                    for j in range(min(len(aprobs), mc)):
                        arrays['attack']['action_probs'][
                            ai, j] = aprobs[j]

                    idx['attack'] += 1

                elif dt == 'DECLARE_BLOCKERS' and \
                        'block' in arrays:
                    # Only 256-dim pair format
                    if not cand or len(cand[0]) <= 200:
                        continue
                    bi = idx['block']
                    mc = max_cand['block']
                    arrays['block']['game_state'][bi] = gs
                    arrays['block']['global_features'][bi] = gf
                    arrays['block']['outcome'][bi] = outcome
                    arrays['block']['file_id'][bi] = fi

                    # Pairs (256-dim)
                    for j in range(min(n_cand, mc)):
                        cf = cand[j]
                        cl = min(len(cf), 256)
                        arrays['block']['pairs'][
                            bi, j, :cl] = sanitize(
                                np.array(cf[:cl],
                                         dtype=np.float32))
                        arrays['block']['pair_mask'][
                            bi, j] = True

                    # Action mask (multi-select)
                    for si in sel:
                        if 0 <= si < mc:
                            arrays['block']['action_mask'][
                                bi, si] = 1.0

                    # Action probs
                    for j in range(min(len(aprobs), mc)):
                        arrays['block']['action_probs'][
                            bi, j] = aprobs[j]

                    idx['block'] += 1

        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    # Flush all mmap arrays
    for dtype_arrays in arrays.values():
        for arr in dtype_arrays.values():
            arr.flush()

    return idx


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

    data_dir = args.data_dir
    output_dir = args.output_dir

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    print(f"  Found {len(files)} trajectory files",
          flush=True)

    if not files:
        print("  ERROR: No files found", flush=True)
        sys.exit(1)

    # Pass 1: Scan
    counts, max_cand = scan_files(files)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create arrays
    print(f"\n  Creating memory-mapped arrays...",
          flush=True)
    arrays = create_arrays(output_dir, counts, max_cand)

    # Pass 2: Fill
    final_idx = fill_arrays(
        files, arrays, counts, max_cand)

    # Write metadata
    metadata = {
        'source_dir': str(data_dir),
        'n_files': len(files),
        'counts': counts,
        'final_counts': final_idx,
        'max_candidates': max_cand,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'game_state_dim': 21184,
        'global_feature_dim': 64,
        'priority_feature_dim': 64,
        'attack_feature_dim': 128,
        'block_feature_dim': 256,
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Preprocessing Complete ===", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Value samples: {final_idx['value']}",
          flush=True)
    print(f"  Priority samples: {final_idx['priority']}",
          flush=True)
    print(f"  Attack samples: {final_idx['attack']}",
          flush=True)
    print(f"  Block samples: {final_idx['block']}",
          flush=True)

    # Report disk usage
    total_bytes = 0
    for root, dirs, fnames in os.walk(output_dir):
        for fn in fnames:
            total_bytes += os.path.getsize(
                os.path.join(root, fn))
    print(f"  Disk usage: {total_bytes/1024**3:.1f} GB",
          flush=True)


if __name__ == '__main__':
    main()
