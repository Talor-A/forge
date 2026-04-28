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

Usage:
    python training/preprocess_trajectories.py \
        --data-dir /path/to/trajectories \
        --output-dir /path/to/preprocessed \
        --workers 16
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

try:
    import orjson
    def _loads(s):
        return orjson.loads(s)
except ImportError:
    def _loads(s):
        return json.loads(s)

# Canonical dimension constants
GLOBAL_DIM = 96
CARD_DIM = 256
GAME_STATE_DIM = 37216  # 96 + 145*256

DECISION_KEYS = ('priority', 'attack', 'block', 'target',
                 'card_select', 'mulligan', 'binary')


def _scan_one(filepath):
    """Scan a single file. Returns (per_file_counts, max_cand).
    per_file_counts has 'total' + each decision type."""
    counts = {'total': 0}
    for k in DECISION_KEYS:
        counts[k] = 0
    max_cand = {k: 0 for k in DECISION_KEYS if k != 'binary'}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return filepath, counts, max_cand
        for line in lines[1:]:
            counts['total'] += 1
            rec = _loads(line)
            dt = rec.get('decisionType', '')
            cand = rec.get('candidateFeatures', [])
            nc = len(cand)
            if dt == 'PRIORITY_ACTION':
                counts['priority'] += 1
                if nc > max_cand['priority']:
                    max_cand['priority'] = nc
            elif dt == 'DECLARE_ATTACKERS':
                counts['attack'] += 1
                if nc > max_cand['attack']:
                    max_cand['attack'] = nc
            elif dt == 'DECLARE_BLOCKERS':
                if cand and len(cand[0]) > CARD_DIM + 10:
                    counts['block'] += 1
                    if nc > max_cand['block']:
                        max_cand['block'] = nc
            elif dt == 'TARGET_SELECTION':
                if nc > 1:
                    counts['target'] += 1
                    if nc > max_cand['target']:
                        max_cand['target'] = nc
            elif dt == 'CARD_SELECTION':
                if nc >= 1:
                    counts['card_select'] += 1
                    if nc > max_cand['card_select']:
                        max_cand['card_select'] = nc
            elif dt == 'MULLIGAN':
                counts['mulligan'] += 1
                if nc > max_cand['mulligan']:
                    max_cand['mulligan'] = nc
            elif dt == 'BINARY_CHOICE':
                counts['binary'] += 1
    except Exception as e:
        print(f"    Warning: {filepath}: {e}", flush=True)
    return filepath, counts, max_cand


def scan_files(files, workers=1, progress_cb=None):
    """Pass 1 (parallel): per-file counts + global max_cand."""
    print(f"  Pass 1: Scanning {len(files)} files "
          f"({workers} workers)...", flush=True)

    per_file = [None] * len(files)
    totals = {'total': 0}
    for k in DECISION_KEYS:
        totals[k] = 0
    g_max = {k: 0 for k in DECISION_KEYS if k != 'binary'}

    def _accumulate(i, c, m):
        per_file[i] = c
        for k, v in c.items():
            totals[k] += v
        for k, v in m.items():
            if v > g_max[k]:
                g_max[k] = v

    if workers <= 1:
        for fi, fp in enumerate(files):
            _, c, m = _scan_one(fp)
            _accumulate(fi, c, m)
            if (fi + 1) % 200 == 0:
                print(f"    {fi+1}/{len(files)}", flush=True)
            if progress_cb and fi % 20 == 0:
                progress_cb(fi, len(files))
    else:
        with Pool(workers) as pool:
            for fi, (_, c, m) in enumerate(
                    pool.imap(_scan_one, files, chunksize=8)):
                _accumulate(fi, c, m)
                if (fi + 1) % 500 == 0:
                    print(f"    {fi+1}/{len(files)}", flush=True)
                if progress_cb and fi % 50 == 0:
                    progress_cb(fi, len(files))

    if progress_cb:
        progress_cb(len(files), len(files))
    print(f"  Counts: {totals}", flush=True)
    print(f"  Max candidates: {g_max}", flush=True)
    return totals, g_max, per_file


GAMMA = 0.99  # discount factor for value returns


def _extract_game_id(filepath):
    name = os.path.basename(str(filepath)).replace('.jsonl', '')
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[-1]
    return name


def _build_game_id_map(files):
    ts_to_game = {}
    game_ids = []
    next_id = 0
    for filepath in files:
        ts = _extract_game_id(filepath)
        if ts not in ts_to_game:
            ts_to_game[ts] = next_id
            next_id += 1
        game_ids.append(ts_to_game[ts])
    return game_ids


def sanitize(arr):
    np.clip(arr, -10, 10, out=arr)
    np.nan_to_num(arr, copy=False)
    return arr


# Worker globals (initialized once per process)
_W = {}


def _init_worker(output_dir, shapes, has_keys):
    """Open all mmaps in r+ mode in this worker."""
    sh = os.path.join(output_dir, 'shared')
    _W['gs'] = np.load(os.path.join(sh, 'game_state.npy'),
                       mmap_mode='r+')
    _W['gf'] = np.load(os.path.join(sh, 'global_features.npy'),
                       mmap_mode='r+')
    _W['outcome'] = np.load(os.path.join(sh, 'outcome.npy'),
                            mmap_mode='r+')
    _W['file_id'] = np.load(os.path.join(sh, 'file_id.npy'),
                            mmap_mode='r+')

    pd = os.path.join(output_dir, 'priority')
    _W['p_gs_idx'] = np.load(os.path.join(pd, 'gs_index.npy'), mmap_mode='r+')
    _W['p_cand'] = np.load(os.path.join(pd, 'candidates.npy'), mmap_mode='r+')
    _W['p_cmask'] = np.load(os.path.join(pd, 'candidate_mask.npy'), mmap_mode='r+')
    _W['p_sel'] = np.load(os.path.join(pd, 'selected_idx.npy'), mmap_mode='r+')
    _W['p_aprobs'] = np.load(os.path.join(pd, 'action_probs.npy'), mmap_mode='r+')

    ad = os.path.join(output_dir, 'attack')
    _W['a_gs_idx'] = np.load(os.path.join(ad, 'gs_index.npy'), mmap_mode='r+')
    _W['a_crt'] = np.load(os.path.join(ad, 'creatures.npy'), mmap_mode='r+')
    _W['a_cmask'] = np.load(os.path.join(ad, 'creature_mask.npy'), mmap_mode='r+')
    _W['a_amask'] = np.load(os.path.join(ad, 'action_mask.npy'), mmap_mode='r+')
    _W['a_aprobs'] = np.load(os.path.join(ad, 'action_probs.npy'), mmap_mode='r+')

    bd = os.path.join(output_dir, 'block')
    _W['b_gs_idx'] = np.load(os.path.join(bd, 'gs_index.npy'), mmap_mode='r+')
    _W['b_pairs'] = np.load(os.path.join(bd, 'pairs.npy'), mmap_mode='r+')
    _W['b_pmask'] = np.load(os.path.join(bd, 'pair_mask.npy'), mmap_mode='r+')
    _W['b_amask'] = np.load(os.path.join(bd, 'action_mask.npy'), mmap_mode='r+')
    _W['b_aprobs'] = np.load(os.path.join(bd, 'action_probs.npy'), mmap_mode='r+')

    if has_keys['target']:
        td = os.path.join(output_dir, 'target')
        _W['t_gs_idx'] = np.load(os.path.join(td, 'gs_index.npy'), mmap_mode='r+')
        _W['t_cand'] = np.load(os.path.join(td, 'candidates.npy'), mmap_mode='r+')
        _W['t_cmask'] = np.load(os.path.join(td, 'candidate_mask.npy'), mmap_mode='r+')
        _W['t_sel'] = np.load(os.path.join(td, 'selected_idx.npy'), mmap_mode='r+')
        _W['t_spell'] = np.load(os.path.join(td, 'spell_features.npy'), mmap_mode='r+')

    if has_keys['card_select']:
        csd = os.path.join(output_dir, 'card_select')
        _W['cs_gs_idx'] = np.load(os.path.join(csd, 'gs_index.npy'), mmap_mode='r+')
        _W['cs_cand'] = np.load(os.path.join(csd, 'candidates.npy'), mmap_mode='r+')
        _W['cs_cmask'] = np.load(os.path.join(csd, 'candidate_mask.npy'), mmap_mode='r+')
        _W['cs_amask'] = np.load(os.path.join(csd, 'action_mask.npy'), mmap_mode='r+')

    if has_keys['mulligan']:
        muld = os.path.join(output_dir, 'mulligan')
        _W['mul_gs_idx'] = np.load(os.path.join(muld, 'gs_index.npy'), mmap_mode='r+')
        _W['mul_hand'] = np.load(os.path.join(muld, 'hand_features.npy'), mmap_mode='r+')
        _W['mul_hmask'] = np.load(os.path.join(muld, 'hand_mask.npy'), mmap_mode='r+')
        _W['mul_keep'] = np.load(os.path.join(muld, 'keep_decision.npy'), mmap_mode='r+')

    if has_keys['binary']:
        bind = os.path.join(output_dir, 'binary')
        _W['bin_gs_idx'] = np.load(os.path.join(bind, 'gs_index.npy'), mmap_mode='r+')
        _W['bin_decision'] = np.load(os.path.join(bind, 'decision.npy'), mmap_mode='r+')

    _W['shapes'] = shapes


def _process_file(task):
    """Write one file's records to its assigned mmap offsets."""
    (filepath, game_id, offsets) = task
    shapes = _W['shapes']
    mp = shapes['mp']; ma = shapes['ma']; mb = shapes['mb']
    mt = shapes['mt']; mcs = shapes['mcs']; mmul = shapes['mmul']
    n_tgt = shapes['n_tgt']; n_cs = shapes['n_cs']
    n_mul = shapes['n_mul']; n_bin = shapes['n_bin']

    si = offsets['total']
    pi = offsets['priority']
    ai = offsets['attack']
    bi = offsets['block']
    ti = offsets['target']
    ci = offsets['card_select']
    mi = offsets['mulligan']
    bni = offsets['binary']

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return 0
        records = [_loads(l) for l in lines[1:]]

        n_recs = len(records)
        returns = np.zeros(n_recs, dtype=np.float32)
        G = 0.0
        for t in range(n_recs - 1, -1, -1):
            r = records[t].get('intermediateReward', 0.0)
            tr = records[t].get('terminalReward', 0.0)
            G = r + tr + GAMMA * G
            returns[t] = G

        gs = _W['gs']; gf = _W['gf']
        outcome = _W['outcome']; file_id = _W['file_id']

        for rec_idx, rec in enumerate(records):
            dt = rec.get('decisionType', '')
            cand = rec.get('candidateFeatures', [])
            sel = rec.get('selectedIndices', [])
            aprobs = rec.get('actionProbabilities', [])

            flat_raw = np.asarray(rec.get('gameStateFlat', []),
                                  dtype=np.float32)
            if len(flat_raw) == 0:
                continue
            flat = np.zeros(GAME_STATE_DIM, dtype=np.float32)
            fl = min(len(flat_raw), GAME_STATE_DIM)
            flat[:fl] = flat_raw[:fl]
            sanitize(flat)

            gf_raw = np.asarray(rec.get('globalFeatures', []),
                                dtype=np.float32)
            g = np.zeros(GLOBAL_DIM, dtype=np.float32)
            gl = min(len(gf_raw), GLOBAL_DIM)
            if gl > 0:
                g[:gl] = gf_raw[:gl]
            sanitize(g)

            gs[si] = flat
            gf[si] = g
            outcome[si] = returns[rec_idx]
            file_id[si] = game_id
            shared_row = si
            si += 1

            nc = len(cand)

            if dt == 'PRIORITY_ACTION':
                _W['p_gs_idx'][pi] = shared_row
                for j in range(min(nc, mp)):
                    cf = cand[j]
                    cl = min(len(cf), 64)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['p_cand'][pi, j, :cl] = arr
                    _W['p_cmask'][pi, j] = True
                si_val = sel[0] if sel else nc - 1
                if si_val >= mp:
                    si_val = mp - 1
                _W['p_sel'][pi] = si_val
                for j in range(min(len(aprobs), mp)):
                    _W['p_aprobs'][pi, j] = aprobs[j]
                pi += 1

            elif dt == 'DECLARE_ATTACKERS':
                _W['a_gs_idx'][ai] = shared_row
                for j in range(min(nc, ma)):
                    cf = cand[j]
                    cl = min(len(cf), CARD_DIM)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['a_crt'][ai, j, :cl] = arr
                    _W['a_cmask'][ai, j] = True
                for s in sel:
                    if 0 <= s < ma:
                        _W['a_amask'][ai, s] = 1.0
                for j in range(min(len(aprobs), ma)):
                    _W['a_aprobs'][ai, j] = aprobs[j]
                ai += 1

            elif dt == 'DECLARE_BLOCKERS':
                if not cand or len(cand[0]) <= CARD_DIM + 10:
                    continue
                _W['b_gs_idx'][bi] = shared_row
                for j in range(min(nc, mb)):
                    cf = cand[j]
                    cl = min(len(cf), CARD_DIM * 2)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['b_pairs'][bi, j, :cl] = arr
                    _W['b_pmask'][bi, j] = True
                for s in sel:
                    if 0 <= s < mb:
                        _W['b_amask'][bi, s] = 1.0
                for j in range(min(len(aprobs), mb)):
                    _W['b_aprobs'][bi, j] = aprobs[j]
                bi += 1

            elif dt == 'TARGET_SELECTION' and n_tgt > 0:
                if nc <= 1:
                    continue
                _W['t_gs_idx'][ti] = shared_row
                for j in range(min(nc, mt)):
                    cf = cand[j]
                    cl = min(len(cf), CARD_DIM)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['t_cand'][ti, j, :cl] = arr
                    _W['t_cmask'][ti, j] = True
                si_val = sel[0] if sel else 0
                if si_val >= mt:
                    si_val = mt - 1
                _W['t_sel'][ti] = si_val
                sf = rec.get('spellFeatures')
                if sf is not None:
                    sl = min(len(sf), 64)
                    arr = np.asarray(sf[:sl], dtype=np.float32)
                    sanitize(arr)
                    _W['t_spell'][ti, :sl] = arr
                ti += 1

            elif dt == 'CARD_SELECTION' and n_cs > 0:
                if nc < 1:
                    continue
                _W['cs_gs_idx'][ci] = shared_row
                for j in range(min(nc, mcs)):
                    cf = cand[j]
                    cl = min(len(cf), CARD_DIM)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['cs_cand'][ci, j, :cl] = arr
                    _W['cs_cmask'][ci, j] = True
                for s in sel:
                    if 0 <= s < mcs:
                        _W['cs_amask'][ci, s] = 1.0
                ci += 1

            elif dt == 'MULLIGAN' and n_mul > 0:
                _W['mul_gs_idx'][mi] = shared_row
                for j in range(min(nc, mmul)):
                    cf = cand[j]
                    cl = min(len(cf), CARD_DIM)
                    arr = np.asarray(cf[:cl], dtype=np.float32)
                    sanitize(arr)
                    _W['mul_hand'][mi, j, :cl] = arr
                    _W['mul_hmask'][mi, j] = True
                _W['mul_keep'][mi] = 1.0 if (sel and sel[0] == 1) else 0.0
                mi += 1

            elif dt == 'BINARY_CHOICE' and n_bin > 0:
                _W['bin_gs_idx'][bni] = shared_row
                _W['bin_decision'][bni] = 1.0 if (sel and sel[0] == 1) else 0.0
                bni += 1

        return si - offsets['total']
    except Exception as e:
        print(f"    Warning: {filepath}: {e}", flush=True)
        return 0


def _create_arrays(output_dir, counts, max_cand):
    """Create all mmap files (zero-filled) up front."""
    nt = counts['total']
    np_ = counts['priority']; na = counts['attack']; nb = counts['block']
    n_tgt = counts['target']; n_cs = counts['card_select']
    n_mul = counts['mulligan']; n_bin = counts['binary']

    for d in ['shared', 'priority', 'attack', 'block',
              'target', 'card_select', 'mulligan', 'binary']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    def _mk(path, dtype, shape):
        a = np.lib.format.open_memmap(path, mode='w+',
                                      dtype=dtype, shape=shape)
        a.flush()
        del a

    sh = os.path.join(output_dir, 'shared')
    _mk(os.path.join(sh, 'game_state.npy'), np.float32, (nt, GAME_STATE_DIM))
    _mk(os.path.join(sh, 'global_features.npy'), np.float32, (nt, GLOBAL_DIM))
    _mk(os.path.join(sh, 'outcome.npy'), np.float32, (nt,))
    _mk(os.path.join(sh, 'file_id.npy'), np.int32, (nt,))

    pd = os.path.join(output_dir, 'priority')
    mp = max(max_cand['priority'], 1)
    _mk(os.path.join(pd, 'gs_index.npy'), np.int64, (np_,))
    _mk(os.path.join(pd, 'candidates.npy'), np.float32, (np_, mp, 64))
    _mk(os.path.join(pd, 'candidate_mask.npy'), np.bool_, (np_, mp))
    _mk(os.path.join(pd, 'selected_idx.npy'), np.int32, (np_,))
    _mk(os.path.join(pd, 'action_probs.npy'), np.float32, (np_, mp))

    ad = os.path.join(output_dir, 'attack')
    ma = max(max_cand['attack'], 1)
    _mk(os.path.join(ad, 'gs_index.npy'), np.int64, (na,))
    _mk(os.path.join(ad, 'creatures.npy'), np.float32, (na, ma, CARD_DIM))
    _mk(os.path.join(ad, 'creature_mask.npy'), np.bool_, (na, ma))
    _mk(os.path.join(ad, 'action_mask.npy'), np.float32, (na, ma))
    _mk(os.path.join(ad, 'action_probs.npy'), np.float32, (na, ma))

    bd = os.path.join(output_dir, 'block')
    mb = max(max_cand['block'], 1)
    _mk(os.path.join(bd, 'gs_index.npy'), np.int64, (nb,))
    _mk(os.path.join(bd, 'pairs.npy'), np.float32, (nb, mb, CARD_DIM * 2))
    _mk(os.path.join(bd, 'pair_mask.npy'), np.bool_, (nb, mb))
    _mk(os.path.join(bd, 'action_mask.npy'), np.float32, (nb, mb))
    _mk(os.path.join(bd, 'action_probs.npy'), np.float32, (nb, mb))

    mt = 0
    if n_tgt > 0:
        mt = max(max_cand['target'], 1)
        td = os.path.join(output_dir, 'target')
        _mk(os.path.join(td, 'gs_index.npy'), np.int64, (n_tgt,))
        _mk(os.path.join(td, 'candidates.npy'), np.float32, (n_tgt, mt, CARD_DIM))
        _mk(os.path.join(td, 'candidate_mask.npy'), np.bool_, (n_tgt, mt))
        _mk(os.path.join(td, 'selected_idx.npy'), np.int32, (n_tgt,))
        _mk(os.path.join(td, 'spell_features.npy'), np.float32, (n_tgt, 64))

    mcs = 0
    if n_cs > 0:
        mcs = max(max_cand['card_select'], 1)
        csd = os.path.join(output_dir, 'card_select')
        _mk(os.path.join(csd, 'gs_index.npy'), np.int64, (n_cs,))
        _mk(os.path.join(csd, 'candidates.npy'), np.float32, (n_cs, mcs, CARD_DIM))
        _mk(os.path.join(csd, 'candidate_mask.npy'), np.bool_, (n_cs, mcs))
        _mk(os.path.join(csd, 'action_mask.npy'), np.float32, (n_cs, mcs))

    mmul = 7
    if n_mul > 0:
        mmul = max(max_cand['mulligan'], 7)
        muld = os.path.join(output_dir, 'mulligan')
        _mk(os.path.join(muld, 'gs_index.npy'), np.int64, (n_mul,))
        _mk(os.path.join(muld, 'hand_features.npy'), np.float32, (n_mul, mmul, CARD_DIM))
        _mk(os.path.join(muld, 'hand_mask.npy'), np.bool_, (n_mul, mmul))
        _mk(os.path.join(muld, 'keep_decision.npy'), np.float32, (n_mul,))

    if n_bin > 0:
        bind = os.path.join(output_dir, 'binary')
        _mk(os.path.join(bind, 'gs_index.npy'), np.int64, (n_bin,))
        _mk(os.path.join(bind, 'decision.npy'), np.float32, (n_bin,))

    return {
        'mp': mp, 'ma': ma, 'mb': mb, 'mt': mt,
        'mcs': mcs, 'mmul': mmul,
        'n_tgt': n_tgt, 'n_cs': n_cs,
        'n_mul': n_mul, 'n_bin': n_bin,
    }


def preprocess(files, output_dir, counts, max_cand,
               per_file_counts, workers=1, progress_cb=None):
    """Pass 2 (parallel): write each file into preassigned mmap offsets."""
    shapes = _create_arrays(output_dir, counts, max_cand)
    has_keys = {
        'target': counts['target'] > 0,
        'card_select': counts['card_select'] > 0,
        'mulligan': counts['mulligan'] > 0,
        'binary': counts['binary'] > 0,
    }

    game_ids = _build_game_id_map(files)
    n_games = len(set(game_ids))
    print(f"  Game IDs: {n_games} unique games from "
          f"{len(files)} files", flush=True)

    # Build per-file offsets via prefix sum.
    offsets_list = []
    running = {'total': 0}
    for k in DECISION_KEYS:
        running[k] = 0
    for fc in per_file_counts:
        offsets_list.append(dict(running))
        for k in running:
            running[k] += fc[k]

    tasks = [(files[i], game_ids[i], offsets_list[i])
             for i in range(len(files))]

    print(f"\n  Pass 2: Writing {len(files)} files "
          f"({workers} workers)...", flush=True)

    if workers <= 1:
        _init_worker(output_dir, shapes, has_keys)
        for fi, t in enumerate(tasks):
            _process_file(t)
            if (fi + 1) % 200 == 0:
                print(f"    {fi+1}/{len(files)}", flush=True)
            if progress_cb and fi % 20 == 0:
                progress_cb(fi, len(files))
    else:
        with Pool(workers, initializer=_init_worker,
                  initargs=(output_dir, shapes, has_keys)) as pool:
            for fi, _ in enumerate(
                    pool.imap_unordered(_process_file, tasks,
                                        chunksize=4)):
                if (fi + 1) % 500 == 0:
                    print(f"    {fi+1}/{len(files)}", flush=True)
                if progress_cb and fi % 50 == 0:
                    progress_cb(fi, len(files))

    if progress_cb:
        progress_cb(len(files), len(files))

    return {'total': running['total'],
            'priority': running['priority'],
            'attack': running['attack'],
            'block': running['block'],
            'target': running['target'],
            'card_select': running['card_select'],
            'mulligan': running['mulligan'],
            'binary': running['binary']}


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectories to mmap')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--output-dir',
        default='../../rl_data/preprocessed')
    parser.add_argument('--workers', type=int, default=0,
        help='Parallel workers (default: os.cpu_count())')
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    t0 = time.time()
    print("=== Trajectory Preprocessing ===", flush=True)
    print(f"  Workers: {workers}", flush=True)

    path = Path(args.data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    print(f"  Found {len(files)} trajectory files", flush=True)
    if not files:
        print("  ERROR: No files found", flush=True)
        sys.exit(1)

    counts, max_cand, per_file = scan_files(files, workers=workers)

    os.makedirs(args.output_dir, exist_ok=True)

    final = preprocess(files, args.output_dir, counts, max_cand,
                       per_file, workers=workers)

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
    print(f"  Shared: {final['total']} records", flush=True)
    print(f"  Priority: {final['priority']}", flush=True)
    print(f"  Attack: {final['attack']}", flush=True)
    print(f"  Block: {final['block']}", flush=True)
    print(f"  Target: {final['target']}", flush=True)
    print(f"  Card Select: {final['card_select']}", flush=True)
    print(f"  Mulligan: {final['mulligan']}", flush=True)
    print(f"  Binary: {final['binary']}", flush=True)

    total_bytes = 0
    for root, dirs, fnames in os.walk(args.output_dir):
        for fn in fnames:
            total_bytes += os.path.getsize(
                os.path.join(root, fn))
    print(f"  Disk: {total_bytes/1024**3:.1f} GB", flush=True)


if __name__ == '__main__':
    main()
