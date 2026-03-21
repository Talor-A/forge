#!/usr/bin/env python3
"""
Decision Head Training Dashboard — Tkinter GUI for monitoring
attack and block head imitation learning.

Launch:
    python training/train_decisions_ui.py \
        --data-dir /path/to/trajectories \
        --device cuda --epochs 50
"""

import argparse
import json
import os
import sys
import threading
import time
import random
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

import tkinter as tk
from tkinter import ttk

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from model.mtg_model import MTGModel
from model.gpu_config import auto_detect_profile


# ── Shared state ─────────────────────────────────────

@dataclass
class TrainingState:
    status: str = "Idle"
    phase: str = ""
    chart_dirty: bool = False
    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False

    # Loading
    files_total: int = 0
    files_loaded: int = 0
    attack_samples: int = 0
    block_samples: int = 0
    priority_samples: int = 0

    # Config
    device: str = "cpu"
    gpu_name: str = ""
    encoder_params: int = 0
    head_params: int = 0

    # Current head being trained
    current_head: str = ""
    epoch: int = 0
    total_epochs: int = 50
    epoch_progress: float = 0.0

    # Metrics per head
    attack_train_losses: List[float] = field(
        default_factory=list)
    attack_train_accs: List[float] = field(
        default_factory=list)
    attack_val_losses: List[float] = field(
        default_factory=list)
    attack_val_accs: List[float] = field(
        default_factory=list)
    attack_best_acc: float = 0.0

    block_train_losses: List[float] = field(
        default_factory=list)
    block_train_accs: List[float] = field(
        default_factory=list)
    block_val_losses: List[float] = field(
        default_factory=list)
    block_val_accs: List[float] = field(
        default_factory=list)
    block_best_acc: float = 0.0

    priority_train_losses: List[float] = field(
        default_factory=list)
    priority_train_accs: List[float] = field(
        default_factory=list)
    priority_val_losses: List[float] = field(
        default_factory=list)
    priority_val_accs: List[float] = field(
        default_factory=list)
    priority_best_acc: float = 0.0

    current_train_loss: float = 0.0
    current_train_acc: float = 0.0
    current_val_loss: float = 0.0
    current_val_acc: float = 0.0
    epoch_time: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0

    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0


def log(state, msg):
    """Thread-safe logging to console and state."""
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 200:
        state.log_lines = state.log_lines[-200:]
    state.log_dirty = True


# ── Data loading ─────────────────────────────────────

def parse_game_state(flat, global_feats):
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


def load_decisions(data_dir, state, max_files=None,
                   heads=None, chunk_size=200):
    """Load decision samples in chunks to bound memory.

    For priority (140K+ samples), loads chunk_size files
    at a time. For attack/block (small), loads all at once.
    """
    from training.chunked_loader import load_decision_samples

    state.phase = "loading"
    state.status = "Loading trajectory files..."

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    state.files_total = len(files)

    attack_samples = []
    block_samples = []
    priority_samples = []

    # Load in chunks to bound memory
    for ci in range(0, len(files), chunk_size):
        chunk_files = files[ci:ci + chunk_size]
        a, b, p = load_decision_samples(
            chunk_files, heads=
            list(heads) if heads else None)
        attack_samples.extend(a)
        block_samples.extend(b)
        priority_samples.extend(p)
        state.files_loaded = min(ci + chunk_size,
                                 len(files))
        state.attack_samples = len(attack_samples)
        state.block_samples = len(block_samples)
        state.priority_samples = len(priority_samples)

    state.status = (
        f"Loaded {len(attack_samples)} attack, "
        f"{len(block_samples)} block, "
        f"{len(priority_samples)} priority decisions")
    return attack_samples, block_samples, priority_samples


def load_decisions_old(data_dir, state, max_files=None,
                   heads=None):
    """Original load function — kept as fallback."""
    state.phase = "loading"
    state.status = "Loading trajectory files..."

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    state.files_total = len(files)

    attack_samples = []
    block_samples = []
    priority_samples = []

    for i, filepath in enumerate(files):
        state.files_loaded = i + 1
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

                # Skip decision types we're not training
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
                g = np.zeros(64, dtype=np.float32)
                gl = min(len(gf), 64)
                if gl > 0:
                    g[:gl] = gf[:gl]

                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                if dt == 'PRIORITY_ACTION':
                    # Priority: 64-dim action features,
                    # single-select (softmax CE)
                    n = len(cand)
                    actions = np.zeros(
                        (n, 64), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 64)
                        actions[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(actions, -10, 10, out=actions)
                    actions = np.nan_to_num(actions)

                    # Selected index (single-select)
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
                    state.priority_samples = len(
                        priority_samples)
                    continue

                if dt == 'DECLARE_ATTACKERS':
                    # Attack: 128-dim card features,
                    # multi-select (binary BCE)
                    n = len(cand)
                    creatures = np.zeros(
                        (n, 128), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 128)
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
                    state.attack_samples = len(
                        attack_samples)

                elif dt == 'DECLARE_BLOCKERS':
                    # Block: 256-dim pair features
                    # (blocker+attacker concat),
                    # multi-select assignment
                    n = len(cand)
                    feat_dim = len(cand[0]) if cand else 0
                    if feat_dim < 200:
                        continue  # skip old-format data

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
                    state.block_samples = len(
                        block_samples)
        except Exception:
            pass

    state.status = (
        f"Loaded {len(attack_samples)} attack, "
        f"{len(block_samples)} block, "
        f"{len(priority_samples)} priority decisions")
    return attack_samples, block_samples, priority_samples


# ── Batch processing ─────────────────────────────────

def make_batch(model, batch, device, use_amp, head):
    max_c = max(s['n_creatures'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)

    cf = torch.zeros(bs, max_c, 128, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool,
                      device=device)
    tgt = torch.zeros(bs, max_c, device=device)
    gf = torch.zeros(bs, 64, device=device)

    mb = torch.zeros(bs, 30, 128, device=device)
    mbm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 30, 128, device=device)
    obm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, 128, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 40, 128, device=device)
    mgm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 40, 128, device=device)
    ogm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, 128, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        nc = s['n_creatures']
        cf[i, :nc] = torch.from_numpy(
            s['creature_features'])
        cm[i, :nc] = True
        tgt[i, :nc] = torch.from_numpy(s['action_mask'])

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        with torch.no_grad():
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, cf, cm)

        # Only compute loss on real creatures (mask=True)
        # Replace padding logits with 0 to avoid NaN in BCE
        safe_logits = logits.clone()
        safe_logits[~cm] = 0.0
        safe_tgt = tgt.clone()
        safe_tgt[~cm] = 0.0

        loss = nn.functional.binary_cross_entropy_with_logits(
            safe_logits, safe_tgt, reduction='none')
        loss = (loss * cm.float()).sum() / \
               cm.float().sum().clamp(min=1)

    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == tgt) *
                   cm.float()).sum().item()
        total = cm.float().sum().item()

    return loss, correct, total


def make_priority_batch(model, batch, device, use_amp,
                        head):
    """Build a priority batch with CrossEntropyLoss
    (single-select softmax)."""
    max_a = max(s['n_actions'] for s in batch)
    max_a = max(max_a, 1)
    bs = len(batch)

    af = torch.zeros(bs, max_a, 64, device=device)
    am = torch.zeros(bs, max_a, dtype=torch.bool,
                      device=device)
    tgt = torch.zeros(bs, dtype=torch.long, device=device)
    gf = torch.zeros(bs, 64, device=device)

    mb = torch.zeros(bs, 30, 128, device=device)
    mbm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 30, 128, device=device)
    obm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, 128, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 40, 128, device=device)
    mgm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 40, 128, device=device)
    ogm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, 128, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        na = s['n_actions']
        af[i, :na] = torch.from_numpy(
            s['action_features'])
        am[i, :na] = True
        tgt[i] = s['selected_idx']

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        with torch.no_grad():
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, af, am)

        loss = nn.functional.cross_entropy(logits, tgt)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == tgt).sum().item()
        total = bs

    return loss, correct, total


def make_block_batch(model, batch, device, use_amp,
                     head):
    """Build a block batch for the BlockHead.

    Block data has (blocker,attacker) pair features (256-dim).
    BlockHead expects separate blocker/attacker tensors.
    We reconstruct these from the pairs: for each sample,
    extract unique blockers and attackers from the pairs.
    """
    bs = len(batch)

    # Find max blockers and attackers across batch
    # Each pair is (blocker_i, attacker_j) — we need to
    # figure out how many unique blockers/attackers
    # Pairs are ordered: b0a0, b0a1, ..., b1a0, b1a1, ...
    # Last pair is the "no block" zero vector
    max_b, max_a = 0, 0
    for s in batch:
        n = s['n_pairs']
        pf = s['pair_features']
        # Count non-zero unique blocker features
        # Pairs are b*a + no_block, so n_pairs = nb*na + 1
        # We detect dimensions from feature patterns
        # Simpler: extract from contextInfo if available,
        # or infer from pair count
        # Last pair is zero (no-block), so real pairs = n-1
        real = n - 1
        if real <= 0:
            continue
        # Find number of attackers: first blocker's pairs
        # end when blocker features change
        first_blocker = pf[0, :128]
        na = 1
        for j in range(1, real):
            if np.allclose(pf[j, :128], first_blocker,
                           atol=0.01):
                na += 1
            else:
                break
        nb = real // max(na, 1)
        max_b = max(max_b, nb)
        max_a = max(max_a, na)

    max_b = max(max_b, 1)
    max_a = max(max_a, 1)

    bf = torch.zeros(bs, max_b, 128, device=device)
    bm = torch.zeros(bs, max_b, dtype=torch.bool,
                      device=device)
    af = torch.zeros(bs, max_a, 128, device=device)
    am_t = torch.zeros(bs, max_a, dtype=torch.bool,
                       device=device)
    # Target: for each blocker, which attacker (or no-block)
    tgt = torch.full((bs, max_b), max_a, dtype=torch.long,
                     device=device)  # default = no block
    gf = torch.zeros(bs, 64, device=device)

    mb = torch.zeros(bs, 30, 128, device=device)
    mbm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 30, 128, device=device)
    obm = torch.zeros(bs, 30, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, 128, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 40, 128, device=device)
    mgm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 40, 128, device=device)
    ogm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, 128, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        n = s['n_pairs']
        pf_np = s['pair_features']
        sel = s['action_mask']
        real = n - 1
        if real <= 0:
            continue

        # Infer nb, na from pair structure
        first_blocker = pf_np[0, :128]
        na = 1
        for j in range(1, real):
            if np.allclose(pf_np[j, :128], first_blocker,
                           atol=0.01):
                na += 1
            else:
                break
        nb = real // max(na, 1)

        # Extract unique blockers and attackers
        for b_idx in range(min(nb, max_b)):
            bf[i, b_idx] = torch.from_numpy(
                pf_np[b_idx * na, :128])
            bm[i, b_idx] = True
        for a_idx in range(min(na, max_a)):
            af[i, a_idx] = torch.from_numpy(
                pf_np[a_idx, 128:256])
            am_t[i, a_idx] = True

        # Build assignment target from selected pairs
        for pair_idx in range(real):
            if sel[pair_idx] > 0.5:
                b_idx = pair_idx // na
                a_idx = pair_idx % na
                if b_idx < max_b and a_idx < max_a:
                    tgt[i, b_idx] = a_idx

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        with torch.no_grad():
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        # BlockHead: (batch, max_b, max_a+1) logits
        logits = head(state_emb, bf, bm, af, am_t)

        # CE loss per blocker
        # logits: (bs, max_b, max_a+1), tgt: (bs, max_b)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            reduction='none')
        # Mask to real blockers only
        loss = (loss.reshape(bs, max_b)
                * bm.float()).sum() / \
               bm.float().sum().clamp(min=1)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = ((preds == tgt)
                   * bm).sum().item()
        total = bm.sum().item()

    return loss, correct, total


def train_block_head(model, head, samples, args,
                     state, device, use_amp):
    """Train the block head with per-blocker CE loss."""
    state.current_head = 'block'
    state.status = "Training block head..."
    log(state, f"\n=== Training block head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.1))
    train_s = samples[:-n_val]
    val_s = samples[-n_val:]

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_block_model.pt')

    tl_list = state.block_train_losses
    ta_list = state.block_train_accs
    vl_list = state.block_val_losses
    va_list = state.block_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training block head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_block_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * max(total, 1)
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_block_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * max(total, 1)
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.block_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping block training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_priority_head(model, head, samples, args,
                        state, device, use_amp):
    """Train the priority head with CrossEntropyLoss."""
    state.current_head = 'priority'
    state.status = "Training priority head..."
    log(state, f"\n=== Training priority head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.1))
    train_s = samples[:-n_val]
    val_s = samples[-n_val:]

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_priority_model.pt')

    tl_list = state.priority_train_losses
    ta_list = state.priority_train_accs
    vl_list = state.priority_val_losses
    va_list = state.priority_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training priority head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_priority_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_priority_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.priority_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping priority training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


# ── Training thread ──────────────────────────────────

def train_head(model, head, head_name, samples, args,
               state, device, use_amp):
    state.current_head = head_name
    state.status = f"Training {head_name} head..."
    log(state, f"\n=== Training {head_name} head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.1))
    train_s = samples[:-n_val]
    val_s = samples[-n_val:]

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, f'best_{head_name}_model.pt')

    # Get the right metric lists
    if head_name == 'attack':
        tl_list = state.attack_train_losses
        ta_list = state.attack_train_accs
        vl_list = state.attack_val_losses
        va_list = state.attack_val_accs
    else:
        tl_list = state.block_train_losses
        ta_list = state.block_train_accs
        vl_list = state.block_val_losses
        va_list = state.block_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training {head_name} head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0
        n_batches = max(
            1, len(train_s) // args.batch_size)

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        if head_name == 'attack':
            state.attack_best_acc = best_acc
        else:
            state.block_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        # Early stop on NaN
        if np.isnan(tl) or np.isnan(vl):
            log(state, f"  NaN detected! Stopping {head_name} training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_head_mmap(model, head, head_name,
                    train_ds, val_ds, args, state,
                    device, use_amp, batch_fn):
    """Generic mmap-based head training.
    Uses DataLoader directly, no list materialization."""
    import torch.utils.data as tud

    state.current_head = head_name
    state.status = f"Training {head_name} head..."
    log(state, f"\n=== Training {head_name} head (mmap)"
        f" ===")
    log(state, f"Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    train_loader = tud.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=lambda x: x)
    val_loader = tud.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda x: x)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, f'best_{head_name}_model.pt')

    if head_name == 'attack':
        tl_list = state.attack_train_losses
        ta_list = state.attack_train_accs
        vl_list = state.attack_val_losses
        va_list = state.attack_val_accs
    else:
        tl_list = state.block_train_losses
        ta_list = state.block_train_accs
        vl_list = state.block_val_losses
        va_list = state.block_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training {head_name} head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        model.train()
        tloss, tc, tt = 0, 0, 0

        for bi, batch in enumerate(train_loader):
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_loader), 1)

            loss, correct, total = batch_fn(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = batch_fn(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        if head_name == 'attack':
            state.attack_best_acc = best_acc
        else:
            state.block_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                f"  NaN detected! Stopping {head_name}.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_priority_head_mmap(model, head,
                             train_ds, val_ds,
                             args, state, device,
                             use_amp):
    """Train priority head using mmap datasets directly.
    Avoids materializing 140K samples into RAM."""
    import torch.utils.data as tud

    state.current_head = 'priority'
    state.status = "Training priority head..."
    log(state, f"\n=== Training priority head (mmap) ===")
    log(state, f"Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    # DataLoaders with list collation (batch = list of dicts)
    train_loader = tud.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=lambda x: x)
    val_loader = tud.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda x: x)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_priority_model.pt')

    tl_list = state.priority_train_losses
    ta_list = state.priority_train_accs
    vl_list = state.priority_val_losses
    va_list = state.priority_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training priority head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        tloss, tc, tt = 0, 0, 0

        for bi, batch in enumerate(train_loader):
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_loader), 1)

            loss, correct, total = make_priority_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = make_priority_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.priority_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping priority.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def trainer_thread(state, args):
    try:
        profile = auto_detect_profile()
        device = args.device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = profile.use_amp and device.startswith(
            'cuda')

        state.device = device
        state.gpu_name = profile.name
        state.total_epochs = args.epochs

        # Load data via mmap or fallback to JSONL
        heads_list = (args.heads.split(',')
                 if args.heads != 'all'
                 else ['priority', 'attack', 'block'])

        preprocessed_dir = os.path.join(
            os.path.dirname(args.data_dir),
            'preprocessed')
        use_mmap = os.path.exists(
            os.path.join(preprocessed_dir,
                         'metadata.json'))

        if use_mmap:
            from training.mmap_dataset import (
                MmapPriorityDataset,
                MmapAttackDataset,
                MmapBlockDataset)
            import json as json_mod

            state.status = "Loading preprocessed data..."
            with open(os.path.join(preprocessed_dir,
                                   'metadata.json')) as f:
                meta = json_mod.load(f)

            priority_samples = []
            attack_samples = []
            block_samples = []

            if 'priority' in heads_list:
                # Priority uses DataLoader directly
                # (too large to materialize into list)
                priority_train_ds = MmapPriorityDataset(
                    preprocessed_dir, train=True)
                priority_val_ds = MmapPriorityDataset(
                    preprocessed_dir, train=False)
                state.priority_samples = len(
                    priority_train_ds)
                log(state,
                    f"Priority: {len(priority_train_ds)}"
                    f" train, {len(priority_val_ds)}"
                    f" val samples (mmap)")

            if 'attack' in heads_list:
                attack_train_ds = MmapAttackDataset(
                    preprocessed_dir, train=True)
                attack_val_ds = MmapAttackDataset(
                    preprocessed_dir, train=False)
                state.attack_samples = len(
                    attack_train_ds)
                log(state,
                    f"Attack: {len(attack_train_ds)}"
                    f" train, {len(attack_val_ds)}"
                    f" val samples (mmap)")

            if 'block' in heads_list:
                block_train_ds = MmapBlockDataset(
                    preprocessed_dir, train=True)
                block_val_ds = MmapBlockDataset(
                    preprocessed_dir, train=False)
                state.block_samples = len(
                    block_train_ds)
                log(state,
                    f"Block: {len(block_train_ds)}"
                    f" train, {len(block_val_ds)}"
                    f" val samples (mmap)")
        else:
            # Fallback: load from JSONL (chunked)
            log(state,
                "No preprocessed data — loading JSONL")
            heads_filter = (
                list(heads_list)
                if heads_list != ['priority', 'attack',
                                  'block']
                else None)
            attack_samples, block_samples, \
                priority_samples = load_decisions(
                    args.data_dir, state,
                    args.max_files,
                    heads=heads_filter)

        # Load model
        state.phase = "training"
        state.status = "Loading encoder..."
        log(state, f"\nDevice: {device} ({state.gpu_name})")
        log(state, f"AMP: {use_amp}")
        if os.path.exists(args.encoder_checkpoint):
            log(state, f"Loading encoder: "
                f"{args.encoder_checkpoint}")
            model = MTGModel.load(
                args.encoder_checkpoint, device=device)
            log(state, "Encoder loaded.")
        else:
            log(state, "No checkpoint — random init")
            model = MTGModel().to(device)

        state.encoder_params = sum(
            p.numel()
            for p in model.state_encoder.parameters())

        os.makedirs(args.save_dir, exist_ok=True)

        log(state, f"Heads to train: {heads_list}")

        # Train priority head first (softmax CE, not BCE)
        if 'priority' in heads_list:
            if use_mmap and priority_train_ds:
                train_priority_head_mmap(
                    model, model.priority_head,
                    priority_train_ds, priority_val_ds,
                    args, state, device, use_amp)
            elif priority_samples:
                train_priority_head(
                    model, model.priority_head,
                    priority_samples, args, state,
                    device, use_amp)

        # Train attack head
        if 'attack' in heads_list:
            state.epoch = 0
            state.epoch_progress = 0
            if use_mmap and attack_train_ds:
                train_head_mmap(
                    model, model.attack_head, 'attack',
                    attack_train_ds, attack_val_ds,
                    args, state, device, use_amp,
                    make_batch)
            elif attack_samples:
                train_head(model, model.attack_head,
                           'attack', attack_samples,
                           args, state, device, use_amp)

        # Train block head (pair-based assignment)
        if 'block' in heads_list:
            state.epoch = 0
            state.epoch_progress = 0
            if use_mmap and block_train_ds:
                train_head_mmap(
                    model, model.block_head, 'block',
                    block_train_ds, block_val_ds,
                    args, state, device, use_amp,
                    make_block_batch)
            elif block_samples:
                train_block_head(
                    model, model.block_head,
                    block_samples, args, state,
                    device, use_amp)

        # Save combined model
        save_path = os.path.join(
            args.save_dir, 'model_with_decisions.pt')
        model.save(save_path)

        log(state, f"\n=== Training Complete ===")
        log(state, f"Attack best val acc: {state.attack_best_acc:.1%}")
        log(state, f"Block best val acc: {state.block_best_acc:.1%}")
        log(state, f"Priority best val acc: {state.priority_best_acc:.1%}")
        log(state, f"Model saved: {save_path}")

        state.status = "Training complete!"
        state.phase = "done"
        state.chart_dirty = True

    except Exception as e:
        log(state, f"\nERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        state.chart_dirty = True
        import traceback
        traceback.print_exc()
        with open('/tmp/rl_decision_error.log', 'w') as f:
            traceback.print_exc(file=f)


# ── Tkinter Dashboard ────────────────────────────────

class DecisionDashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        self.root.title(
            "MTG RL — Decision Head Training")
        self.root.geometry("900x750")
        self.root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Header.TLabel',
            font=('Helvetica', 16, 'bold'),
            background='#1e1e2e', foreground='#cdd6f4')
        style.configure('Stat.TLabel',
            font=('Consolas', 11),
            background='#1e1e2e', foreground='#a6adc8')
        style.configure('Value.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#89b4fa')
        style.configure('Good.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#a6e3a1')
        style.configure('Status.TLabel',
            font=('Consolas', 10),
            background='#1e1e2e', foreground='#f9e2af')
        style.configure('Dark.TFrame',
            background='#1e1e2e')
        style.configure("blue.Horizontal.TProgressbar",
            troughcolor='#313244', background='#89b4fa')

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        m = ttk.Frame(self.root, style='Dark.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(m,
            text="MTG RL — Decision Head Training",
            style='Header.TLabel').pack(pady=(0, 10))

        self.status_var = tk.StringVar(
            value="Initializing...")
        ttk.Label(m, textvariable=self.status_var,
            style='Status.TLabel').pack()

        pf = ttk.Frame(m, style='Dark.TFrame')
        pf.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(
            pf, length=860, mode='determinate',
            style="blue.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=2)
        self.prog_label = tk.StringVar(value="")
        ttk.Label(pf, textvariable=self.prog_label,
            style='Stat.TLabel').pack(anchor='w')

        # Stats
        sf = ttk.Frame(m, style='Dark.TFrame')
        sf.pack(fill=tk.X, pady=5)
        self.stat_vars = {}
        stats = [
            ('Head', '—'), ('Epoch', '—'),
            ('Train Loss', '—'), ('Train Acc', '—'),
            ('Val Loss', '—'), ('Val Acc', '—'),
            ('Best Attack', '—'), ('Best Block', '—'),
            ('Best Priority', '—'), ('Epoch Time', '—'),
            ('ETA', '—'), ('GPU Mem', '—'),
            ('Head Params', '—'),
        ]
        for i, (label, default) in enumerate(stats):
            r, c = divmod(i, 4)
            lbl = ttk.Label(sf, text=f"{label}:",
                style='Stat.TLabel')
            lbl.grid(row=r, column=c*2, sticky='w',
                padx=(10, 2), pady=2)
            var = tk.StringVar(value=default)
            val = ttk.Label(sf, textvariable=var,
                style='Value.TLabel')
            val.grid(row=r, column=c*2+1, sticky='w',
                padx=(0, 15), pady=2)
            self.stat_vars[label] = var

        # Charts
        if HAS_MATPLOTLIB:
            cf = ttk.Frame(m, style='Dark.TFrame')
            cf.pack(fill=tk.BOTH, expand=True, pady=5)

            self.fig = Figure(figsize=(8.5, 6), dpi=100,
                facecolor='#1e1e2e')

            self.ax_atk_loss = self.fig.add_subplot(231)
            self.ax_atk_acc = self.fig.add_subplot(234)
            self.ax_blk_loss = self.fig.add_subplot(232)
            self.ax_blk_acc = self.fig.add_subplot(235)
            self.ax_pri_loss = self.fig.add_subplot(233)
            self.ax_pri_acc = self.fig.add_subplot(236)

            for ax, title in [
                    (self.ax_atk_loss, 'Attack Loss'),
                    (self.ax_atk_acc, 'Attack Accuracy'),
                    (self.ax_blk_loss, 'Block Loss'),
                    (self.ax_blk_acc, 'Block Accuracy'),
                    (self.ax_pri_loss, 'Priority Loss'),
                    (self.ax_pri_acc, 'Priority Accuracy')]:
                ax.set_facecolor('#313244')
                ax.set_title(title, color='#cdd6f4',
                    fontsize=9)
                ax.tick_params(colors='#6c7086',
                    labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color('#45475a')

            self.fig.tight_layout(pad=2.0)
            self.canvas = FigureCanvasTkAgg(
                self.fig, master=cf)
            self.canvas.get_tk_widget().pack(
                fill=tk.BOTH, expand=True)

    def _update_loop(self):
        s = self.state
        self.status_var.set(s.status)

        if s.phase == "loading":
            pct = s.files_loaded / max(
                s.files_total, 1) * 100
            self.progress['value'] = pct
            self.prog_label.set(
                f"Loading: {s.files_loaded}/"
                f"{s.files_total} files | "
                f"Atk: {s.attack_samples} | "
                f"Blk: {s.block_samples} | "
                f"Pri: {s.priority_samples}")
        elif s.phase == "training":
            # Three heads: priority=0-33, attack=33-66, block=66-100
            head_offsets = {'priority': 0, 'attack': 33,
                            'block': 66}
            head_offset = head_offsets.get(
                s.current_head, 0)
            total_pct = head_offset + (
                (s.epoch - 1 + s.epoch_progress)
                / max(s.total_epochs, 1) * 33)
            self.progress['value'] = total_pct
            self.prog_label.set(
                f"{s.current_head.title()} head: "
                f"epoch {s.epoch}/{s.total_epochs}")
        elif s.phase == "done":
            self.progress['value'] = 100

        self.stat_vars['Head'].set(
            s.current_head.title() or '—')
        self.stat_vars['Epoch'].set(
            f"{s.epoch}/{s.total_epochs}")
        self.stat_vars['Train Loss'].set(
            f"{s.current_train_loss:.4f}")
        self.stat_vars['Train Acc'].set(
            f"{s.current_train_acc:.1%}")
        self.stat_vars['Val Loss'].set(
            f"{s.current_val_loss:.4f}")
        self.stat_vars['Val Acc'].set(
            f"{s.current_val_acc:.1%}")
        self.stat_vars['Best Attack'].set(
            f"{s.attack_best_acc:.1%}")
        self.stat_vars['Best Block'].set(
            f"{s.block_best_acc:.1%}")
        self.stat_vars['Best Priority'].set(
            f"{s.priority_best_acc:.1%}")
        self.stat_vars['Epoch Time'].set(
            f"{s.epoch_time:.1f}s")
        self.stat_vars['ETA'].set(
            f"{s.eta:.0f}s" if s.eta > 0 else "—")
        self.stat_vars['Head Params'].set(
            f"{s.head_params:,}" if s.head_params else "—")
        if s.gpu_mem_total_mb > 0:
            self.stat_vars['GPU Mem'].set(
                f"{s.gpu_mem_used_mb:.0f}/"
                f"{s.gpu_mem_total_mb:.0f} MB")

        if HAS_MATPLOTLIB and s.chart_dirty:
            s.chart_dirty = False
            self._draw_charts(s)

        self.root.after(500, self._update_loop)

    def _draw_charts(self, s):
        for ax, tl, ta, vl, va, title_l, title_a in [
            (self.ax_atk_loss, s.attack_train_losses,
             s.attack_train_accs, s.attack_val_losses,
             s.attack_val_accs,
             'Attack Loss', 'Attack Accuracy'),
            (self.ax_blk_loss, s.block_train_losses,
             s.block_train_accs, s.block_val_losses,
             s.block_val_accs,
             'Block Loss', 'Block Accuracy'),
            (self.ax_pri_loss, s.priority_train_losses,
             s.priority_train_accs, s.priority_val_losses,
             s.priority_val_accs,
             'Priority Loss', 'Priority Accuracy'),
        ]:
            # Loss chart
            ax.clear()
            ax.set_facecolor('#313244')
            ax.set_title(title_l, color='#cdd6f4',
                fontsize=9)
            if tl:
                ep = range(1, len(tl) + 1)
                ax.plot(ep, tl, color='#89b4fa',
                    linewidth=1.5, label='Train')
                if vl:
                    ax.plot(ep, vl, color='#f38ba8',
                        linewidth=1.5, label='Val')
                ax.legend(fontsize=7,
                    facecolor='#313244',
                    edgecolor='#45475a',
                    labelcolor='#cdd6f4')
            ax.tick_params(colors='#6c7086', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#45475a')

        for ax, ta, va, title_a in [
            (self.ax_atk_acc, s.attack_train_accs,
             s.attack_val_accs, 'Attack Accuracy'),
            (self.ax_blk_acc, s.block_train_accs,
             s.block_val_accs, 'Block Accuracy'),
            (self.ax_pri_acc, s.priority_train_accs,
             s.priority_val_accs, 'Priority Accuracy'),
        ]:
            ax.clear()
            ax.set_facecolor('#313244')
            ax.set_title(title_a, color='#cdd6f4',
                fontsize=9)
            if ta:
                ep = range(1, len(ta) + 1)
                ax.plot(ep, ta, color='#89b4fa',
                    linewidth=1.5, label='Train')
                if va:
                    ax.plot(ep, va, color='#f38ba8',
                        linewidth=1.5, label='Val')
                ax.set_ylim(0.4, 1.05)
                ax.axhline(y=0.5, color='#585b70',
                    linestyle='--', linewidth=0.8)
                ax.legend(fontsize=7,
                    facecolor='#313244',
                    edgecolor='#45475a',
                    labelcolor='#cdd6f4')
            ax.tick_params(colors='#6c7086', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#45475a')

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Decision Head Training Dashboard')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--encoder-checkpoint',
        default='../../rl_data/checkpoints/'
                'best_value_model.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-files', type=int,
        default=None)
    parser.add_argument('--heads', default='all',
        help='Comma-separated heads to train: '
             'priority,attack,block or "all"')
    args = parser.parse_args()

    state = TrainingState()

    t = threading.Thread(target=trainer_thread,
        args=(state, args), daemon=True)
    t.start()

    root = tk.Tk()
    app = DecisionDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
