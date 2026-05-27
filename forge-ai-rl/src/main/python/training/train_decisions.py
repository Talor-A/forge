#!/usr/bin/env python3
"""
Train Decision Heads — imitation learning on heuristic AI choices.

Loads the pre-trained value network (frozen encoder), then trains
the specialized decision heads to predict what the heuristic AI
chose at each decision point.

Heads trained:
- AttackHead: which creatures to attack with (binary per creature)
- BlockHead: which creatures to block with
- PriorityHead: which spell/ability to play (future)

Usage:
    python training/train_decisions.py \
        --data-dir /path/to/trajectories \
        --encoder-checkpoint /path/to/best_value_model.pt \
        --device cuda \
        --epochs 50
"""

import argparse
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.gpu_config import auto_detect_profile
from training.mmap_dataset import (
    parse_game_state, GAME_STATE_DIM, CARD_DIM, GLOBAL_DIM,
    ZONES_CONFIG, MmapAttackDataset, MmapBlockDataset,
    SharedState,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────
#
# Decision data is loaded via memory-mapped numpy arrays produced by
# preprocess_trajectories.py. Train/val split is built into the
# Mmap*Dataset classes; samples are batched as list-of-dicts via
# DataLoader(collate_fn=lambda x: x) to match the existing batch fns.


def _list_collate(batch):
    """DataLoader collate that returns the raw list of dicts.

    The downstream `_attack_batch` / `_block_batch` fns expect a list
    of per-sample dicts so they can pad to the batch's max creature
    count. PyTorch's default collate would stack along dim 0, which
    breaks on variable-length creature arrays.
    """
    return batch


# ── Training ─────────────────────────────────────────

def train_attack_head(model, train_ds, val_ds, args,
                      device, use_amp):
    """Train the attack head via imitation learning."""
    print('\n  === Training Attack Head ===', flush=True)

    # Freeze encoder, unfreeze attack head
    for param in model.state_encoder.parameters():
        param.requires_grad = False
    for param in model.value_network.parameters():
        param.requires_grad = False
    for param in model.attack_head.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.attack_head.parameters())
    print(f'  Trainable params: {trainable:,} '
          f'(attack head only)', flush=True)

    optimizer = optim.AdamW(
        model.attack_head.parameters(),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f'  Train: {len(train_ds)} | '
          f'Val: {len(val_ds)} (by file_id)', flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_list_collate,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_list_collate)

    best_val_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_attack_model.pt')

    print(flush=True)
    print('  Epoch | Train Loss | Train Acc |'
          ' Val Loss  | Val Acc |  Time', flush=True)
    print('  ──────┼────────────┼───────────┼'
          '───────────┼─────────┼──────', flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        tloss, tcorrect, ttotal = 0, 0, 0

        for batch in train_loader:
            if len(batch) < 2:
                continue
            loss, correct, total = _attack_batch(
                model, batch, device, use_amp)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(
                    model.attack_head.parameters(), 1.0)
                assert torch.isfinite(gn) and gn > 0, \
                    f"attack imitation grad norm dead: {gn}"
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.attack_head.parameters(), 1.0)
                assert torch.isfinite(gn) and gn > 0, \
                    f"attack imitation grad norm dead: {gn}"
                optimizer.step()

            tloss += loss.item() * total
            tcorrect += correct
            ttotal += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vcorrect, vtotal = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = _attack_batch(
                    model, batch, device, use_amp)
                vloss += loss.item() * total
                vcorrect += correct
                vtotal += total

        ta = tcorrect / max(ttotal, 1)
        va = vcorrect / max(vtotal, 1)
        tl = tloss / max(ttotal, 1)
        vl = vloss / max(vtotal, 1)
        elapsed = time.time() - t0

        status = ''
        if va > best_val_acc:
            best_val_acc = va
            model.save(save_path)
            status = ' *'

        print(f'  {epoch:>4d}   | {tl:>10.4f} | {ta:>8.1%} |'
              f' {vl:>9.4f} | {va:>6.1%} |'
              f' {elapsed:>4.1f}s{status}', flush=True)

    print(f'\n  Best val accuracy: {best_val_acc:.1%}',
          flush=True)
    return best_val_acc


def _attack_batch(model, batch, device, use_amp):
    """Process one batch of attack decisions.

    Each sample dict comes from MmapAttackDataset and carries:
      creature_features: (n, CARD_DIM), action_mask: (n,),
      n_creatures: int, game_state_flat: (GAME_STATE_DIM,),
      global_features: (GLOBAL_DIM,).
    """
    # Pad creatures to max in batch
    max_c = max(s['n_creatures'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)

    cd = CARD_DIM
    creature_feats = torch.zeros(bs, max_c, cd,
                                  device=device)
    creature_mask = torch.zeros(bs, max_c,
                                 dtype=torch.bool,
                                 device=device)
    targets = torch.zeros(bs, max_c, device=device)
    global_feats = torch.zeros(bs, GLOBAL_DIM, device=device)

    # Parse game states for encoder
    my_board = torch.zeros(bs, 40, cd, device=device)
    my_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                              device=device)
    opp_board = torch.zeros(bs, 40, cd, device=device)
    opp_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                               device=device)
    hand = torch.zeros(bs, 15, cd, device=device)
    hand_m = torch.zeros(bs, 15, dtype=torch.bool,
                          device=device)
    my_gy = torch.zeros(bs, 20, cd, device=device)
    my_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                           device=device)
    opp_gy = torch.zeros(bs, 20, cd, device=device)
    opp_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                            device=device)
    stack = torch.zeros(bs, 10, cd, device=device)
    stack_m = torch.zeros(bs, 10, dtype=torch.bool,
                           device=device)

    for i, s in enumerate(batch):
        nc = s['n_creatures']
        creature_feats[i, :nc] = torch.from_numpy(
            s['creature_features'])
        creature_mask[i, :nc] = True
        targets[i, :nc] = torch.from_numpy(
            s['action_mask'])

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        global_feats[i] = torch.from_numpy(g)
        my_board[i] = torch.from_numpy(
            zones['my_board'])
        my_board_m[i] = torch.from_numpy(
            masks['my_board_mask'])
        opp_board[i] = torch.from_numpy(
            zones['opp_board'])
        opp_board_m[i] = torch.from_numpy(
            masks['opp_board_mask'])
        hand[i] = torch.from_numpy(zones['hand'])
        hand_m[i] = torch.from_numpy(
            masks['hand_mask'])
        my_gy[i] = torch.from_numpy(zones['my_gy'])
        my_gy_m[i] = torch.from_numpy(
            masks['my_gy_mask'])
        opp_gy[i] = torch.from_numpy(zones['opp_gy'])
        opp_gy_m[i] = torch.from_numpy(
            masks['opp_gy_mask'])
        stack[i] = torch.from_numpy(zones['stack'])
        stack_m[i] = torch.from_numpy(
            masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode game state (frozen)
        with torch.no_grad():
            state = model.encode_state(
                global_feats,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)

        # Attack head forward
        logits = model.attack_head(
            state, creature_feats, creature_mask)

        # Binary cross-entropy per creature
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            reduction='none')
        # Only count real creatures
        loss = (loss * creature_mask.float()).sum() / \
               creature_mask.float().sum().clamp(min=1)

    assert torch.isfinite(loss), "attack imitation loss non-finite"
    # Accuracy: per-creature binary accuracy
    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == targets) *
                   creature_mask.float()).sum().item()
        total = creature_mask.float().sum().item()

    return loss, correct, total


def train_block_head(model, train_ds, val_ds, args,
                     device, use_amp):
    """Train the block head (same structure as attack)."""
    print('\n  === Training Block Head ===', flush=True)

    for param in model.state_encoder.parameters():
        param.requires_grad = False
    for param in model.attack_head.parameters():
        param.requires_grad = False
    for param in model.block_head.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.block_head.parameters())
    print(f'  Trainable params: {trainable:,}', flush=True)

    optimizer = optim.AdamW(
        model.block_head.parameters(),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}'
          ' (by file_id)', flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_list_collate,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_list_collate)

    best_val_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_block_model.pt')

    print(flush=True)
    print('  Epoch | Train Loss | Train Acc |'
          ' Val Loss  | Val Acc |  Time', flush=True)
    print('  ──────┼────────────┼───────────┼'
          '───────────┼─────────┼──────', flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        tl, tc, tt = 0, 0, 0
        for batch in train_loader:
            if len(batch) < 2:
                continue
            loss, correct, total = _block_batch(
                model, batch, device, use_amp)
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(
                    model.block_head.parameters(), 1.0)
                assert torch.isfinite(gn) and gn > 0, \
                    f"block imitation grad norm dead: {gn}"
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.block_head.parameters(), 1.0)
                assert torch.isfinite(gn) and gn > 0, \
                    f"block imitation grad norm dead: {gn}"
                optimizer.step()
            tl += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = _block_batch(
                    model, batch, device, use_amp)
                vl += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        elapsed = time.time() - t0

        status = ''
        if va > best_val_acc:
            best_val_acc = va
            model.save(save_path)
            status = ' *'

        print(f'  {epoch:>4d}   | {tl/max(tt,1):>10.4f} |'
              f' {ta:>8.1%} |'
              f' {vl/max(vt,1):>9.4f} | {va:>6.1%} |'
              f' {elapsed:>4.1f}s{status}', flush=True)

    print(f'\n  Best val accuracy: {best_val_acc:.1%}',
          flush=True)
    return best_val_acc


def _block_batch(model, batch, device, use_amp):
    """Same as _attack_batch but for block pairs."""
    max_c = max(s['n_pairs'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)
    cd = CARD_DIM

    creature_feats = torch.zeros(bs, max_c, cd,
                                  device=device)
    creature_mask = torch.zeros(bs, max_c,
                                 dtype=torch.bool,
                                 device=device)
    targets = torch.zeros(bs, max_c, device=device)
    global_feats = torch.zeros(bs, GLOBAL_DIM, device=device)

    my_board = torch.zeros(bs, 40, cd, device=device)
    my_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                              device=device)
    opp_board = torch.zeros(bs, 40, cd, device=device)
    opp_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                               device=device)
    hand = torch.zeros(bs, 15, cd, device=device)
    hand_m = torch.zeros(bs, 15, dtype=torch.bool,
                          device=device)
    my_gy = torch.zeros(bs, 20, cd, device=device)
    my_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                           device=device)
    opp_gy = torch.zeros(bs, 20, cd, device=device)
    opp_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                            device=device)
    stack = torch.zeros(bs, 10, cd, device=device)
    stack_m = torch.zeros(bs, 10, dtype=torch.bool,
                           device=device)

    for i, s in enumerate(batch):
        # MmapBlockDataset returns pair_features (CARD_DIM*2)
        # but the legacy attack_head consumer below expects
        # CARD_DIM. Take the first CARD_DIM channels — the
        # blocker half. Pre-mmap code did the same truncation
        # implicitly via the (n, CARD_DIM) tensor allocation.
        nc = s['n_pairs']
        pair = s['pair_features']
        creature_feats[i, :nc] = torch.from_numpy(
            pair[:, :cd].copy())
        creature_mask[i, :nc] = True
        targets[i, :nc] = torch.from_numpy(
            s['action_mask'])

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        global_feats[i] = torch.from_numpy(g)
        my_board[i] = torch.from_numpy(
            zones['my_board'])
        my_board_m[i] = torch.from_numpy(
            masks['my_board_mask'])
        opp_board[i] = torch.from_numpy(
            zones['opp_board'])
        opp_board_m[i] = torch.from_numpy(
            masks['opp_board_mask'])
        hand[i] = torch.from_numpy(zones['hand'])
        hand_m[i] = torch.from_numpy(
            masks['hand_mask'])
        my_gy[i] = torch.from_numpy(zones['my_gy'])
        my_gy_m[i] = torch.from_numpy(
            masks['my_gy_mask'])
        opp_gy[i] = torch.from_numpy(zones['opp_gy'])
        opp_gy_m[i] = torch.from_numpy(
            masks['opp_gy_mask'])
        stack[i] = torch.from_numpy(zones['stack'])
        stack_m[i] = torch.from_numpy(
            masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        with torch.no_grad():
            state = model.encode_state(
                global_feats,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)

        # Block head: use same interface as attack head
        # (binary per creature: block or not)
        logits = model.attack_head(
            state, creature_feats, creature_mask)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        loss = (loss * creature_mask.float()).sum() / \
               creature_mask.float().sum().clamp(min=1)

    assert torch.isfinite(loss), "block imitation loss non-finite"
    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == targets) *
                   creature_mask.float()).sum().item()
        total = creature_mask.float().sum().item()

    return loss, correct, total


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train MTG RL Decision Heads')
    parser.add_argument('--data-dir',
        default='../../rl_data/preprocessed',
        help='Directory produced by preprocess_trajectories.py')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--encoder-checkpoint',
        default='../../rl_data/checkpoints/best_value_model.pt',
        help='Pre-trained value model with encoder')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--heads', default='attack,block',
        help='Comma-separated heads to train')
    args = parser.parse_args()

    profile = auto_detect_profile()
    device = args.device or (
        'cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = profile.use_amp and device.startswith('cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    print('┌────────────────────────────────────────────┐',
          flush=True)
    print('│    MTG RL — Decision Head Training         │',
          flush=True)
    print('└────────────────────────────────────────────┘',
          flush=True)
    print(f'  Device: {device} ({profile.name})',
          flush=True)
    print(f'  AMP: {use_amp}', flush=True)
    print(f'  Encoder: {args.encoder_checkpoint}',
          flush=True)

    # Load pre-trained model
    if os.path.exists(args.encoder_checkpoint):
        print(f'  Loading pre-trained encoder...',
              flush=True)
        model = MTGModel.load(
            args.encoder_checkpoint, device=device)
        print(f'  Loaded.', flush=True)
    else:
        print(f'  No checkpoint found, using random init',
              flush=True)
        model = MTGModel.from_size("xl").to(device)

    heads = args.heads.split(',')

    if not os.path.exists(
            os.path.join(args.data_dir, 'metadata.json')):
        print(f'  {args.data_dir} is not a preprocessed '
              f'dir (missing metadata.json). Run '
              f'preprocess_trajectories.py first.',
              flush=True)
        return

    shared = SharedState(args.data_dir)

    if 'attack' in heads:
        train_ds = MmapAttackDataset(
            args.data_dir, train=True,
            val_fraction=args.val_split, shared=shared)
        val_ds = MmapAttackDataset(
            args.data_dir, train=False,
            val_fraction=args.val_split, shared=shared)
        if len(train_ds) > 0:
            train_attack_head(
                model, train_ds, val_ds, args,
                device, use_amp)
        else:
            print('  No attack data found', flush=True)

    if 'block' in heads:
        train_ds = MmapBlockDataset(
            args.data_dir, train=True,
            val_fraction=args.val_split, shared=shared)
        val_ds = MmapBlockDataset(
            args.data_dir, train=False,
            val_fraction=args.val_split, shared=shared)
        if len(train_ds) > 0:
            train_block_head(
                model, train_ds, val_ds, args,
                device, use_amp)
        else:
            print('  No block data found', flush=True)

    # Save final combined model
    model.save(os.path.join(
        args.save_dir, 'model_with_decisions.pt'))
    print(f'\n  Saved to {args.save_dir}/'
          f'model_with_decisions.pt', flush=True)


if __name__ == '__main__':
    main()
