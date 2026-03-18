"""
Train Value Network — First training phase.

Trains the game state encoder + value network to predict
win/loss from end-of-game state snapshots. This is the
simplest possible training task and validates the full pipeline.

Usage:
    python training/train_value.py \
        --data-dir ../../rl_data/trajectories \
        --device cuda \
        --epochs 50 \
        --batch-size 64
"""

import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent to path
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.gpu_config import auto_detect_profile
from training.dataset import TrajectoryDataset, create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scaler,
                device, use_amp):
    """Train for one epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        # Move to device
        global_f = batch['global_features'].to(device)
        my_board = batch['my_board'].to(device)
        my_board_m = batch['my_board_mask'].to(device)
        opp_board = batch['opp_board'].to(device)
        opp_board_m = batch['opp_board_mask'].to(device)
        hand = batch['hand'].to(device)
        hand_m = batch['hand_mask'].to(device)
        my_gy = batch['my_gy'].to(device)
        my_gy_m = batch['my_gy_mask'].to(device)
        opp_gy = batch['opp_gy'].to(device)
        opp_gy_m = batch['opp_gy_mask'].to(device)
        stack = batch['stack'].to(device)
        stack_m = batch['stack_mask'].to(device)
        targets = batch['value_target'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass
            state_embed = model.encode_state(
                global_f,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)

            # Value prediction
            value_pred = model.get_value(
                state_embed).squeeze(-1)

            # MSE loss on value prediction
            loss = nn.functional.mse_loss(
                value_pred, targets)

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()

        # Track metrics
        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        # Accuracy: did we predict the right sign?
        predicted_win = value_pred > 0
        actual_win = targets > 0
        total_correct += (
            predicted_win == actual_win).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        global_f = batch['global_features'].to(device)
        my_board = batch['my_board'].to(device)
        my_board_m = batch['my_board_mask'].to(device)
        opp_board = batch['opp_board'].to(device)
        opp_board_m = batch['opp_board_mask'].to(device)
        hand = batch['hand'].to(device)
        hand_m = batch['hand_mask'].to(device)
        my_gy = batch['my_gy'].to(device)
        my_gy_m = batch['my_gy_mask'].to(device)
        opp_gy = batch['opp_gy'].to(device)
        opp_gy_m = batch['opp_gy_mask'].to(device)
        stack = batch['stack'].to(device)
        stack_m = batch['stack_mask'].to(device)
        targets = batch['value_target'].to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            state_embed = model.encode_state(
                global_f,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)
            value_pred = model.get_value(
                state_embed).squeeze(-1)
            loss = nn.functional.mse_loss(
                value_pred, targets)

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        predicted_win = value_pred > 0
        actual_win = targets > 0
        total_correct += (
            predicted_win == actual_win).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train MTG RL Value Network')
    parser.add_argument(
        '--data-dir',
        default='../../rl_data/trajectories',
        help='Trajectory data directory')
    parser.add_argument(
        '--save-dir',
        default='../../rl_data/checkpoints',
        help='Checkpoint directory')
    parser.add_argument(
        '--log-dir',
        default='../../rl_data/runs/value_train',
        help='TensorBoard log directory')
    parser.add_argument(
        '--epochs', type=int, default=50)
    parser.add_argument(
        '--batch-size', type=int, default=None)
    parser.add_argument(
        '--lr', type=float, default=1e-3)
    parser.add_argument(
        '--device', default=None)
    parser.add_argument(
        '--val-split', type=float, default=0.1,
        help='Validation split ratio')
    parser.add_argument(
        '--save-every', type=int, default=10,
        help='Save checkpoint every N epochs')
    args = parser.parse_args()

    # Auto-detect GPU
    profile = auto_detect_profile()
    device = args.device or (
        'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size or profile.batch_size
    use_amp = profile.use_amp and device.startswith('cuda')

    logger.info(f"Device: {device} | Batch: {batch_size} "
                f"| AMP: {use_amp}")
    logger.info(f"GPU: {profile.name}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    full_dataset = TrajectoryDataset(args.data_dir)
    if len(full_dataset) == 0:
        logger.error("No training data found!")
        return

    # Train/val split
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.startswith('cuda'),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=device.startswith('cuda'))

    logger.info(
        f"Train: {n_train} samples | Val: {n_val} samples")

    # Create model
    model = MTGModel().to(device)
    params = model.count_parameters()
    logger.info(f"Model: {params['total']:,} parameters")
    logger.info(
        f"  Encoder: {params['state_encoder']:,} | "
        f"Value: {params['value_network']:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    best_val_acc = 0
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler,
            device, use_amp)
        val_loss, val_acc = evaluate(
            model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Acc: {val_acc:.3f} | "
            f"LR: {lr:.6f} | "
            f"{elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar(
            'train/loss', train_loss, epoch)
        writer.add_scalar(
            'train/accuracy', train_acc, epoch)
        writer.add_scalar(
            'val/loss', val_loss, epoch)
        writer.add_scalar(
            'val/accuracy', val_acc, epoch)
        writer.add_scalar(
            'train/lr', lr, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(os.path.join(
                args.save_dir, 'best_value_model.pt'))
            logger.info(
                f"  -> New best val accuracy: {val_acc:.3f}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            model.save(os.path.join(
                args.save_dir,
                f'value_model_epoch_{epoch}.pt'))

    # Final save
    model.save(os.path.join(
        args.save_dir, 'value_model_final.pt'))
    logger.info("=" * 60)
    logger.info(
        f"Training complete. Best val acc: {best_val_acc:.3f}")
    logger.info(
        f"Model saved to {args.save_dir}")
    writer.close()


if __name__ == '__main__':
    main()
