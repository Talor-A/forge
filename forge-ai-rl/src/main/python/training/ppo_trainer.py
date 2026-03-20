#!/usr/bin/env python3
"""
PPO Self-Play Training Loop

Alternates between:
1. Playing games (Java subprocess) with RL agent using model server
2. Loading trajectory data with action probabilities
3. Computing advantages via value network
4. PPO policy gradient updates on attack/block heads
5. Saving updated model and repeating

Usage:
    python training/ppo_trainer.py \
        --checkpoint /path/to/model_with_decisions.pt \
        --device cuda \
        --rounds 20 \
        --games-per-round 200
"""

import argparse
import json
import os
import sys
import time
import subprocess
import signal
import random
import socket
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.gpu_config import auto_detect_profile
from serving.model_server import ModelServer

import threading


# ── Config ───────────────────────────────────────────

PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
FORGE_JAR = os.path.join(
    PROJECT_ROOT,
    'forge-gui-desktop/target/'
    'forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar')
DECKS = ['green_stompy.dck', 'white_weenie.dck',
         'blue_tempo.dck', 'red_aggro.dck']


# ── Data loading for PPO ─────────────────────────────

def parse_game_state(flat, gf):
    """Parse flat state into zone tensors."""
    card_dim = 128
    zones_cfg = [('my_board', 30), ('opp_board', 30),
                 ('hand', 15), ('my_gy', 40),
                 ('opp_gy', 40), ('stack', 10)]
    g = np.zeros(64, dtype=np.float32)
    gl = min(len(gf), 64)
    if gl > 0:
        g[:gl] = gf[:gl]
    zones, masks = {}, {}
    offset = 64
    for name, count in zones_cfg:
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
        zones[name] = zd
        masks[name + '_mask'] = zm
    return g, zones, masks


def load_ppo_data(traj_dir):
    """Load trajectory data for PPO training.
    Returns game-level samples with state and outcome.
    Works with both mid-game and end-game-only recordings."""
    path = Path(traj_dir)
    files = sorted(path.glob('traj_*.jsonl'))

    attack_samples = []
    block_samples = []
    priority_samples = []
    value_samples = []

    for filepath in files:
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

                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(gf, -10, 10, out=gf)
                gf = np.nan_to_num(gf)

                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                # Always collect value training data
                if len(flat) > 0:
                    value_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'outcome': outcome,
                    })

                if len(cand) < 1:
                    continue

                # Old policy probabilities for PPO ratio
                old_probs = np.array(
                    rec.get('actionProbabilities', []),
                    dtype=np.float32)

                if dt == 'PRIORITY_ACTION':
                    # Priority: 64-dim action features,
                    # single-select
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

                    # Old log prob for the selected action
                    old_lp = 0.0
                    if len(old_probs) > selected_idx:
                        p = max(old_probs[selected_idx],
                                1e-8)
                        old_lp = float(np.log(p))

                    priority_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'action_features': actions,
                        'selected_idx': selected_idx,
                        'n_actions': n,
                        'outcome': outcome,
                        'old_log_prob': old_lp,
                    })
                    continue

                # Combat: 128-dim card features, multi-select
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

                action_mask = np.zeros(
                    n, dtype=np.float32)
                for idx in sel:
                    if 0 <= idx < n:
                        action_mask[idx] = 1.0

                # Old log prob: sum of per-creature
                # log probs for the joint action
                old_lp = 0.0
                if len(old_probs) >= n:
                    for j in range(n):
                        p = max(old_probs[j], 1e-8)
                        if action_mask[j] > 0.5:
                            old_lp += float(np.log(p))
                        else:
                            old_lp += float(
                                np.log(max(1-p, 1e-8)))

                sample = {
                    'global_features': gf,
                    'game_state_flat': flat,
                    'creature_features': creatures,
                    'action_mask': action_mask,
                    'n_creatures': n,
                    'outcome': outcome,
                    'old_log_prob': old_lp,
                }
                if dt == 'DECLARE_ATTACKERS':
                    attack_samples.append(sample)
                elif dt == 'DECLARE_BLOCKERS':
                    block_samples.append(sample)
        except Exception:
            pass

    return (attack_samples, block_samples,
            priority_samples, value_samples)


# ── PPO batch computation ────────────────────────────

def compute_ppo_batch(model, head, samples, device,
                      use_amp, clip_eps=0.2):
    """
    Compute PPO loss for a batch of attack/block decisions.

    For each creature, the old policy chose attack (1) or not (0).
    We compute the new policy's log prob for that same action,
    then apply the PPO clipped objective.
    """
    if not samples:
        return torch.tensor(0.0, device=device), {}, 0

    max_c = max(s['n_creatures'] for s in samples)
    max_c = max(max_c, 1)
    bs = len(samples)

    cf = torch.zeros(bs, max_c, 128, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool,
                      device=device)
    actions = torch.zeros(bs, max_c, device=device)
    outcomes = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, 64, device=device)

    # Zone tensors for encoder
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

    for i, s in enumerate(samples):
        nc = s['n_creatures']
        cf[i, :nc] = torch.from_numpy(
            s['creature_features'])
        cm[i, :nc] = True
        actions[i, :nc] = torch.from_numpy(
            s['action_mask'])
        outcomes[i] = s['outcome']
        old_log_probs[i] = s.get('old_log_prob', 0.0)

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(
            masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(
            masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(
            masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(
            masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode state
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        # Value estimate (critic)
        value = model.get_value(state).squeeze(-1)

        # Policy logits (actor)
        logits = head(state, cf, cm)

        # Compute log probs for chosen actions
        # Binary: log P(attack) = log_sigmoid(logit)
        #         log P(not attack) = log_sigmoid(-logit)
        safe_logits = logits.clone()
        safe_logits[~cm] = 0.0

        log_probs = (
            F.logsigmoid(safe_logits) * actions +
            F.logsigmoid(-safe_logits) * (1 - actions))
        # Sum per-creature log probs for total action prob
        log_probs = (log_probs * cm.float()).sum(dim=1)

        # Advantage: outcome - value estimate (MC return)
        advantage = (outcomes - value.detach())
        # Normalize advantage for stability
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        # PPO clipped objective with importance sampling
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(value, outcomes)

        # Entropy bonus (encourage exploration)
        probs = torch.sigmoid(safe_logits)
        entropy = -(
            probs * F.logsigmoid(safe_logits) +
            (1 - probs) * F.logsigmoid(-safe_logits))
        entropy = (entropy * cm.float()).sum(dim=1).mean()

        # Total loss — entropy coefficient 0.1 encourages
        # exploration (prevents sigmoid saturation)
        total_loss = (
            policy_loss +
            0.5 * value_loss -
            0.01 * entropy)

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantage.mean().item(),
        'mean_value': value.mean().item(),
        'win_rate': (outcomes > 0).float().mean().item(),
    }
    return total_loss, metrics, bs


def compute_ppo_priority_batch(model, head, samples,
                               device, use_amp,
                               clip_eps=0.2):
    """
    Compute PPO loss for a batch of priority decisions.

    Uses Categorical distribution (single-select softmax)
    instead of Bernoulli (binary per-creature).
    """
    if not samples:
        return torch.tensor(0.0, device=device), {}, 0

    max_a = max(s['n_actions'] for s in samples)
    max_a = max(max_a, 1)
    bs = len(samples)

    af = torch.zeros(bs, max_a, 64, device=device)
    am = torch.zeros(bs, max_a, dtype=torch.bool,
                      device=device)
    actions = torch.zeros(bs, dtype=torch.long,
                          device=device)
    outcomes = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, 64, device=device)

    # Zone tensors for encoder
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

    for i, s in enumerate(samples):
        na = s['n_actions']
        af[i, :na] = torch.from_numpy(
            s['action_features'])
        am[i, :na] = True
        actions[i] = s['selected_idx']
        outcomes[i] = s['outcome']
        old_log_probs[i] = s.get('old_log_prob', 0.0)

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(
            masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(
            masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(
            masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(
            masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode state
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        # Value estimate (critic)
        value = model.get_value(state).squeeze(-1)

        # Policy logits (actor) — masked softmax
        logits = head(state, af, am)

        # Categorical distribution over actions
        dist = torch.distributions.Categorical(
            logits=logits)
        log_probs = dist.log_prob(actions)

        # Advantage: outcome - value estimate (MC return)
        advantage = (outcomes - value.detach())
        # Normalize advantage for stability
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        # PPO clipped objective with importance sampling
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(value, outcomes)

        # Entropy bonus
        entropy = dist.entropy().mean()

        total_loss = (
            policy_loss +
            0.5 * value_loss -
            0.01 * entropy)

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantage.mean().item(),
        'mean_value': value.mean().item(),
        'win_rate': (outcomes > 0).float().mean().item(),
    }
    return total_loss, metrics, bs


# ── Game runner subprocess ───────────────────────────

class ModelServerError(Exception):
    """Raised when Java reports the model server is down."""
    pass


def run_games(n_games, traj_dir, mode='evaluate',
              port=50051, quiet=True,
              progress_callback=None,
              log_callback=None):
    """Run games via Java subprocess.
    Raises ModelServerError if the server is detected as down.
    progress_callback(completed, total) called as games complete.
    log_callback(line) called for each stdout line from Java."""
    os.makedirs(traj_dir, exist_ok=True)
    # Clean old trajectories
    for f in Path(traj_dir).glob('traj_*.jsonl'):
        f.unlink()

    deck_args = []
    for d in DECKS:
        deck_args.extend(['-d', d])

    cmd = [
        'java', '-Xmx8192m',
        '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
        '--add-opens', 'java.base/java.util=ALL-UNNAMED',
        '--add-opens', 'java.base/java.text=ALL-UNNAMED',
        '--add-opens',
        'java.base/java.lang.reflect=ALL-UNNAMED',
        '--add-opens',
        'java.desktop/javax.imageio.spi=ALL-UNNAMED',
        '-jar', FORGE_JAR,
        'rltrain', mode,
    ] + deck_args + [
        '-n', str(n_games),
        '-t', '4',
        '-o', traj_dir,
        '-host', 'localhost',
        '-port', str(port),
    ]
    # Don't use -q so we can parse progress
    # (evaluate mode prints Game N/M every 10 games)

    cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')

    # Stream output for progress tracking
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)

    stdout_lines = []
    try:
        for line in proc.stdout:
            stdout_lines.append(line)
            if log_callback:
                log_callback(line.rstrip())
            # Parse progress from evaluate mode output
            if progress_callback and 'Game ' in line:
                try:
                    # "Game 10/100: RL win rate: ..."
                    parts = line.split('Game ')[1].split('/')
                    done = int(parts[0])
                    progress_callback(done, n_games)
                except (IndexError, ValueError):
                    pass
            # Also count trajectory files as progress
            elif (progress_callback
                  and 'Wrote trajectory' in line):
                done = len(list(
                    Path(traj_dir).glob('traj_*.jsonl')))
                # Each game produces ~4 files
                est = min(done // 4, n_games)
                if est > 0:
                    progress_callback(est, n_games)

        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()

    stdout = ''.join(stdout_lines)

    # Check for model server abort
    if 'ABORT' in stdout:
        raise ModelServerError(
            "Model server is down — Java aborted the run. "
            "Check server logs.")

    # Parse win rate from output
    win_rate = None
    server_errors = 0
    for line in stdout_lines:
        if 'RL Wins:' in line:
            try:
                pct = line.split('(')[1].split('%')[0]
                win_rate = float(pct) / 100
            except (IndexError, ValueError):
                pass
        elif 'MODEL_SERVER_ERROR' in line:
            server_errors += 1

    if server_errors > 0:
        print(f"  WARNING: {server_errors} model server "
              f"errors during run", flush=True)

    return win_rate, stdout


# ── Model server management ──────────────────────────

def start_model_server(model, device, port=50051):
    """Start model server in background thread."""
    server = ModelServer(model, port=port, device=device)
    thread = threading.Thread(
        target=server.start, daemon=True)
    thread.start()
    time.sleep(1)  # Wait for server to bind
    # Verify server is actually listening
    if not check_server_health(port):
        raise RuntimeError(
            f"Model server failed to start on port {port}")
    return server


def check_server_health(port, host='localhost'):
    """Verify the model server is reachable and responding."""
    import struct
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        # Send a minimal ping request
        ping = json.dumps({
            'decisionType': 'PING',
            'globalFeatures': [],
            'candidateFeatures': [],
        }).encode('utf-8')
        sock.sendall(struct.pack('>I', len(ping)))
        sock.sendall(ping)
        # Read response
        length_bytes = sock.recv(4)
        if len(length_bytes) == 4:
            resp_len = struct.unpack('>I', length_bytes)[0]
            resp = sock.recv(resp_len)
            sock.close()
            return True
        sock.close()
        return False
    except Exception:
        return False


def find_free_port():
    """Find a free TCP port."""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# ── Main PPO loop ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PPO Self-Play Training')
    parser.add_argument('--checkpoint',
        default=os.path.join(
            PROJECT_ROOT,
            'rl_data/checkpoints/model_with_decisions.pt'))
    parser.add_argument('--save-dir',
        default=os.path.join(
            PROJECT_ROOT, 'rl_data/checkpoints'))
    parser.add_argument('--traj-dir',
        default=os.path.join(
            PROJECT_ROOT, 'rl_data/ppo_trajectories'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--rounds', type=int, default=20,
        help='Number of collect→train rounds')
    parser.add_argument('--games-per-round', type=int,
        default=200)
    parser.add_argument('--ppo-epochs', type=int,
        default=4,
        help='PPO update epochs per round')
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-games', type=int,
        default=50,
        help='Evaluation games per round')
    parser.add_argument('--port', type=int, default=0,
        help='Model server port (0=auto)')
    args = parser.parse_args()

    profile = auto_detect_profile()
    device = args.device or (
        'cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = profile.use_amp and device.startswith('cuda')
    port = args.port or find_free_port()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)

    print('┌────────────────────────────────────────┐',
          flush=True)
    print('│     MTG RL — PPO Self-Play Training    │',
          flush=True)
    print('└────────────────────────────────────────┘',
          flush=True)
    print(f'  Device: {device} ({profile.name})',
          flush=True)
    print(f'  Rounds: {args.rounds}', flush=True)
    print(f'  Games/round: {args.games_per_round}',
          flush=True)
    print(f'  PPO epochs: {args.ppo_epochs}', flush=True)
    print(f'  Eval games: {args.eval_games}', flush=True)
    print(f'  Server port: {port}', flush=True)

    # Load model
    print(f'\n  Loading model: {args.checkpoint}',
          flush=True)
    if os.path.exists(args.checkpoint):
        model = MTGModel.load(
            args.checkpoint, device=device)
    else:
        print('  No checkpoint, using random init',
              flush=True)
        model = MTGModel().to(device)

    # Unfreeze everything for PPO
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Start model server
    print(f'  Starting model server on port {port}...',
          flush=True)
    server = start_model_server(model, device, port)

    best_win_rate = 0.0
    history = []

    print(flush=True)
    print('  Round │ Games │ Attacks │ Blocks │'
          ' Priority │ Policy Loss │ Value Loss │'
          ' Entropy │ Eval WR │ Status', flush=True)
    print('  ──────┼───────┼─────────┼────────┼'
          '──────────┼─────────────┼────────────┼'
          '─────────┼─────────┼───────', flush=True)

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()

        # ── Step 1: Collect games with RL agent ──
        # Use 'evaluate' mode so RL plays vs heuristic
        # (captures RL's combat decisions)
        try:
            _, stdout = run_games(
                args.games_per_round, args.traj_dir,
                mode='evaluate', port=port)
        except ModelServerError as e:
            print(f'\n  FATAL: {e}', flush=True)
            print('  Stopping PPO — model server is down.',
                  flush=True)
            break

        # ── Step 2: Load trajectories ──
        attack_data, block_data, priority_data, \
            value_data = load_ppo_data(args.traj_dir)

        if not attack_data and not block_data \
                and not priority_data:
            print(f'  {rnd:>4d}   │ {args.games_per_round:>5d} │'
                  f'    0    │   0    │'
                  f'     0    │'
                  f' no data     │            │'
                  f'         │         │ SKIP',
                  flush=True)
            continue

        # ── Step 3: PPO updates ──
        model.train()
        total_pl, total_vl, total_ent = 0, 0, 0
        n_updates = 0

        for ppo_epoch in range(args.ppo_epochs):
            # Attack head updates
            random.shuffle(attack_data)
            for bi in range(0, len(attack_data),
                            args.batch_size):
                batch = attack_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = compute_ppo_batch(
                    model, model.attack_head, batch,
                    device, use_amp)

                if torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    optimizer.step()

                total_pl += metrics['policy_loss']
                total_vl += metrics['value_loss']
                total_ent += metrics['entropy']
                n_updates += 1

            # Block head updates
            random.shuffle(block_data)
            for bi in range(0, len(block_data),
                            args.batch_size):
                batch = block_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = compute_ppo_batch(
                    model, model.attack_head, batch,
                    device, use_amp)

                if torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    optimizer.step()

                total_pl += metrics['policy_loss']
                total_vl += metrics['value_loss']
                total_ent += metrics['entropy']
                n_updates += 1

            # Priority head updates (Categorical, not Bernoulli)
            random.shuffle(priority_data)
            for bi in range(0, len(priority_data),
                            args.batch_size):
                batch = priority_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = \
                    compute_ppo_priority_batch(
                        model, model.priority_head,
                        batch, device, use_amp)

                if torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 0.5)
                    optimizer.step()

                total_pl += metrics['policy_loss']
                total_vl += metrics['value_loss']
                total_ent += metrics['entropy']
                n_updates += 1

        avg_pl = total_pl / max(n_updates, 1)
        avg_vl = total_vl / max(n_updates, 1)
        avg_ent = total_ent / max(n_updates, 1)

        # ── Step 4: Evaluate vs heuristic ──
        model.eval()
        try:
            eval_wr, _ = run_games(
                args.eval_games, args.traj_dir + '_eval',
                mode='evaluate', port=port)
            eval_wr = eval_wr or 0.0
        except ModelServerError as e:
            print(f'\n  FATAL: {e}', flush=True)
            print('  Stopping PPO — model server down '
                  'during eval.', flush=True)
            break

        # Save
        status = ''
        if eval_wr > best_win_rate:
            best_win_rate = eval_wr
            model.save(os.path.join(
                args.save_dir, 'best_ppo_model.pt'))
            status = '★ best'
        if rnd % 5 == 0:
            model.save(os.path.join(
                args.save_dir,
                f'ppo_model_round_{rnd}.pt'))
            if not status:
                status = 'saved'

        elapsed = time.time() - t0
        history.append({
            'round': rnd,
            'eval_wr': eval_wr,
            'policy_loss': avg_pl,
            'value_loss': avg_vl,
        })

        print(
            f'  {rnd:>4d}   │ {args.games_per_round:>5d} │'
            f' {len(attack_data):>7d} │ {len(block_data):>6d} │'
            f' {len(priority_data):>8d} │'
            f' {avg_pl:>11.4f} │ {avg_vl:>10.4f} │'
            f' {avg_ent:>7.3f} │'
            f' {eval_wr:>6.1%} │ {status}'
            f'  ({elapsed:.0f}s)',
            flush=True)

    # Final summary
    print(flush=True)
    print('  ╔════════════════════════════════╗',
          flush=True)
    print(f'  ║ Best win rate: {best_win_rate:>6.1%}'
          f'          ║', flush=True)
    print(f'  ║ Total rounds: {args.rounds:>3d}'
          f'             ║', flush=True)
    print('  ╚════════════════════════════════╝',
          flush=True)

    # Save final
    model.save(os.path.join(
        args.save_dir, 'ppo_model_final.pt'))

    # Write history
    with open(os.path.join(
            args.save_dir, 'ppo_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
