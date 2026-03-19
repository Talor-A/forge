#!/usr/bin/env python3
"""
PPO Training Dashboard — visual monitoring of self-play training.

Shows:
- Win rate vs heuristic over rounds (the key metric)
- Policy loss, value loss, entropy curves
- Per-round stats (games played, decisions, timing)
- Console log panel

Launch:
    python training/ppo_ui.py \
        --checkpoint /path/to/model_with_decisions.pt \
        --device cuda --rounds 20 --games-per-round 200
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List

import tkinter as tk
from tkinter import ttk

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg)
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class PPOState:
    status: str = "Idle"
    phase: str = ""  # collecting, training, evaluating, done
    chart_dirty: bool = False

    round: int = 0
    total_rounds: int = 20
    games_this_round: int = 0
    attacks_this_round: int = 0
    blocks_this_round: int = 0

    # History
    win_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    current_win_rate: float = 0.0
    best_win_rate: float = 0.0
    best_round: int = 0
    current_policy_loss: float = 0.0
    current_value_loss: float = 0.0
    current_entropy: float = 0.0

    elapsed: float = 0.0
    round_time: float = 0.0
    eta: float = 0.0

    device: str = ""
    gpu_name: str = ""

    # Console log
    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 300:
        state.log_lines = state.log_lines[-300:]
    state.log_dirty = True


# ── PPO training thread (delegates to ppo_trainer) ───

def ppo_thread(state, args):
    try:
        from training.ppo_trainer import (
            load_ppo_data, compute_ppo_batch,
            run_games, start_model_server,
            find_free_port, parse_game_state,
            PROJECT_ROOT)
        from model.mtg_model import MTGModel
        from model.gpu_config import auto_detect_profile
        import torch
        import torch.optim as optim
        import torch.nn.functional as F
        import random

        profile = auto_detect_profile()
        device = args.device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = (profile.use_amp
                   and device.startswith('cuda'))
        port = args.port or find_free_port()

        state.device = device
        state.gpu_name = profile.name
        state.total_rounds = args.rounds

        log(state, f"Device: {device} ({profile.name})")
        log(state, f"Port: {port}")
        log(state, f"Rounds: {args.rounds}, "
            f"Games/round: {args.games_per_round}")

        # Load model
        log(state, f"Loading: {args.checkpoint}")
        if os.path.exists(args.checkpoint):
            model = MTGModel.load(
                args.checkpoint, device=device)
        else:
            model = MTGModel().to(device)
        log(state, "Model loaded.")

        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=1e-5)
        scaler = (torch.amp.GradScaler('cuda')
                  if use_amp else None)

        # Start server
        log(state, f"Starting model server on :{port}")
        server = start_model_server(model, device, port)
        log(state, "Server ready.")

        traj_dir = os.path.join(
            PROJECT_ROOT, 'rl_data/ppo_trajectories')
        eval_dir = traj_dir + '_eval'
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()

        for rnd in range(1, args.rounds + 1):
            state.round = rnd
            t0 = time.time()

            # Collect
            state.phase = "collecting"
            state.status = (
                f"Round {rnd}: collecting "
                f"{args.games_per_round} games...")
            log(state, f"\n--- Round {rnd}/{args.rounds} "
                f"---")
            log(state, "  Collecting games...")

            _, stdout = run_games(
                args.games_per_round, traj_dir,
                mode='evaluate', port=port)

            attack_data, block_data = load_ppo_data(
                traj_dir)
            state.attacks_this_round = len(attack_data)
            state.blocks_this_round = len(block_data)
            log(state,
                f"  Data: {len(attack_data)} attacks, "
                f"{len(block_data)} blocks")

            if not attack_data and not block_data:
                log(state, "  No data — skipping")
                continue

            # PPO update
            state.phase = "training"
            state.status = (
                f"Round {rnd}: PPO update...")
            model.train()

            total_pl, total_vl, total_ent = 0, 0, 0
            n_updates = 0

            for ppo_ep in range(args.ppo_epochs):
                for data in [attack_data, block_data]:
                    random.shuffle(data)
                    for bi in range(0, len(data),
                                    args.batch_size):
                        batch = data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_ppo_batch(
                                model, model.attack_head,
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

            # Evaluate
            state.phase = "evaluating"
            state.status = (
                f"Round {rnd}: evaluating...")
            model.eval()

            eval_wr, _ = run_games(
                args.eval_games, eval_dir,
                mode='evaluate', port=port)
            eval_wr = eval_wr or 0.0

            # Update state
            state.current_win_rate = eval_wr
            state.current_policy_loss = avg_pl
            state.current_value_loss = avg_vl
            state.current_entropy = avg_ent
            state.win_rates.append(eval_wr)
            state.policy_losses.append(avg_pl)
            state.value_losses.append(avg_vl)
            state.entropies.append(avg_ent)

            if eval_wr > state.best_win_rate:
                state.best_win_rate = eval_wr
                state.best_round = rnd
                model.save(os.path.join(
                    save_dir, 'best_ppo_model.pt'))

            if rnd % 5 == 0:
                model.save(os.path.join(
                    save_dir,
                    f'ppo_model_round_{rnd}.pt'))

            state.round_time = time.time() - t0
            state.elapsed = time.time() - start_time
            state.eta = (
                (args.rounds - rnd) * state.round_time)
            state.chart_dirty = True

            log(state,
                f"  Policy: {avg_pl:.4f} | "
                f"Value: {avg_vl:.4f} | "
                f"Entropy: {avg_ent:.3f} | "
                f"Win rate: {eval_wr:.1%}"
                f"{'  ★ BEST' if eval_wr >= state.best_win_rate else ''}")

            time.sleep(0.05)

        model.save(os.path.join(
            save_dir, 'ppo_model_final.pt'))
        state.status = "Training complete!"
        state.phase = "done"
        state.chart_dirty = True
        log(state, f"\nBest win rate: "
            f"{state.best_win_rate:.1%} "
            f"(round {state.best_round})")

    except Exception as e:
        log(state, f"ERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        import traceback
        traceback.print_exc()


# ── Dashboard ────────────────────────────────────────

class PPODashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        root.title("MTG RL — PPO Self-Play Training")
        root.geometry("950x750")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel',
            font=('Helvetica', 16, 'bold'),
            background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel',
            font=('Consolas', 11),
            background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#89b4fa')
        style.configure('St.TLabel',
            font=('Consolas', 10),
            background='#1e1e2e', foreground='#f9e2af')
        style.configure('D.TFrame',
            background='#1e1e2e')
        style.configure("b.Horizontal.TProgressbar",
            troughcolor='#313244', background='#89b4fa')

        self._build(root)
        self._tick()

    def _build(self, root):
        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(m, text="MTG RL — PPO Self-Play",
            style='H.TLabel').pack(pady=(0, 6))
        self.status_v = tk.StringVar(value="Starting...")
        ttk.Label(m, textvariable=self.status_v,
            style='St.TLabel').pack()

        pf = ttk.Frame(m, style='D.TFrame')
        pf.pack(fill=tk.X, pady=4)
        self.prog = ttk.Progressbar(pf, length=900,
            style="b.Horizontal.TProgressbar")
        self.prog.pack(fill=tk.X)
        self.prog_v = tk.StringVar()
        ttk.Label(pf, textvariable=self.prog_v,
            style='S.TLabel').pack(anchor='w')

        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=4)
        self.svars = {}
        for i, (k, v) in enumerate([
            ('Round', '—'), ('Win Rate', '—'),
            ('Best WR', '—'), ('Policy Loss', '—'),
            ('Value Loss', '—'), ('Entropy', '—'),
            ('Round Time', '—'), ('ETA', '—'),
        ]):
            r, c = divmod(i, 4)
            ttk.Label(sf, text=f"{k}:",
                style='S.TLabel').grid(
                row=r, column=c*2, sticky='w',
                padx=(8, 2), pady=2)
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv,
                style='V.TLabel').grid(
                row=r, column=c*2+1, sticky='w',
                padx=(0, 12), pady=2)
            self.svars[k] = sv

        if HAS_MPL:
            cf = ttk.Frame(m, style='D.TFrame')
            cf.pack(fill=tk.BOTH, expand=True, pady=4)
            self.fig = Figure(figsize=(9, 4), dpi=100,
                facecolor='#1e1e2e')
            self.ax_wr = self.fig.add_subplot(121)
            self.ax_loss = self.fig.add_subplot(122)
            for ax in [self.ax_wr, self.ax_loss]:
                ax.set_facecolor('#313244')
                ax.tick_params(colors='#6c7086',
                    labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color('#45475a')
            self.fig.tight_layout(pad=2)
            self.canvas = FigureCanvasTkAgg(
                self.fig, master=cf)
            self.canvas.get_tk_widget().pack(
                fill=tk.BOTH, expand=True)

        # Console log
        lf = ttk.Frame(m, style='D.TFrame')
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf, height=8,
            bg='#181825', fg='#a6adc8',
            font=('Consolas', 9),
            insertbackground='#cdd6f4',
            selectbackground='#45475a',
            wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _tick(self):
        s = self.state
        self.status_v.set(s.status)

        pct = s.round / max(s.total_rounds, 1) * 100
        self.prog['value'] = pct
        self.prog_v.set(
            f"Round {s.round}/{s.total_rounds} "
            f"({s.phase})")

        self.svars['Round'].set(
            f"{s.round}/{s.total_rounds}")
        self.svars['Win Rate'].set(
            f"{s.current_win_rate:.1%}")
        self.svars['Best WR'].set(
            f"{s.best_win_rate:.1%} (r{s.best_round})")
        self.svars['Policy Loss'].set(
            f"{s.current_policy_loss:.4f}")
        self.svars['Value Loss'].set(
            f"{s.current_value_loss:.4f}")
        self.svars['Entropy'].set(
            f"{s.current_entropy:.3f}")
        self.svars['Round Time'].set(
            f"{s.round_time:.0f}s")
        self.svars['ETA'].set(
            f"{s.eta:.0f}s" if s.eta > 0 else "—")

        if HAS_MPL and s.chart_dirty and s.win_rates:
            s.chart_dirty = False

            self.ax_wr.clear()
            self.ax_wr.set_facecolor('#313244')
            self.ax_wr.set_title('Win Rate vs Heuristic',
                color='#cdd6f4', fontsize=10)
            rr = range(1, len(s.win_rates) + 1)
            self.ax_wr.plot(rr, s.win_rates,
                color='#a6e3a1', linewidth=2,
                marker='o', markersize=4)
            self.ax_wr.axhline(y=0.5, color='#f38ba8',
                linestyle='--', linewidth=1,
                label='50% baseline')
            self.ax_wr.set_ylim(0.0, 1.0)
            self.ax_wr.set_ylabel('Win Rate',
                color='#a6adc8', fontsize=9)
            self.ax_wr.set_xlabel('Round',
                color='#a6adc8', fontsize=9)
            self.ax_wr.legend(fontsize=8,
                facecolor='#313244',
                edgecolor='#45475a',
                labelcolor='#cdd6f4')
            self.ax_wr.tick_params(colors='#6c7086',
                labelsize=8)
            for sp in self.ax_wr.spines.values():
                sp.set_color('#45475a')

            self.ax_loss.clear()
            self.ax_loss.set_facecolor('#313244')
            self.ax_loss.set_title('Training Losses',
                color='#cdd6f4', fontsize=10)
            self.ax_loss.plot(rr, s.policy_losses,
                color='#89b4fa', linewidth=1.5,
                label='Policy')
            self.ax_loss.plot(rr, s.value_losses,
                color='#f38ba8', linewidth=1.5,
                label='Value')
            self.ax_loss.legend(fontsize=8,
                facecolor='#313244',
                edgecolor='#45475a',
                labelcolor='#cdd6f4')
            self.ax_loss.set_xlabel('Round',
                color='#a6adc8', fontsize=9)
            self.ax_loss.tick_params(colors='#6c7086',
                labelsize=8)
            for sp in self.ax_loss.spines.values():
                sp.set_color('#45475a')

            self.fig.tight_layout(pad=2)
            self.canvas.draw()

        if s.log_dirty:
            s.log_dirty = False
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.insert('1.0',
                '\n'.join(s.log_lines[-50:]))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(1000, self._tick)


def main():
    from training.ppo_trainer import PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
        default=os.path.join(PROJECT_ROOT,
            'rl_data/checkpoints/model_with_decisions.pt'))
    parser.add_argument('--save-dir',
        default=os.path.join(PROJECT_ROOT,
            'rl_data/checkpoints'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--games-per-round', type=int,
        default=200)
    parser.add_argument('--ppo-epochs', type=int,
        default=4)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-games', type=int,
        default=50)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

    state = PPOState()
    t = threading.Thread(target=ppo_thread,
        args=(state, args), daemon=True)
    t.start()
    root = tk.Tk()
    PPODashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
