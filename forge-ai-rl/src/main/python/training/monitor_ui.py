#!/usr/bin/env python3
"""
Lightweight training monitor — reads ppo_training_state.json and redraws
charts every few seconds. Runs independently of the training process.

Usage:
    python training/monitor_ui.py --state-file /path/to/ppo_training_state.json
"""

import argparse
import json
import os
import sys
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

REFRESH_MS = 3000  # redraw every 3 seconds


class MonitorUI:
    def __init__(self, root, state_file):
        self.root = root
        self.state_file = state_file
        root.title("MTG RL — Training Monitor")
        root.geometry("1100x680")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel', font=('Helvetica', 15, 'bold'),
            background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel', font=('Consolas', 11),
            background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel', font=('Consolas', 12, 'bold'),
            background='#1e1e2e', foreground='#89b4fa')
        style.configure('D.TFrame', background='#1e1e2e')

        self._build(root)
        self._refresh()

    def _build(self, root):
        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        ttk.Label(m, text="MTG RL — Training Monitor",
            style='H.TLabel').pack(pady=(0, 4))

        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(m, textvariable=self.status_var,
            style='S.TLabel').pack()

        # Stats row
        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=6)
        self.svars = {}
        for i, (k, v) in enumerate([
            ('Round', '—'), ('Best WR', '—'),
            ('Latest WR', '—'), ('Elo', '—'),
            ('Policy Loss', '—'), ('Value Loss', '—'),
            ('Entropy', '—'), ('Atk Rate', '—'),
            ('Spells/Turn', '—'), ('Idle Turns', '—'),
        ]):
            r, c = divmod(i, 5)
            ttk.Label(sf, text=f"{k}:",
                style='S.TLabel').grid(row=r, column=c*2,
                sticky='w', padx=(10, 2), pady=2)
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv,
                style='V.TLabel').grid(row=r, column=c*2+1,
                sticky='w', padx=(0, 14), pady=2)
            self.svars[k] = sv

        # Charts
        cf = ttk.Frame(m, style='D.TFrame')
        cf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.fig = Figure(figsize=(10, 4), dpi=100,
            facecolor='#1e1e2e')
        self.ax_wr = self.fig.add_subplot(131)
        self.ax_loss = self.fig.add_subplot(132)
        self.ax_gp = self.fig.add_subplot(133)
        for ax, title in [
            (self.ax_wr, 'Win Rate vs Heuristic'),
            (self.ax_loss, 'Training Losses'),
            (self.ax_gp, 'Gameplay Metrics'),
        ]:
            ax.set_facecolor('#313244')
            ax.set_title(title, color='#cdd6f4', fontsize=10)
            ax.tick_params(colors='#6c7086', labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#45475a')
        self.ax_gp2 = self.ax_gp.twinx()
        self.ax_gp2.tick_params(colors='#6c7086', labelsize=7)
        self.ax_gp2.spines['right'].set_color('#45475a')
        self.fig.tight_layout(pad=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(m,
            text=f"Watching: {self.state_file}  •  refresh every {REFRESH_MS//1000}s",
            style='S.TLabel').pack(pady=(4, 0))

    def _refresh(self):
        try:
            with open(self.state_file) as f:
                s = json.load(f)
            self._update(s)
        except Exception as e:
            self.status_var.set(f"Error reading state: {e}")
        self.root.after(REFRESH_MS, self._refresh)

    def _update(self, s):
        rnd = s.get('completed_rounds', 0)
        best_wr = s.get('best_win_rate', 0)
        best_r = s.get('best_round', 0)
        win_rates = s.get('win_rates', [])
        policy_losses = s.get('policy_losses', [])
        value_losses = s.get('value_losses', [])
        entropies = s.get('entropies', [])
        elo = s.get('current_elo', 1000)
        atk_rates = s.get('attack_rates', [])
        spells = s.get('spells_per_turns', [])
        idle = s.get('idle_turns_pcts', [])

        latest_wr = win_rates[-1] if win_rates else 0
        latest_pl = policy_losses[-1] if policy_losses else 0
        latest_vl = value_losses[-1] if value_losses else 0
        latest_ent = entropies[-1] if entropies else 0

        self.status_var.set(
            f"Round {rnd} complete  •  last updated from disk")
        self.svars['Round'].set(str(rnd))
        self.svars['Best WR'].set(
            f"{best_wr:.1%} (r{best_r})")
        self.svars['Latest WR'].set(f"{latest_wr:.1%}")
        self.svars['Elo'].set(f"{elo:.0f}")
        self.svars['Policy Loss'].set(f"{latest_pl:.4f}")
        self.svars['Value Loss'].set(f"{latest_vl:.4f}")
        self.svars['Entropy'].set(f"{latest_ent:.3f}")
        self.svars['Atk Rate'].set(
            f"{atk_rates[-1]:.0f}%" if atk_rates else "—")
        self.svars['Spells/Turn'].set(
            f"{spells[-1]:.2f}" if spells else "—")
        self.svars['Idle Turns'].set(
            f"{idle[-1]:.0f}%" if idle else "—")

        if not win_rates:
            return

        # All x-axes use true round numbers so charts are comparable
        n_rounds = rnd  # total rounds completed
        all_rounds = range(1, n_rounds + 1)
        # eval rounds: every eval_interval (infer from ratio)
        eval_interval = max(1, round(n_rounds / len(win_rates))) \
            if win_rates else 1
        eval_rounds = [i * eval_interval for i in
                       range(1, len(win_rates) + 1)]
        gp_rounds = [i * eval_interval for i in
                     range(1, len(atk_rates) + 1)] if atk_rates \
                    else eval_rounds

        self.ax_wr.clear()
        self.ax_wr.set_facecolor('#313244')
        self.ax_wr.set_title('Win Rate vs Heuristic',
            color='#cdd6f4', fontsize=10)
        self.ax_wr.axhline(y=0.5, color='#f38ba8',
            linestyle='--', linewidth=1, label='50%', zorder=1)
        self.ax_wr.axhline(y=best_wr, color='#f9e2af',
            linestyle=':', linewidth=1,
            label=f'best {best_wr:.1%}', zorder=2)
        self.ax_wr.plot(eval_rounds, win_rates, color='#a6e3a1',
            linewidth=2, marker='o', markersize=4, zorder=3)
        self.ax_wr.set_xlim(1, n_rounds + 3)
        self.ax_wr.set_ylim(0, 0.6)
        self.ax_wr.set_ylabel('Win Rate',
            color='#a6adc8', fontsize=9)
        self.ax_wr.set_xlabel('Round',
            color='#a6adc8', fontsize=9)
        self.ax_wr.legend(fontsize=8, facecolor='#313244',
            edgecolor='#45475a', labelcolor='#cdd6f4')
        self.ax_wr.tick_params(colors='#6c7086', labelsize=8)
        for sp in self.ax_wr.spines.values():
            sp.set_color('#45475a')

        self.ax_loss.clear()
        self.ax_loss.set_facecolor('#313244')
        self.ax_loss.set_title('Training Losses',
            color='#cdd6f4', fontsize=10)
        if policy_losses:
            self.ax_loss.plot(all_rounds, policy_losses,
                color='#89b4fa', linewidth=1.5, label='Policy')
        if value_losses:
            self.ax_loss.plot(all_rounds, value_losses,
                color='#f38ba8', linewidth=1.5, label='Value')
        if entropies:
            ax2 = self.ax_loss.twinx()
            ax2.plot(all_rounds, entropies, color='#a6e3a1',
                linewidth=1.5, linestyle='--', label='Entropy')
            ax2.set_ylabel('Entropy', color='#a6e3a1', fontsize=8)
            ax2.tick_params(colors='#6c7086', labelsize=7)
            ax2.spines['right'].set_color('#45475a')
        self.ax_loss.set_xlim(1, n_rounds + 3)
        self.ax_loss.autoscale(axis='y')
        self.ax_loss.legend(fontsize=8, facecolor='#313244',
            edgecolor='#45475a', labelcolor='#cdd6f4')
        self.ax_loss.set_xlabel('Round',
            color='#a6adc8', fontsize=9)
        self.ax_loss.tick_params(colors='#6c7086', labelsize=8)
        for sp in self.ax_loss.spines.values():
            sp.set_color('#45475a')

        self.ax_gp.clear()
        self.ax_gp2.clear()
        self.ax_gp.set_facecolor('#313244')
        self.ax_gp.set_title('Gameplay Metrics',
            color='#cdd6f4', fontsize=10)
        if atk_rates:
            self.ax_gp.plot(gp_rounds, atk_rates, color='#f9e2af',
                linewidth=1.5, label='Atk %', marker='o',
                markersize=3)
        if idle:
            ir = range(1, len(idle) + 1)
            self.ax_gp.plot(gp_rounds[:len(idle)], idle,
                color='#f38ba8', linewidth=1.5, label='Idle %',
                marker='s', markersize=3)
        if spells:
            self.ax_gp2.plot(gp_rounds[:len(spells)], spells,
                color='#a6e3a1', linewidth=1.5,
                label='Spells/Turn', marker='^', markersize=3)
        self.ax_gp.set_xlim(1, n_rounds + 3)
        self.ax_gp.autoscale(axis='y')
        self.ax_gp2.autoscale(axis='y')
        self.ax_gp.set_ylabel('Percent',
            color='#a6adc8', fontsize=8)
        self.ax_gp.set_xlabel('Round',
            color='#a6adc8', fontsize=9)
        self.ax_gp2.set_ylabel('Spells/Turn',
            color='#a6e3a1', fontsize=8)
        self.ax_gp.legend(fontsize=7, loc='upper left',
            facecolor='#313244', edgecolor='#45475a',
            labelcolor='#cdd6f4')
        self.ax_gp.tick_params(colors='#6c7086', labelsize=8)
        for sp in self.ax_gp.spines.values():
            sp.set_color('#45475a')

        self.fig.tight_layout(pad=2)
        self.canvas.draw()


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.ppo_trainer import PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-file',
        default=os.path.join(PROJECT_ROOT,
            'rl_data/checkpoints/ppo_training_state.json'))
    args = parser.parse_args()

    root = tk.Tk()
    MonitorUI(root, args.state_file)
    root.mainloop()


if __name__ == '__main__':
    main()
