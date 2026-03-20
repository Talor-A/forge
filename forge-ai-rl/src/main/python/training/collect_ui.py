#!/usr/bin/env python3
"""
Data Collection Dashboard — visual monitoring of trajectory collection.

Launches the Java game runner and displays live progress:
- Games completed / total
- Win/loss counts
- Trajectory files and decision counts
- Average turns per game
- Collection speed

Usage:
    python training/collect_ui.py --games 1000
"""

import argparse
import os
import sys
import threading
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from training.ppo_trainer import PROJECT_ROOT, FORGE_JAR

import subprocess


@dataclass
class CollectState:
    status: str = "Starting..."
    phase: str = "init"

    games_done: int = 0
    games_total: int = 0

    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    errors: int = 0

    traj_files: int = 0
    attack_decisions: int = 0
    block_decisions: int = 0
    priority_decisions: int = 0

    games_per_sec: float = 0.0
    eta_sec: float = 0.0
    elapsed_sec: float = 0.0
    avg_turns: float = 0.0

    # For chart
    speed_history: List[float] = field(default_factory=list)
    files_history: List[int] = field(default_factory=list)
    chart_dirty: bool = False

    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 200:
        state.log_lines = state.log_lines[-200:]
    state.log_dirty = True


def collect_thread(state, args):
    """Run Java data collection and monitor progress."""
    try:
        output_dir = os.path.join(PROJECT_ROOT, 'rl_data/trajectories')
        os.makedirs(output_dir, exist_ok=True)

        state.games_total = args.games
        state.phase = "collecting"
        state.status = f"Collecting {args.games} games..."

        log(state, f"Output: {output_dir}")
        log(state, f"Games: {args.games}, Threads: 16")

        decks = ['green_stompy.dck', 'white_weenie.dck',
                 'blue_tempo.dck', 'red_aggro.dck']
        deck_args = []
        for d in decks:
            deck_args.extend(['-d', d])

        cmd = [
            'java', '-Xmx8192m',
            '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
            '--add-opens', 'java.base/java.util=ALL-UNNAMED',
            '--add-opens', 'java.base/java.text=ALL-UNNAMED',
            '--add-opens', 'java.base/java.lang.reflect=ALL-UNNAMED',
            '--add-opens', 'java.desktop/javax.imageio.spi=ALL-UNNAMED',
            '-jar', FORGE_JAR,
            'rltrain', 'collect',
        ] + deck_args + [
            '-n', str(args.games),
            '-t', '16',
            '-o', output_dir,
        ]

        cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')
        t0 = time.time()

        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)

        # Monitor thread: poll files for progress + track time
        def monitor():
            while state.phase == "collecting":
                try:
                    # Count trajectory files for continuous progress
                    files = list(Path(output_dir).glob('traj_*.jsonl'))
                    file_count = len(files)
                    if file_count > state.traj_files:
                        state.traj_files = file_count
                    # 2 files per game (one per player)
                    games_from_files = file_count // 2
                    if games_from_files > state.games_done:
                        state.games_done = games_from_files

                    elapsed = time.time() - t0
                    state.elapsed_sec = elapsed
                    if elapsed > 0 and state.games_done > 0:
                        state.games_per_sec = (
                            state.games_done / elapsed)
                    if state.games_per_sec > 0:
                        remaining = (state.games_total
                                     - state.games_done)
                        state.eta_sec = remaining / state.games_per_sec
                    if state.games_done > 0:
                        state.status = (
                            f"Game {state.games_done}/"
                            f"{state.games_total}"
                            f" ({state.games_per_sec:.1f}/s)")
                    state.speed_history.append(state.games_per_sec)
                    state.files_history.append(state.traj_files)
                    state.chart_dirty = True
                except Exception:
                    pass
                time.sleep(1)

        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Parse progress table rows (pipe-delimited with │)
            # Format: Game │ Done/Total │ Games/s │ P1 │ P2 │ Draw│ Err │ Turns │ Files │ ETA
            if '│' in line and not line.startswith('─'):
                try:
                    cols = [c.strip() for c in line.split('│')]
                    if len(cols) >= 10:
                        # cols: ['Game', 'Done/Total', 'Games/s', 'P1', 'P2',
                        #        'Draw', 'Err', 'Turns', 'Files', 'ETA']
                        # Skip header row
                        if cols[1] and '/' in cols[1]:
                            parts = cols[1].split('/')
                            done = int(parts[0])
                            total = int(parts[1])
                            state.games_done = done
                            state.games_total = max(
                                state.games_total, total)
                        if cols[2]:
                            try:
                                state.games_per_sec = float(cols[2])
                            except ValueError:
                                pass
                        if cols[3]:
                            try:
                                state.p1_wins = int(cols[3])
                            except ValueError:
                                pass
                        if cols[4]:
                            try:
                                state.p2_wins = int(cols[4])
                            except ValueError:
                                pass
                        if cols[5]:
                            try:
                                state.draws = int(cols[5])
                            except ValueError:
                                pass
                        if cols[6]:
                            try:
                                state.errors = int(cols[6])
                            except ValueError:
                                pass
                        if cols[7]:
                            try:
                                state.avg_turns = float(cols[7])
                            except ValueError:
                                pass
                        if cols[8]:
                            try:
                                state.traj_files = int(cols[8])
                            except ValueError:
                                pass
                except Exception:
                    pass

            # Parse summary lines (after collection complete)
            # "Games: 1000 | P1: 500 | P2: 420 | Draw: 70 | Errors: 10"
            if 'P1:' in line and 'P2:' in line:
                try:
                    for part in line.split('|'):
                        part = part.strip()
                        if part.startswith('P1:'):
                            state.p1_wins = int(part[3:].strip())
                        elif part.startswith('P2:'):
                            state.p2_wins = int(part[3:].strip())
                        elif part.startswith('Draw:'):
                            state.draws = int(part[5:].strip())
                        elif part.startswith('Errors:'):
                            state.errors = int(part[7:].strip())
                except (ValueError, IndexError):
                    pass

            # "Avg turns: 8.5 | Files: 200"
            if 'Avg turns:' in line:
                try:
                    for part in line.split('|'):
                        part = part.strip()
                        if part.startswith('Avg turns:'):
                            state.avg_turns = float(
                                part[10:].strip())
                        elif part.startswith('Files:'):
                            state.traj_files = int(
                                part[6:].strip())
                except (ValueError, IndexError):
                    pass

            if 'Collection Complete' in line:
                state.phase = "analyzing"
                state.status = "Analyzing trajectories..."

            log(state, line)

        proc.wait()
        state.phase = "analyzing"

        # Analyze trajectory files
        state.status = "Counting decisions..."
        attacks, blocks, priority = 0, 0, 0
        tapped_leak = 0
        total_cand, total_sel = 0, 0
        files = sorted(Path(output_dir).glob('traj_*.jsonl'))
        state.traj_files = len(files)

        for f in files:
            for fline in open(f):
                try:
                    r = json.loads(fline)
                    dt = r.get('decisionType', '')
                    cand = r.get('candidateFeatures', [])
                    sel = r.get('selectedIndices', [])
                    if dt == 'DECLARE_ATTACKERS' and len(cand) > 0:
                        attacks += 1
                        total_cand += len(cand)
                        total_sel += len(sel)
                        for i, cf in enumerate(cand):
                            if len(cf) > 17 and cf[17] > 0.5 and i in sel:
                                tapped_leak += 1
                    elif dt == 'DECLARE_BLOCKERS' and len(cand) > 0:
                        blocks += 1
                    elif dt == 'PRIORITY_ACTION':
                        priority += 1
                except Exception:
                    pass

        state.attack_decisions = attacks
        state.block_decisions = blocks
        state.priority_decisions = priority
        state.elapsed_sec = time.time() - t0

        state.phase = "done"
        state.status = "Collection complete!"

        log(state, "")
        log(state, f"=== Summary ===")
        log(state, f"Files: {state.traj_files}")
        log(state, f"Attacks: {attacks}, Blocks: {blocks}")
        log(state, f"Avg candidates/attack: {total_cand/max(attacks,1):.1f}")
        log(state, f"Avg selected/attack: {total_sel/max(attacks,1):.1f}")
        log(state, f"Attack rate: {total_sel*100/max(total_cand,1):.1f}%")
        log(state, f"Tapped leak: {tapped_leak} (MUST be 0)")
        log(state, f"Time: {state.elapsed_sec:.0f}s")

    except Exception as e:
        log(state, f"ERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        import traceback
        traceback.print_exc()


class CollectDashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        root.title("MTG RL — Data Collection")
        root.geometry("900x700")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel', font=('Helvetica', 16, 'bold'),
                         background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel', font=('Consolas', 11),
                         background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel', font=('Consolas', 11, 'bold'),
                         background='#1e1e2e', foreground='#89b4fa')
        style.configure('D.TFrame', background='#1e1e2e')
        style.configure("b.Horizontal.TProgressbar",
                         troughcolor='#313244', background='#89b4fa')

        self._build(root)
        self._tick()

    def _build(self, root):
        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(m, text="MTG RL — Data Collection",
                  style='H.TLabel').pack(pady=(0, 6))

        self.status_v = tk.StringVar(value="Starting...")
        ttk.Label(m, textvariable=self.status_v,
                  style='S.TLabel').pack()

        # Progress bar
        pf = ttk.Frame(m, style='D.TFrame')
        pf.pack(fill=tk.X, pady=4)
        self.prog = ttk.Progressbar(pf, length=850,
                                     style="b.Horizontal.TProgressbar")
        self.prog.pack(fill=tk.X)
        self.prog_v = tk.StringVar()
        ttk.Label(pf, textvariable=self.prog_v,
                  style='S.TLabel').pack(anchor='w')

        # Stats grid
        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=4)
        self.svars = {}
        stats = [
            ('Games', '—'), ('Speed', '—'),
            ('P1 Wins', '—'), ('P2 Wins', '—'),
            ('Draws', '—'), ('Errors', '—'),
            ('Avg Turns', '—'), ('ETA', '—'),
            ('Files', '—'), ('Attacks', '—'),
            ('Blocks', '—'), ('Elapsed', '—'),
        ]
        for i, (k, v) in enumerate(stats):
            r, c = divmod(i, 4)
            ttk.Label(sf, text=f"{k}:", style='S.TLabel').grid(
                row=r, column=c*2, sticky='w', padx=(8, 2), pady=2)
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv, style='V.TLabel').grid(
                row=r, column=c*2+1, sticky='w', padx=(0, 12), pady=2)
            self.svars[k] = sv

        # Console log
        lf = ttk.Frame(m, style='D.TFrame')
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf, height=15,
                                 bg='#181825', fg='#a6adc8',
                                 font=('Consolas', 9),
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _tick(self):
        s = self.state
        self.status_v.set(s.status)

        if s.games_total > 0:
            pct = s.games_done / s.games_total * 100
            self.prog['value'] = pct
            self.prog_v.set(
                f"{s.games_done}/{s.games_total} games "
                f"({pct:.0f}%)")

        self.svars['Games'].set(
            f"{s.games_done}/{s.games_total}")
        self.svars['Speed'].set(
            f"{s.games_per_sec:.1f}/s")
        self.svars['P1 Wins'].set(str(s.p1_wins))
        self.svars['P2 Wins'].set(str(s.p2_wins))
        self.svars['Draws'].set(str(s.draws))
        self.svars['Errors'].set(str(s.errors))
        self.svars['Avg Turns'].set(
            f"{s.avg_turns:.1f}" if s.avg_turns > 0 else "—")
        self.svars['ETA'].set(
            f"{s.eta_sec:.0f}s" if s.eta_sec > 0 else "—")
        self.svars['Files'].set(str(s.traj_files))
        self.svars['Attacks'].set(str(s.attack_decisions))
        self.svars['Blocks'].set(str(s.block_decisions))
        self.svars['Elapsed'].set(
            f"{s.elapsed_sec:.0f}s" if s.elapsed_sec > 0 else "—")

        if s.log_dirty:
            s.log_dirty = False
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.insert('1.0',
                                  '\n'.join(s.log_lines[-40:]))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(500, self._tick)


def main():
    parser = argparse.ArgumentParser(
        description='MTG RL Data Collection Dashboard')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of games to collect')
    args = parser.parse_args()

    state = CollectState()
    t = threading.Thread(target=collect_thread,
                         args=(state, args), daemon=True)
    t.start()

    root = tk.Tk()
    CollectDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
