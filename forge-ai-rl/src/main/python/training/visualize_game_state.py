#!/usr/bin/env python3
"""
Game State Visualizer — shows board state as the model sees it,
with model predictions.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from training.ppo_trainer import parse_game_state

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Feature decoding ───────────────────────────────

KEYWORDS = [
    "Flying", "First Strike", "Double Strike", "Trample",
    "Haste", "Vigilance", "Deathtouch", "Lifelink", "Reach",
    "Hexproof", "Indestructible", "Flash", "Menace", "Defender",
    "Prowess",
]

COLOR_HEX = {
    "W": ("#f9f3e0", "#1a1a1a"),
    "U": ("#0e68ab", "#ffffff"),
    "B": ("#2b2b2b", "#cccccc"),
    "R": ("#d32029", "#ffffff"),
    "G": ("#00733e", "#ffffff"),
    "C": ("#9e9e9e", "#1a1a1a"),
}

LAND_COLORS = {
    "W": "Plains", "U": "Island", "B": "Swamp",
    "R": "Mountain", "G": "Forest",
}


def decode_card(feats):
    if len(feats) < 30:
        return None

    type_names = ["Creature", "Instant", "Sorcery",
                  "Enchantment", "Artifact", "Planeswalker", "Land"]
    types = [t for i, t in enumerate(type_names)
             if i < len(feats) and feats[i] > 0.5]

    color_chars = ["W", "U", "B", "R", "G", "C"]
    colors = [c for i, c in enumerate(color_chars)
              if 7+i < len(feats) and feats[7+i] > 0.5]

    cmc = int(round(feats[13] * 16)) if len(feats) > 13 else 0

    power = int(round(feats[14] * 25 - 5)) if "Creature" in types and len(feats) > 14 else None
    toughness = int(round(feats[15] * 25 - 5)) if "Creature" in types and len(feats) > 15 else None

    tapped = feats[17] > 0.5 if len(feats) > 17 else False
    sick = feats[18] > 0.5 if len(feats) > 18 else False

    p1p1 = int(round(feats[22] * 20)) if len(feats) > 22 else 0

    kws = [kw for i, kw in enumerate(KEYWORDS)
           if 29+i < len(feats) and feats[29+i] > 0.5]

    # Type label
    if "Land" in types:
        c = colors[0] if colors and colors[0] != "C" else "C"
        label = LAND_COLORS.get(c, "Land")
    elif types:
        label = "/".join(types)
    else:
        label = "?"

    return {
        "label": label, "types": types, "colors": colors,
        "cmc": cmc, "power": power, "toughness": toughness,
        "tapped": tapped, "sick": sick, "p1p1": p1p1,
        "keywords": kws,
    }


# ActionEncoder 64-dim feature layout
API_TYPES = [
    "DealDamage", "Draw", "Counter", "ChangeZone",
    "Pump", "PumpAll", "Destroy", "DestroyAll",
    "Sacrifice", "Discard", "GainLife", "LoseLife",
    "Token", "Animate", "Attach", "Tap",
    "Untap", "Mill", "Regenerate", "Protection",
    "Fight", "Charm", "Scry", "Explore",
    "AddOrRemoveCounter", "ManaReflected", "Mana",
    "ChangeTargets", "Fog", "ChangeZone",
]


def decode_action(feats):
    """Decode a 64-dim ActionEncoder feature vector."""
    if len(feats) < 18:
        return None

    # Check pass action (feature[63] = 1.0)
    if len(feats) > 63 and feats[63] > 0.5:
        return {"label": "PASS", "is_pass": True,
                "types": [], "colors": [], "cmc": 0,
                "sa_type": "pass", "api": None,
                "targets": False}

    type_names = ["Creature", "Instant", "Sorcery",
                  "Enchantment", "Artifact", "Planeswalker",
                  "Land"]
    types = [t for i, t in enumerate(type_names)
             if i < len(feats) and feats[i] > 0.5]

    color_chars = ["W", "U", "B", "R", "G", "C"]
    colors = [c for i, c in enumerate(color_chars)
              if 7+i < len(feats) and feats[7+i] > 0.5]

    cmc = int(round(feats[13] * 16)) if len(feats) > 13 \
        else 0

    sa_type = "spell" if feats[14] > 0.5 \
        else "activated" if feats[15] > 0.5 \
        else "triggered" if feats[16] > 0.5 \
        else "mana" if feats[17] > 0.5 \
        else "other"

    # API type [18-47]
    api = None
    for i, name in enumerate(API_TYPES):
        if 18+i < len(feats) and feats[18+i] > 0.5:
            api = name
            break

    targets = feats[48] > 0.5 if len(feats) > 48 else False

    # Build label
    color_str = "".join(colors) if colors else ""
    type_str = "/".join(types) if types else "?"
    label = f"{color_str} {type_str}" if color_str \
        else type_str
    if api:
        label += f" ({api})"
    label += f" [{cmc}]"

    return {
        "label": label, "is_pass": False,
        "types": types, "colors": colors, "cmc": cmc,
        "sa_type": sa_type, "api": api,
        "targets": targets,
    }


# ── Card rendering (MTG card style) ───────────────

CARD_W, CARD_H = 85, 120


def draw_card_image(info, highlight=None):
    """Draw a card as PIL image in MTG card style."""
    if not HAS_PIL:
        return None

    c = info["colors"][0] if info["colors"] else "C"
    bg_color, fg_color = COLOR_HEX.get(c, COLOR_HEX["C"])

    # Border
    if highlight == "attack":
        border = "#00ff00"
    elif highlight == "block":
        border = "#4488ff"
    else:
        border = "#333333"

    img = Image.new("RGB", (CARD_W, CARD_H), border)
    draw = ImageDraw.Draw(img)

    # Card body
    m = 3  # border margin
    draw.rectangle([m, m, CARD_W-m-1, CARD_H-m-1], fill=bg_color)

    # Try to load a small font, fall back to default
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 8)
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 10)
    except (IOError, OSError):
        font_sm = ImageFont.load_default()
        font_md = font_sm
        font_lg = font_sm

    y = 4

    # Title bar: type + CMC
    type_str = "/".join(info["types"])[:12]
    draw.text((4, y), type_str, fill=fg_color, font=font_md)
    cmc_str = f"{{{info['cmc']}}}"
    draw.text((CARD_W - 22, y), cmc_str, fill=fg_color, font=font_md)
    y += 12

    # Color identity
    color_str = "".join(info["colors"]) if info["colors"] else "Colorless"
    draw.text((4, y), color_str, fill=fg_color, font=font_sm)
    y += 10

    # Separator
    draw.line([(4, y), (CARD_W-4, y)], fill=fg_color, width=1)
    y += 3

    # Art area (just colored block)
    art_h = 24
    darker = _darken(bg_color, 0.8)
    draw.rectangle([4, y, CARD_W-4, y+art_h], fill=darker)

    # Show land name in art area for lands
    if "Land" in info["types"]:
        draw.text((6, y+6), info["label"],
                  fill=fg_color, font=font_lg)
    y += art_h + 3

    # Separator
    draw.line([(4, y), (CARD_W-4, y)], fill=fg_color, width=1)
    y += 3

    # Keywords (compact)
    for kw in info["keywords"][:2]:
        draw.text((4, y), kw, fill=fg_color, font=font_sm)
        y += 10

    # +1/+1 counters
    if info["p1p1"] > 0:
        draw.text((4, y), f"+{info['p1p1']}/+{info['p1p1']}",
                  fill="#ffdd00", font=font_sm)
        y += 10

    # State (tapped/sick)
    if info["tapped"]:
        draw.text((4, y), "TAPPED", fill="#ff6666", font=font_sm)
        y += 10
    if info["sick"]:
        draw.text((4, y), "SICK", fill="#ffaa44", font=font_sm)
        y += 10

    # P/T box (bottom right for creatures)
    if info["power"] is not None:
        pt = f"{info['power']}/{info['toughness']}"
        box_w = 30
        bx = CARD_W - box_w - 4
        by = CARD_H - 18
        draw.rectangle([bx, by, CARD_W-4, CARD_H-4],
                       fill="#000000", outline=fg_color)
        draw.text((bx+3, by+1), pt, fill="#ffffff", font=font_lg)

    return img


def _darken(hex_color, factor):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ── Data loading ───────────────────────────────────

def load_samples(data_dir, max_samples=200):
    path = Path(data_dir)
    files = sorted(path.glob("traj_*.jsonl"))
    random.shuffle(files)

    samples = []
    for f in files:
        if len(samples) >= max_samples:
            break
        try:
            lines = open(f).readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get("won", False)

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get("decisionType", "")
                if dt not in ("DECLARE_ATTACKERS",
                              "DECLARE_BLOCKERS",
                              "PRIORITY_ACTION"):
                    continue
                cand = rec.get("candidateFeatures", [])
                if len(cand) < 1:
                    continue

                samples.append({
                    "type": dt,
                    "info": rec.get("contextInfo", ""),
                    "global_features": np.array(
                        rec.get("globalFeatures", []),
                        dtype=np.float32),
                    "game_state_flat": np.array(
                        rec.get("gameStateFlat", []),
                        dtype=np.float32),
                    "candidates": cand,
                    "selected": rec.get(
                        "selectedIndices", []),
                    "won": won,
                })
                if len(samples) >= max_samples:
                    break
        except Exception:
            pass
    return samples


# ── Main Viewer ────────────────────────────────────

class GameStateViewer:
    def __init__(self, root, samples, model, device,
                 model_path=None, data_dir=None):
        self.root = root
        self.all_samples = samples
        self.samples = samples
        self.model = model
        self.device = device
        self.model_path = model_path
        self.data_dir = data_dir
        self.idx = 0
        self.card_photos = []  # keep references

        root.title("MTG RL — Game State Visualizer")
        root.geometry("1500x950")
        root.configure(bg="#1e1e2e")

        self._build(root)
        if samples:
            self._show()

    def _build(self, root):
        # Nav bar
        nav = tk.Frame(root, bg="#1e1e2e")
        nav.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Random", command=self._rand).pack(side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(nav, text="Attacks", command=self._filter_attacks).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Blocks", command=self._filter_blocks).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Priority", command=self._filter_priority).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Pri Pass", command=self._filter_priority_pass).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Pri Play", command=self._filter_priority_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="All", command=self._filter_all).pack(side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(nav, text="Reload Model",
                   command=self._reload_model).pack(
                       side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Reload Data",
                   command=self._reload_data).pack(
                       side=tk.LEFT, padx=3)

        self.model_v = tk.StringVar(
            value=os.path.basename(model_path)
            if model_path else "no model")
        tk.Label(nav, textvariable=self.model_v,
                 bg="#1e1e2e", fg="#a6e3a1",
                 font=("Consolas", 9)).pack(
                     side=tk.RIGHT, padx=5)

        self.nav_v = tk.StringVar(value="0/0")
        tk.Label(nav, textvariable=self.nav_v, bg="#1e1e2e",
                 fg="#a6adc8", font=("Consolas", 11)).pack(side=tk.LEFT, padx=15)

        self.type_v = tk.StringVar()
        tk.Label(nav, textvariable=self.type_v, bg="#1e1e2e",
                 fg="#f9e2af", font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=5)

        self.info_v = tk.StringVar()
        tk.Label(nav, textvariable=self.info_v, bg="#1e1e2e",
                 fg="#89b4fa", font=("Consolas", 11)).pack(side=tk.LEFT, padx=10)

        self.outcome_v = tk.StringVar()
        self.outcome_lbl = tk.Label(nav, textvariable=self.outcome_v,
                                     bg="#1e1e2e", font=("Consolas", 14, "bold"))
        self.outcome_lbl.pack(side=tk.RIGHT, padx=10)

        # Status bar
        self.status_v = tk.StringVar()
        tk.Label(root, textvariable=self.status_v, bg="#181825",
                 fg="#cdd6f4", font=("Consolas", 11), anchor="w",
                 padx=10, pady=4).pack(fill=tk.X)

        # Main: board on left, predictions on right
        main = tk.Frame(root, bg="#1e1e2e")
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: board areas
        board_col = tk.Frame(main, bg="#1e1e2e")
        board_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Opponent board
        self._make_section(board_col, "Opponent's Board", "#f38ba8")
        self.opp_creatures = tk.Frame(board_col, bg="#252535")
        self.opp_creatures.pack(fill=tk.X, padx=5)
        self.opp_lands = tk.Frame(board_col, bg="#202030")
        self.opp_lands.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(fill=tk.X, pady=3)

        # My board
        self._make_section(board_col, "My Board", "#a6e3a1")
        self.my_creatures = tk.Frame(board_col, bg="#252535")
        self.my_creatures.pack(fill=tk.X, padx=5)
        self.my_lands = tk.Frame(board_col, bg="#202030")
        self.my_lands.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(fill=tk.X, pady=3)

        # Stack
        self._make_section(board_col, "Stack", "#cba6f7")
        self.stack_frame = tk.Frame(board_col, bg="#252535")
        self.stack_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(
            fill=tk.X, pady=3)

        # My hand
        self._make_section(board_col, "My Hand", "#f9e2af")
        self.hand_frame = tk.Frame(board_col, bg="#252535")
        self.hand_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(
            fill=tk.X, pady=3)

        # Priority candidates (shown for PRIORITY_ACTION)
        self._make_section(board_col,
                           "Priority Candidates", "#fab387")
        self.priority_frame = tk.Frame(
            board_col, bg="#252535")
        self.priority_frame.pack(
            fill=tk.X, padx=5, pady=(0, 5))

        # Right: model predictions
        pred_col = tk.Frame(main, bg="#181825", width=320)
        pred_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        pred_col.pack_propagate(False)

        tk.Label(pred_col, text="Model Prediction",
                 bg="#181825", fg="#89b4fa",
                 font=("Consolas", 12, "bold")).pack(padx=5, pady=5)

        self.pred_text = tk.Text(pred_col, bg="#181825", fg="#cdd6f4",
                                  font=("Consolas", 10), wrap=tk.WORD,
                                  state=tk.DISABLED)
        self.pred_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _make_section(self, parent, title, color):
        tk.Label(parent, text=title, bg="#1e1e2e",
                 fg=color, font=("Consolas", 10, "bold"),
                 anchor="w").pack(fill=tk.X, padx=5)

    def _filter_attacks(self):
        self.filtered = [s for s in self.all_samples if s["type"] == "DECLARE_ATTACKERS"]
        self.samples = self.filtered if self.filtered else self.all_samples
        self.idx = 0
        self._show()

    def _filter_blocks(self):
        self.filtered = [s for s in self.all_samples if s["type"] == "DECLARE_BLOCKERS"]
        self.samples = self.filtered if self.filtered else self.all_samples
        self.idx = 0
        self._show()

    def _filter_priority(self):
        f = [s for s in self.all_samples
             if s["type"] == "PRIORITY_ACTION"]
        self.samples = f if f else self.all_samples
        self.idx = 0
        self._show()

    def _filter_priority_pass(self):
        f = [s for s in self.all_samples
             if s["type"] == "PRIORITY_ACTION"
             and s["selected"]
             and s["selected"][0] == len(
                 s["candidates"]) - 1]
        self.samples = f if f else self.all_samples
        self.idx = 0
        self._show()

    def _filter_priority_play(self):
        f = [s for s in self.all_samples
             if s["type"] == "PRIORITY_ACTION"
             and s["selected"]
             and s["selected"][0] < len(
                 s["candidates"]) - 1]
        self.samples = f if f else self.all_samples
        self.idx = 0
        self._show()

    def _reload_model(self):
        """Reload model from disk (picks up PPO updates)."""
        if not self.model_path:
            return
        # Try PPO best first, then the specified path
        candidates = [
            os.path.join(os.path.dirname(self.model_path),
                         'best_ppo_model.pt'),
            self.model_path,
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self.model = MTGModel.load(
                        path, device=self.device)
                    self.model.eval()
                    self.model_v.set(
                        os.path.basename(path) +
                        " (reloaded)")
                    if self.samples:
                        self._show()
                    return
                except Exception as e:
                    print(f"Reload failed: {e}")

    def _reload_data(self):
        """Reload trajectory data from disk."""
        if not self.data_dir:
            return
        # Also check PPO trajectories
        dirs = [self.data_dir]
        ppo_dir = os.path.join(
            os.path.dirname(self.data_dir),
            'ppo_trajectories')
        if os.path.isdir(ppo_dir):
            dirs.append(ppo_dir)
        ppo_eval = ppo_dir + '_eval'
        if os.path.isdir(ppo_eval):
            dirs.append(ppo_eval)

        all_samples = []
        for d in dirs:
            all_samples.extend(
                load_samples(d, max_samples=100))
        if all_samples:
            self.all_samples = all_samples
            self.samples = all_samples
            self.idx = 0
            self._show()

    def _filter_all(self):
        self.samples = self.all_samples
        self.idx = 0
        self._show()

    def _prev(self):
        self.idx = max(0, self.idx - 1)
        self._show()

    def _next(self):
        self.idx = min(len(self.samples) - 1, self.idx + 1)
        self._show()

    def _rand(self):
        self.idx = random.randint(0, len(self.samples) - 1)
        self._show()

    def _show(self):
        s = self.samples[self.idx]
        self.card_photos = []  # clear references

        self.nav_v.set(f"{self.idx+1}/{len(self.samples)}")
        dt = s.get("type", "?")
        type_labels = {
            "DECLARE_ATTACKERS": "ATTACK",
            "DECLARE_BLOCKERS": "BLOCK",
            "PRIORITY_ACTION": "PRIORITY",
        }
        self.type_v.set(type_labels.get(dt, dt))
        self.info_v.set(s["info"])
        won = s["won"]
        self.outcome_v.set("WON" if won else "LOST")
        self.outcome_lbl.configure(fg="#a6e3a1" if won else "#f38ba8")

        gf = s["global_features"]
        flat = s["game_state_flat"]

        # Status
        PHASE_NAMES = [
            "Untap", "Upkeep", "Draw", "Main 1",
            "Begin Combat", "Declare Attackers",
            "Declare Blockers", "First Strike Dmg",
            "Combat Damage", "End Combat",
            "Main 2", "End of Turn", "Cleanup",
        ]
        if len(gf) >= 35:
            turn = int(round(gf[4] * 30))
            active = "My turn" if gf[5] > 0.5 \
                else "Opp turn"
            phase = "?"
            for i, pn in enumerate(PHASE_NAMES):
                if 6+i < len(gf) and gf[6+i] > 0.5:
                    phase = pn
                    break
            lands_untap = int(round(gf[29] * 15))
            stack_size = int(round(gf[32] * 10))
            self.status_v.set(
                f"Turn {turn} | {active} | {phase}  ||  "
                f"Life: {gf[0]*50-10:.0f} vs "
                f"{gf[1]*50-10:.0f}  |  "
                f"Hand: {gf[19]*15:.0f} vs "
                f"{gf[20]*15:.0f}  |  "
                f"Creatures: {gf[23]*20:.0f} vs "
                f"{gf[24]*20:.0f}  |  "
                f"Lands untapped: {lands_untap}  |  "
                f"Stack: {stack_size}")

        # Parse zones
        card_dim = 128
        zones = {}
        offset = 64
        for zname, count in [("my_board", 30), ("opp_board", 30),
                              ("my_hand", 15), ("my_gy", 40),
                              ("opp_gy", 40), ("stack", 10)]:
            cards = []
            for j in range(count):
                start = offset + j * card_dim
                end = start + card_dim
                if end <= len(flat):
                    cf = flat[start:end]
                    if np.any(cf != 0):
                        info = decode_card(cf)
                        if info:
                            cards.append(info)
            zones[zname] = cards
            offset += count * card_dim

        # Split my board creatures — mark which are attacking
        selected = set(s.get("selected", []))
        my_board = zones.get("my_board", [])
        my_creatures = [c for c in my_board if "Land" not in c["types"]]
        my_lands = [c for c in my_board if "Land" in c["types"]]

        opp_board = zones.get("opp_board", [])
        opp_creatures = [c for c in opp_board if "Land" not in c["types"]]
        opp_lands = [c for c in opp_board if "Land" in c["types"]]

        hand = zones.get("my_hand", [])

        # Render
        self._render_cards(self.opp_creatures, opp_creatures)
        self._render_cards(self.opp_lands, opp_lands)

        # For my creatures, mark attackers based on candidate index
        # The candidates list maps to creatures that can attack
        cand_highlights = {}
        for i in selected:
            cand_highlights[i] = "attack"
        self._render_cards(self.my_creatures, my_creatures,
                           candidate_highlights=cand_highlights)
        self._render_cards(self.my_lands, my_lands)

        self._render_cards(self.hand_frame, hand)

        # Stack
        stack_cards = zones.get("stack", [])
        self._render_cards(self.stack_frame, stack_cards)

        # Priority candidates
        self._render_priority(s)

        # Model prediction
        self._predict(s)

    def _render_cards(self, frame, cards, candidate_highlights=None):
        for w in frame.winfo_children():
            w.destroy()

        if not cards:
            tk.Label(frame, text="(empty)", bg=frame["bg"],
                     fg="#585b70", font=("Consolas", 9)).pack(
                         side=tk.LEFT, padx=10, pady=20)
            return

        for i, info in enumerate(cards):
            hl = None
            if candidate_highlights and i in candidate_highlights:
                hl = candidate_highlights[i]

            if HAS_PIL:
                pil_img = draw_card_image(info, highlight=hl)
                if pil_img:
                    photo = ImageTk.PhotoImage(pil_img)
                    self.card_photos.append(photo)
                    lbl = tk.Label(frame, image=photo, bg=frame["bg"])
                    lbl.pack(side=tk.LEFT, padx=1, pady=2)
                    continue

            # Text fallback
            c = info["colors"][0] if info["colors"] else "C"
            bg, fg = COLOR_HEX.get(c, COLOR_HEX["C"])
            text = info["label"]
            if info["power"] is not None:
                text += f"\n{info['power']}/{info['toughness']}"
            lbl = tk.Label(frame, text=text, bg=bg, fg=fg,
                           font=("Consolas", 8), width=14, height=5,
                           relief="raised", borderwidth=2)
            lbl.pack(side=tk.LEFT, padx=1, pady=2)

    def _render_priority(self, s):
        """Render priority action candidates as text labels."""
        for w in self.priority_frame.winfo_children():
            w.destroy()

        if s.get("type") != "PRIORITY_ACTION":
            tk.Label(self.priority_frame,
                     text="(not a priority decision)",
                     bg=self.priority_frame["bg"],
                     fg="#585b70",
                     font=("Consolas", 9)).pack(
                         side=tk.LEFT, padx=10, pady=8)
            return

        candidates = s.get("candidates", [])
        selected = s.get("selected", [])
        sel_idx = selected[0] if selected else -1

        for i, cf in enumerate(candidates):
            info = decode_action(cf)
            if not info:
                continue

            is_chosen = (i == sel_idx)
            is_pass = info.get("is_pass", False)

            if is_chosen:
                bg = "#a6e3a1" if not is_pass else "#f9e2af"
                fg = "#1e1e2e"
            elif is_pass:
                bg = "#45475a"
                fg = "#a6adc8"
            else:
                bg = "#313244"
                fg = "#cdd6f4"

            text = info["label"]
            if is_chosen:
                text = f">> {text}"

            lbl = tk.Label(
                self.priority_frame, text=text,
                bg=bg, fg=fg,
                font=("Consolas", 9, "bold"
                      if is_chosen else ""),
                padx=6, pady=3, relief="raised",
                borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=4)

    def _predict(self, s):
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete("1.0", tk.END)

        if self.model is None:
            self.pred_text.insert("1.0", "No model loaded")
            self.pred_text.config(state=tk.DISABLED)
            return

        gf = s["global_features"]
        flat = s["game_state_flat"]

        g = np.zeros(64, dtype=np.float32)
        gl = min(len(gf), 64)
        if gl > 0:
            g[:gl] = gf[:gl]
        np.clip(g, -10, 10, out=g)

        _, zones, masks = parse_game_state(flat, g)

        with torch.no_grad():
            t = lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)

            state = self.model.encode_state(
                t(g),
                t(zones["my_board"]), t(masks["my_board_mask"]),
                t(zones["opp_board"]), t(masks["opp_board_mask"]),
                t(zones["hand"]), t(masks["hand_mask"]),
                t(zones["my_gy"]), t(masks["my_gy_mask"]),
                t(zones["opp_gy"]), t(masks["opp_gy_mask"]),
                t(zones["stack"]), t(masks["stack_mask"]))

            value = self.model.get_value(state).item()

            lines = []
            lines.append(f"Win probability: {(value+1)/2:.0%}")
            lines.append(f"Value: {value:+.3f}")
            lines.append("")
            correct = (value > 0) == s["won"]
            lines.append(f"Model: {'WIN' if value > 0 else 'LOSS'}")
            lines.append(f"Actual: {'WON' if s['won'] else 'LOST'}")
            lines.append(f"{'CORRECT' if correct else 'WRONG'}")
            lines.append("")

            candidates = s.get("candidates", [])
            dt = s.get("type", "")

            if candidates and dt == "DECLARE_ATTACKERS":
                n = len(candidates)
                cf_t = torch.zeros(1, n, 128, device=self.device)
                cm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), 128)
                    cf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                logits = self.model.attack_head(state, cf_t, cm_t)
                probs = torch.sigmoid(logits)

                lines.append("=== Attack Decision ===")
                lines.append("")
                for j in range(n):
                    p = probs[0, j].item()
                    info = decode_card(candidates[j])
                    sel = j in s["selected"]
                    marker = ">>" if sel else "  "
                    label = info["label"] if info else "?"
                    if info and info["power"] is not None:
                        label += f" {info['power']}/{info['toughness']}"
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(f"   [{bar:20s}] {p:.0%}")
                    lines.append("")

                lines.append(f"Heuristic: {list(s['selected'])}")
                model_sel = [j for j in range(n) if probs[0,j].item() > 0.5]
                lines.append(f"Model:     {model_sel}")
                if set(s["selected"]) == set(model_sel):
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            elif candidates and dt == "PRIORITY_ACTION":
                n = len(candidates)
                af_t = torch.zeros(
                    1, n, 64, device=self.device)
                am_t = torch.zeros(
                    1, n, dtype=torch.bool,
                    device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), 64)
                    af_t[0, j, :cl] = torch.tensor(
                        cf[:cl], dtype=torch.float32)
                    am_t[0, j] = True

                logits = self.model.priority_head(
                    state, af_t, am_t)
                probs = torch.softmax(logits, dim=-1)

                lines.append("=== Priority Decision ===")
                lines.append("")
                sel_idx = s["selected"][0] \
                    if s["selected"] else -1
                for j in range(n):
                    p = probs[0, j].item()
                    info = decode_action(candidates[j])
                    sel = (j == sel_idx)
                    marker = ">>" if sel else "  "
                    label = info["label"] if info else "?"
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(
                        f"   [{bar:20s}] {p:.0%}")
                    lines.append("")

                model_pick = probs[0].argmax().item()
                model_info = decode_action(
                    candidates[model_pick])
                model_label = model_info["label"] \
                    if model_info else "?"
                heur_info = decode_action(
                    candidates[sel_idx]) if sel_idx >= 0 \
                    else None
                heur_label = heur_info["label"] \
                    if heur_info else "PASS"
                lines.append(
                    f"Heuristic: [{sel_idx}] {heur_label}")
                lines.append(
                    f"Model:     [{model_pick}] "
                    f"{model_label}")
                if model_pick == sel_idx:
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            elif candidates and dt == "DECLARE_BLOCKERS":
                n = len(candidates)
                cf_t = torch.zeros(1, n, 128, device=self.device)
                cm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), 128)
                    cf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                # Block head uses card_select_head (softmax)
                logits = self.model.card_select_head(state, cf_t, cm_t)
                probs = torch.softmax(logits, dim=-1)

                lines.append("=== Block Decision ===")
                lines.append("")
                for j in range(n):
                    p = probs[0, j].item()
                    info = decode_card(candidates[j])
                    sel = j in s["selected"]
                    marker = ">>" if sel else "  "
                    label = info["label"] if info else "?"
                    if info and info["power"] is not None:
                        label += f" {info['power']}/{info['toughness']}"
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(f"   [{bar:20s}] {p:.0%}")
                    if sel:
                        lines.append(f"   >> BLOCKING")
                    lines.append("")

                lines.append(f"Heuristic: {list(s['selected'])}")
                # Model picks highest prob candidates
                model_sel = [j for j in range(n)
                             if probs[0,j].item() > 1.0/n]
                lines.append(f"Model:     {model_sel}")
                if set(s["selected"]) == set(model_sel):
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            self.pred_text.insert("1.0", "\n".join(lines))
        self.pred_text.config(state=tk.DISABLED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
        default="../../rl_data/trajectories")
    parser.add_argument("--model",
        default="../../rl_data/checkpoints/"
                "model_with_decisions.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int,
        default=200)
    args = parser.parse_args()

    # Load data from multiple sources
    print("Loading samples...", flush=True)
    samples = load_samples(args.data_dir,
                           args.max_samples)
    # Also load PPO trajectories if available
    ppo_dir = os.path.join(
        os.path.dirname(args.data_dir),
        'ppo_trajectories')
    if os.path.isdir(ppo_dir):
        ppo_samples = load_samples(ppo_dir, 100)
        samples.extend(ppo_samples)
        print(f"  + {len(ppo_samples)} PPO samples",
              flush=True)

    attacks = sum(1 for s in samples
                  if s["type"] == "DECLARE_ATTACKERS")
    blocks = sum(1 for s in samples
                 if s["type"] == "DECLARE_BLOCKERS")
    priority = sum(1 for s in samples
                   if s["type"] == "PRIORITY_ACTION")
    print(f"  {len(samples)} total samples "
          f"({attacks} attacks, {blocks} blocks, "
          f"{priority} priority)", flush=True)

    model_path = args.model
    model = None
    # Try best_ppo_model first, then specified path
    ppo_model = os.path.join(
        os.path.dirname(model_path),
        'best_ppo_model.pt')
    for p in [ppo_model, model_path]:
        if p and os.path.exists(p):
            print(f"Loading model: {p}", flush=True)
            model = MTGModel.load(p, device=args.device)
            model.eval()
            model_path = p
            break

    root = tk.Tk()
    GameStateViewer(root, samples, model, args.device,
                    model_path=model_path,
                    data_dir=args.data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
