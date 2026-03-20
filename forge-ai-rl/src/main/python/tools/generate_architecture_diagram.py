#!/usr/bin/env python3
"""Generate a detailed architecture diagram of the MTG RL model."""

from PIL import Image, ImageDraw, ImageFont
import os

# Canvas
W, H = 2400, 1800
img = Image.new('RGB', (W, H), '#0d1117')
draw = ImageDraw.Draw(img)

# Fonts
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_head = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except (IOError, OSError):
    font_title = font_head = font_sub = font_sm = font_xs = ImageFont.load_default()

# Colors
C = {
    'encoder': '#1f6feb',
    'encoder_light': '#388bfd',
    'value': '#a371f7',
    'priority': '#f0883e',
    'attack': '#3fb950',
    'block': '#58a6ff',
    'target': '#d2a8ff',
    'cardsel': '#79c0ff',
    'mulligan': '#f778ba',
    'binary': '#ffa657',
    'input': '#8b949e',
    'output': '#e6edf3',
    'layer': '#161b22',
    'layer_border': '#30363d',
    'text': '#e6edf3',
    'text_dim': '#8b949e',
    'arrow': '#484f58',
    'highlight': '#f0883e',
}


def draw_box(x, y, w, h, fill, border=None, radius=8):
    """Draw a rounded rectangle."""
    border = border or fill
    draw.rounded_rectangle([x, y, x+w, y+h], radius=radius,
                           fill=fill, outline=border, width=2)


def draw_layer(x, y, w, text, subtext=None, color='#161b22',
               border='#30363d', text_color='#e6edf3'):
    """Draw a layer box with text."""
    h = 28 if not subtext else 40
    draw_box(x, y, w, h, color, border)
    draw.text((x + w//2, y + 6), text, fill=text_color,
              font=font_sm, anchor='mt')
    if subtext:
        draw.text((x + w//2, y + 22), subtext, fill=C['text_dim'],
                  font=font_xs, anchor='mt')
    return y + h + 4


def draw_arrow(x1, y1, x2, y2, color=None):
    """Draw an arrow line."""
    color = color or C['arrow']
    draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    # arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 8
    draw.polygon([
        (x2, y2),
        (x2 - size * math.cos(angle - 0.4), y2 - size * math.sin(angle - 0.4)),
        (x2 - size * math.cos(angle + 0.4), y2 - size * math.sin(angle + 0.4)),
    ], fill=color)


def draw_section(x, y, w, h, title, color, params=None):
    """Draw a section with title bar."""
    draw_box(x, y, w, h, '#0d1117', color, radius=10)
    draw_box(x, y, w, 32, color, color, radius=10)
    # Fix bottom corners of title bar
    draw.rectangle([x+1, y+24, x+w-1, y+32], fill=color)
    label = title
    if params:
        label += f"  ({params})"
    draw.text((x + w//2, y + 8), label, fill='#ffffff',
              font=font_head, anchor='mt')
    return y + 40


# === Title ===
draw.text((W//2, 25), "MTG RL Model Architecture — 11M Parameters",
          fill=C['text'], font=font_title, anchor='mt')
draw.text((W//2, 55), "Hierarchical transformer encoder with specialized decision heads",
          fill=C['text_dim'], font=font_sub, anchor='mt')

# === INPUT SECTION (left) ===
ix, iy = 30, 100
draw_section(ix, iy, 280, 520, "Game State Input", C['input'])
ty = iy + 48

inputs = [
    ("Global Features", "64-dim", "Life, turn, phase, hand/lib sizes,\ncreature counts, mana, stack"),
    ("My Board", "30 × 128-dim", "Creatures, lands, enchantments\nwith type/color/P&T/keywords/state"),
    ("Opponent Board", "30 × 128-dim", "Same encoding as my board"),
    ("My Hand", "15 × 128-dim", "Cards in hand"),
    ("My Graveyard", "40 × 128-dim", "Graveyard cards"),
    ("Opp Graveyard", "40 × 128-dim", "Opponent's graveyard"),
    ("Stack", "10 × 128-dim", "Spells/abilities resolving"),
]

for name, dim, desc in inputs:
    draw_box(ix+10, ty, 260, 58, '#161b22', C['input'], radius=6)
    draw.text((ix+20, ty+4), name, fill=C['text'], font=font_sm)
    draw.text((ix+260, ty+4), dim, fill=C['highlight'], font=font_sm, anchor='rt')
    draw.text((ix+20, ty+22), desc, fill=C['text_dim'], font=font_xs)
    ty += 64

# Total
draw.text((ix+140, ty+5), "Total: 21,184 floats", fill=C['highlight'],
          font=font_sub, anchor='mt')

# === ENCODER (center-left) ===
ex, ey = 350, 100
enc_h = 520
inner_y = draw_section(ex, ey, 340, enc_h, "Game State Encoder", C['encoder'], "3.4M params")

# Per-zone encoders
draw_box(ex+10, inner_y, 320, 120, '#161b22', C['encoder_light'], radius=6)
draw.text((ex+170, inner_y+6), "6 × CardSetEncoder", fill=C['encoder_light'],
          font=font_sub, anchor='mt')
ly = inner_y + 28
ly = draw_layer(ex+20, ly, 300, "Linear(128 → 128)", color='#1a2332', border=C['encoder'])
ly = draw_layer(ex+20, ly, 300, "TransformerEncoder × 2", "4 heads, GELU, d_ff=512",
                color='#1a2332', border=C['encoder'])
ly = draw_layer(ex+20, ly, 300, "Masked Mean Pool → (batch, 128)", color='#1a2332', border=C['encoder'])

inner_y += 130

# Global encoder
draw_box(ex+10, inner_y, 320, 52, '#161b22', C['encoder_light'], radius=6)
draw.text((ex+170, inner_y+6), "Global Encoder", fill=C['encoder_light'],
          font=font_sub, anchor='mt')
draw_layer(ex+20, inner_y+26, 300, "Linear(64 → 128) → GELU → LayerNorm",
           color='#1a2332', border=C['encoder'])
inner_y += 62

# Cross-zone attention
draw_box(ex+10, inner_y, 320, 72, '#161b22', C['encoder_light'], radius=6)
draw.text((ex+170, inner_y+6), "Cross-Zone Attention", fill=C['encoder_light'],
          font=font_sub, anchor='mt')
ly = inner_y + 28
ly = draw_layer(ex+20, ly, 300, "Stack 7 zone embeddings → (batch, 7, 128)",
                color='#1a2332', border=C['encoder'])
ly = draw_layer(ex+20, ly, 300, "TransformerEncoder × 1", "4 heads, GELU",
                color='#1a2332', border=C['encoder'])
inner_y += 82

# Output projection
draw_box(ex+10, inner_y, 320, 52, '#161b22', C['encoder_light'], radius=6)
draw.text((ex+170, inner_y+6), "Output Projection", fill=C['encoder_light'],
          font=font_sub, anchor='mt')
draw_layer(ex+20, inner_y+26, 300, "Linear(896 → 512) → GELU → LN → Linear(512 → 512)",
           color='#1a2332', border=C['encoder'])
inner_y += 62

# Output label
draw.text((ex+170, inner_y+5), "State Embedding: (batch, 512)", fill=C['encoder_light'],
          font=font_sub, anchor='mt')

# Arrow from input to encoder
draw_arrow(ix+280, iy+260, ex, ey+260, C['encoder_light'])

# === STATE EMBEDDING HUB ===
hub_x, hub_y = 730, 360
draw_box(hub_x, hub_y, 200, 40, C['encoder'], C['encoder_light'], radius=20)
draw.text((hub_x+100, hub_y+12), "512-dim State Embedding",
          fill='#ffffff', font=font_sm, anchor='mt')

# Arrow from encoder to hub
draw_arrow(ex+340, ey+300, hub_x, hub_y+20, C['encoder_light'])

# === DECISION HEADS (right side) ===
heads = [
    {
        'name': 'Value Network', 'color': C['value'], 'params': '198K',
        'input': 'State (512)',
        'layers': [
            'Linear(512→256) → GELU → LN',
            'Linear(256→256) → GELU → LN',
            'Linear(256→1) → Tanh',
        ],
        'output': 'Win probability [-1, +1]',
    },
    {
        'name': 'Priority Head', 'color': C['priority'], 'params': '543K',
        'input': 'State (512) + Actions (N×64)',
        'layers': [
            'Action proj: Linear(64→256)',
            'Cross-Attn: actions→state (4 heads)',
            'Score: Linear(512→256→1)',
        ],
        'output': 'Softmax over N actions + pass',
    },
    {
        'name': 'Attack Head', 'color': C['attack'], 'params': '1.9M',
        'input': 'State (512) + Creatures (N×128)',
        'layers': [
            'Card proj: Linear(128→256)',
            'Self-Attn TransformerEnc × 2 (4h)',
            'Classifier: Linear(512→256→1)',
        ],
        'output': 'Per-creature sigmoid [0,1]',
    },
    {
        'name': 'Block Head', 'color': C['block'], 'params': '658K',
        'input': 'State + Blockers + Attackers (128)',
        'layers': [
            'Blocker/Attacker proj: Linear(128→256)',
            'Cross-Attn: blockers→attackers (4h)',
            'Pairwise score: Linear(768→256→1)',
        ],
        'output': 'Per-blocker: attacker assignment',
    },
    {
        'name': 'Card Select Head', 'color': C['cardsel'], 'params': '1.5M',
        'input': 'State (512) + Cards (N×128)',
        'layers': [
            'Card proj: Linear(128→256)',
            'Self-Attn TransformerEnc × 1 (4h)',
            'Score + GRU for multi-select',
        ],
        'output': 'Card selection probabilities',
    },
    {
        'name': 'Target Head', 'color': C['target'], 'params': '674K',
        'input': 'State (512) + Targets (N×64)',
        'layers': [
            'Target proj: Linear(64→256)',
            'Pointer attention (scaled dot)',
            'GRU context for multi-target',
        ],
        'output': 'Target selection probabilities',
    },
    {
        'name': 'Mulligan Head', 'color': C['mulligan'], 'params': '2.0M',
        'input': 'State (512) + Hand (N×128)',
        'layers': [
            'Card proj: Linear(128→256)',
            'Self-Attn TransformerEnc × 2 (4h)',
            'Keep/mull + bottom card scores',
        ],
        'output': 'Keep logit + per-card bottom score',
    },
    {
        'name': 'Binary Head', 'color': C['binary'], 'params': '198K',
        'input': 'State (512)',
        'layers': [
            'Linear(512→256) → GELU → LN',
            'Linear(256→256) → GELU',
            'Linear(256→1)',
        ],
        'output': 'Yes/No logit',
    },
]

hx_start = 980
hy_start = 80
col_w = 340
col_gap = 10
row_h = 210

for i, head in enumerate(heads):
    col = i // 4
    row = i % 4
    hx = hx_start + col * (col_w + col_gap)
    hy = hy_start + row * row_h

    inner_y = draw_section(hx, hy, col_w, row_h - 10, head['name'],
                           head['color'], head['params'])

    # Input
    draw.text((hx+10, inner_y), f"In: {head['input']}",
              fill=C['text_dim'], font=font_xs)
    inner_y += 16

    # Layers
    for layer in head['layers']:
        inner_y = draw_layer(hx+8, inner_y, col_w-16, layer,
                             color='#161b22', border=head['color'])

    # Output
    draw.text((hx+10, inner_y+2), f"Out: {head['output']}",
              fill=head['color'], font=font_xs)

    # Arrow from hub to head
    arrow_target_x = hx
    arrow_target_y = hy + 16
    if col == 0:
        draw_arrow(hub_x+200, hub_y+20, arrow_target_x, arrow_target_y,
                   head['color'])
    else:
        draw_arrow(hub_x+200, hub_y+20, arrow_target_x, arrow_target_y,
                   head['color'])

# === Legend ===
lx, ly = 30, H - 140
draw_box(lx, ly, 600, 120, '#161b22', C['layer_border'], radius=10)
draw.text((lx+300, ly+8), "Architecture Summary", fill=C['text'],
          font=font_head, anchor='mt')

legend_items = [
    ("Shared Encoder", C['encoder'], "3.4M params — Transformer with per-zone set attention"),
    ("Value Network (Critic)", C['value'], "198K params — Evaluates game state win probability"),
    ("Decision Heads (Actor)", C['priority'], "7.5M params — Specialized policies per decision type"),
]
for i, (name, color, desc) in enumerate(legend_items):
    ly_item = ly + 34 + i * 28
    draw.rectangle([lx+15, ly_item, lx+30, ly_item+15], fill=color)
    draw.text((lx+40, ly_item), name, fill=C['text'], font=font_sm)
    draw.text((lx+220, ly_item), desc, fill=C['text_dim'], font=font_sm)

# Training info
draw.text((lx+620, ly+10), "Training Pipeline", fill=C['text'],
          font=font_head)
info = [
    "1. Value Network: MSE loss on game outcomes (±1)",
    "2. Priority Head: CrossEntropy (softmax single-select)",
    "3. Attack Head: BCE (binary per-creature)",
    "4. Block Head: CE per-blocker (attacker assignment)",
    "5. PPO: REINFORCE + value baseline + entropy bonus",
]
for i, line in enumerate(info):
    draw.text((lx+620, ly+36 + i*18), line, fill=C['text_dim'], font=font_sm)

# Save
out_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'rl_data', 'mtg_model_architecture.png')
out_path = os.path.abspath(out_path)
img.save(out_path, 'PNG')
print(f"Saved to: {out_path}")
