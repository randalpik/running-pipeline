"""Visual tokens — colors, fonts, sizes — for every plot script.

Single source of truth. To re-skin the pipeline, edit this file.
"""
from __future__ import annotations


# --- Dark-theme chrome ---
BG         = '#1a1a1a'
FG         = '#eee'
FG_SOFT    = '#bbb'    # subtitle / secondary descriptive text
FG_DIM     = '#aaa'    # tick labels, less-prominent labels
FG_MUTE    = '#999'    # muted in-tooltip captions
GRID       = '#333'
GRID_FAINT = 'rgba(255,255,255,0.08)'
PANEL_BG   = 'rgba(26,26,26,0.92)'
TOOLTIP_BG = 'rgba(26,26,26,0.96)'
BORDER     = '#555'

# --- Reference / overlay lines ---
CS_LINE       = 'rgb(255,180,80)'   # orange — Critical Speed reference
CS_LINE_WIDTH = 2.5
TREND_LINE    = 'rgb(220,220,220)'  # white-ish — recovery rolling trend
TREND_WIDTH   = 2.0

# --- Race surfaces (canonical hex). RGBA derivations use rgba(). ---
SURFACES = {
    'Track':    '#ff5b4d',
    'Road':     '#4aa3ff',
    'XC':       '#4ade80',
    'Downhill': '#b87de9',
    'Unknown':  '#888888',
}

# --- Training-quality categories (separate semantic axis from surfaces;
#     do not unify with SURFACES) ---
CAT_COLORS = {
    'interval':           '#d62728',
    'tempo':              '#2ca02c',
    'rep':                '#9467bd',
    'continuous_fartlek': '#ff7f0e',
    'lr_20-22.9':         '#17becf',
    'lr_23+':             '#1f77b4',
    'hill_lc':            '#e377c2',
    'hill_rc':            '#bcbd22',
    'hill_pwr1':          '#8c564b',
}


def rgba(hex_color: str, alpha: float) -> str:
    """'#4aa3ff', 0.7 -> 'rgba(74,163,255,0.7)'."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'
