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
FRONTIER_LINE = '#9933ff'           # vibrant purple (tab-shell accent #93f) — performance frontier
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
#     do not unify with SURFACES). Palette mirrors the conditional-formatting
#     (CF) scheme of the source running log so plot and spreadsheet read
#     consistently. Long runs and hills each share one color since the plot
#     uses a single combined trace for each. ---
CAT_COLORS = {
    # Tuned for visibility on the dark plot background (#1a1a1a) while still
    # tracking the log's CF hue families (yellow→tempo, gold→interval,
    # peach→fartlek, blue→hills, purple→long). Original CF pastels were too
    # close in hue and too low in saturation; these are spread out and
    # punched up.
    'interval':           '#F1C40F',  # saturated gold (CF pale yellow → vivid)
    'tempo':              "#83CC1D",  # lime / yellow-green (separated from interval)
    'rep':                "#F38A12",  # carrot orange (CF peach → vivid)
    'continuous_fartlek': "#FF5643",  # warm orange (CF peach → vivid, distinct from rep)
    'long':               '#674EA7',  # deep purple (CF purple family)
    'hill_cont':          '#3D85C6',  # saturated medium blue (CF light blue → vivid)
    'hill_rep':           '#1ABC9C',  # vivid teal (CF muted teal #A2C4C9 → vivid)
}

# --- Qualitative-trends weather/conditions tag colors. The raw hexes are the
#     conditional-formatting cell tints from the source running log, kept here
#     as the single re-skin point. The Weather panel remaps the light
#     cloud-family tints to dark-legible
#     tones at plot time (see plot_qualitative_trends.wcolor); the Conditions
#     panel uses these directly for marker rings. ---
WEATHER_CF_COLORS = {
    'clear':      '#CFE2F3',
    'cloudy':     '#CCCCCC',
    'overcast':   '#999999',
    'foggy':      '#EFEFEF',
    'light rain': '#6D9EEB',
    'heavy rain': '#1155CC',
    'flurries':   '#B4A7D6',
    'inside':     '#B6D7A8',
    'smoky':      '#E1D7C5',
}
CONDITIONS_CF_COLORS = {
    'wet':    '#6D9EEB',
    'muddy':  '#A36F06',
    'snow':   '#D9D2E9',
    'icy':    '#CFE2F3',
    'inside': '#B6D7A8',
}


# --- Workout condition-tag rings (Workouts tab). One color per tag, used
#     both for the marker ring and the matching tooltip tag text so the two
#     always read as the same signal. 'xc' deliberately reuses the XC surface
#     green — it marks XC-corrected sessions that stay in Training. ---
TAG_COLORS = {
    'uncertain accuracy': '#FF4500',         # bright red-orange (kept clearly
                                             # off the carrot-orange rep fill)
    'snow':               '#8FD3FF',         # light blue
    'outlier':            '#E8A33C',         # amber — matches the exclusion
                                             # tooltip text; distinct from the
                                             # red-orange accuracy ring
    'xc':                 SURFACES['XC'],    # XC green
    'enriched':           '#AAAAAA',         # light gray — informational:
                                             # session successfully watch-
                                             # enriched (white overpowered)
    'partners':           '#F4D03F',         # bright yellow — partner-paced
                                             # run excluded from Training
                                             # (distinct from the amber
                                             # outlier ring's orange cast)
}


def rgba(hex_color: str, alpha: float) -> str:
    """'#4aa3ff', 0.7 -> 'rgba(74,163,255,0.7)'."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'
