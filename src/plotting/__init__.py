"""Shared utilities for plot scripts in src/plots/."""
from .render import render_plot, CursorTooltip
from .layout import apply_default_layout, right_margin_for_anchored_box
from . import tokens
from .tokens import (
    BG, FG, FG_SOFT, FG_DIM, FG_MUTE, GRID, GRID_FAINT, PANEL_BG, TOOLTIP_BG, BORDER,
    CS_LINE, CS_LINE_WIDTH, TREND_LINE, TREND_WIDTH,
    SURFACES, CAT_COLORS, TAG_COLORS, rgba,
)
from .formatters import (sec_to_mss, sec_to_mss_full, signed_sec, fmt_min,
                         route_label, route_paren)
from .axes import (thin_yearly_ticks, yearly_x_axis_kwargs,
                   nice_time_interval, time_ticks_at_interval, nice_time_ticks,
                   TIME_LADDER)
from .markers import (
    pr_marker, is_pr_eligible, marker_half_px,
    PR_EXCLUDED_SURFACES, PR_LEGEND_NAME, PR_LEGEND_RANK,
    PR_LINE_WIDTH, PR_RING_PADDING, PR_LINE_COLOR,
)
from .smoothing import GAP_BREAK_DAYS, adaptive_gauss_smoother, gaussian_rolling_trend

__all__ = [
    'render_plot', 'CursorTooltip', 'apply_default_layout',
    'right_margin_for_anchored_box',
    'tokens', 'rgba',
    'BG', 'FG', 'FG_SOFT', 'FG_DIM', 'FG_MUTE', 'GRID', 'GRID_FAINT', 'PANEL_BG',
    'TOOLTIP_BG', 'BORDER',
    'CS_LINE', 'CS_LINE_WIDTH', 'TREND_LINE', 'TREND_WIDTH',
    'SURFACES', 'CAT_COLORS', 'TAG_COLORS',
    'sec_to_mss', 'sec_to_mss_full', 'signed_sec', 'fmt_min',
    'route_label', 'route_paren',
    'thin_yearly_ticks', 'yearly_x_axis_kwargs',
    'nice_time_interval', 'time_ticks_at_interval', 'nice_time_ticks', 'TIME_LADDER',
    'pr_marker', 'is_pr_eligible', 'marker_half_px',
    'PR_EXCLUDED_SURFACES', 'PR_LEGEND_NAME', 'PR_LEGEND_RANK',
    'PR_LINE_WIDTH', 'PR_RING_PADDING', 'PR_LINE_COLOR',
    'GAP_BREAK_DAYS', 'adaptive_gauss_smoother', 'gaussian_rolling_trend',
]
