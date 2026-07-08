"""Shared x-axis helpers for date-based plots.

Every plot with a yearly date axis (most of `src/plots/`) should pull its
tick config from here so styling stays unified. The collision-clearing
trick — invisible outside ticks of length 8 — pushes the year labels just
far enough below the axis line that they no longer crowd the lowest
y-axis tick label in the lower-left corner.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .tokens import FG, GRID
from .formatters import sec_to_mss


# ---------- time / pace y-axis ticks ----------

# Nice tick intervals (seconds) for pace/duration axes, ascending. 15 keeps
# the long-runs density; 10 keeps workouts / training-quality; 30 keeps
# races / CS / recovery. A formula picks from this ladder per data range so
# the gridline density a plot wants is preserved across any profile's range,
# instead of hardcoding a fixed interval (and, worse, a fixed range).
TIME_LADDER = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]


def nice_time_interval(lo, hi, *, target, ladder=TIME_LADDER):
    """Smallest ladder interval (seconds) giving at most ``target`` gridlines
    across [lo, hi]. Falls back to the largest ladder entry for huge spans."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return ladder[0]
    span = hi - lo
    for iv in ladder:
        if span / iv <= target:
            return iv
    return ladder[-1]


def time_ticks_at_interval(lo, hi, interval):
    """Tickvals (seconds) snapped outward to multiples of ``interval``, with
    M:SS / H:MM:SS labels via ``sec_to_mss``."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [int(round(lo))], [sec_to_mss(lo)]
    tick_min = math.floor(lo / interval) * interval
    tick_max = math.ceil(hi / interval) * interval
    n = int(round((tick_max - tick_min) / interval)) + 1
    ticks = [tick_min + i * interval for i in range(n)]
    return ticks, [sec_to_mss(t) for t in ticks]


def nice_time_ticks(lo, hi, *, target, ladder=TIME_LADDER):
    """Pace/time axis ticks: pick a nice interval for ~``target`` gridlines,
    snap the range outward to it, return (tickvals_sec, M:SS labels). The
    snapped range is ``[ticks[0], ticks[-1]]`` — use it as the axis bounds so
    gridlines land on round values and nothing is clipped."""
    iv = nice_time_interval(lo, hi, target=target, ladder=ladder)
    return time_ticks_at_interval(lo, hi, iv)


def thin_yearly_ticks(x_lo, x_hi, *, max_labels=11):
    """Yearly gridlines (Jan 1 each year), with empty-string labels on
    intervening years when there'd be more than ``max_labels`` labels.
    Empty-string labels still draw the tick + gridline, just without
    text — keeping visual rhythm consistent across panels with different
    x-spans. Labels anchor to the most recent year, then walk backward.
    """
    y_lo = int(pd.Timestamp(x_lo).year)
    y_hi = int(pd.Timestamp(x_hi).year)
    years = list(range(y_lo, y_hi + 1))
    if not years:
        return [], []
    tickvals = [pd.Timestamp(f'{y}-01-01') for y in years]
    n = len(years)
    if n <= max_labels:
        ticktext = [str(y) for y in years]
    else:
        step = -(-n // max_labels)  # ceil division
        ticktext = [str(y) if ((n - 1 - i) % step == 0) else ''
                    for i, y in enumerate(years)]
    return tickvals, ticktext


def auto_date_x_axis_kwargs(x_lo, x_hi, *, nticks=12, **extra) -> dict[str, Any]:
    """``yearly_x_axis_kwargs`` styling with Plotly's span-derived auto date
    ticks instead of baked Jan-1 array ticks — for zoomable date axes
    (Misc Trends), where tick density must re-derive from the CURRENT range
    on every relayout (years at a decade, months at a year, weeks at a
    quarter) and a sub-year profile needs month ticks at first render
    rather than a lone Jan-1 tick. ``nticks`` bounds the density: >= 12
    keeps a full decade at one gridline per year (Plotly's unhinted default
    thins a 10-year span to 2-year ticks)."""
    return yearly_x_axis_kwargs(x_lo, x_hi, tickmode='auto', tickvals=None,
                                ticktext=None, nticks=nticks, **extra)


def yearly_x_axis_kwargs(x_lo, x_hi, *, max_labels=11, **extra) -> dict[str, Any]:
    """Standard yearly x-axis config — pass to ``update_xaxes(...)`` or
    spread into ``xaxis=dict(...)``. ``extra`` overrides any returned key.

    The ``ticks='outside' + ticklen=8 + tickcolor=transparent`` combo
    shifts year labels ~8px below the axis line without drawing a
    visible tick mark — clears the lower-left collision with the bottom
    y-tick label that single-panel plots otherwise hit.
    """
    tickvals, ticktext = thin_yearly_ticks(x_lo, x_hi, max_labels=max_labels)
    base = dict(
        range=[x_lo, x_hi],
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(color=FG, size=12),
        showgrid=True,
        gridcolor=GRID,
        zerolinecolor=GRID,
        ticks='outside',
        ticklen=8,
        tickcolor='rgba(0,0,0,0)',
    )
    base.update(extra)
    return base
