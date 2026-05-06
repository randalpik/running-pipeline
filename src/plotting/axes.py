"""Shared x-axis helpers for date-based plots.

Every plot with a yearly date axis (most of `src/plots/`) should pull its
tick config from here so styling stays unified. The collision-clearing
trick — invisible outside ticks of length 8 — pushes the year labels just
far enough below the axis line that they no longer crowd the lowest
y-axis tick label in the lower-left corner.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .tokens import FG, GRID


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
