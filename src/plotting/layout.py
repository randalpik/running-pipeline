"""Default Plotly layout helpers shared by every plot in src/plots/.

Each plot script still calls ``fig.update_layout(...)`` for its own
axes, legend, margins — this just stamps the dark-theme defaults so the
same paper_bgcolor/plot_bgcolor/font boilerplate isn't copy-pasted six
times.

**Titles live in HTML, not Plotly.** ``layout.title`` is intentionally
left unset; the per-plot title is rendered by ``render_plot(...,
title=..., subtitle=...)`` as a plain-HTML overlay bar above the
plot. See base.css ``.rp-title-bar`` and the ``--rp-title-h`` custom
property. Each plot's ``margin.t`` only needs to reserve room for
subplot-titles or top-axis labels — typically 20px for single-panel,
30–50px for plots with ``subplot_titles``.
"""
from __future__ import annotations

from .tokens import BG, FG


def apply_default_layout(fig, **overrides):
    """Stamp dark-theme defaults onto ``fig`` then apply caller overrides.

    ``overrides`` is forwarded to ``fig.update_layout(...)`` after defaults
    are set, so callers can override anything they want using the same
    kwarg shape as ``update_layout``. Pass each plot's axes/legend/
    margin overrides exactly as you would to ``update_layout``.
    """
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG),
        autosize=True,
    )
    if overrides:
        fig.update_layout(**overrides)
    return fig
