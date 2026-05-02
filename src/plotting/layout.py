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


# Right-margin padding around an anchored overlay box: 12px from viewport
# right + 16px buffer between box and data area. MUST stay in sync with
# BOX_RIGHT_OFFSET + MARGIN_BUFFER in _scaffold/overlay_anchor.js.
ANCHORED_BOX_RIGHT_PADDING = 28


def right_margin_for_anchored_box(box_width_px: int, *, legend_min_px: int = 0) -> int:
    """Right margin (px) needed to fit an anchored `data-rp-anchor` box of
    ``box_width_px`` outside the data area. Pass ``legend_min_px`` as the
    margin you'd otherwise reserve for the legend; the larger of the two
    wins. Pair the returned value with the same ``box_width_px`` in the
    box's CSS ``width:`` so the margin and box always agree.
    """
    return max(legend_min_px, box_width_px + ANCHORED_BOX_RIGHT_PADDING)


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
