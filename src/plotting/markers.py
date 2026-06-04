"""PR (personal-record) markers and eligibility rules — shared by race
plots and any future plot that wants to show PR markers.

Plotly draws marker.line centered on the size boundary, so setting
size = base_size + RING_PADDING with line_width = LINE_WIDTH puts the
ring's inner edge a hair inside the underlying marker's outer edge —
eliminating any visible halo gap.
"""
from __future__ import annotations


PR_LINE_WIDTH   = 1.5      # px — thin outline, not a halo
PR_RING_PADDING = 1        # ring size = base_size + this
PR_LINE_COLOR   = 'white'
PR_LEGEND_NAME  = 'PR effort'
PR_LEGEND_RANK  = 2100     # below CS-derived (2000) and Estimated (2001)


def pr_marker(base_size):
    """Plotly marker dict for the PR overlay (transparent fill, white ring)."""
    return dict(
        symbol='diamond',
        size=base_size + PR_RING_PADDING,
        color='rgba(0,0,0,0)',
        line=dict(width=PR_LINE_WIDTH, color=PR_LINE_COLOR),
    )


# Plotly renders the 'diamond' symbol at radius (size/2)*1.3 — it up-scales
# diamonds ~1.3× vs circles for visual parity — so a diamond's half-width is
# 0.65*size, NOT size/2. 'circle' (and most others) use radius size/2.
PLOTLY_DIAMOND_SCALE = 1.3


def marker_half_px(base_size, *, symbol='diamond', ringed=False, line_width=0.0):
    """Half the RENDERED width (px) of a marker — for axis edge-padding so the
    leftmost/rightmost markers aren't clipped.

    Accounts for the Plotly diamond up-scale (see ``PLOTLY_DIAMOND_SCALE``): a
    diamond's half-width is ``0.65*size``, a circle's is ``size/2``. The
    marker's own outline (``line_width``) straddles the path, so half of it sits
    outside the symbol on each side.

    ``ringed=True`` sizes for the PR overlay ring instead — diameter
    ``base_size + PR_RING_PADDING`` with a ``PR_LINE_WIDTH`` outline. The
    leftmost race marker is always a PR, so on race plots its ring is what must
    clear the edge.
    """
    if ringed:
        size, line = base_size + PR_RING_PADDING, PR_LINE_WIDTH
    else:
        size, line = base_size, line_width
    scale = PLOTLY_DIAMOND_SCALE if symbol == 'diamond' else 1.0
    return size / 2.0 * scale + line / 2.0


# Surfaces excluded from PR eligibility. Downhill races (net-downhill,
# course-aided) project to absurdly fast 5K-equivalents that would
# dominate the running-min sequence and invalidate every subsequent
# on-the-flat PR. They're plotted normally; they just don't earn a ring.
PR_EXCLUDED_SURFACES = {'Downhill'}


def is_pr_eligible(surface) -> bool:
    return str(surface) not in PR_EXCLUDED_SURFACES
