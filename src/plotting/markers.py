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


# Surfaces excluded from PR eligibility. Downhill races (net-downhill,
# course-aided) project to absurdly fast 5K-equivalents that would
# dominate the running-min sequence and invalidate every subsequent
# on-the-flat PR. They're plotted normally; they just don't earn a ring.
PR_EXCLUDED_SURFACES = {'Downhill'}


def is_pr_eligible(surface) -> bool:
    return str(surface) not in PR_EXCLUDED_SURFACES
