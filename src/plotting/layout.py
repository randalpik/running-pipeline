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

import re

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


_AXIS_KEY_RE = re.compile(r'[xy]axis\d*$')


def reshape_patch(mobile_fig, *, height_px, extra=None, title_font_px=None):
    """Dotted-path layout patch that reshapes a figure's subplot grid.

    ``mobile_fig`` is a THROWAWAY ``make_subplots`` figure with the mobile
    rows/cols and the SAME ``subplot_titles`` in the same fill order as the
    desktop figure. ``make_subplots`` numbers axes in row-major cell order,
    so the panel at cell *i* owns ``xaxis{i+1}``/``yaxis{i+1}`` in ANY grid
    shape — traces, axis ids and everything keyed on them are invariant
    under the reshape. Only the axis ``domain``s, the paper-anchored
    subplot-title annotations, margins and the explicit pixel ``height``
    differ, and those are exactly what this returns (as
    ``MobileLayout.patch`` input for ``_scaffold/mobile.js``).

    ``extra`` merges additional dotted paths (margins, legend placement…);
    ``title_font_px`` optionally shrinks the subplot-title font.
    """
    lay = mobile_fig.layout.to_plotly_json()
    patch = {}
    for key, val in lay.items():
        if _AXIS_KEY_RE.fullmatch(key) and val.get('domain') is not None:
            patch[f'{key}.domain'] = list(val['domain'])
    for i, ann in enumerate(lay.get('annotations', ())):
        patch[f'annotations[{i}].x'] = ann.get('x')
        patch[f'annotations[{i}].y'] = ann.get('y')
        if title_font_px is not None:
            patch[f'annotations[{i}].font.size'] = title_font_px
    patch['height'] = height_px
    if extra:
        patch.update(extra)
    return patch


def assert_reshape_compatible(desktop_fig, mobile_fig):
    """Fail the build loudly when a reshape patch would mis-target.

    The patch addresses subplot-title annotations by index and axes by key,
    so the mobile scratch figure must have (a) the same axis-key set and
    (b) annotations that are an index-aligned prefix of the desktop's (a
    plot may append extra annotations after the ``make_subplots`` titles;
    those are untouched by the patch).
    """
    d = desktop_fig.layout.to_plotly_json()
    m = mobile_fig.layout.to_plotly_json()
    d_axes = {k for k in d if _AXIS_KEY_RE.fullmatch(k)}
    m_axes = {k for k in m if _AXIS_KEY_RE.fullmatch(k)}
    if d_axes != m_axes:
        raise AssertionError(
            f'mobile reshape axis-key mismatch: desktop {sorted(d_axes)} '
            f'vs mobile {sorted(m_axes)}')
    d_ann = d.get('annotations', ()) or ()
    m_ann = m.get('annotations', ()) or ()
    if len(m_ann) > len(d_ann):
        raise AssertionError(
            f'mobile reshape has {len(m_ann)} annotations, desktop only '
            f'{len(d_ann)}')
    for i, (da, ma) in enumerate(zip(d_ann, m_ann)):
        if da.get('text') != ma.get('text'):
            raise AssertionError(
                f'mobile reshape annotation[{i}] text mismatch: '
                f'{da.get("text")!r} vs {ma.get("text")!r}')


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
