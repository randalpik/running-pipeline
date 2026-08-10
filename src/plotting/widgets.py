"""Tiny HTML primitives for plot overlay UIs.

Each function returns plain HTML built from the shared ``.rp-*`` classes
in ``_scaffold/base.css``. There is **no inline ``<style>``, no inline
``<script>``, no f-string brace soup**. Plot scripts use these to build
the structural skeleton; live behavior comes from the co-located
``<plot>.js`` file picking up the well-known ``id`` and ``data-*``
attributes.

Convention: helpers do not escape values. Plot scripts must pre-escape
any user-derived content (today there is none — every label is a
hardcoded constant or a number formatted in Python).
"""
from __future__ import annotations

from html import escape
from typing import Iterable, Optional, Sequence


def sidebar(
    overlay_id: str,
    *,
    body: str,
    anchor: str = 'below-legend',
    compact: bool = False,
    width_px: Optional[int] = None,
    top_px: Optional[int] = None,
    right_px: Optional[int] = None,
    extra_attrs: str = '',
) -> str:
    """Fixed-position right-rail panel.

    ``overlay_id`` becomes the element ``id`` (used by the plot's JS to
    bind handlers). ``body`` is the inner HTML — typically built from
    other widget helpers + small plot-specific bits.

    ``anchor='below-legend'`` opts the panel into the
    ``overlay_anchor.js`` flow that positions it just below the Plotly
    legend; pass ``anchor=''`` to disable. ``compact`` swaps to the
    smaller-padding variant used for secondary panels (e.g.,
    training_quality's route-offset table).

    ``width_px`` pins the sidebar's width via an inline style. The
    same value is what plots pass to ``right_margin_for_anchored_box``
    so the plot reserves matching room — keeping width as a single
    Python int avoids drift between the two.

    ``top_px`` / ``right_px`` set static positioning for sidebars that
    are NOT legend-anchored (pass ``anchor=''`` together with these).
    """
    classes = 'rp-sidebar'
    if compact:
        classes += ' rp-sidebar-compact'
    anchor_attr = f' data-rp-anchor="{anchor}"' if anchor else ''
    style_parts = []
    if width_px is not None:
        style_parts.append(f'width: {width_px}px')
    if top_px is not None:
        style_parts.append(f'top: {top_px}px')
    if right_px is not None:
        style_parts.append(f'right: {right_px}px')
    style = f' style="{"; ".join(style_parts)}"' if style_parts else ''
    extra = f' {extra_attrs}' if extra_attrs else ''
    return (
        f'<div id="{overlay_id}" class="{classes}"{anchor_attr}{style}{extra}>\n'
        f'{body}\n'
        f'</div>'
    )


def title(text: str) -> str:
    return f'<div class="rp-sidebar-title">{text}</div>'


def subtitle(text: str) -> str:
    return f'<div class="rp-sidebar-sub">{text}</div>'


def divider() -> str:
    return '<div class="rp-sidebar-divider"></div>'


def button_row(
    buttons: Sequence[tuple],
    *,
    pill: bool = False,
) -> str:
    """Row of buttons.

    ``buttons`` is a sequence of ``(button_id, label)`` or
    ``(button_id, label, attrs_dict)`` tuples. Each button gets the
    ``rp-btn`` class (default, outlined) or ``rp-btn-pill`` (set
    ``pill=True`` for borderless toggle bars).

    Active state is set client-side by the plot's JS adding the
    ``is-active`` class — keep it out of static HTML so the CSS class
    system stays the single source of truth.
    """
    btn_class = 'rp-btn-pill' if pill else 'rp-btn'
    parts = []
    for entry in buttons:
        if len(entry) == 2:
            bid, label = entry
            extra = ''
        else:
            bid, label, attrs = entry
            extra = ''.join(f' {k}="{v}"' for k, v in attrs.items())
        parts.append(
            f'<button id="{bid}" class="{btn_class}"{extra}>{label}</button>'
        )
    return '<div class="rp-btn-row">\n  ' + '\n  '.join(parts) + '\n</div>'


def toggle_bar(
    overlay_id: str,
    buttons: Sequence[tuple],
    *,
    default_id: Optional[str] = None,
) -> str:
    """Top-right pill toggle group (scope/mode selector).

    Differs from ``button_row(pill=True)`` only in the outer wrapper:
    a ``.rp-toggle-bar`` panel positioned fixed top-right.

    ``buttons`` entries are ``(value, label)`` — ``value`` is what
    handlers read via ``data-value=...``. ``default_id`` is the
    ``value`` of the initially-active button.
    """
    parts = []
    for value, label in buttons:
        active = ' is-active' if value == default_id else ''
        parts.append(
            f'<button class="rp-btn-pill{active}" '
            f'data-value="{value}">{label}</button>'
        )
    inner = '\n  '.join(parts)
    return (
        f'<div id="{overlay_id}" class="rp-toggle-bar">\n  '
        f'{inner}\n</div>'
    )


def search_box(
    overlay_id: str,
    *,
    placeholder: str = '',
    count_id: Optional[str] = None,
) -> str:
    """Fixed top-right free-text search field.

    Same screen position as ``toggle_bar`` (``.rp-search`` shares the
    top-right chrome). ``type="search"`` gives the native clear (✕)
    control. The plot's JS binds an ``input`` listener via the
    ``overlay_id``.

    ``count_id`` adds a ``.rp-search-count`` span left of the input —
    an empty slot the plot's JS fills with a live result count.
    """
    ph = f' placeholder="{escape(placeholder)}"' if placeholder else ''
    count = (f'  <span id="{count_id}" class="rp-search-count"></span>\n'
             if count_id else '')
    return (
        f'<div id="{overlay_id}" class="rp-search">\n'
        f'{count}'
        f'  <input type="search"{ph} autocomplete="off" spellcheck="false">\n'
        f'</div>'
    )


def checkbox_rows(
    items: Iterable[tuple],
    *,
    data_attr: str,
    checked: bool = True,
) -> str:
    """Stack of labeled checkbox rows.

    ``items`` is an iterable of ``(value, label)`` or
    ``(value, label, meta)`` tuples. ``meta``, if present, is rendered
    as a small grey suffix (e.g., a count or β value).

    ``data_attr`` becomes ``data-<data_attr>="<value>"`` on each
    checkbox so JS can route the toggle. ``checked`` sets the initial
    state of every row.
    """
    rows = []
    chk = ' checked' if checked else ''
    for entry in items:
        if len(entry) == 2:
            value, label = entry
            meta_html = ''
        else:
            value, label, meta = entry
            meta_html = f' <span class="rp-row-meta">{meta}</span>'
        rows.append(
            f'<label class="rp-row">'
            f'<input type="checkbox" data-{data_attr}="{value}"{chk}>'
            f' {label}{meta_html}</label>'
        )
    return '\n'.join(rows)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    *,
    align: Optional[Sequence[Optional[str]]] = None,
) -> str:
    """Compact zebra-striped coefficient table.

    ``align`` is a per-column list of ``'left'`` / ``'right'`` /
    ``'center'`` / ``None``. ``'right'``-aligned columns get the
    ``num`` class so the CSS handles the alignment (and the
    monospace-friendly padding).
    """
    align_cols: list[Optional[str]] = list(align) if align else [None] * len(headers)

    def _cell_class(a):
        return ' class="num"' if a == 'right' else ''

    def _th_style(a):
        return ' style="text-align:right"' if a == 'right' else ''

    head = '<tr>' + ''.join(
        f'<th{_th_style(a)}>{h}</th>' for h, a in zip(headers, align_cols)
    ) + '</tr>'
    body_rows = []
    for row in rows:
        if len(row) == 1:
            # Single-cell row spans every column — for entries whose value is
            # an expression rendered inside the cell (e.g. a rate formula on
            # its own right-aligned line) rather than a separate column.
            cells = f'<td colspan="{len(headers)}">{row[0]}</td>'
        else:
            cells = ''.join(
                f'<td{_cell_class(a)}>{c}</td>'
                for c, a in zip(row, align_cols)
            )
        body_rows.append(f'<tr>{cells}</tr>')
    return (
        '<table class="rp-table">\n'
        f'  <thead>{head}</thead>\n'
        f'  <tbody>{"".join(body_rows)}</tbody>\n'
        '</table>'
    )


def stats_footer(lines: Sequence[str]) -> str:
    """Small footer block at the bottom of a sidebar.

    ``lines`` is rendered with ``<br>`` separators inside a single
    ``.rp-sidebar-stats`` div.
    """
    return (
        '<div class="rp-sidebar-stats">\n'
        + '<br>\n'.join(lines)
        + '\n</div>'
    )


def detail_row(label: str, value: str) -> str:
    """Label/value pair inside a sidebar ``<details>`` block.

    Renders ``<b>{label}</b> {value}`` — both passed through as HTML
    so callers can include subscripts, math symbols, etc.
    """
    return f'<div class="rp-detail-row"><b>{label}</b>{value}</div>'


def noteworthy(text: str) -> str:
    return f'<div class="rp-sidebar-noteworthy">{text}</div>'


def js_globals(values: dict) -> str:
    """Serialize Python values as ``window.__PLOT_*`` globals.

    Keys map to JS global names without the ``__PLOT_`` prefix:
    ``js_globals({'BINS': [...]})`` produces
    ``window.__PLOT_BINS = [...];``. The plot's sibling .js file reads
    these at startup. Use this instead of interpolating values into
    the JS source.
    """
    import json as _json
    parts = []
    for key, value in values.items():
        payload = _json.dumps(value, separators=(',', ':'), default=str)
        parts.append(f'window.__PLOT_{key} = {payload};')
    return '<script>\n' + '\n'.join(parts) + '\n</script>'
