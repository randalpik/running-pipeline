"""Single rendering authority for every plot script.

Each plot script builds a Plotly figure, then calls :func:`render_plot` which
owns: writing the HTML via ``fig.write_html``, injecting the shared dark-theme
CSS, optionally injecting a cursor-following tooltip scaffold, and folding in
any plot-specific overlay HTML/CSS/JS (sidebars, restyle handlers, custom
event listeners).

The renderer is *agnostic* about overlays — ``overlay_html`` is an opaque
string. Plot-specific behavior (recovery's normalization sidebar, race's
distance-filter + PR-recompute, geography's bar-snapping mousemove) lives in
each plot's source verbatim and is passed in here as a string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SCAFFOLD_DIR = Path(__file__).resolve().parent / '_scaffold'
_BASE_CSS = (_SCAFFOLD_DIR / 'base.css').read_text()
_CURSOR_TOOLTIP_JS = (_SCAFFOLD_DIR / 'cursor_tooltip.js').read_text()

# Tiny iframe-side script: forward Alt+ArrowLeft/Right to the host shell so
# tab cycling keeps working when focus is inside the plot. Skipped when the
# document is loaded standalone (window.parent === window).
_TAB_KEY_FORWARDER_JS = """
<script>
(function () {
  if (window.parent === window) return;
  window.addEventListener('keydown', function (e) {
    if (!e.altKey) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    try { window.parent.postMessage({ type: 'rp-tab-key', key: e.key }, '*'); }
    catch (err) {}
  });
})();
</script>
"""


@dataclass
class CursorTooltip:
    """Per-plot input for the smart spikeline tooltip scaffold.

    Two-mode behavior:

    - **Smooth** (default): cursor is not near any snap-eligible marker.
      Spikeline follows the cursor; tooltip is rendered by ``build_js``,
      which must define ``function buildTooltip(day) { return html }``
      where ``day`` is days-since-1970-01-01. Returning ``''`` suppresses
      the tooltip.
    - **Snap**: cursor is within ``snap_px`` of a marker on a trace
      tagged ``meta.snap_eligible = True``. Spikeline jumps to the marker's
      x; tooltip = the marker's ``customdata[i]`` (per-point HTML the plot
      pre-renders).

    ``payload`` is JSON-serialized as ``window.__TT_DATA`` for the smooth
    builder. ``first_day`` / ``last_day`` clamp the day index passed to
    ``buildTooltip`` (use to suppress hover outside the data range).

    Set ``always_snap=True`` for plots whose data is purely discrete (e.g.
    geography bars): the scaffold never falls back to smooth mode and
    suppresses the tooltip when no snap-eligible point is found.
    """
    payload: object = None
    build_js: str = ''
    first_day: Optional[int] = None
    last_day: Optional[int] = None
    spike: bool = True
    snap_px: int = 30
    always_snap: bool = False


def render_plot(
    fig,
    out_path,
    *,
    title_slug: str,
    page_title: str,
    cursor_tooltip: Optional[CursorTooltip] = None,
    overlay_html: str = '',
    extra_head_css: str = '',
) -> Path:
    """Write ``fig`` as a self-contained HTML document at ``out_path``.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    out_path : str | Path
        Where to write the HTML. Parent directory created if missing.
    title_slug : str
        Short id used as the iframe ``<title>`` and tab-shell href stem.
    page_title : str
        Human-readable title written into ``<title>``. (Tab labels live in
        the shell manifest, not here.)
    cursor_tooltip : CursorTooltip | None
        If supplied, the cursor-tooltip scaffold is injected. Otherwise no
        tooltip-related markup is written and the plot is responsible for
        any custom hover behavior via ``overlay_html``.
    overlay_html : str
        Opaque HTML/CSS/JS injected just before ``</body>``. Use this for
        plot-specific UI (sidebars, restyle handlers, event listeners).
    extra_head_css : str
        Plot-specific CSS injected into ``<head>`` after the shared base CSS.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``"directory"`` writes plotly.min.js once into out_path's parent and
    # has every HTML reference it via a relative <script src>. All six plots
    # write to output/ so they share a single ~3.5MB bundle (cached by the
    # browser on first iframe load), instead of inlining it six times.
    fig.write_html(
        str(out_path),
        include_plotlyjs='directory',
        full_html=True,
        config={'responsive': True},
    )

    head_css = _BASE_CSS
    if extra_head_css:
        head_css = head_css + '\n' + extra_head_css

    body_pre_close_parts = [_TAB_KEY_FORWARDER_JS]
    if overlay_html:
        body_pre_close_parts.append(overlay_html)
    if cursor_tooltip is not None:
        body_pre_close_parts.append(_render_cursor_tooltip(cursor_tooltip))

    body_pre_close = '\n'.join(body_pre_close_parts)

    head_inject = (
        f'<title>{page_title}</title>\n'
        f'<meta name="rp-slug" content="{title_slug}">\n'
        f'<style>\n{head_css}\n</style>\n'
    )

    html = out_path.read_text()
    html = html.replace('</head>', head_inject + '</head>', 1)
    if body_pre_close:
        html = html.replace('</body>', body_pre_close + '\n</body>', 1)
    out_path.write_text(html)
    return out_path


def _render_cursor_tooltip(ct: CursorTooltip) -> str:
    payload_json = json.dumps(ct.payload, separators=(',', ':'), default=str)
    range_obj = {}
    if ct.first_day is not None:
        range_obj['firstDay'] = ct.first_day
    if ct.last_day is not None:
        range_obj['lastDay'] = ct.last_day
    range_json = json.dumps(range_obj if range_obj else {})
    spike_json = 'true' if ct.spike else 'false'
    always_snap_json = 'true' if ct.always_snap else 'false'

    return (
        '<div class="rp-tooltip"></div>\n'
        '<div class="rp-spike"></div>\n'
        '<script>\n'
        f'window.__TT_DATA = {payload_json};\n'
        f'window.__TT_RANGE = {range_json};\n'
        f'window.__TT_SPIKE = {spike_json};\n'
        f'window.__TT_SNAP_PX = {ct.snap_px};\n'
        f'window.__TT_ALWAYS_SNAP = {always_snap_json};\n'
        f'{ct.build_js}\n'
        f'{_CURSOR_TOOLTIP_JS}\n'
        '</script>\n'
    )
