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
_OVERLAY_ANCHOR_JS = (_SCAFFOLD_DIR / 'overlay_anchor.js').read_text()
_TAP_HOVER_JS = (_SCAFFOLD_DIR / 'tap_hover.js').read_text()
_AXIS_PAD_JS = (_SCAFFOLD_DIR / 'axis_pad.js').read_text()
_MOBILE_HEAD_JS = (_SCAFFOLD_DIR / 'mobile_head.js').read_text()
_MOBILE_JS = (_SCAFFOLD_DIR / 'mobile.js').read_text()
_TOUCH_SCROLL_JS = (_SCAFFOLD_DIR / 'touch_scroll.js').read_text()

# Shared <head> boilerplate for every page the pipeline emits — render_plot()
# and dashboard.py's hand-rolled document import this. Single source so the
# viewport meta can't be added to one and forgotten in the other.
#
# (The meta is inert in a subframe — Chromium sizes a frame's viewport from
# the frame itself — and removing it here was measured to change nothing on
# device. It matters only when a plot page is opened standalone.)
_HEAD_META = (
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<link rel="icon" href="data:,">\n'
    '<meta name="darkreader-lock">\n'
)

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

# Iframe-side ready signal: postMessage the host shell once Plotly's first
# layout pass finishes, so it can hide the loading spinner. The iframe `load`
# event fires hundreds of ms before this on the heavy plots, so we listen for
# `plotly_afterplot` instead. Idempotent — the shell ignores duplicates.
_PLOT_READY_FORWARDER_JS = """
<script>
(function () {
  if (window.parent === window) return;
  function bind() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || typeof gd.on !== 'function') { setTimeout(bind, 50); return; }
    var sent = false;
    function notify() {
      if (sent) return;
      sent = true;
      try { window.parent.postMessage({ type: 'rp-plot-ready' }, '*'); }
      catch (err) {}
    }
    gd.on('plotly_afterplot', notify);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
</script>
"""


# Iframe-side shell-mode receiver. `html.rp-framed` marks "this document is
# inside the tab shell" (set synchronously, so CSS gated on :not(.rp-framed) —
# the standalone rotate-your-phone hint — never flashes in the shell).
# `html.rp-mobile` mirrors the shell's mobile breakpoint, pushed via the
# 'rp-shell-mode' postMessage (see _scaffold/shell.js) — a plot page can't
# derive it itself: its viewport is already rotated to landscape, and
# (pointer: coarse) alone over-matches on touch laptops.
_SHELL_MODE_RECEIVER_JS = """
<script>
(function () {
  if (window.parent === window) return;
  document.documentElement.classList.add('rp-framed');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'rp-shell-mode') return;
    document.documentElement.classList.toggle('rp-mobile', d.mobile === true);
  });
})();
</script>
"""


@dataclass
class MobileLayout:
    """Layout reshape applied when the viewport is mobile-short (see
    ``_scaffold/mobile.js``; breakpoint ``(max-height: 520px)``).

    ``patch`` maps dotted Plotly layout paths (``'xaxis3.domain'``,
    ``'annotations[2].y'``, ``'margin.r'``, ``'height'``) to mobile values;
    traces are never touched — build the paths with
    :func:`src.plotting.layout.reshape_patch` so the panel→axis-id mapping
    stays invariant. ``variants`` is the same keyed by page name for plots
    that swap whole figures client-side (Misc Trends); ``variant_global``
    names the window global holding the active key. Exactly one of
    ``patch`` / ``variants``.

    ``scroll=True`` (default) also puts the page into scroll mode while the
    mobile layout is applied: html/body become scrollable and the plot div
    takes the patched ``height`` (which should exceed the viewport — that's
    the point).
    """
    patch: Optional[dict] = None
    variants: Optional[dict] = None
    variant_global: Optional[str] = None
    scroll: bool = True

    def to_json_obj(self) -> dict:
        obj: dict = {'scroll': self.scroll}
        if self.patch is not None:
            obj['patch'] = self.patch
        if self.variants is not None:
            obj['variants'] = self.variants
        if self.variant_global is not None:
            obj['variant_global'] = self.variant_global
        return obj


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

    Set ``spike_full_plot=True`` for stacked-subplot plots whose spike
    should visually span every panel (e.g. qualitative_trends). Default
    is per-subplot clipping so multi-panel plots don't draw a spike
    across panels the cursor isn't currently in.

    Set ``spike_snap_day=True`` to quantize the smooth-mode spike to the
    day the tooltip describes (one position per calendar day) instead of
    following the cursor pixel — use when the x-axis is day-granular and
    a free-floating spike reads as false precision (e.g. annual).
    """
    payload: object = None
    build_js: str = ''
    first_day: Optional[int] = None
    last_day: Optional[int] = None
    spike: bool = True
    snap_px: int = 30
    # Snap radius for TAPS (tap-as-hover on touch devices) — wider than the
    # mouse radius because a fingertip is less precise. Mouse behavior is
    # unaffected.
    touch_snap_px: int = 44
    always_snap: bool = False
    spike_full_plot: bool = False
    spike_snap_day: bool = False


def render_plot(
    fig,
    out_path,
    *,
    title_slug: str,
    page_title: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    cursor_tooltip: Optional[CursorTooltip] = None,
    overlay_html: str = '',
    overlay_js_files: Optional[list] = None,
    extra_head_css: str = '',
    extra_head_css_files: Optional[list] = None,
    axis_pad: Optional[list] = None,
    plotly_config: Optional[dict] = None,
    mobile_layout: Optional[MobileLayout] = None,
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
        Human-readable title written into the document ``<title>`` element.
    title : str | None
        Main title rendered as a plain-HTML overlay at the top of the
        iframe (NOT via Plotly's ``layout.title``). HTML is passed through
        — embed ``<i>``, ``<b>``, etc. as needed.
    subtitle : str | None
        Optional subtitle on a second line below the main title. HTML
        passed through (CSS gives ``<i>`` an accent color).
    cursor_tooltip : CursorTooltip | None
        If supplied, the cursor-tooltip scaffold is injected. Otherwise no
        tooltip-related markup is written and the plot is responsible for
        any custom hover behavior via ``overlay_html``.
    overlay_html : str
        Opaque HTML injected just before ``</body>``. Use this for the
        structural skeleton of plot-specific UI. Prefer composing it from
        :mod:`src.plotting.widgets` helpers and styling via shared
        ``.rp-*`` classes in ``_scaffold/base.css`` rather than embedding
        ``<style>`` / ``<script>`` blocks inline.
    overlay_js_files : list[str | Path] | None
        Paths to ``.js`` files (one per plot, sibling-located is the
        canonical layout). Each is read and wrapped in a single
        ``<script>`` tag, injected after ``overlay_html`` so the DOM it
        binds to already exists. Use this instead of embedding JS in
        Python f-strings.
    extra_head_css : str
        Plot-specific CSS injected into ``<head>`` after the shared base CSS.
    extra_head_css_files : list[str | Path] | None
        Paths to ``.css`` files appended to ``<head>`` after
        ``extra_head_css``. Use for plot-specific rules that don't
        generalize into shared ``_scaffold/base.css`` (e.g. geography's
        legend tree).
    axis_pad : list[dict] | None
        Opt-in pixel-accurate x-axis edge padding. Each dict is
        ``{'axis': 'xaxis', 'loMs': int, 'hiMs': int, 'halfPx': float}``
        where ``loMs``/``hiMs`` are the TIGHT data range as epoch-ms and
        ``halfPx`` is half the widest edge marker (so edge diamonds aren't
        clipped). Set the figure's axis range to the same tight ``[loMs, hiMs]``;
        the injected ``_scaffold/axis_pad.js`` converts ``halfPx`` to a date
        delta from the rendered pixel width and re-pads on window resize.
    mobile_layout : MobileLayout | None
        Opt-in mobile reshape (scrollable spread-out panels). The desktop
        figure is untouched — the reshape ships as a JS global that
        ``_scaffold/mobile.js`` applies client-side on short viewports.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``"directory"`` writes plotly.min.js once into out_path's parent and
    # has every HTML reference it via a relative <script src>. All six plots
    # write to output/ so they share a single ~3.5MB bundle (cached by the
    # browser on first iframe load), instead of inlining it six times.
    config = {'responsive': True, 'displayModeBar': False}
    if plotly_config:
        config.update(plotly_config)
    fig.write_html(
        str(out_path),
        include_plotlyjs='directory',
        full_html=True,
        config=config,
    )

    head_css = _BASE_CSS
    if extra_head_css:
        head_css = head_css + '\n' + extra_head_css
    if extra_head_css_files:
        for css_path in extra_head_css_files:
            head_css = head_css + '\n' + Path(css_path).read_text()

    body_pre_close_parts = [
        _TAB_KEY_FORWARDER_JS,
        _PLOT_READY_FORWARDER_JS,
        _SHELL_MODE_RECEIVER_JS,
        f'<script>\n{_OVERLAY_ANCHOR_JS}\n</script>',
        # Before overlay_js_files: plot-specific JS (geography's custom
        # tooltip) consumes window.rpTapHover at bind time.
        f'<script>\n{_TAP_HOVER_JS}\n</script>',
    ]
    if title is not None:
        body_pre_close_parts.append(_render_title_bar(title, subtitle))
    if overlay_html:
        body_pre_close_parts.append(overlay_html)
    if overlay_js_files:
        for js_path in overlay_js_files:
            body_pre_close_parts.append(
                f'<script>\n{Path(js_path).read_text()}\n</script>'
            )
    if cursor_tooltip is not None:
        body_pre_close_parts.append(_render_cursor_tooltip(cursor_tooltip))
    if axis_pad:
        body_pre_close_parts.append(
            f'<script>\nwindow.__PLOT_AXIS_PAD = {json.dumps(axis_pad)};\n'
            f'{_AXIS_PAD_JS}\n</script>'
        )
    # Last: the mobile layout engine re-renders the figure, so everything
    # that binds to the plot must already be listening for 'rp-layout-mode'.
    # touch_scroll.js precedes it — mobile.js attaches the page's pan
    # handler through it at startup.
    body_pre_close_parts.append(
        f'<script>\n{_TOUCH_SCROLL_JS}\n{_MOBILE_JS}\n</script>')

    body_pre_close = '\n'.join(body_pre_close_parts)

    # __RP_PLOT_CONFIG lets scaffold JS re-render the figure with the exact
    # config it was born with (staticPlot on Misc Trends, scrollZoom on the
    # world map) instead of guessing. Emitted in <head> together with the
    # mobile layout so mobile_head.js can hide the desktop-first paint
    # before Plotly's inline newPlot (a <body> script) runs.
    head_globals = f'window.__RP_PLOT_CONFIG = {json.dumps(config)};\n'
    if mobile_layout is not None:
        head_globals += ('window.__PLOT_MOBILE_LAYOUT = '
                         f'{json.dumps(mobile_layout.to_json_obj())};\n')
    head_inject = (
        f'<title>{page_title}</title>\n'
        f'{_HEAD_META}'
        f'<meta name="rp-slug" content="{title_slug}">\n'
        f'<style>\n{head_css}\n</style>\n'
        f'<script>\n{head_globals}{_MOBILE_HEAD_JS}\n</script>\n'
    )

    html = out_path.read_text()
    html = html.replace('</head>', head_inject + '</head>', 1)
    if body_pre_close:
        html = html.replace('</body>', body_pre_close + '\n</body>', 1)
    # Plotly's write_html omits the doctype, which puts the iframe into
    # quirks mode. Prepend it so the page renders in standards mode.
    if not html.lstrip().lower().startswith('<!doctype'):
        html = '<!doctype html>\n' + html
    out_path.write_text(html)
    return out_path


def _render_title_bar(title: str, subtitle: Optional[str]) -> str:
    """Inline HTML for the per-plot title overlay (see base.css)."""
    parts = [f'<div class="rp-title-main">{title}</div>']
    if subtitle:
        parts.append(f'<div class="rp-title-sub">{subtitle}</div>')
    return '<div class="rp-title-bar">\n' + '\n'.join(parts) + '\n</div>'


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
    spike_full_json = 'true' if ct.spike_full_plot else 'false'
    spike_snap_day_json = 'true' if ct.spike_snap_day else 'false'

    return (
        '<div class="rp-tooltip"></div>\n'
        '<div class="rp-spike"></div>\n'
        '<script>\n'
        f'window.__TT_DATA = {payload_json};\n'
        f'window.__TT_RANGE = {range_json};\n'
        f'window.__TT_SPIKE = {spike_json};\n'
        f'window.__TT_SNAP_PX = {ct.snap_px};\n'
        f'window.__TT_TOUCH_SNAP_PX = {ct.touch_snap_px};\n'
        f'window.__TT_ALWAYS_SNAP = {always_snap_json};\n'
        f'window.__TT_SPIKE_FULL_PLOT = {spike_full_json};\n'
        f'window.__TT_SPIKE_SNAP_DAY = {spike_snap_day_json};\n'
        f'{ct.build_js}\n'
        f'{_CURSOR_TOOLTIP_JS}\n'
        '</script>\n'
    )
