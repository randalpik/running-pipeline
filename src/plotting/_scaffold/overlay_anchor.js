// Legend-aware positioning for overlay boxes.
//
// Opt-in per element via `data-rp-anchor="below-legend"`. After Plotly's
// first render and on every relayout / legend interaction / window resize,
// each anchored element's `top` is set to sit just below the main legend's
// bottom edge. On short viewports the box may rise — but only as far as
// needed to keep MIN_VISIBLE px of it on-screen (never high enough to
// cover the legend wholesale), and its `max-height` is set from the final
// top so the sidebar's own overflow-y scrolling reaches the rest.
// Boxes right-align to the viewport edge with a 12px offset.
//
// The figure's `margin.r` and the box's CSS `width` must both come from
// the same Python source (see `right_margin_for_anchored_box` in
// src/plotting/layout.py) so the box always fits in the right margin
// without runtime relayout. BOX_RIGHT_OFFSET below MUST stay in sync with
// ANCHORED_BOX_RIGHT_PADDING in layout.py.
//
// `data-rp-gap` (default 12) controls the px gap between legend bottom and
// the box's top edge.

(function () {
  var BOX_RIGHT_OFFSET = 12;  // gap between box's right edge and viewport right
  var MIN_TOP = 68;           // never above the title bar (--rp-title-h + 8)
  var BOTTOM_PAD = 12;        // gap kept below the box at the viewport bottom
  var MIN_VISIBLE = 90;       // guaranteed on-screen box height when clamped

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  // The `.legend` class is used by Plotly for the main trace legend AND for
  // colorbars / range selectors. Scope to the main `infolayer` legend.
  function getLegend(plot) {
    return plot.querySelector('.infolayer > .legend');
  }

  // Bottom edge to anchor below. The legend <g>'s bounding rect ignores
  // Plotly's legend scrollbox clip, so on short viewports it can extend past
  // the window bottom; the legend's `.bg` rect reflects the clipped height.
  function legendBottom(plot) {
    var legend = getLegend(plot);
    if (!legend) return null;
    var bg = legend.querySelector('.bg');
    var rect = (bg || legend).getBoundingClientRect();
    if (!rect.width && !rect.height) return null;
    return rect.bottom;
  }

  function position() {
    var plot = getPlot();
    if (!plot) return;
    var anchored = document.querySelectorAll('[data-rp-anchor="below-legend"]');
    if (!anchored.length) return;
    var lb = legendBottom(plot);
    if (lb == null) return;

    // Uncap → measure → write, so natural height is never read through a
    // stale max-height and multiple boxes don't thrash each other.
    var natural = [];
    var i, el;
    for (i = 0; i < anchored.length; i++) anchored[i].style.maxHeight = 'none';
    for (i = 0; i < anchored.length; i++) natural.push(anchored[i].offsetHeight);
    var vh = window.innerHeight;
    for (i = 0; i < anchored.length; i++) {
      el = anchored[i];
      var gap = parseFloat(el.dataset.rpGap);
      if (isNaN(gap)) gap = 12;
      // Below the legend by default. When that leaves less than the
      // guaranteed strip, rise just enough for min(natural, MIN_VISIBLE) —
      // overlapping at most the legend's last rows, never burying it.
      var top = lb + gap;
      var maxTop = vh - BOTTOM_PAD - Math.min(natural[i], MIN_VISIBLE);
      if (top > maxTop) top = Math.max(MIN_TOP, maxTop);
      el.style.top = top + 'px';
      el.style.right = BOX_RIGHT_OFFSET + 'px';
      el.style.maxHeight = Math.max(MIN_VISIBLE, vh - top - BOTTOM_PAD) + 'px';
    }
  }

  function bind() {
    var plot = getPlot();
    if (!plot) { setTimeout(bind, 50); return; }
    // Plotly fires its events on the gd element.
    plot.on('plotly_afterplot', position);
    plot.on('plotly_relayout', position);
    plot.on('plotly_legendclick', function () { setTimeout(position, 0); });
    plot.on('plotly_legenddoubleclick', function () { setTimeout(position, 0); });
    // First positioning pass — `plotly_afterplot` may have fired before our
    // listener attached on slow loads.
    position();
  }

  // Window-level listeners live outside bind() so a re-bind (below) doesn't
  // stack duplicates.
  window.addEventListener('resize', position);
  // The mobile layout engine's Plotly.newPlot clears gd.on bindings.
  window.addEventListener('rp-layout-mode', function () { bind(); });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
