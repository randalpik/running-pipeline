// Legend-aware positioning for overlay boxes.
//
// Opt-in per element via `data-rp-anchor="below-legend"`. After Plotly's
// first render and on every relayout / legend interaction / window resize,
// each anchored element's `top` is set to sit just below the main legend's
// bottom edge. Boxes right-align to the viewport edge with a 12px offset.
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

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  // The `.legend` class is used by Plotly for the main trace legend AND for
  // colorbars / range selectors. Scope to the main `infolayer` legend.
  function getLegend(plot) {
    return plot.querySelector('.infolayer > .legend');
  }

  function position() {
    var plot = getPlot();
    if (!plot) return;
    var anchored = document.querySelectorAll('[data-rp-anchor="below-legend"]');
    if (!anchored.length) return;
    var legend = getLegend(plot);
    if (!legend) return;
    var legendRect = legend.getBoundingClientRect();
    if (!legendRect.width && !legendRect.height) return;

    for (var i = 0; i < anchored.length; i++) {
      var el = anchored[i];
      var gap = parseFloat(el.dataset.rpGap);
      if (isNaN(gap)) gap = 12;
      el.style.top = (legendRect.bottom + gap) + 'px';
      el.style.right = BOX_RIGHT_OFFSET + 'px';
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
    window.addEventListener('resize', position);
    // First positioning pass — `plotly_afterplot` may have fired before our
    // listener attached on slow loads.
    position();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
