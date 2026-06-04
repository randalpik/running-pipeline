// Pixel-accurate, resize-aware x-axis edge padding.
//
// The figure ships with each padded x-axis set to its TIGHT data range
// [loMs, hiMs] (epoch-ms). The visible gutter we actually want is "≥ half a
// marker width" so edge diamonds/dots aren't clipped — a PIXEL quantity that
// only the browser can resolve, and that changes whenever the plot is resized.
//
// For each configured axis we convert that half-marker pixel pad into a date
// delta using the axis's current pixel length, then relayout the range to
// [loMs − padMs, hiMs + padMs]. We always pad from the ORIGINAL tight range in
// the global (never the already-padded current range), so repeated resizes
// don't drift. `xa._length` is the axis drawing width, fixed by the plot area
// independent of the range, so one pass converges.
//
// Reads window.__PLOT_AXIS_PAD = [{axis:'xaxis', loMs, hiMs, halfPx}, ...].
(function () {
  var CFG = window.__PLOT_AXIS_PAD || [];
  if (!CFG.length) return;

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  function apply() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var upd = {};
    CFG.forEach(function (c) {
      var xa = gd._fullLayout[c.axis];
      if (!xa || !xa._length) return;
      var span = c.hiMs - c.loMs;
      if (span <= 0 || xa._length <= 2 * c.halfPx) return;
      // We want the rendered gutter to be exactly halfPx pixels. Adding the pad
      // widens the range to (span + 2·padMs) over the SAME pixel length, so
      // ms-per-px grows; solving padMs·_length/(span + 2·padMs) = halfPx gives:
      var padMs = c.halfPx * span / (xa._length - 2 * c.halfPx);
      upd[c.axis + '.range'] = [c.loMs - padMs, c.hiMs + padMs];
    });
    if (Object.keys(upd).length && window.Plotly) Plotly.relayout(gd, upd);
  }

  function bind() {
    var gd = pdiv();
    // Wait for Plotly's first layout pass — _length is only populated then.
    if (!gd || !gd._fullLayout || typeof gd.on !== 'function') {
      setTimeout(bind, 50);
      return;
    }
    apply();
    // Re-pad after Plotly's own resize recompute (debounced).
    var t = null;
    window.addEventListener('resize', function () {
      if (t) clearTimeout(t);
      t = setTimeout(apply, 120);   // after cursor_tooltip's Plots.resize
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
