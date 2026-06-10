// Normalize toggle for the Long Runs plot.
//
// One checkbox subtracts the TQ long-run model's covariate contributions
// (temperature, marathon/short-race fatigue — betas fit on the long runs
// themselves) from each point's pace. Per-point adjustments arrive
// index-aligned with the markers trace via window.__PLOT_LR_NORM_ADJ
// (sec/mi, positive = modeled slower than baseline, so normalizing moves
// the point faster).
//
// window.__lrNormOn mirrors the checkbox so the cursor-tooltip build_js
// (which lives in plot_long_runs.py's payload, keyed by day) can show the
// applied shift only while normalization is active.
(function () {
  window.__lrNormOn = false;
  var ADJ = window.__PLOT_LR_NORM_ADJ;
  var cb = document.querySelector('#lr-gradient input[data-lrnorm]');
  if (!ADJ || !cb) return;

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  function findTrace(plot) {
    for (var i = 0; i < plot.data.length; i++) {
      var t = plot.data[i];
      if (t.meta && t.meta.role === 'long_runs') return i;
    }
    return -1;
  }

  function update() {
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(update, 100); return; }
    var ti = findTrace(plot);
    if (ti < 0) return;
    var raw = plot.data[ti].meta.raw_y;
    var on = cb.checked;
    window.__lrNormOn = on;
    var y = raw.slice();
    if (on) {
      for (var i = 0; i < y.length; i++) {
        if (y[i] != null && ADJ[i] != null) y[i] = raw[i] - ADJ[i] / 60;
      }
    }
    Plotly.restyle(plot, { y: [y] }, [ti]);
  }

  cb.addEventListener('change', update);
})();
