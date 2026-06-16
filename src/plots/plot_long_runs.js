// Normalize + Show-tags toggles for the Long Runs plot.
//
// The markers' displayed y/color are the WATCH-CORRECTED source of truth,
// baked in by plot_long_runs.py. meta.disp_y carries that un-normalized
// corrected base so this toggle can re-derive without recomputing.
//
// Normalize (ON by default) subtracts the full flat / sea-level race-
// equivalent adjustment via window.__PLOT_LR_NORM_ADJ (sec/mi): physical
// route (grade, off-road footing, altitude) + training-state (temperature,
// race fatigue), the same decomposition the Training page applies. Python
// bakes the normalized y into the initial trace (so first paint is correct
// with no JS recompute); toggling off restores meta.disp_y.
//
// The condition-tag halo rings are separate traces whose meta.idx maps
// each ring point to its marker-trace position; their y is recomputed from
// the same base so rings track the markers. The rings ship VISIBLE but
// parked at an off-axis sentinel so their legend entries exist from load
// (constant legend size). Show-tags swaps the ring DATA between the
// sentinel and the real points.
//
// window.__lrNormOn mirrors the checkbox so the cursor-tooltip build_js
// (keyed by day in plot_long_runs.py's payload) can render the matching
// "Normalized adjustment" line.
(function () {
  window.__lrNormOn = true;
  var ADJ = window.__PLOT_LR_NORM_ADJ;
  var normCb = document.querySelector('#lr-gradient input[data-lrnorm]');
  var tagsCb = document.querySelector('#lr-gradient input[data-lrtags]');
  if (!normCb && !tagsCb) return;

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  function update() {
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(update, 100); return; }
    var main = -1, rings = [];
    for (var i = 0; i < plot.data.length; i++) {
      var m = plot.data[i].meta;
      if (m && m.role === 'long_runs') main = i;
      if (m && m.role === 'lr_ring') rings.push(i);
    }
    if (main < 0) return;
    var meta = plot.data[main].meta;
    var normOn = !!(normCb && normCb.checked);
    window.__lrNormOn = normOn;

    var base = meta.disp_y;
    var y = [];
    for (var k = 0; k < base.length; k++) {
      var v = base[k];
      if (v != null && normOn && ADJ && ADJ[k] != null) v = v - ADJ[k] / 60;
      y.push(v);
    }
    Plotly.restyle(plot, { y: [y] }, [main]);
    var tagsOn = !!(tagsCb && tagsCb.checked);
    for (var rI = 0; rI < rings.length; rI++) {
      var rMeta = plot.data[rings[rI]].meta;
      if (tagsOn) {
        var ry = [];
        for (var j = 0; j < rMeta.idx.length; j++) ry.push(y[rMeta.idx[j]]);
        Plotly.restyle(plot, { x: [rMeta.x_real], y: [ry] }, [rings[rI]]);
      } else {
        Plotly.restyle(plot, { x: [['1900-01-01']], y: [[6.0]] }, [rings[rI]]);
      }
    }
  }

  if (normCb) normCb.addEventListener('change', update);
  if (tagsCb) tagsCb.addEventListener('change', update);
})();
