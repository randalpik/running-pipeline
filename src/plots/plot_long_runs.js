// Watch-correction + Normalize toggles for the Long Runs plot.
//
// Watch correction swaps the markers' display base from logged values to
// the watch-corrected ones (y = pace AND marker.color = distance gradient),
// both carried index-aligned in the markers trace's meta (raw_y / corr_y /
// raw_color / corr_color — built in plot_long_runs.py). Points without a
// correction keep their logged values in the corr arrays' fallbacks.
//
// Normalize subtracts the full flat / sea-level race-equivalent adjustment
// via window.__PLOT_LR_NORM_ADJ (sec/mi): physical route (grade, off-road
// footing, altitude) + training-state (temperature, race fatigue), the same
// decomposition the Training page applies. It stacks on top of whichever base
// the Watch toggle selects.
//
// The condition-tag halo rings are separate traces whose meta.idx maps
// each ring point to its marker-trace position; their y is recomputed from
// the same base so rings track the markers through both toggles. The rings
// ship VISIBLE but parked at an off-axis sentinel so their legend entries
// exist from load (constant legend size — toggling never reflows the
// legend, the gradient sidebar, or the plot). Show-tags swaps the ring
// DATA between the sentinel and the real points.
//
// window.__lrWatchOn / window.__lrNormOn mirror the checkboxes so the
// cursor-tooltip build_js (keyed by day in plot_long_runs.py's payload)
// can render the matching variant.
(function () {
  window.__lrNormOn = false;
  window.__lrWatchOn = false;
  var ADJ = window.__PLOT_LR_NORM_ADJ;
  var normCb = document.querySelector('#lr-gradient input[data-lrnorm]');
  var watchCb = document.querySelector('#lr-gradient input[data-lrwatch]');
  var tagsCb = document.querySelector('#lr-gradient input[data-lrtags]');
  if (!normCb && !watchCb && !tagsCb) return;

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
    var watchOn = !!(watchCb && watchCb.checked);
    var normOn = !!(normCb && normCb.checked);
    window.__lrWatchOn = watchOn;
    window.__lrNormOn = normOn;

    var base = watchOn ? meta.corr_y : meta.raw_y;
    var y = [];
    for (var k = 0; k < meta.raw_y.length; k++) {
      // corr_y is null where no correction exists — fall back to logged.
      var v = (base[k] != null) ? base[k] : meta.raw_y[k];
      if (v != null && normOn && ADJ && ADJ[k] != null) v = v - ADJ[k] / 60;
      y.push(v);
    }
    var color = watchOn ? meta.corr_color : meta.raw_color;
    Plotly.restyle(plot, { y: [y], 'marker.color': [color] }, [main]);
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
  if (watchCb) watchCb.addEventListener('change', update);
  if (tagsCb) tagsCb.addEventListener('change', update);
})();
