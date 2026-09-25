// Colour-by dropdown: re-encode marker colour from a sidebar <select>.
//
// Generic over plots. A plot opts in by
//   1. shipping window.__PLOT_COLOR_BY (see plotting/color_by.build_color_by),
//   2. rendering widgets.gradient_strip('cb-strip', ...) + widgets.select(
//      'cb-select', ..., data_attr='colorby') into its sidebar, and
//   3. tagging every marker trace whose points carry the encoding with
//      meta.color_by === true.
//
// Traces are resolved by that meta flag, never by index — trace counts vary
// per profile (Long Runs only emits a halo-ring trace for tags that occur).
//
// Colours are precomputed per variable in Python and shipped as #rrggbb
// arrays, so this file never evaluates a gradient: the sidebar strip and the
// markers are by construction the same mapping. Only marker.color is
// restyled, leaving each plot's own handlers (which drive y and
// marker.opacity) free to run in any order relative to this one.

(function () {
  var CB = window.__PLOT_COLOR_BY;
  if (!CB || !CB.vars) return;

  var sel = document.querySelector('select[data-colorby]');
  var strip = document.getElementById('cb-strip');
  if (!sel || !strip) return;

  var titleEl = document.getElementById('cb-strip-title');
  var barEl = document.getElementById('cb-strip-bar');
  var ticksEl = document.getElementById('cb-strip-ticks');
  // Multi-ramp legend (a variable with `strips`, e.g. Recent effort): shown in
  // place of the single bar + ticks.
  var stackEl = document.getElementById('cb-strip-stack');

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  function markerTraces(plot) {
    var idx = [];
    for (var i = 0; i < plot.data.length; i++) {
      var m = plot.data[i].meta;
      if (m && m.color_by) idx.push(i);
    }
    return idx;
  }

  function tickHtml(ticks) {
    // Mirrors widgets.gradient_strip: end labels get data-at and NO inline
    // `left`, so base.css can pin them inside the bar's edges (an inline
    // `left: 100%` would beat that rule and clip against the sidebar).
    var html = '';
    for (var i = 0; i < ticks.length; i++) {
      var pos = ticks[i][0], label = ticks[i][1];
      if (pos <= 0) html += '<span data-at="0">' + label + '</span>';
      else if (pos >= 1) html += '<span data-at="1">' + label + '</span>';
      else html += '<span style="left: ' + (100 * pos) + '%">' + label + '</span>';
    }
    return html;
  }

  function renderTicks(ticks) { ticksEl.innerHTML = tickHtml(ticks); }

  function renderStack(strips) {
    // Mirrors widgets.gradient_rows.
    stackEl.innerHTML = '';
    for (var i = 0; i < strips.length; i++) {
      var row = document.createElement('div');
      row.className = 'rp-gradient-row';
      var label = document.createElement('span');
      label.className = 'rp-gradient-label';
      label.textContent = strips[i].title;
      var body = document.createElement('div');
      var bar = document.createElement('div');
      bar.className = 'rp-gradient-bar';
      bar.style.background = strips[i].gradient;
      var ticks = document.createElement('div');
      ticks.className = 'rp-gradient-ticks';
      ticks.innerHTML = tickHtml(strips[i].ticks);
      body.appendChild(bar);
      body.appendChild(ticks);
      row.appendChild(label);
      row.appendChild(body);
      stackEl.appendChild(row);
    }
  }

  function apply() {
    var v = CB.vars[sel.value];
    if (!v) return;
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(apply, 100); return; }

    if (titleEl) titleEl.textContent = v.title;
    var stacked = !!(v.strips && stackEl);
    if (barEl) { barEl.style.background = v.gradient; barEl.style.display = stacked ? 'none' : ''; }
    if (ticksEl) { renderTicks(v.ticks); ticksEl.style.display = stacked ? 'none' : ''; }
    if (stackEl) {
      stackEl.style.display = stacked ? '' : 'none';
      if (stacked) renderStack(v.strips);
    }

    var idx = markerTraces(plot);
    if (!idx.length) return;
    // One colour array per target trace. Recovery draws the same runs twice
    // (absolute pace + residual), so both panels stay in lockstep.
    var colors = [];
    for (var i = 0; i < idx.length; i++) colors.push(v.colors);
    Plotly.restyle(plot, { 'marker.color': colors }, idx);
  }

  sel.addEventListener('change', apply);

  // Re-apply after a mobile reshape: mobile.js re-renders the figure from the
  // original spec, which would otherwise restore the load-time colours.
  document.addEventListener('rp-layout-mode', function () {
    setTimeout(apply, 0);
  });

  // No initial apply() — the figure ships already coloured by CB.default and
  // the strip's static HTML already matches it, so load does no extra work.
})();
