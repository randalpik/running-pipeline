// Training-quality plot: "Normalize to CS" checkbox.
//
// Each trace ships both raw_y and norm_y in its meta. Toggling the
// checkbox restyles every such trace's y data and re-relayouts the
// y-axis to match the active mode. Initial Python render matches the
// default checked state — no first-paint restyle.
//
// Axis configs come from window.__PLOT_AXIS_RAW and __PLOT_AXIS_NORM.
(function () {
  function getPlot() { return document.querySelector('.plotly-graph-div'); }
  function applyState(checked) {
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(function () { applyState(checked); }, 100); return; }
    var newY = [], indices = [];
    for (var i = 0; i < plot.data.length; i++) {
      var meta = plot.data[i].meta;
      if (meta && meta.raw_y && meta.norm_y) {
        newY.push(checked ? meta.norm_y : meta.raw_y);
        indices.push(i);
      }
    }
    if (indices.length) {
      Plotly.restyle(plot, { y: newY }, indices);
    }
    var ax = checked ? window.__PLOT_AXIS_NORM : window.__PLOT_AXIS_RAW;
    Plotly.relayout(plot, {
      'yaxis.range': ax.range,
      'yaxis.tickvals': ax.tickvals,
      'yaxis.ticktext': ax.ticktext,
      'yaxis.title.text': ax.title,
    });
  }
  var cb = document.getElementById('tq-norm-cb');
  cb.addEventListener('change', function () { applyState(cb.checked); });

  // Mobile: the pill's fixed slot overlaps the title bar, so fold it into
  // the legend-anchored #tq-routes box (moving the node keeps the checkbox
  // listener) and restore it on the way back. Reads the one mobile signal —
  // html.rp-mobile via rpMobile — never a live viewport query, which would
  // reintroduce the relayout loop documented in _scaffold/mobile.js.
  function isMobile() {
    return window.rpMobile ? window.rpMobile.isMobile()
      : document.documentElement.classList.contains('rp-mobile');
  }
  var toggle = document.getElementById('tq-norm-toggle');
  var home = toggle && { parent: toggle.parentNode, next: toggle.nextSibling };
  function placeToggle() {
    if (!toggle) return;
    var routes = document.getElementById('tq-routes');
    if (isMobile() && routes) {
      toggle.classList.add('rp-inline');
      routes.appendChild(toggle);
    } else if (toggle.parentNode !== home.parent) {
      toggle.classList.remove('rp-inline');
      home.parent.insertBefore(toggle, home.next);
    }
  }
  // This file is injected before _scaffold/mobile.js, so rpMobile may not
  // exist yet — wait briefly for it rather than missing mode changes.
  var hookTries = 0;
  (function hookMode() {
    if (window.rpMobile) { window.rpMobile.onChange(placeToggle); return; }
    if (++hookTries < 40) setTimeout(hookMode, 50);
  })();
  placeToggle();
})();
