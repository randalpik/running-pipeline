// Per-panel PR overlay for the by-distance plot.
//
// Each race trace is tagged with meta.panel_name + meta.pr_eligible. Each
// PR overlay is tagged with meta.panel_name + meta.is_pr_overlay. On any
// plotly_restyle that isn't self-inflicted (i.e. didn't only touch PR
// overlays), we walk all traces, group races by panel, compute the
// chronological running min on visible PR-eligible races, and update each
// panel's overlay in a single batched restyle.
//
// The PR-effort legend sentinel still returns false from legendclick so
// clicking it doesn't toggle visibility.
(function () {
  function findPlot() { return document.querySelector('.plotly-graph-div'); }

  // Plotly may pack numeric arrays as a typedarray spec. _inputArray is the
  // original, typically Float64Array (Array.isArray returns false on it).
  function asArray(v) {
    if (v == null) return null;
    if (Array.isArray(v)) return v;
    if (v._inputArray && typeof v._inputArray.length === 'number') return v._inputArray;
    if (typeof v.length === 'number') return v;
    return null;
  }

  function recomputePanelPRs() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) return;
    var byPanel = {};
    plot.data.forEach(function (t, i) {
      var m = t.meta;
      if (!m || !m.panel_name) return;
      var p = byPanel[m.panel_name];
      if (!p) { p = byPanel[m.panel_name] = { races: [], overlayIdx: -1 }; }
      if (m.is_pr_overlay) {
        p.overlayIdx = i;
      } else if (m.pr_eligible) {
        var v = t.visible;
        if (v === false || v === 'legendonly') return;
        var xs = asArray(t.x);
        var ys = asArray(t.y);
        if (!xs || !ys) return;
        for (var k = 0; k < xs.length; k++) {
          var y = ys[k];
          if (y == null || isNaN(y)) continue;
          p.races.push({ x: xs[k], y: y, ts: new Date(xs[k]).getTime() });
        }
      }
    });

    var indices = [], xs_all = [], ys_all = [];
    Object.keys(byPanel).forEach(function (name) {
      var info = byPanel[name];
      if (info.overlayIdx < 0) return;
      info.races.sort(function (a, b) { return a.ts - b.ts; });
      var best = Infinity, prX = [], prY = [];
      for (var i = 0; i < info.races.length; i++) {
        var p = info.races[i];
        if (p.y < best) { best = p.y; prX.push(p.x); prY.push(p.y); }
      }
      indices.push(info.overlayIdx);
      xs_all.push(prX);
      ys_all.push(prY);
    });
    if (indices.length === 0) return;
    Plotly.restyle(plot, { x: xs_all, y: ys_all }, indices);
  }

  function isSelfRestyle(eventData) {
    var indices = (eventData && eventData[1]) || null;
    if (!indices || !indices.length) return false;
    var plot = findPlot();
    if (!plot) return false;
    return indices.every(function (i) {
      var m = plot.data[i] && plot.data[i].meta;
      return m && m.is_pr_overlay;
    });
  }

  function attach() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(attach, 100); return; }
    plot.on('plotly_legendclick', function (ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
    plot.on('plotly_legenddoubleclick', function (ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
    // Run synchronously inside the restyle event so our overlay update
    // is batched with the visibility-toggle paint. Deferring via
    // setTimeout(0) splits this into two paints — visibility changes
    // first, PR diamonds catch up a frame later. Plotly's plotly_restyle
    // is dispatched after its internal state mutation completes, so it
    // is safe to issue another restyle from inside the handler.
    plot.on('plotly_restyle', function (ev) {
      if (isSelfRestyle(ev)) return;
      recomputePanelPRs();
    });
  }
  attach();
})();
