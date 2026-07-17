// Distance-filter sidebar + location/event search for the all-races plot.
//
// Each checkbox toggles the visibility of all traces tagged with
// meta.filter_bin matching the box's data-bin attribute. Sentinel
// traces (without meta.filter_bin) are left alone, which is what
// keeps each surface's legend entry alive even when all its bins are
// unchecked.
//
// The search box (#race-search) is a per-POINT filter, so it can't ride
// trace visibility: it masks each bin trace's x/y/customdata down to
// the points whose meta.search_loc / meta.search_event contain the
// query (case-insensitive substring). Original arrays are cached on
// first use; an empty query restores them. Search composes with the
// checkbox filter (trace `visible`) and the native surface legend
// toggles (`legendonly`) as an AND — they act on orthogonal axes.
//
// The PR overlay (meta.is_pr_overlay) is recomputed on any visibility
// OR masking change — checkbox-, search-, and legend-click-driven all
// funnel through the plotly_restyle listener.
(function () {
  function findPlot() { return document.querySelector('.plotly-graph-div'); }

  // Plotly may store numeric arrays in a binary-packed typedarray-spec
  // format ({dtype, bdata, _inputArray}) for compactness — t.y[i] then
  // returns undefined and t.y.length is undefined. The original numbers
  // are on _inputArray, which is typically a Float64Array (a TypedArray,
  // NOT a plain Array — Array.isArray() returns false on it). Both
  // TypedArrays and plain Arrays support [i] indexing and .length, so
  // checking for length is enough.
  function asArray(v) {
    if (v == null) return null;
    if (Array.isArray(v)) return v;
    if (v._inputArray && typeof v._inputArray.length === 'number') return v._inputArray;
    if (typeof v.length === 'number') return v;
    return null;
  }

  // Walk every trace tagged meta.filter_bin and gather (date, y) tuples
  // from those currently visible AND PR-eligible (Downhill races are
  // excluded). Respects bin checkboxes AND surface legend toggles, both
  // of which mutate trace.visible.
  function gatherVisibleRaces(plot) {
    var pts = [];
    plot.data.forEach(function (t) {
      if (!t.meta || !t.meta.filter_bin) return;
      if (t.meta.pr_eligible === false) return;
      var v = t.visible;
      if (v === false || v === 'legendonly') return;
      var xs = asArray(t.x);
      var ys = asArray(t.y);
      if (!xs || !ys) return;
      for (var i = 0; i < xs.length; i++) {
        var y = ys[i];
        if (y == null || isNaN(y)) continue;
        pts.push({ x: xs[i], y: y, ts: new Date(xs[i]).getTime() });
      }
    });
    return pts;
  }

  function computePRs(pts) {
    pts.sort(function (a, b) { return a.ts - b.ts; });
    var best = Infinity, prX = [], prY = [];
    for (var i = 0; i < pts.length; i++) {
      if (pts[i].y < best) {
        best = pts[i].y;
        prX.push(pts[i].x);
        prY.push(pts[i].y);
      }
    }
    return { x: prX, y: prY };
  }

  function findOverlayIdx(plot) {
    for (var i = 0; i < plot.data.length; i++) {
      var t = plot.data[i];
      if (t.meta && t.meta.is_pr_overlay) return i;
    }
    return -1;
  }

  function recomputePRs() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) return;
    var idx = findOverlayIdx(plot);
    if (idx < 0) return;
    var pts = gatherVisibleRaces(plot);
    var pr = computePRs(pts);
    Plotly.restyle(plot, { x: [pr.x], y: [pr.y] }, [idx]);
  }

  function update() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(update, 100); return; }
    var checked = new Set();
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function (cb) {
      if (cb.checked) checked.add(cb.dataset.bin);
    });
    var updates = plot.data.map(function (t) {
      var bin = t.meta && t.meta.filter_bin;
      if (!bin) return true;          // CS lines, sentinels, PR overlay — leave alone
      return checked.has(bin);
    });
    // Restyle visibility — the plotly_restyle event listener will trigger
    // recomputePRs once the change is applied.
    Plotly.restyle(plot, { 'visible': updates });
  }

  // Live "n total" readout inside #race-search: counts the points of
  // every bin trace that survives all three filters (search masking
  // shrinks the arrays; checkboxes/legend toggle trace visibility).
  function updateCount() {
    var plot = findPlot();
    var el = document.getElementById('race-search-count');
    if (!plot || !plot.data || !el) return;
    var n = 0;
    plot.data.forEach(function (t) {
      if (!t.meta || !t.meta.filter_bin) return;
      var v = t.visible;
      if (v === false || v === 'legendonly') return;
      var xs = asArray(t.x);
      if (xs) n += xs.length;
    });
    el.textContent = n + ' total';
  }

  // ----- location/event search -----
  // Cache of each bin trace's full arrays, built once the plot is ready.
  // meta.search_* stay full-length on the trace, but x/y/customdata get
  // rewritten by masking restyles — so the originals live here.
  var searchCache = null;
  function ensureSearchCache(plot) {
    if (searchCache) return searchCache;
    searchCache = [];
    plot.data.forEach(function (t, idx) {
      if (!t.meta || !t.meta.filter_bin) return;
      var xs = asArray(t.x), ys = asArray(t.y), cd = asArray(t.customdata);
      if (!xs || !ys) return;
      searchCache.push({
        idx: idx,
        x: Array.prototype.slice.call(xs),
        y: Array.prototype.slice.call(ys),
        cd: cd ? Array.prototype.slice.call(cd) : null,
        loc: t.meta.search_loc || [],
        event: t.meta.search_event || [],
      });
    });
    return searchCache;
  }

  function applySearch() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(applySearch, 100); return; }
    var input = document.querySelector('#race-search input');
    if (!input) return;
    var q = input.value.trim().toLowerCase();
    var cache = ensureSearchCache(plot);
    var xs = [], ys = [], cds = [], indices = [];
    cache.forEach(function (c) {
      var x = c.x, y = c.y, cd = c.cd;
      if (q) {
        x = []; y = []; cd = c.cd ? [] : null;
        for (var i = 0; i < c.x.length; i++) {
          var loc = c.loc[i] || '', ev = c.event[i] || '';
          if (loc.indexOf(q) === -1 && ev.indexOf(q) === -1) continue;
          x.push(c.x[i]);
          y.push(c.y[i]);
          if (cd) cd.push(c.cd[i]);
        }
      }
      xs.push(x); ys.push(y); cds.push(cd); indices.push(c.idx);
    });
    if (!indices.length) return;
    // The plotly_restyle listener recomputes PRs over the masked pool.
    Plotly.restyle(plot, { x: xs, y: ys, customdata: cds }, indices);
  }

  function attachLegendInterceptor() {
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(attachLegendInterceptor, 100); return; }
    // Cancel clicks on the PR-legend sentinel (returning false stops Plotly
    // from toggling visibility AND prevents the legend marker from dimming).
    plot.on('plotly_legendclick', function (ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
    plot.on('plotly_legenddoubleclick', function (ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
    // plotly_restyle fires AFTER the visibility change is applied — both
    // for our own checkbox-driven restyles AND for Plotly's internal
    // legend-click toggles. To avoid recursion when recomputePRs itself
    // calls Plotly.restyle on the overlay, we identify our own restyles
    // by checking the event payload: a restyle scoped to ONLY the PR
    // overlay's trace index is ours, so skip. Anything else is a real
    // visibility change and we recompute.
    plot.on('plotly_restyle', function (eventData) {
      var indices = (eventData && eventData[1]) || null;
      var prIdx = findOverlayIdx(plot);
      if (indices && indices.length === 1 && indices[0] === prIdx) return;
      // Defer slightly so any in-flight Plotly state updates settle.
      setTimeout(function () { recomputePRs(); updateCount(); }, 0);
    });
    updateCount();
  }

  document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function (cb) {
    cb.addEventListener('change', update);
  });
  document.getElementById('bf-all').addEventListener('click', function () {
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function (cb) { cb.checked = true; });
    update();
  });
  document.getElementById('bf-none').addEventListener('click', function () {
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function (cb) { cb.checked = false; });
    update();
  });
  var searchInput = document.querySelector('#race-search input');
  if (searchInput) searchInput.addEventListener('input', applySearch);
  attachLegendInterceptor();
})();
