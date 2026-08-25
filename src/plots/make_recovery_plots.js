// Normalization sidebar for the recovery plot.
//
// Toggles normalization factors (temp, elevation, terrain, altitude,
// recent_effort, time_of_day, era, wind) and 3 visibility filters (bad conditions,
// non-solo, outliers). On each change, walks every recovery point's
// customdata, subtracts the selected factor contributions, and
// restyles both the pace panel and the residual panel. Trend lines
// are recomputed via a Gaussian rolling smoother to mirror the
// Python-side trend exactly (σ provided via window.__PLOT_TREND_SIGMA_DAYS),
// and the cursor tooltip's payload is re-synced to match (see syncTooltip).
//
// Hidden points get y=null + opacity=0 so hover hit-testing skips them
// AND the rolling-mean trend ignores them.
(function () {
  // ORDER MUST MATCH customdata channel order in Python (channels 0-7)
  var FACTOR_ORDER = ['temp', 'elevation', 'terrain', 'altitude',
                      'recent_effort', 'time_of_day', 'era', 'wind'];
  var ERA_INDEX = FACTOR_ORDER.indexOf('era');
  // Channel 8: mixed/trail descent-braking refund — an elevation×terrain
  // interaction with no checkbox of its own. Applied only when BOTH the
  // Elevation and Terrain toggles are on (it exists only because a route
  // both descends and is rough, so normalizing it requires removing both).
  var TERRAIN_DESCENT_INDEX = 8;

  var TREND_SIGMA_MS = window.__PLOT_TREND_SIGMA_DAYS * 86400000;
  var TREND_TRUNC_MS = 4 * TREND_SIGMA_MS;  // truncate kernel at 4σ
  var TREND_STEP_MS = 1 * 86400000;          // daily step
  var TREND_TWO_SIGSQ = 2 * TREND_SIGMA_MS * TREND_SIGMA_MS;
  var BASE_OPACITY = 0.6;

  function getPlot() { return document.querySelector('.plotly-graph-div'); }

  function findTraces(plot) {
    var idx = { pace: -1, residual: -1, trendPace: -1, trendResid: -1 };
    plot.data.forEach(function (t, i) {
      if (!t.meta) return;
      if (t.meta.role === 'pace') idx.pace = i;
      else if (t.meta.role === 'residual') idx.residual = i;
      else if (t.meta.role === 'trend_pace') idx.trendPace = i;
      else if (t.meta.role === 'trend_resid') idx.trendResid = i;
    });
    return idx;
  }

  function rollingTrend(dateMs, ys, mask) {
    // Gaussian-kernel smoother, σ = TREND_SIGMA_MS, truncated at ±4σ.
    // mask: optional boolean array. True = include this point in the trend.
    if (dateMs.length === 0) return { x: [], y: [] };
    var t0 = dateMs[0], t1 = dateMs[dateMs.length - 1];
    var trendX = [], trendY = [];
    var lo = 0, hi = 0;
    for (var t = t0; t <= t1; t += TREND_STEP_MS) {
      var lo_target = t - TREND_TRUNC_MS, hi_target = t + TREND_TRUNC_MS;
      while (lo < dateMs.length && dateMs[lo] < lo_target) lo++;
      while (hi < dateMs.length && dateMs[hi] <= hi_target) hi++;
      var sumWY = 0, sumW = 0, count = 0;
      for (var k = lo; k < hi; k++) {
        if (mask && !mask[k]) continue;
        if (ys[k] == null || isNaN(ys[k])) continue;
        var dt = dateMs[k] - t;
        var w = Math.exp(-(dt * dt) / TREND_TWO_SIGSQ);
        sumWY += ys[k] * w;
        sumW += w;
        count++;
      }
      if (count >= 5) {
        trendX.push(new Date(t));
        trendY.push(sumWY / sumW);
      }
    }
    return { x: trendX, y: trendY };
  }

  function update() {
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) { setTimeout(update, 100); return; }
    var idx = findTraces(plot);
    if (idx.pace < 0 || idx.residual < 0) return;

    var checked = {};
    document.querySelectorAll('#norm-filter input[type=checkbox]').forEach(function (cb) {
      checked[cb.dataset.factor] = cb.checked;
    });

    var paceTrace = plot.data[idx.pace];
    var residTrace = plot.data[idx.residual];
    var rawPace = paceTrace.meta.raw_y;
    var rawResid = residTrace.meta.raw_y;
    var dateMs = residTrace.meta.date_ms;
    var isBadCond = residTrace.meta.is_bad_cond;
    var isPartner = residTrace.meta.is_partner_run;
    var isOutlier = residTrace.meta.is_outlier;
    var custom = paceTrace.customdata;

    var hideBadCond = !!checked['hide_bad_cond'];
    var hidePartner = !!checked['hide_partner'];
    var hideOutlier = !!checked['hide_outlier'];

    var n = rawPace.length;
    var newPace = new Array(n);
    var newResid = new Array(n);
    var newOpacity = new Array(n);
    var visibleMask = new Array(n);
    for (var i = 0; i < n; i++) {
      var hidden = (hideBadCond && isBadCond[i]) ||
                   (hidePartner && isPartner[i]) ||
                   (hideOutlier && isOutlier[i]);
      if (hidden) {
        // null y suppresses both rendering AND hover hit-testing
        newPace[i] = null;
        newResid[i] = null;
        newOpacity[i] = 0;
        visibleMask[i] = false;
        continue;
      }
      var adjPace = 0, adjResid = 0;
      var c = custom[i];
      for (var j = 0; j < FACTOR_ORDER.length; j++) {
        if (!checked[FACTOR_ORDER[j]]) continue;
        if (j === ERA_INDEX) {
          adjResid += c[j];
        } else {
          adjPace += c[j];
          adjResid += c[j];
        }
      }
      // Descent-braking refund: gated on Elevation AND Terrain both on.
      if (checked['elevation'] && checked['terrain']) {
        adjPace += c[TERRAIN_DESCENT_INDEX];
        adjResid += c[TERRAIN_DESCENT_INDEX];
      }
      newPace[i] = rawPace[i] - adjPace;
      newResid[i] = rawResid[i] - adjResid;
      newOpacity[i] = BASE_OPACITY;
      visibleMask[i] = true;
    }

    Plotly.restyle(plot,
                   { y: [newPace, newResid],
                     'marker.opacity': [newOpacity, newOpacity] },
                   [idx.pace, idx.residual]);

    if (idx.trendPace >= 0 && idx.trendResid >= 0) {
      var tp = rollingTrend(dateMs, newPace, visibleMask);
      var tr = rollingTrend(dateMs, newResid, visibleMask);
      Plotly.restyle(plot, { x: [tp.x, tr.x], y: [tp.y, tr.y] },
                     [idx.trendPace, idx.trendResid]);
      syncTooltip(tp, tr, newResid, dateMs);
    }
  }

  // The cursor tooltip reads its rows from window.__TT_DATA, which Python bakes
  // once at first paint. Everything above moves the points and both trend lines,
  // so the payload has to move with them — otherwise the tooltip keeps reporting
  // the un-normalized, all-points numbers against a normalized chart (measured
  // drift with every factor on: median 7.5, max 66 sec/mi on the trend rows and
  // mean 11.2, max 68 on the per-run residual).
  //
  // Trend arrays are placed by DATE, not by position: rollingTrend emits only
  // the days whose window held >= 5 visible points, so a filter that thins a
  // stretch leaves real holes, and those days should read '—' in the tooltip
  // exactly as the line breaks on the chart.
  function syncTooltip(tp, tr, newResid, dateMs) {
    var P = window.__TT_DATA;
    if (!P) return;
    function toGrid(t, len) {
      var out = new Array(len);
      for (var i = 0; i < len; i++) out[i] = null;
      for (var k = 0; k < t.x.length; k++) {
        var g = Math.round(t.x[k].getTime() / 86400000) - P.first_day;
        if (g >= 0 && g < len) out[g] = t.y[k];
      }
      return out;
    }
    P.trend_pace = toGrid(tp, P.trend_pace.length);
    P.trend_resid = toGrid(tr, P.trend_resid.length);
    // Sessions are matched to trace points by day (recovery has at most one
    // run per day) rather than by index, so the two orderings can't silently
    // drift apart. A hidden point carries a null residual, which the tooltip
    // renders as '—'.
    var byDay = {};
    for (var i = 0; i < dateMs.length; i++) {
      byDay[Math.round(dateMs[i] / 86400000)] = i;
    }
    for (var j = 0; j < P.sessions.length; j++) {
      var pi = byDay[P.sessions[j].day];
      if (pi === undefined) continue;
      var v = newResid[pi];
      P.sessions[j].resid = (v == null || isNaN(v)) ? null : v;
    }
  }

  function setGroup(mode, on) {
    var sel = '#norm-filter input[data-mode="' + mode + '"]';
    document.querySelectorAll(sel).forEach(function (cb) { cb.checked = on; });
    update();
  }

  // Initial paint shows the no-normalization-applied scatter and trend
  // exactly as Python wrote them — no JS recompute needed. update() only
  // fires from now on when the user toggles a checkbox or button.

  document.querySelectorAll('#norm-filter input[type=checkbox]').forEach(function (cb) {
    cb.addEventListener('change', update);
  });
  // null-safe: a data-aware profile may omit the norm or filter section
  // entirely, so the corresponding All/None buttons won't exist.
  function bindClick(id, fn) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('click', fn);
  }
  bindClick('nf-norm-all', function () { setGroup('norm', true); });
  bindClick('nf-norm-none', function () { setGroup('norm', false); });
  bindClick('nf-hide-all', function () { setGroup('filter', true); });
  bindClick('nf-hide-none', function () { setGroup('filter', false); });

  // Tooltip rendering is handled by the smart spikeline scaffold (see
  // src/plotting/_scaffold/cursor_tooltip.js); this overlay only owns the
  // normalization sidebar and the plotly_restyle recompute loop.
})();
