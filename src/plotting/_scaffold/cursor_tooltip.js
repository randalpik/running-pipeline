/* Smart spikeline + tooltip scaffold.
 *
 * Two modes:
 *   - SMOOTH: cursor is far from any data point. Spike follows cursor X;
 *             tooltip is built from window.buildTooltip(day) — the same
 *             payload-driven smooth-mode renderer each plot supplies.
 *   - SNAP:   cursor is within window.__TT_SNAP_PX of a marker on a trace
 *             tagged meta.snap_eligible === true. Spike snaps to the
 *             point's x; tooltip = trace.customdata[i] (per-point HTML
 *             the plot pre-renders).
 *
 * Geography (and any other always-snap plot) sets window.__TT_ALWAYS_SNAP
 * = true; the scaffold then never falls back to smooth even if no point is
 * within threshold. (For geography, every cursor x maps to exactly one bin,
 * so snap is always well-defined.)
 *
 * Multi-panel support: each trace carries `xaxis` + `yaxis` ids ('x', 'x2',
 * etc.); we look up the corresponding _fullLayout.xaxis / xaxis2 / ... so
 * pixel positions are computed via the right subplot's c2p.
 */
(function () {
  var SNAP_PX        = window.__TT_SNAP_PX || 30;
  var ALWAYS_SNAP    = window.__TT_ALWAYS_SNAP === true;
  var SHOW_SPIKE     = window.__TT_SPIKE !== false;
  var range          = window.__TT_RANGE || { firstDay: -Infinity, lastDay: Infinity };

  var tt    = document.querySelector('.rp-tooltip');
  var spike = document.querySelector('.rp-spike');
  if (!tt) return;

  var rafScheduled = false;
  var lastContent  = '';
  var pending = { show: false, x: 0, y: 0, html: '', spikeX: 0 };
  var ttW = 0, ttH = 0;

  function paint() {
    rafScheduled = false;
    if (!pending.show) {
      tt.style.display = 'none';
      if (spike) spike.style.display = 'none';
      return;
    }
    if (pending.html !== lastContent) {
      tt.innerHTML = pending.html;
      lastContent = pending.html;
      ttW = tt.offsetWidth;
      ttH = tt.offsetHeight;
    }
    var x = pending.x + 15;
    var y = pending.y + 10;
    if (x + ttW > window.innerWidth)  x = pending.x - ttW - 15;
    if (y + ttH > window.innerHeight) y = window.innerHeight - ttH - 10;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    tt.style.display = 'block';
    if (spike && SHOW_SPIKE && pending.spikeX != null) {
      spike.style.transform = 'translateX(' + pending.spikeX + 'px)';
      // Bound the spike vertically to the active subplot, so multi-panel
      // plots don't draw a line straight across panels they don't apply
      // to. Single-panel plots get a spike covering the whole plot area.
      if (pending.spikeTop != null && pending.spikeHeight != null) {
        spike.style.top = pending.spikeTop + 'px';
        spike.style.height = pending.spikeHeight + 'px';
      } else {
        spike.style.top = '0px';
        spike.style.height = '100vh';
      }
      spike.style.display = 'block';
    }
  }
  function schedule() {
    if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(paint); }
  }

  // Convert plotly trace xaxis id ('x', 'x2', ...) to layout key.
  function axisKey(prefix, traceId) {
    if (!traceId || traceId === prefix) return prefix + 'axis';
    return prefix + 'axis' + traceId.slice(1);
  }

  function asArray(v) {
    if (v == null) return null;
    if (Array.isArray(v)) return v;
    if (v._inputArray && typeof v._inputArray.length === 'number') return v._inputArray;
    if (typeof v.length === 'number') return v;
    return null;
  }

  function numericize(v) {
    if (v instanceof Date) return v.getTime();
    if (typeof v === 'string') {
      var t = new Date(v).getTime();
      return isNaN(t) ? v : t;
    }
    return v;
  }

  // Snap-HTML lookup: prefer trace.customdata[i] when it's a string;
  // otherwise fall back to trace.text[i]. This lets plots reserve
  // customdata for structured per-point data (e.g. recovery's per-point
  // factor-contribution array) while still supplying snap HTML via text.
  function snapHtmlForPoint(t, i) {
    var cd = t.customdata;
    if (cd) {
      var cv = cd[i];
      if (typeof cv === 'string') return cv;
    }
    var ts = t.text;
    if (Array.isArray(ts)) return ts[i];
    if (typeof ts === 'string') return ts;
    return null;
  }

  function findNearestSnapPoint(pdiv, mouseX, mouseY) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var bestD2 = Infinity, best = null;
    var traces = pdiv.data;
    for (var ti = 0; ti < traces.length; ti++) {
      var t = traces[ti];
      if (!t.meta || t.meta.snap_eligible !== true) continue;
      var v = t.visible;
      if (v === false || v === 'legendonly') continue;
      var xs = asArray(t.x), ys = asArray(t.y);
      if (!xs || !ys) continue;
      var xa = fl[axisKey('x', t.xaxis)];
      var ya = fl[axisKey('y', t.yaxis)];
      if (!xa || !ya || !xa.c2p) continue;
      var n = xs.length;
      for (var i = 0; i < n; i++) {
        var y = ys[i];
        if (y == null || isNaN(y)) continue;
        var xv = numericize(xs[i]);
        var pxX = xa.c2p(xv);
        var pxY = ya.c2p(y);
        if (pxX == null || pxY == null || isNaN(pxX) || isNaN(pxY)) continue;
        var sx = rect.left + xa._offset + pxX;
        var sy = rect.top  + ya._offset + pxY;
        var dx = sx - mouseX, dy = sy - mouseY;
        var d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
          bestD2 = d2;
          best = {
            traceIdx: ti, pointIdx: i,
            screenX: sx, screenY: sy,
            html: snapHtmlForPoint(t, i),
          };
        }
      }
    }
    if (!best) return null;
    if (ALWAYS_SNAP) return best;
    return Math.sqrt(bestD2) <= SNAP_PX ? best : null;
  }

  function inPlotBounds(pdiv, e) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var sz = fl._size;
    var pl = rect.left + sz.l;
    var pr = rect.left + sz.l + sz.w;
    var pt = rect.top  + sz.t;
    var pb = rect.top  + sz.t + sz.h;
    return e.clientX >= pl && e.clientX <= pr &&
           e.clientY >= pt && e.clientY <= pb;
  }

  // Return the cartesian subplot whose pixel rect contains (mx, my), or
  // null. Pulls subplot pairs from fl._subplots.cartesian (e.g. ['xy',
  // 'x2y2', ...]) when present; falls back to single-panel ('xy') so
  // single-panel plots get their plot-area bounds as the "subplot"
  // (used to clip the spike to the plot area instead of the viewport).
  function findSubplotAt(pdiv, mx, my) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var pairs = (fl._subplots && fl._subplots.cartesian) || ['xy'];
    for (var i = 0; i < pairs.length; i++) {
      var sp = pairs[i];
      var match = sp.match(/^(x\d*)(y\d*)$/);
      if (!match) continue;
      var xaKey = 'xaxis' + (match[1].slice(1) || '');
      var yaKey = 'yaxis' + (match[2].slice(1) || '');
      var xa = fl[xaKey], ya = fl[yaKey];
      if (!xa || !ya || xa._length == null || ya._length == null) continue;
      var xl = rect.left + xa._offset;
      var xr = xl + xa._length;
      var yt = rect.top  + ya._offset;
      var yb = yt + ya._length;
      if (mx >= xl && mx <= xr && my >= yt && my <= yb) {
        return {
          xaxisId: match[1], yaxisId: match[2],
          xa: xa, ya: ya,
          left: xl, right: xr, top: yt, bottom: yb,
        };
      }
    }
    return null;
  }

  function subplotForTrace(pdiv, trace) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var xaKey = axisKey('x', trace.xaxis);
    var yaKey = axisKey('y', trace.yaxis);
    var xa = fl[xaKey], ya = fl[yaKey];
    if (!xa || !ya) return null;
    return {
      xaxisId: trace.xaxis || 'x',
      yaxisId: trace.yaxis || 'y',
      xa: xa, ya: ya,
      left:   rect.left + xa._offset,
      right:  rect.left + xa._offset + xa._length,
      top:    rect.top  + ya._offset,
      bottom: rect.top  + ya._offset + ya._length,
    };
  }

  function dayFromCursorX(sp, mouseX) {
    if (!sp || !sp.xa || !sp.xa.p2c) return null;
    var dataX = sp.xa.p2c(mouseX - sp.left);
    var dayIdx = Math.round(dataX / 86400000);
    if (dayIdx < range.firstDay) dayIdx = range.firstDay;
    if (dayIdx > range.lastDay)  dayIdx = range.lastDay;
    return dayIdx;
  }

  function dayFromMs(ms) {
    return Math.round(numericize(ms) / 86400000);
  }

  function callBuilder(day, isSnap, pointHtml, ctx) {
    if (typeof buildTooltip !== 'function') return '';
    try {
      // buildTooltip(day, isSnap, pointHtml, ctx) — ctx carries the
      // active subplot's xaxisId/yaxisId for multi-panel plots that
      // need per-panel content. Older single-arg builders still work.
      return buildTooltip(day, isSnap, pointHtml, ctx) || '';
    } catch (err) { return ''; }
  }

  function bind() {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (!pdiv || !pdiv._fullLayout) { setTimeout(bind, 100); return; }

    pdiv.addEventListener('mousemove', function (e) {
      var sp = findSubplotAt(pdiv, e.clientX, e.clientY);
      if (!sp) {
        pending.show = false;
        schedule();
        return;
      }
      var snap = findNearestSnapPoint(pdiv, e.clientX, e.clientY);
      if (snap) {
        var t = pdiv.data[snap.traceIdx];
        var xs = asArray(t.x);
        var snapDay = (xs ? dayFromMs(xs[snap.pointIdx]) : null);
        if (snapDay == null && range.firstDay !== -Infinity) snapDay = range.firstDay;
        // The snap point belongs to its trace's subplot, which may
        // differ from the cursor's subplot if the mouse hovered into a
        // neighbour while still snapping — use the trace's subplot for
        // spike-bound calculation so the line stays where the point is.
        var snapSp = subplotForTrace(pdiv, t) || sp;
        var ctx = { xaxisId: snapSp.xaxisId, yaxisId: snapSp.yaxisId };
        var html = callBuilder(snapDay, true, snap.html, ctx);
        if (!html) { pending.show = false; schedule(); return; }
        pending.html        = html;
        pending.x           = snap.screenX;
        pending.y           = snap.screenY;
        pending.spikeX      = snap.screenX;
        pending.spikeTop    = snapSp.top;
        pending.spikeHeight = snapSp.bottom - snapSp.top;
        pending.show        = true;
        schedule();
        return;
      }
      if (ALWAYS_SNAP) {
        pending.show = false;
        schedule();
        return;
      }
      var day = dayFromCursorX(sp, e.clientX);
      if (day == null) { pending.show = false; schedule(); return; }
      var ctx = { xaxisId: sp.xaxisId, yaxisId: sp.yaxisId };
      var html = callBuilder(day, false, null, ctx);
      if (!html) { pending.show = false; schedule(); return; }
      pending.html        = html;
      pending.x           = e.clientX;
      pending.y           = e.clientY;
      pending.spikeX      = e.clientX;
      pending.spikeTop    = sp.top;
      pending.spikeHeight = sp.bottom - sp.top;
      pending.show        = true;
      schedule();
    });

    pdiv.addEventListener('mouseleave', function () {
      pending.show = false;
      schedule();
    });

    if (window.Plotly) window.Plotly.Plots.resize(pdiv);
  }

  bind();
  window.addEventListener('resize', function () {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (pdiv && window.Plotly) window.Plotly.Plots.resize(pdiv);
  });
})();
