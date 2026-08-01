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
  var TOUCH_SNAP_PX  = window.__TT_TOUCH_SNAP_PX || 44;  // fat-finger radius
  var TOUCH_SLOP_PX  = 24;   // taps this close to a subplot still resolve
  var TOUCH_GAP_PX   = 24;   // tooltip clearance above the touch point
  var ALWAYS_SNAP    = window.__TT_ALWAYS_SNAP === true;
  var SHOW_SPIKE     = window.__TT_SPIKE !== false;
  var SPIKE_FULL     = window.__TT_SPIKE_FULL_PLOT === true;
  var SPIKE_SNAP_DAY = window.__TT_SPIKE_SNAP_DAY === true;
  var range          = window.__TT_RANGE || { firstDay: -Infinity, lastDay: Infinity };

  var tt    = document.querySelector('.rp-tooltip');
  var spike = document.querySelector('.rp-spike');
  if (!tt) return;

  // Off-screen measurement clone. Width/height transitions on `tt` only fire
  // when the inline pixel value changes — never through `auto` (CSS doesn't
  // interpolate auto↔px) — so we measure on this hidden ghost and pin pixels
  // onto the outer `tt`, which then animates between sizes.
  //
  // The visible tooltip is two layers (see base.css): the outer `.rp-tooltip`
  // is the animating, clipped box (border + overflow:hidden + transition); the
  // inner `.tt-inner` holds the text. The ghost mirrors that — an outer clone
  // (keeps the `rp-tooltip` class so `.rp-tooltip .tt-inner` rules apply)
  // wrapping a `.tt-inner`. `measure()` (below) sizes the inner to its ACTUAL
  // post-wrap layout rather than the raw `max-content` width, so the box hugs
  // the widest rendered line — see the function for why that matters.
  var ghost = tt.cloneNode(false);
  ghost.style.cssText =
    'visibility:hidden;position:absolute;left:-9999px;top:0;' +
    'transition:none;display:block;width:auto;height:auto;overflow:visible;';
  var ghostInner = document.createElement('div');
  ghostInner.className = 'tt-inner';
  ghost.appendChild(ghostInner);
  document.body.appendChild(ghost);

  var rafScheduled = false;
  var lastContent  = '';
  // `place`: 'cursor' offsets the box beside the pointer (mouse); 'touch'
  // centers it above the touch point so a finger never covers it.
  var pending = { show: false, x: 0, y: 0, html: '', spikeX: 0, place: 'cursor' };
  var ttW = 0, ttH = 0;

  // Max width of a list of client rects (line boxes).
  function maxRectWidth(rects) {
    var w = 0;
    for (var i = 0; i < rects.length; i++) if (rects[i].width > w) w = rects[i].width;
    return w;
  }

  function isBlockish(el) {
    var d = getComputedStyle(el).display;
    return d === 'block' || d === 'list-item' || d === 'table' ||
           d.indexOf('flex') >= 0 || d.indexOf('grid') >= 0;
  }

  // Natural one-line width of a flex row (e.g. `.tt-row`). The row stretches to
  // the box (space-between), and its value is often a bare text node (an
  // anonymous flex item, invisible to `.children`) — so measure it shrink-wrapped
  // at `max-content`, which counts every flex item including the anonymous text.
  function flexRowWidth(row) {
    var pw = row.style.width, pd = row.style.display;
    row.style.display = 'inline-flex';
    row.style.width = 'max-content';
    var w = row.getBoundingClientRect().width;
    row.style.width = pw;
    row.style.display = pd;
    return w;
  }

  // Widest *rendered* line in `el`'s subtree, as currently laid out. A line can
  // span several inline fragments (text + <b> + <span>), so we measure whole
  // line boxes (Range.getClientRects over a block's inline contents) rather than
  // individual text nodes; flex rows are measured at their natural width; nested
  // block containers recurse. Used only when content overflows the cap (so it
  // has to wrap) — the common, fits-without-wrapping case uses `max-content`.
  function widestLine(el) {
    var d = getComputedStyle(el).display;
    if (d.indexOf('flex') >= 0 || d.indexOf('grid') >= 0) return flexRowWidth(el);
    var kids = el.children, hasBlockKid = false;
    for (var i = 0; i < kids.length; i++) if (isBlockish(kids[i])) { hasBlockKid = true; break; }
    var rng = document.createRange();
    if (!hasBlockKid) {            // pure inline content — measure its line boxes
      rng.selectNodeContents(el);
      return maxRectWidth(rng.getClientRects());
    }
    var w = 0;                     // mixed: recurse blocks, measure loose text runs
    for (var j = 0; j < kids.length; j++) w = Math.max(w, widestLine(kids[j]));
    for (var n = el.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3 && n.nodeValue && n.nodeValue.replace(/\s+/g, '')) {
        rng.selectNode(n);
        w = Math.max(w, maxRectWidth(rng.getClientRects()));
      }
    }
    return w;
  }

  // Size the tooltip for `html`. The box hugs its content: it is the content's
  // `max-content` width (which correctly accounts for bare text-node values and
  // multi-fragment lines) when that fits within `max-width`; otherwise the
  // content must wrap, and the box hugs the widest *rendered* line so it never
  // sits wider than its text (no dead space) and never clips a line. Clamped to
  // `[min-width, max-width]`. Returns the outer border-box dims (`w`,`h`), the
  // inner content-box width to pin on the visible tooltip (`innerW`), and the
  // diagnostics the layout test asserts on (`content`,`pad`,`min`,`cap`).
  function measure(html) {
    ghostInner.innerHTML = html;
    // Reset inline overrides from a prior call so computed style reflects the
    // stylesheet (incl. any per-plot `.tt-inner` head override, e.g. min-width).
    ghostInner.style.width = '';
    ghostInner.style.maxWidth = '';
    var cs  = getComputedStyle(ghostInner);
    var cap = parseFloat(cs.maxWidth); if (!isFinite(cap)) cap = 1e9;
    var min = parseFloat(cs.minWidth) || 0;
    var pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);

    // Pass 1: natural (unwrapped) width. `max-content` is reliable here — it
    // includes anonymous text-node flex-item values and full multi-fragment
    // lines that piecewise measurement would miss.
    ghostInner.style.maxWidth = 'none';
    ghostInner.style.width = 'max-content';
    var natural = Math.ceil(ghostInner.getBoundingClientRect().width);  // border-box

    var content, innerW;
    if (natural <= cap) {
      // Fits without wrapping — box = natural, so every line shows in full and
      // nothing clips. (min-width may floor it; that's the intended floor.)
      content = natural - pad;
      innerW  = Math.ceil(Math.min(Math.max(natural, min), cap));
    } else {
      // Overflows the cap → must wrap. Lay out at the cap, then hug the widest
      // rendered line so the box doesn't sit at the full cap with dead space.
      ghostInner.style.maxWidth = '';
      ghostInner.style.width = cap + 'px';
      content = widestLine(ghostInner);
      innerW  = Math.ceil(Math.min(Math.max(content + pad, min), cap));
    }

    ghostInner.style.maxWidth = '';
    ghostInner.style.width = innerW + 'px';
    var h = Math.ceil(ghostInner.getBoundingClientRect().height);
    // +2 = the outer `.rp-tooltip` 1px border on each side.
    return { w: innerW + 2, h: h, innerW: innerW, content: content, pad: pad, min: min, cap: cap };
  }
  // Exposed so the layout test can exercise sizing without a live Plotly plot.
  window.__RP_TT_MEASURE = measure;

  function paint() {
    rafScheduled = false;
    if (!pending.show) {
      tt.style.display = 'none';
      if (spike) spike.style.display = 'none';
      return;
    }
    if (pending.html !== lastContent) {
      var m = measure(pending.html);
      ttW = m.w;
      ttH = m.h;
      // Pin the visible inner to the measured content width so it wraps to the
      // same layout the ghost did. Left at `max-content`, the inner would size
      // to the unwrapped width and overflow (then clip inside) the outer box.
      // The outer animates from its previous size to (ttW, ttH), revealing /
      // clipping the fixed-width inner as it goes, so text never reflows.
      tt.innerHTML = '<div class="tt-inner" style="width:' + m.innerW + 'px">' + pending.html + '</div>';
      tt.style.width  = ttW + 'px';
      tt.style.height = ttH + 'px';
      lastContent = pending.html;
    }
    var x, y;
    if (pending.place === 'touch') {
      // Above the touch point, centered; flip below when clipped at the top.
      x = pending.x - ttW / 2;
      y = pending.y - ttH - TOUCH_GAP_PX;
      if (y < 0) y = pending.y + TOUCH_GAP_PX;
      if (x + ttW > window.innerWidth) x = window.innerWidth - ttW - 4;
      if (y + ttH > window.innerHeight) y = window.innerHeight - ttH - 4;
    } else {
      x = pending.x + 15;
      y = pending.y + 10;
      if (x + ttW > window.innerWidth)  x = pending.x - ttW - 15;
      if (y + ttH > window.innerHeight) y = window.innerHeight - ttH - 10;
    }
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

  function findNearestSnapPoint(pdiv, mouseX, mouseY, radiusPx) {
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
    return Math.sqrt(bestD2) <= radiusPx ? best : null;
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
  // `slopPx` (touch only; 0 on mouse for byte-identical desktop behavior)
  // lets a point in a gutter/margin resolve to the nearest subplot within
  // that distance — a fat finger just outside the axes still gets a tooltip.
  function findSubplotAt(pdiv, mx, my, slopPx) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var pairs = (fl._subplots && fl._subplots.cartesian) || ['xy'];
    var best = null, bestD = Infinity;
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
      var found = {
        xaxisId: match[1], yaxisId: match[2],
        xa: xa, ya: ya,
        left: xl, right: xr, top: yt, bottom: yb,
      };
      if (mx >= xl && mx <= xr && my >= yt && my <= yb) return found;
      if (slopPx) {
        // Distance from the point to the subplot rect (0 inside).
        var dx = mx < xl ? xl - mx : (mx > xr ? mx - xr : 0);
        var dy = my < yt ? yt - my : (my > yb ? my - yb : 0);
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d <= slopPx && d < bestD) { bestD = d; best = found; }
      }
    }
    return best;
  }

  // Plot-area bounds (top/bottom of the entire cartesian region, not a
  // single subplot). Used when SPIKE_FULL is set so the spike spans every
  // stacked panel.
  function plotAreaBounds(pdiv) {
    var fl = pdiv._fullLayout;
    var rect = pdiv.getBoundingClientRect();
    var sz = fl._size;
    return { top: rect.top + sz.t, bottom: rect.top + sz.t + sz.h };
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

  function hide() {
    pending.show = false;
    if (window.__RP_TT_STATE) window.__RP_TT_STATE.shown = false;
    schedule();
  }

  // The whole two-mode decision as a pure function of a client point —
  // shared verbatim by mousemove (isTouch=false, today's behavior) and the
  // tap path (isTouch=true: bigger snap radius, gutter slop, above-finger
  // placement).
  function showAt(pdiv, clientX, clientY, isTouch) {
    // A plot-specific zoom drag owns the cursor while this flag is set
    // (e.g. Misc Trends' drag-to-zoom band) — hide tooltip + spike.
    if (window.__rpZoomDragging) { hide(); return; }
    var sp = findSubplotAt(pdiv, clientX, clientY, isTouch ? TOUCH_SLOP_PX : 0);
    if (!sp) { hide(); return; }
    var place = isTouch ? 'touch' : 'cursor';
    var snap = findNearestSnapPoint(pdiv, clientX, clientY,
                                    isTouch ? TOUCH_SNAP_PX : SNAP_PX);
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
      if (!html) { hide(); return; }
      var sb = SPIKE_FULL ? plotAreaBounds(pdiv) : snapSp;
      pending.html        = html;
      pending.x           = snap.screenX;
      pending.y           = snap.screenY;
      pending.place       = place;
      pending.spikeX      = snap.screenX;
      pending.spikeTop    = sb.top;
      pending.spikeHeight = sb.bottom - sb.top;
      pending.show        = true;
      window.__RP_TT_STATE = { shown: true, isSnap: true, place: place };
      schedule();
      return;
    }
    if (ALWAYS_SNAP) { hide(); return; }
    var day = dayFromCursorX(sp, clientX);
    if (day == null) { hide(); return; }
    var ctx2 = { xaxisId: sp.xaxisId, yaxisId: sp.yaxisId };
    var html2 = callBuilder(day, false, null, ctx2);
    if (!html2) { hide(); return; }
    var sb2 = SPIKE_FULL ? plotAreaBounds(pdiv) : sp;
    // Day-quantized spike: place the line at the (clamped) day the
    // tooltip describes instead of the free cursor pixel.
    var spikeX = clientX;
    if (SPIKE_SNAP_DAY) {
      var dayPx = sp.xa.c2p(day * 86400000);
      if (dayPx != null && !isNaN(dayPx)) spikeX = sp.left + dayPx;
    }
    pending.html        = html2;
    pending.x           = clientX;
    pending.y           = clientY;
    pending.place       = place;
    pending.spikeX      = spikeX;
    pending.spikeTop    = sb2.top;
    pending.spikeHeight = sb2.bottom - sb2.top;
    pending.show        = true;
    window.__RP_TT_STATE = { shown: true, isSnap: false, place: place };
    schedule();
  }

  var bindTries = 0;
  function bind() {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (!pdiv || !pdiv._fullLayout) {
      // No plot on this page (e.g. the layout-test fixture) — give up after
      // ~10s instead of polling forever.
      if (++bindTries > 100) return;
      setTimeout(bind, 100);
      return;
    }

    pdiv.addEventListener('mousemove', function (e) {
      // Ignore the emulated mouse events that trail a touch — the tap path
      // already handled it (and a stray mousemove would move the tooltip
      // out from under a deliberate tap).
      if (window.rpTapHover && window.rpTapHover.mouseSuppressed()) return;
      showAt(pdiv, e.clientX, e.clientY, false);
    });

    pdiv.addEventListener('mouseleave', function () {
      if (window.rpTapHover && window.rpTapHover.mouseSuppressed()) return;
      hide();
    });

    // Tap acts as hover (see _scaffold/tap_hover.js: taps show, second tap
    // same spot / tap outside / swipe hides, and Plotly's touch zoom-box is
    // disabled on coarse-pointer devices).
    if (window.rpTapHover) {
      window.rpTapHover.bind(pdiv, {
        show: function (x, y) { showAt(pdiv, x, y, true); },
        hide: hide,
      });
    }

    // The tooltip/spike are position:fixed and placed from
    // getBoundingClientRect — both go stale the moment the page scrolls
    // (mobile scroll-mode pages), so just dismiss. Capture phase: in scroll
    // mode the scroller is <body>, and scroll events don't bubble.
    window.addEventListener('scroll', hide, { capture: true, passive: true });

    // Mobile layout engine re-rendered the figure (Plotly.newPlot) — axis
    // geometry changed wholesale.
    window.addEventListener('rp-layout-mode', function () {
      hide();
      if (window.Plotly) window.Plotly.Plots.resize(pdiv);
    });

    if (window.Plotly) window.Plotly.Plots.resize(pdiv);
  }

  bind();
  window.addEventListener('resize', function () {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (pdiv && window.Plotly) window.Plotly.Plots.resize(pdiv);
  });
})();
