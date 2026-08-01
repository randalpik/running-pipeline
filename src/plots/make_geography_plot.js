// Geography plot interactions:
//   - bar pixel snap (uniform slot-width-derived pixel gap)
//   - year/month mode toggle
//   - custom legend (visibility + group expand)
//   - bar-aligned tooltip + spikeline on mousemove
//
// Uses window.__PLOT_GEO for all per-mode data (bins, hover_html,
// tickvals, ticktext). Mode state is local to this IIFE.
(function () {
  var GEO = window.__PLOT_GEO;
  var mode = 'year';

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  // ===== Bin geometry, read from the LIVE axis =====
  // All in plot-area pixels — the space bar-path `d` coordinates and
  // _size.l offsets share. Two traps this exists to avoid:
  //
  //  1. plotWidth / binCount only equals the bin pitch when the WHOLE range
  //     is on screen. Zooming broke everything derived from it: bars were
  //     re-snapped onto the unzoomed grid (huge gaps, mangled bars) and the
  //     tooltip resolved the wrong bin.
  //  2. The axis TYPE differs per mode — yearly is a category axis
  //     ('2016' etc.), but monthly x-values ('2016-01') make Plotly infer a
  //     DATE axis. c2p() takes an index on one and epoch-ms on the other,
  //     so bin centres have to be resolved per type.
  var geomCache = null;
  function binGeom(xa) {
    var bins = GEO[mode].bins;
    var key = mode + '|' + xa.type + '|' + xa.range.join(',') + '|' + xa._length;
    if (geomCache && geomCache.key === key) return geomCache.g;
    var n = bins.length;
    var centers = new Array(n);
    for (var i = 0; i < n; i++) {
      centers[i] = (xa.type === 'category') ? xa.c2p(i) : xa.d2p(bins[i]);
    }
    var pitch = n > 1 ? (centers[n - 1] - centers[0]) / (n - 1) : (xa._length || 1);
    // Integer slot boundaries, computed ONCE and shared by the two bars that
    // meet at each one. Rounding a shared edge independently from either side
    // (cx+pitch/2 vs cx−pitch/2) disagrees whenever it lands on a .5 tie —
    // e.g. bin 33 at centre 261.4, pitch 7.8, edge exactly 257.5 — which made
    // the gap alternate 0/1px across the monthly view.
    var edges = new Array(n + 1);
    for (var j = 0; j < n; j++) edges[j] = Math.round(centers[j] - pitch / 2);
    edges[n] = Math.round(centers[n - 1] + pitch / 2);
    var g = { centers: centers, pitch: pitch, edges: edges };
    geomCache = { key: key, g: g };
    return g;
  }
  function binCenters(xa) { return binGeom(xa).centers; }
  function binAtPx(xa, plotPx) {
    var c = binCenters(xa);
    var best = -1, bestD = Infinity;
    for (var i = 0; i < c.length; i++) {
      var d = Math.abs(c[i] - plotPx);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // Gap between neighbouring bars, chosen from that slot width rather than
  // per mode: a hairline keeps ~7px monthly bars from merging, while wide
  // bars (the yearly view, or anything zoomed in) can afford real air. One
  // rule for both modes and every zoom level, so the spacing always looks
  // deliberate instead of jumping when you switch views.
  var GAP_STEPS = [[30, 1], [100, 2]];   // [slot width below…, gap]
  var GAP_WIDE = 4;                      // …and at or above the last step
  function gapForPitch(pitch) {
    for (var i = 0; i < GAP_STEPS.length; i++) {
      if (pitch < GAP_STEPS[i][0]) return GAP_STEPS[i][1];
    }
    return GAP_WIDE;
  }

  // ===== Bar pixel snap =====
  var BAR_PATH_RE = /^M([-\d.]+),([-\d.]+)V([-\d.]+)H([-\d.]+)V([-\d.]+)Z$/;

  function snapBars() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var fl = gd._fullLayout;
    var xa = fl.xaxis;
    var plotW = fl._size.w;
    var nBins = GEO[mode].bins.length;
    if (nBins === 0 || plotW <= 0 || !xa || !xa.c2p) return;
    var geom = binGeom(xa);
    var gap = gapForPitch(geom.pitch);
    if (gap > geom.pitch - 1) gap = Math.max(0, Math.floor(geom.pitch - 1));

    var paths = gd.querySelectorAll('.barlayer .point path');
    paths.forEach(function (path) {
      var d = path.getAttribute('d');
      if (!d) return;
      var m = d.match(BAR_PATH_RE);
      if (!m) return;
      var x1 = parseFloat(m[1]);
      var y1 = parseFloat(m[2]);
      var y2 = parseFloat(m[3]);
      var x2 = parseFloat(m[4]);
      if (Math.abs(y1 - y2) < 0.5) return;
      // Plotly binds each bar's bin to the path; fall back to hit-testing
      // its centre. Either way the slot comes from the axis, so this holds
      // at any zoom without a separate zoomed/unzoomed code path.
      var datum = path.__data__;
      var idx = (datum && typeof datum.p === 'number' && datum.p === (datum.p | 0))
        ? datum.p : binAtPx(xa, (x1 + x2) / 2);
      if (idx < 0 || idx >= nBins) return;
      var newLeft = geom.edges[idx];
      var newRight = geom.edges[idx + 1] - gap;
      if (newRight <= newLeft) newRight = newLeft + 1;
      var newD = 'M' + newLeft + ',' + y1 + 'V' + y2 + 'H' + newRight + 'V' + y1 + 'Z';
      path.setAttribute('d', newD);
    });
  }

  // ===== Mode toggle =====
  function applyMode() {
    var gd = pdiv();
    if (!gd || !gd.data) return;
    var n = gd.data.length;
    var x = [], y = [];
    for (var i = 0; i < n; i++) {
      var m = gd.data[i].meta || {};
      x.push(mode === 'year' ? m.x_year : m.x_month);
      y.push(mode === 'year' ? m.y_year : m.y_month);
    }
    Plotly.restyle(gd, { x: x, y: y });
    var modeData = GEO[mode];
    Plotly.relayout(gd, {
      // Pin the axis CATEGORICAL on every mode switch. Monthly x-values
      // ('2016-01') otherwise make Plotly re-type the axis to `date`, and a
      // date axis spaces bars by real month length (28-31 days) while giving
      // them all one width — leaving 0-3 days of slack that varies month to
      // month. Invisible at full range, but zooming magnified it into
      // visibly ragged gaps (measured 1-7px), which is the very problem the
      // pixel snapping below was invented to hide. As categories the slots
      // are uniform by construction at any zoom, and the array ticks (whose
      // values ARE the category names) line up exactly with the bars.
      'xaxis.type': 'category',
      'xaxis.tickvals': modeData.tickvals,
      'xaxis.ticktext': modeData.ticktext,
    });
  }

  document.querySelectorAll('#geo-toggle .rp-btn-pill').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var newMode = btn.getAttribute('data-value');
      if (newMode === mode) return;
      mode = newMode;
      document.querySelectorAll('#geo-toggle .rp-btn-pill').forEach(function (b) {
        b.classList.toggle('is-active', b === btn);
      });
      applyMode();
    });
  });

  // ===== Legend interaction =====
  function setVisible(indices, visible) {
    var gd = pdiv();
    if (!gd) return;
    Plotly.restyle(gd, { visible: visible ? true : 'legendonly' }, indices);
  }

  document.querySelectorAll('#geo-legend .legend-item, #geo-legend .legend-singleton').forEach(function (el) {
    el.addEventListener('click', function () {
      var idx = parseInt(el.getAttribute('data-trace-idx'));
      var hidden = el.classList.contains('hidden');
      setVisible([idx], hidden);
      el.classList.toggle('hidden', !hidden);
    });
  });

  // Chevron click toggles expand/collapse of the next-sibling subgroup.
  // stopPropagation so the row's visibility-toggle handler doesn't fire.
  document.querySelectorAll('#geo-legend .chev.expandable').forEach(function (chev) {
    chev.addEventListener('click', function (e) {
      e.stopPropagation();
      var item = chev.closest('.legend-item');
      if (!item) return;
      var sub = item.nextElementSibling;
      if (!sub || !sub.classList.contains('legend-subgroup')) return;
      var willExpand = !sub.classList.contains('expanded');
      sub.classList.toggle('expanded', willExpand);
      chev.classList.toggle('expanded', willExpand);
    });
  });

  document.querySelectorAll('#geo-legend .legend-title').forEach(function (el) {
    el.addEventListener('click', function () {
      var groupDiv = el.parentElement;
      var items = groupDiv.querySelectorAll('.legend-item');
      var allHidden = Array.from(items).every(function (it) {
        return it.classList.contains('hidden');
      });
      var visible = allHidden;
      var indices = [];
      items.forEach(function (it) {
        indices.push(parseInt(it.getAttribute('data-trace-idx')));
      });
      setVisible(indices, visible);
      items.forEach(function (it) {
        it.classList.toggle('hidden', !visible);
      });
    });
  });

  // ===== Custom hover =====
  function findBinFromPx(plotPx) {
    var nBins = GEO[mode].bins.length;
    if (nBins === 0) return -1;
    var gd = pdiv();
    var xa = gd && gd._fullLayout && gd._fullLayout.xaxis;
    if (!xa || !xa.c2p) return -1;
    return binAtPx(xa, plotPx);   // zoom- and axis-type-aware (see above)
  }

  function bindHover() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) { setTimeout(bindHover, 100); return; }
    var tt = document.getElementById('geo-tooltip');
    var spike = document.getElementById('geo-spike');
    var TOUCH_SLOP_PX = 24;   // taps just outside the bars still resolve
    var TOUCH_GAP_PX = 24;    // tooltip clearance above the touch point

    function hideTT() {
      tt.style.display = 'none';
      spike.style.display = 'none';
    }

    // Same two-mode point handler for mouse and tap. Bin-indexed
    // always-snap by construction — every plot-area x maps to one bin.
    function showAt(clientX, clientY, isTouch) {
      var fl = gd._fullLayout;
      if (!fl) return;
      var rect = gd.getBoundingClientRect();
      var bg = fl._size;
      var pl = rect.left + bg.l, pr = rect.left + bg.l + bg.w;
      var pt = rect.top + bg.t, pb = rect.top + bg.t + bg.h;
      if (isTouch) {
        // A fat finger at the first/last bar's edge lands just outside the
        // plot area — clamp within the slop instead of hiding.
        if (clientX >= pl - TOUCH_SLOP_PX && clientX <= pr + TOUCH_SLOP_PX) {
          clientX = Math.min(Math.max(clientX, pl), pr);
        }
        if (clientY >= pt - TOUCH_SLOP_PX && clientY <= pb + TOUCH_SLOP_PX) {
          clientY = Math.min(Math.max(clientY, pt), pb);
        }
      }
      if (clientX < pl || clientX > pr ||
          clientY < pt || clientY > pb) {
        hideTT();
        return;
      }
      var plotPx = clientX - pl;
      var binIdx = findBinFromPx(plotPx);
      if (binIdx < 0 || binIdx >= GEO[mode].hover_html.length) {
        hideTT();
        return;
      }
      var html = GEO[mode].hover_html[binIdx];
      if (!html) {
        hideTT();
        return;
      }
      tt.innerHTML = html;
      tt.style.display = 'block';
      var ttW = tt.offsetWidth, ttH = tt.offsetHeight;
      var x, y;
      if (isTouch) {
        // Above the touch point, centered, so the finger never covers it.
        x = clientX - ttW / 2;
        y = clientY - ttH - TOUCH_GAP_PX;
        if (y < 0) y = clientY + TOUCH_GAP_PX;
        if (x + ttW > window.innerWidth) x = window.innerWidth - ttW - 4;
        if (x < 0) x = 0;
        if (y + ttH > window.innerHeight) y = window.innerHeight - ttH - 4;
      } else {
        x = clientX + 14;
        y = clientY + 12;
        if (x + ttW > window.innerWidth)  x = clientX - ttW - 14;
        if (y + ttH > window.innerHeight) y = window.innerHeight - ttH - 10;
      }
      tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';

      // Bar centre from the live axis, so the spike tracks the zoom.
      var binCenterPx = pl + binCenters(fl.xaxis)[binIdx];
      spike.style.transform = 'translateX(' + binCenterPx + 'px)';
      spike.style.display = 'block';
    }

    gd.addEventListener('mousemove', function (e) {
      if (window.rpTapHover && window.rpTapHover.mouseSuppressed()) return;
      showAt(e.clientX, e.clientY, false);
    });

    gd.addEventListener('mouseleave', function () {
      if (window.rpTapHover && window.rpTapHover.mouseSuppressed()) return;
      hideTT();
    });

    if (window.rpTapHover) {
      window.rpTapHover.bind(gd, {
        show: function (x, y) { showAt(x, y, true); },
        hide: hideTT,
      });
    }

    function bindAfterplot() {
      if (gd.on) {
        gd.on('plotly_afterplot', function () {
          requestAnimationFrame(snapBars);
        });
      }
    }
    bindAfterplot();
    // The mobile layout engine's Plotly.newPlot (margin.r patch) clears
    // gd.on bindings — re-bind and re-snap.
    window.addEventListener('rp-layout-mode', function () {
      bindAfterplot();
      requestAnimationFrame(snapBars);
    });
    requestAnimationFrame(snapBars);
    requestAnimationFrame(snapBars);
  }

  bindHover();
})();
