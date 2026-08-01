// Geography plot interactions:
//   - bar pixel snap (uniform per-mode pixel gap)
//   - year/month mode toggle
//   - custom legend (visibility + group expand)
//   - bar-aligned tooltip + spikeline on mousemove
//
// Uses window.__PLOT_GEO for all per-mode data (bins, hover_html,
// tickvals, ticktext, gap_px). Mode state is local to this IIFE.
(function () {
  var GEO = window.__PLOT_GEO;
  var mode = 'year';

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  // ===== Bar pixel snap (uniform per-mode pixel gap) =====
  var BAR_PATH_RE = /^M([-\d.]+),([-\d.]+)V([-\d.]+)H([-\d.]+)V([-\d.]+)Z$/;

  function snapBars() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var fl = gd._fullLayout;
    var bg = fl._size;
    var plotW = bg.w;
    var nBins = GEO[mode].bins.length;
    if (nBins === 0 || plotW <= 0) return;
    var pitch = plotW / nBins;
    var gap = GEO[mode].gap_px;

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
      var center = (x1 + x2) / 2;
      var binIdx = Math.round(center / pitch - 0.5);
      if (binIdx < 0) binIdx = 0;
      if (binIdx >= nBins) binIdx = nBins - 1;
      var newLeft = Math.round(binIdx * pitch);
      var newRight = Math.round((binIdx + 1) * pitch) - gap;
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
    var plotW = gd._fullLayout._size.w;
    var pitch = plotW / nBins;
    var idx = Math.floor(plotPx / pitch);
    if (idx < 0) idx = 0;
    if (idx >= nBins) idx = nBins - 1;
    return idx;
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

      var pitch = bg.w / GEO[mode].bins.length;
      var binCenterPx = pl + (binIdx + 0.5) * pitch;
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
