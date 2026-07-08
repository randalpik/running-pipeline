// Misc. Trends page toggle (Weather / Other), per-subplot inset labels,
// custom x-only drag-to-zoom, and the client-side gradient re-raster.
//
// Each page is its OWN complete figure (a clean make_subplots with exactly its
// panel count), serialized into window.__PLOT_TRENDS_FIGS. Switching pages
// Plotly.newPlot's the target figure into the single plot div — a fresh render,
// so the swapped-in page's gridlines lay down exactly like a normal page load.
// We do NOT resize axes / toggle domains at runtime: plotly.js does not redraw
// a y-axis's gridlines when its domain changes, which blanked the grid in every
// shared-figure approach. window.__rpActiveTab tells the cursor tooltip which
// tab's rows to render.
//
// The cursor-tooltip scaffold binds mousemove/mouseleave to the plot DIV (not
// via gd.on), so those listeners survive newPlot and keep working against the
// new figure's live state. Only the inset re-positioning uses gd.on, which
// newPlot clears — so we re-bind it after each swap.
//
// Inset labels are HTML overlay divs (.rp-inset), each tagged data-rp-row /
// data-rp-page / data-rp-key, positioned at their subplot's top-left from the
// live Plotly layout. The standalone single-panel pages load this same file:
// there is no #trends-toggle and no __PLOT_TRENDS_FIGS there, so only the inset
// positioning and drag-to-zoom run (and the lone inset always shows).
//
// Drag-to-zoom: the page stays staticPlot (Plotly owns NO mouse events — the
// custom tooltip/spike/toggle depend on that), so zoom is custom too. A
// mousedown+drag inside the plot area draws the .rp-zoomband selection;
// mouseup issues a programmatic Plotly.relayout of the master x-axis range
// (programmatic calls work under staticPlot — it only disables user input).
// shared_xaxes makes the bottom row the master (matches=null); the other rows
// follow via matches. Ticks/gridlines need NO handling here: every x-axis is
// tickmode='auto' (auto_date_x_axis_kwargs), so density re-derives from the
// current range on every relayout — the same rule at first render (years for
// a decade profile, months for a sub-year one) and at any zoom depth.
// window.__rpZoomDragging tells the tooltip scaffold to hide during a drag.
// Reset: double-click in the plot area, or the .rp-zoom-reset pill.
//
// Client-side re-raster: the baked gradient PNGs are 3840px wide — fine at the
// full home range (a downscale) but blurry/stairstepped the moment a zoom
// stretches them. So every zoom REDRAWS each envelope panel's gradient at
// native display resolution from window.__PLOT_RASTER_DATA (per-day env edges
// + ramp anchors / per-day solar minute-anchors, all computed in Python — no
// modeling here): an offscreen canvas fills the panel rect with the same
// piecewise-linear ramp (one linear gradient; per-day-column gradients for
// solar), masks it to the band polygon between the lo/hi day polylines with
// destination-in (canvas AA = the area-coverage AA the baked raster
// supersamples for), and the result replaces images[img] source/x/sizex in
// the SAME relayout as the range change. The white trend lines stay Plotly
// vector shapes above the image. Reset restores the baked sources snapshotted
// per page before any zoom mutates them.
(function () {
  var FIGS = window.__PLOT_TRENDS_FIGS || {};
  var CONFIG = window.__PLOT_TRENDS_CONFIG ||
               { responsive: true, displayModeBar: false, staticPlot: true };
  var RD = window.__PLOT_RASTER_DATA || null;
  var page = window.__rpActiveTab || 'weather';
  window.__rpActiveTab = page;

  var INSET_PAD = 6;      // px gap from the subplot's top-left corner
  var MIN_DRAG_PX = 5;    // click-vs-drag gate
  var MIN_SPAN_MS = 3 * 86400000;  // reject degenerate (<3 day) zooms
  var DAY_MS = 86400000;
  var MAX_DPR = 2;        // cap canvas density (4K laptops; cost, not quality)

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  // Place each visible inset at the top-left of its subplot row, read from the
  // live _fullLayout axis offsets.
  function positionInsets() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var fl = gd._fullLayout;
    var rect = gd.getBoundingClientRect();
    document.querySelectorAll('.rp-inset').forEach(function (el) {
      if (el.style.display === 'none') return;
      var row = el.getAttribute('data-rp-row') || '1';
      var n = (row === '1') ? '' : row;
      var xa = fl['xaxis' + n];
      var ya = fl['yaxis' + n];
      if (!xa || !ya || xa._offset == null || ya._offset == null) return;
      el.style.left = (rect.left + xa._offset + INSET_PAD) + 'px';
      el.style.top = (rect.top + ya._offset + INSET_PAD) + 'px';
    });
  }

  function showInsetsForPage() {
    document.querySelectorAll('.rp-inset').forEach(function (el) {
      var p = el.getAttribute('data-rp-page');
      el.style.display = (!p || p === page) ? '' : 'none';
    });
  }

  // newPlot clears gd.on handlers, so (re-)bind the inset re-positioning after
  // every render. The cursor-tooltip listeners are on the div element and
  // survive newPlot, so they need no re-bind.
  function bindRedraw(gd) {
    if (gd && typeof gd.on === 'function') {
      gd.on('plotly_afterplot', positionInsets);
    }
  }

  // ---- client-side gradient re-raster ------------------------------------

  var origImages = {};    // page -> [{source, x, sizex}] baked snapshots

  function hexRgb(h) {
    h = String(h).replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
            parseInt(h.slice(4, 6), 16)];
  }

  function rgbCss(c) { return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  // Piecewise-linear colour at `v` over [(value, '#hex'), ...] anchors,
  // clamped at the ends — the JS twin of Python's gradient_at.
  function colorAt(anchors, v) {
    if (v <= anchors[0][0]) return anchors[0][1];
    var last = anchors[anchors.length - 1];
    if (v >= last[0]) return last[1];
    for (var i = 0; i < anchors.length - 1; i++) {
      var v0 = anchors[i][0], v1 = anchors[i + 1][0];
      if (v >= v0 && v <= v1) {
        var t = v1 > v0 ? (v - v0) / (v1 - v0) : 0;
        var c0 = hexRgb(anchors[i][1]), c1 = hexRgb(anchors[i + 1][1]);
        return rgbCss([Math.round(c0[0] + (c1[0] - c0[0]) * t),
                       Math.round(c0[1] + (c1[1] - c0[1]) * t),
                       Math.round(c0[2] + (c1[2] - c0[2]) * t)]);
      }
    }
    return last[1];
  }

  // Vertical gradient for an anchor-ramp panel: exact colours at the canvas
  // top/bottom (the anchors may extend past the y-range) plus every interior
  // anchor as a stop. Same piecewise-linear ramp as the baked raster.
  function rampGradient(ctx, anchors, y0, y1, H) {
    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, colorAt(anchors, y1));
    for (var i = 0; i < anchors.length; i++) {
      var off = (y1 - anchors[i][0]) / (y1 - y0);
      if (off > 0 && off < 1) g.addColorStop(off, anchors[i][1]);
    }
    g.addColorStop(1, colorAt(anchors, y0));
    return g;
  }

  // One day's solar gradient from its minute anchors [twilight_begin,
  // sunrise, blue_begin, noon, blue_end, sunset, twilight_end] (nulls for
  // polar-ish missing events) — the JS twin of solar_ramp_from_anchors.
  function solarGradient(ctx, row, y0, y1, H) {
    var C = RD.solar_colors;
    var noon = row[3] == null ? 720 : row[3];
    var pts = [[0, C.night]];
    if (row[0] != null) pts.push([row[0], C.twilight]);
    if (row[1] != null) pts.push([row[1], C.horizon]);
    if (row[2] != null) pts.push([row[2], C.noon]);
    pts.push([noon, C.noon]);
    if (row[4] != null) pts.push([row[4], C.noon]);
    if (row[5] != null) pts.push([row[5], C.horizon]);
    if (row[6] != null) pts.push([row[6], C.twilight]);
    pts.push([1440, C.night]);
    var g = ctx.createLinearGradient(0, 0, 0, H);
    var last = -1;
    for (var i = 0; i < pts.length; i++) {
      var m = Math.max(pts[i][0], last + 1e-4);   // strictly increasing
      last = m;
      var off = (y1 - m) / (y1 - y0);
      if (off < 0) off = 0;
      if (off > 1) off = 1;
      g.addColorStop(off, rgbCss(pts[i][1]));
    }
    return g;
  }

  // Band polygon between the per-day lo/hi env edges, in canvas px for the
  // view window [t0, t1]. Straight lines between day points = the linear
  // edge interpolation the baked raster uses; canvas fill AA replaces its
  // 3x supersampling. Split at null gaps; an isolated day gets a hairline
  // sliver (matching the baked raster's single-column band).
  function bandPath(entry, t0, t1, W, H) {
    var lo = entry.lo, hi = entry.hi, n = lo.length;
    var first = RD.first_day;
    var y0 = entry.y0, y1 = entry.y1;
    function xPx(i) { return ((first + i) * DAY_MS - t0) / (t1 - t0) * W; }
    function yPx(v) { return (y1 - v) / (y1 - y0) * H; }
    // Clip the day walk to just past the view: cut edges land >= 2 days
    // off-canvas (invisible) and path coords stay bounded at deep zooms.
    var dLo = Math.max(0, Math.floor(t0 / DAY_MS - first) - 2);
    var dHi = Math.min(n - 1, Math.ceil(t1 / DAY_MS - first) + 2);
    var p = new Path2D();
    var i = dLo;
    while (i <= dHi) {
      if (lo[i] == null || hi[i] == null) { i++; continue; }
      var j = i;
      while (j + 1 <= dHi && lo[j + 1] != null && hi[j + 1] != null) j++;
      if (j === i) {
        p.rect(xPx(i) - 0.5, yPx(hi[i]), 1,
               Math.max(0, yPx(lo[i]) - yPx(hi[i])));
      } else {
        p.moveTo(xPx(i), yPx(hi[i]));
        for (var k = i + 1; k <= j; k++) p.lineTo(xPx(k), yPx(hi[k]));
        for (var m = j; m >= i; m--) p.lineTo(xPx(m), yPx(lo[m]));
        p.closePath();
      }
      i = j + 1;
    }
    return p;
  }

  // Render one panel's gradient band for [t0, t1] at native display density.
  function drawPanelCanvas(entry, t0, t1, wCss, hCss) {
    var dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    var W = Math.max(1, Math.round(wCss * dpr));
    var H = Math.max(1, Math.round(hCss * dpr));
    var cv = document.createElement('canvas');
    cv.width = W;
    cv.height = H;
    var ctx = cv.getContext('2d');
    if (entry.anchors) {
      ctx.fillStyle = rampGradient(ctx, entry.anchors, entry.y0, entry.y1, H);
      ctx.fillRect(0, 0, W, H);
    } else if (entry.solar) {
      // Colour varies by day: one 1px column per x. Day anchors are
      // interpolated linearly between adjacent days (the baked raster used
      // nearest-day, which shows faint per-day stripes once a day spans
      // many px); a null anchor on either side falls back to the nearest
      // day's value.
      var rows = entry.solar;
      var n = rows.length;
      var row = new Array(7);
      for (var x = 0; x < W; x++) {
        var dayF = (t0 + (x + 0.5) / W * (t1 - t0)) / DAY_MS - RD.first_day;
        if (dayF < 0) dayF = 0;
        if (dayF > n - 1) dayF = n - 1;
        var d0 = Math.floor(dayF);
        var d1 = Math.min(d0 + 1, n - 1);
        var t = dayF - d0;
        var r0 = rows[d0], r1 = rows[d1];
        for (var q = 0; q < 7; q++) {
          row[q] = (r0[q] != null && r1[q] != null)
            ? r0[q] + (r1[q] - r0[q]) * t
            : (t < 0.5 ? r0[q] : r1[q]);
        }
        ctx.fillStyle = solarGradient(ctx, row, entry.y0, entry.y1, H);
        ctx.fillRect(x, 0, 1, H);
      }
    }
    ctx.globalCompositeOperation = 'destination-in';
    ctx.fill(bandPath(entry, t0, t1, W, H));
    ctx.globalCompositeOperation = 'source-over';
    return cv;
  }

  // Relayout patch replacing every envelope image on the current page with a
  // fresh native-resolution render covering exactly [t0, t1].
  function rasterImagePatch(gd, t0, t1) {
    var patch = {};
    if (!RD || !RD.pages) return patch;
    var fl = gd._fullLayout;
    (RD.pages[page] || []).forEach(function (e) {
      var nx = e.row === 1 ? '' : String(e.row);
      var xa = fl['xaxis' + nx], ya = fl['yaxis' + nx];
      if (!xa || !ya || xa._length == null || ya._length == null) return;
      var cv = drawPanelCanvas(e, t0, t1, xa._length, ya._length);
      patch['images[' + e.img + '].source'] = cv.toDataURL('image/png');
      patch['images[' + e.img + '].x'] = t0;
      patch['images[' + e.img + '].sizex'] = t1 - t0;
    });
    return patch;
  }

  // Baked-PNG snapshot per page, taken BEFORE any zoom mutates the live
  // layout (relayout writes into gd.layout, which for toggled pages IS the
  // FIGS[page].layout object). Reset restores from here.
  function snapOrigImages(gd) {
    if (origImages[page]) return;
    origImages[page] = ((gd.layout && gd.layout.images) || []).map(
      function (im) { return { source: im.source, x: im.x, sizex: im.sizex }; });
  }

  // ---- drag-to-zoom ----------------------------------------------------

  var zoomed = false;
  var zoomRangeMs = null; // [t0, t1] ms while zoomed (drives re-raster)
  var homeRange = null;   // master x range at first load (same on both pages)
  var dragStartX = null;  // clientX at mousedown; null = not armed
  var band = null;        // the .rp-zoomband element

  function xKeys(fl) {
    return Object.keys(fl).filter(function (k) {
      return /^xaxis\d*$/.test(k);
    });
  }

  // The master x-axis is the one no other axis chain points it at —
  // shared_xaxes gives every non-bottom row matches='x{k}' and leaves the
  // bottom row's matches unset. Resolved at runtime because panel counts
  // vary per profile (empty panels are dropped) and per page.
  function masterXKey(fl) {
    var ks = xKeys(fl);
    for (var i = 0; i < ks.length; i++) {
      if (!fl[ks[i]].matches) return ks[i];
    }
    return 'xaxis';
  }

  // Full cartesian region spanning all stacked panels (the band is
  // full-height, like the page's spike_full_plot spikeline).
  function plotArea(gd) {
    var fl = gd._fullLayout;
    var rect = gd.getBoundingClientRect();
    var sz = fl._size;
    return { left: rect.left + sz.l, right: rect.left + sz.l + sz.w,
             top: rect.top + sz.t, bottom: rect.top + sz.t + sz.h };
  }

  function pxToMs(gd, clientX) {
    var fl = gd._fullLayout;
    var xa = fl[masterXKey(fl)];
    var rect = gd.getBoundingClientRect();
    return xa.p2c(clientX - (rect.left + xa._offset));
  }

  function setResetVisible(on) {
    var el = document.querySelector('.rp-zoom-reset');
    if (!el) return;
    if (on) {
      // Sit just left of the page toggle when there is one (combined page);
      // single-panel pages have no toggle, so the CSS right:20px stands.
      var tog = document.getElementById('trends-toggle');
      if (tog) {
        var r = tog.getBoundingClientRect();
        el.style.right = Math.round(window.innerWidth - r.left + 12) + 'px';
      }
      el.style.display = 'block';
    } else {
      el.style.display = 'none';
    }
  }

  function captureHome(gd) {
    if (homeRange) return;
    homeRange = gd._fullLayout[masterXKey(gd._fullLayout)].range.slice();
  }

  // Zoom to [t0, t1] ms: master range + the native-resolution gradient
  // re-render in ONE relayout. Ticks/gridlines re-derive on their own
  // (auto date ticks on every axis).
  function applyZoomRange(gd, msRange) {
    var patch = {};
    patch[masterXKey(gd._fullLayout) + '.range'] = msRange.slice();
    Object.assign(patch, rasterImagePatch(gd, msRange[0], msRange[1]));
    return Plotly.relayout(gd, patch);
  }

  function resetZoom() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout || !zoomed || !homeRange) return;
    var patch = {};
    patch[masterXKey(gd._fullLayout) + '.range'] = homeRange.slice();
    // Restore the baked full-range PNGs (a downscale at home range — crisp).
    (origImages[page] || []).forEach(function (o, k) {
      patch['images[' + k + '].source'] = o.source;
      patch['images[' + k + '].x'] = o.x;
      patch['images[' + k + '].sizex'] = o.sizex;
    });
    zoomed = false;
    zoomRangeMs = null;
    setResetVisible(false);
    Plotly.relayout(gd, patch);
  }

  function cancelDrag() {
    dragStartX = null;
    window.__rpZoomDragging = false;
    document.body.style.userSelect = '';
    document.body.classList.remove('rp-zoom-dragging');
    if (band) band.style.display = 'none';
  }

  function bindZoom() {
    band = document.querySelector('.rp-zoomband');
    var btn = document.getElementById('trends-zoom-reset-btn');
    if (btn) btn.addEventListener('click', resetZoom);

    // mousedown arms on the document (insets sit over the plot but are not
    // children of the plot div, so a div-bound listener would dead-zone
    // them); the plot-area bounds check does the real gating.
    document.addEventListener('mousedown', function (e) {
      if (e.button !== 0) return;
      var gd = pdiv();
      if (!gd || !gd._fullLayout) return;
      var pa = plotArea(gd);
      if (e.clientX < pa.left || e.clientX > pa.right ||
          e.clientY < pa.top || e.clientY > pa.bottom) return;
      captureHome(gd);
      dragStartX = e.clientX;
    });

    // move/up on window so a drag that leaves the plot (or the page)
    // still tracks and completes; endpoints are clamped to the plot area.
    window.addEventListener('mousemove', function (e) {
      if (dragStartX == null) return;
      var gd = pdiv();
      if (!gd || !gd._fullLayout) { cancelDrag(); return; }
      if (!window.__rpZoomDragging) {
        if (Math.abs(e.clientX - dragStartX) < MIN_DRAG_PX) return;
        window.__rpZoomDragging = true;   // tooltip scaffold hides on this
        document.body.style.userSelect = 'none';
        document.body.classList.add('rp-zoom-dragging');
      }
      var pa = plotArea(gd);
      var x0 = Math.max(pa.left, Math.min(dragStartX, e.clientX));
      var x1 = Math.min(pa.right, Math.max(dragStartX, e.clientX));
      band.style.display = 'block';
      band.style.left = x0 + 'px';
      band.style.width = Math.max(0, x1 - x0) + 'px';
      band.style.top = pa.top + 'px';
      band.style.height = (pa.bottom - pa.top) + 'px';
    });

    window.addEventListener('mouseup', function (e) {
      if (dragStartX == null) return;
      var wasDragging = window.__rpZoomDragging;
      var startX = dragStartX;
      var gd = pdiv();
      cancelDrag();
      if (!wasDragging || !gd || !gd._fullLayout) return;
      var pa = plotArea(gd);
      var xA = Math.max(pa.left, Math.min(startX, e.clientX));
      var xB = Math.min(pa.right, Math.max(startX, e.clientX));
      if (xB - xA < MIN_DRAG_PX) return;
      var ms0 = pxToMs(gd, xA);
      var ms1 = pxToMs(gd, xB);
      if (ms0 == null || ms1 == null || !(ms1 - ms0 >= MIN_SPAN_MS)) return;
      zoomed = true;
      zoomRangeMs = [ms0, ms1];
      setResetVisible(true);
      applyZoomRange(gd, zoomRangeMs);
    });

    // Horizontal-select cursor while the pointer is inside the plot area —
    // makes the drag-to-zoom affordance visible. Class on <body> so the CSS
    // also covers the inset labels sitting over the plot.
    document.addEventListener('mousemove', function (e) {
      var gd = pdiv();
      if (!gd || !gd._fullLayout) return;
      var pa = plotArea(gd);
      var inside = e.clientX >= pa.left && e.clientX <= pa.right &&
                   e.clientY >= pa.top && e.clientY <= pa.bottom;
      document.body.classList.toggle('rp-zoom-hot', inside);
    });

    // staticPlot disables Plotly's own double-click, so ours is unopposed.
    document.addEventListener('dblclick', function (e) {
      var gd = pdiv();
      if (!gd || !gd._fullLayout || !zoomed) return;
      var pa = plotArea(gd);
      if (e.clientX < pa.left || e.clientX > pa.right ||
          e.clientY < pa.top || e.clientY > pa.bottom) return;
      resetZoom();
    });

    // Zoomed rasters are rendered for a specific panel pixel width — a
    // resize changes it, so re-render (debounced; range/ticks unchanged).
    var resizeTimer = null;
    window.addEventListener('resize', function () {
      if (!zoomRangeMs) return;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        var gd = pdiv();
        if (gd && gd._fullLayout && zoomRangeMs) {
          Plotly.relayout(
            gd, rasterImagePatch(gd, zoomRangeMs[0], zoomRangeMs[1]));
        }
      }, 150);
    });
  }

  // ---- page toggle -------------------------------------------------------

  function switchTo(next, btn) {
    if (next === page) return;
    var gd = pdiv();
    var f = FIGS[next];
    document.querySelectorAll('#trends-toggle .rp-btn-pill').forEach(function (b) {
      b.classList.toggle('is-active', b === btn);
    });
    if (!gd || !f) return;
    // newPlot renders the incoming figure at its baked full range, so
    // capture the current zoom and re-apply it after the swap. A rapid
    // toggle mid-drag must not strand the band/flag state.
    var keepRange = zoomed ? zoomRangeMs : null;
    cancelDrag();
    page = next;
    window.__rpActiveTab = page;
    Plotly.newPlot(gd, f.data, f.layout, CONFIG).then(function () {
      bindRedraw(gd);
      showInsetsForPage();
      positionInsets();
      snapOrigImages(gd);           // pristine bakes, before any zoom patch
      if (keepRange) applyZoomRange(gd, keepRange);
    });
  }

  function bind() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) { setTimeout(bind, 100); return; }
    document.querySelectorAll('#trends-toggle .rp-btn-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        switchTo(btn.getAttribute('data-value'), btn);
      });
    });
    bindRedraw(gd);
    window.addEventListener('resize', positionInsets);
    showInsetsForPage();
    positionInsets();
    snapOrigImages(gd);
    captureHome(gd);
    bindZoom();
  }

  bind();
})();
