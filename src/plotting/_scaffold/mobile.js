// Mobile layout engine: applies a Python-authored layout reshape when the
// viewport is mobile-short, and restores the authored desktop layout on
// crossing back.
//
// Input: window.__PLOT_MOBILE_LAYOUT (emitted by render.py from the
// MobileLayout dataclass) — either
//   { patch: {<dotted layout path>: value, ...}, scroll: bool }
// or, for plots that swap whole figures client-side (Misc Trends),
//   { variants: {<page>: {<path>: value}}, variant_global: '__rpActiveTab',
//     scroll: bool }
// Paths are dotted Plotly layout paths ('xaxis3.domain', 'annotations[2].y',
// 'margin.r', 'height'). Traces are NEVER touched — reshapes are chosen in
// Python so the panel -> axis-id mapping is invariant (make_subplots numbers
// axes in row-major cell order in any grid shape).
//
// Re-render is a full Plotly.newPlot, never a bare relayout: plotly.js does
// not redraw a y-axis's gridlines when its domain changes (see the header
// comment in plot_qualitative_trends.js). newPlot clears gd.on bindings, so
// the engine dispatches 'rp-layout-mode' on window afterwards — the single
// re-bind contract for overlay_anchor.js / axis_pad.js / cursor_tooltip.js /
// per-plot listeners.
//
// Scroll mode: html/body get .rp-scroll and :root gets --rp-plot-h (kept
// equal to layout.height — the figure's paper geometry comes from
// layout.height, the document's scroll height from the CSS var; equal values
// make them agree instead of fighting base.css's !important pin).
//
// Breakpoint MUST stay in sync with _scaffold/mobile_head.js and the
// (max-height: 520px) blocks in _scaffold/base.css.
(function () {
  var ML = window.__PLOT_MOBILE_LAYOUT || null;
  var MOBILE_MQ = window.matchMedia('(max-height: 520px)');
  var DEBOUNCE_MS = 150;

  function plot() { return document.querySelector('.plotly-graph-div'); }

  // ---- dotted-path helpers ('annotations[2].y' -> ['annotations', 2, 'y'])
  function parsePath(path) {
    var parts = [];
    path.split('.').forEach(function (seg) {
      var name = seg.replace(/\[\d+\]/g, '');
      if (name) parts.push(name);
      var idx = seg.match(/\[(\d+)\]/g);
      if (idx) idx.forEach(function (s) {
        parts.push(parseInt(s.slice(1, -1), 10));
      });
    });
    return parts;
  }
  function getPath(obj, path) {
    var p = parsePath(path), cur = obj;
    for (var i = 0; i < p.length; i++) {
      if (cur == null) return undefined;
      cur = cur[p[i]];
    }
    return cur;
  }
  function setPath(obj, path, val) {
    var p = parsePath(path), cur = obj;
    for (var i = 0; i < p.length - 1; i++) {
      if (cur[p[i]] == null) {
        cur[p[i]] = (typeof p[i + 1] === 'number') ? [] : {};
      }
      cur = cur[p[i]];
    }
    cur[p[p.length - 1]] = val;
  }

  function activePatch() {
    if (!ML) return null;
    if (ML.variants) {
      var key = ML.variant_global ? window[ML.variant_global] : null;
      return (key && ML.variants[key]) || null;
    }
    return ML.patch || null;
  }

  function isMobile() { return MOBILE_MQ.matches; }

  // Apply the active patch onto a caller-owned layout object (Misc Trends
  // patches its per-page deep copy before its own newPlot).
  function patchLayout(layout) {
    var patch = isMobile() ? activePatch() : null;
    if (!patch) return layout;
    Object.keys(patch).forEach(function (k) { setPath(layout, k, patch[k]); });
    return layout;
  }

  function setScroll(on, heightPx) {
    document.documentElement.classList.toggle('rp-scroll', on);
    document.body.classList.toggle('rp-scroll', on);
    if (on && heightPx) {
      document.documentElement.style.setProperty('--rp-plot-h', heightPx + 'px');
    } else {
      document.documentElement.style.removeProperty('--rp-plot-h');
      window.scrollTo(0, 0);
    }
  }

  function syncScrollHeight() {
    var gd = plot();
    if (!gd) return;
    var on = isMobile() && !!activePatch() && (!ML || ML.scroll !== false);
    var h = on ? (gd.layout && gd.layout.height) : null;
    setScroll(on && !!h, h);
    // Forward-compat hook: lets the shell own the scroll later (grow the
    // iframe instead) with zero plot-side change.
    if (window.parent !== window) {
      try {
        window.parent.postMessage({ type: 'rp-plot-height', px: h || null }, '*');
      } catch (e) {}
    }
  }

  // ---- engine (only pages that ship a MobileLayout re-render) ----
  var applied = false;
  var busy = false;
  var desktopSnapshot = null;   // authored values of every patched path

  function render(layout) {
    var gd = plot();
    if (!gd || !window.Plotly) return Promise.resolve();
    var cfg = window.__RP_PLOT_CONFIG ||
              { responsive: true, displayModeBar: false };
    return Plotly.newPlot(gd, gd.data, layout, cfg).then(function () {
      syncScrollHeight();
      document.documentElement.classList.remove('rp-mobile-pending');
      window.dispatchEvent(new CustomEvent('rp-layout-mode',
                                           { detail: { mobile: applied } }));
    });
  }

  function snapshotPaths(patch) {
    if (desktopSnapshot) return;
    var gd = plot();
    desktopSnapshot = {};
    Object.keys(patch).forEach(function (k) {
      desktopSnapshot[k] = getPath(gd.layout, k);   // undefined = unset
    });
  }

  function apply() {
    var gd = plot();
    var patch = activePatch();
    if (!gd || !patch) {
      document.documentElement.classList.remove('rp-mobile-pending');
      return Promise.resolve();
    }
    snapshotPaths(patch);
    var lay = JSON.parse(JSON.stringify(gd.layout));
    Object.keys(patch).forEach(function (k) { setPath(lay, k, patch[k]); });
    applied = true;
    return render(lay);
  }

  function restore() {
    var gd = plot();
    if (!gd || !applied) return Promise.resolve();
    var lay = JSON.parse(JSON.stringify(gd.layout));
    Object.keys(desktopSnapshot).forEach(function (k) {
      setPath(lay, k, desktopSnapshot[k]);   // undefined un-sets (autosize etc.)
    });
    applied = false;
    setScroll(false, null);
    return render(lay);
  }

  function sync() {
    if (busy) return;
    if (isMobile() === applied) return;
    busy = true;
    (isMobile() ? apply() : restore()).then(
      function () { busy = false; sync(); },   // re-check: MQ may have flipped mid-render
      function () { busy = false; }
    );
  }

  var debounceTimer = null;
  function onMQChange() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sync, DEBOUNCE_MS);
  }

  var bootTries = 0;
  function boot() {
    var gd = plot();
    if (!gd || !gd._fullLayout || !gd._fullLayout._size) {
      if (++bootTries > 100) return;
      setTimeout(boot, 100);
      return;
    }
    if (isMobile()) sync();
    else document.documentElement.classList.remove('rp-mobile-pending');
  }

  window.rpMobile = {
    isMobile: isMobile,
    patchLayout: patchLayout,
    syncScrollHeight: syncScrollHeight,
    onChange: function (fn) {
      if (MOBILE_MQ.addEventListener) MOBILE_MQ.addEventListener('change', fn);
      else if (MOBILE_MQ.addListener) MOBILE_MQ.addListener(fn);
    },
  };

  if (ML) {
    window.rpMobile.onChange(onMQChange);
    boot();
  }
})();
