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
// WHICH SIGNAL DECIDES "mobile" — the load-bearing detail here.
//
// It must be something this page's own layout cannot change, or the engine
// feeds back on itself. Deriving it from the viewport (`(max-height:520px)`)
// did exactly that in production: the shell sizes its rotated stage in
// dvh/dvw units, so an Android URL-bar animation changes this iframe's
// height -> flips the query -> swaps a ~720px layout -> changes whether the
// page is scrollable -> Chromium shows/hides the browser controls to match
// -> changes dvh again. Reproduced by jogging the viewport height: 15
// layout flips, 440px <-> 1162px, no user input.
//
// So the single source of truth is `html.rp-mobile`, and EVERY mobile rule
// in the CSS keys off the same class. Inside the shell it is pushed by the
// rp-shell-mode message; the shell derives it from `(pointer: coarse) and
// (max-width: 940px)` — device pointer type and screen width, which no plot
// layout can perturb. Standalone pages latch the viewport query once at load
// in _scaffold/mobile_head.js. A flip-rate guard below is the backstop.
(function () {
  var ML = window.__PLOT_MOBILE_LAYOUT || null;
  var HEIGHT_MQ = window.matchMedia('(max-height: 520px)');
  var FRAMED = document.documentElement.classList.contains('rp-framed');
  var DEBOUNCE_MS = 150;
  // Anti-flap backstop: if the mode ever changes more than this within the
  // window, stop reacting and keep whatever is on screen. A wrong-but-stable
  // layout beats a strobing one.
  var MAX_FLIPS = 6, FLIP_WINDOW_MS = 4000;
  var flips = 0, flipWindowStart = 0, latched = false;

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

  function isMobile() {
    if (latched) return applied;   // frozen: report what's on screen
    return document.documentElement.classList.contains('rp-mobile');
  }

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

  // Geometry breadcrumb for the on-device diagnostic (shell.js ?rpdiag=1).
  // Cheap and always on: the layout bugs we have chased here only appear on
  // real hardware, and without a trace there is nothing to look at.
  function trace(evt) {
    var gd = plot();
    var fl = gd && gd._fullLayout;
    var cs = gd && getComputedStyle(gd);
    var t = (window.__rpTrace = window.__rpTrace || []);
    t.push(evt +
      ' win=' + window.innerWidth + 'x' + window.innerHeight +
      ' icb=' + document.documentElement.clientWidth +
      ' body=' + document.body.clientWidth +
      ' gd=' + (gd ? gd.clientWidth : '-') +
      ' css=' + (cs ? parseInt(cs.width, 10) : '-') +
      ' fig=' + (fl ? Math.round(fl.width) + 'x' + Math.round(fl.height) : '-') +
      ' mob=' + (isMobile() ? 1 : 0) +
      ' scr=' + (document.body.classList.contains('rp-scroll') ? 1 : 0));
    if (t.length > 30) t.shift();
  }

  function render(layout) {
    var gd = plot();
    if (!gd || !window.Plotly) return Promise.resolve();
    var cfg = window.__RP_PLOT_CONFIG ||
              { responsive: true, displayModeBar: false };
    trace('render:pre');
    return Plotly.newPlot(gd, gd.data, layout, cfg).then(function () {
      trace('newPlot');
      syncScrollHeight();
      trace('scrollSync');
      document.documentElement.classList.remove('rp-mobile-pending');
      window.dispatchEvent(new CustomEvent('rp-layout-mode',
                                           { detail: { mobile: applied } }));
      setTimeout(function () { trace('settled'); }, 400);
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

  // Returns false once the mode has changed implausibly often — something
  // upstream is oscillating, so freeze rather than strobe the page.
  function flipAllowed() {
    var now = Date.now();
    if (now - flipWindowStart > FLIP_WINDOW_MS) { flipWindowStart = now; flips = 0; }
    if (++flips <= MAX_FLIPS) return true;
    if (!latched) {
      latched = true;
      window.__rpLayoutLatched = true;
      if (window.console && console.warn) {
        console.warn('[rp-mobile] layout mode flipped ' + flips + 'x in ' +
                     FLIP_WINDOW_MS + 'ms — freezing to stop a relayout loop');
      }
    }
    return false;
  }

  function sync() {
    if (busy || latched) return;
    if (isMobile() === applied) return;
    if (!flipAllowed()) return;
    busy = true;
    (isMobile() ? apply() : restore()).then(
      function () { busy = false; sync(); },   // re-check: mode may have moved
      function () { busy = false; sync(); }
    );
  }

  var debounceTimer = null;
  function onModeChange() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sync, DEBOUNCE_MS);
  }

  // Framed: the shell re-broadcasts rp-shell-mode whenever its device-level
  // breakpoint or the orientation changes; the receiver injected by
  // render.py has already updated html.rp-mobile by the time this runs (it
  // registers its listener first). Standalone: re-latch only on a real
  // orientation change — never on the incidental viewport resizes that
  // URL-bar animations produce.
  if (FRAMED) {
    window.addEventListener('message', function (e) {
      if (e.data && e.data.type === 'rp-shell-mode') onModeChange();
    });
  } else {
    window.addEventListener('orientationchange', function () {
      setTimeout(function () {   // let the viewport settle before re-reading
        document.documentElement.classList.toggle('rp-mobile', HEIGHT_MQ.matches);
        onModeChange();
      }, 250);
    });
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

  // Touch scrolling for the tall mobile layouts. Neither the browser nor
  // Plotly will scroll these pages on their own (both measured — see
  // _scaffold/touch_scroll.js): Plotly claims drags over the plot area for
  // its zoom box, and Chromium won't pan a scroller under the shell's
  // rotated stage at all.
  //
  // Split by WHERE the drag starts, not by direction: a drag beginning on
  // Plotly's draglayer (a panel interior) is left entirely alone, so
  // drag-to-zoom keeps both axes there. Everywhere else on the page — the
  // title bar, the axis-label margins, the legend rail, the gutters between
  // panels — a vertical drag scrolls. That keeps Plotly's own event
  // bookkeeping intact (stealing a gesture it already started leaves
  // gd._dragging stuck) and still leaves a generous scroll surface.
  function scrollable() {
    return document.documentElement.scrollHeight - window.innerHeight > 4;
  }
  function onPlotlyDrag(el) {
    return !!(el && el.closest && el.closest('.draglayer'));
  }
  function attachTouchScroll() {
    if (!window.rpTouchScroll) return;
    window.rpTouchScroll.attach(document, {
      capture: true,   // beat the page's other touch handlers to the event
      decide: function (dx, dy, target) {
        if (Math.abs(dy) <= Math.abs(dx)) return false;
        return scrollable() && !onPlotlyDrag(target);
      },
      scrollerFor: function () { return document.scrollingElement; },
      deltaFor: function (stepX, stepY) { return -stepY; },
    });
  }

  window.rpMobile = {
    isMobile: isMobile,
    patchLayout: patchLayout,
    syncScrollHeight: syncScrollHeight,
    onChange: function (fn) {
      if (FRAMED) {
        window.addEventListener('message', function (e) {
          if (e.data && e.data.type === 'rp-shell-mode') fn();
        });
      } else if (HEIGHT_MQ.addEventListener) {
        HEIGHT_MQ.addEventListener('change', fn);
      } else if (HEIGHT_MQ.addListener) {
        HEIGHT_MQ.addListener(fn);
      }
    },
  };

  if (ML) {
    attachTouchScroll();
    boot();
  }
})();
