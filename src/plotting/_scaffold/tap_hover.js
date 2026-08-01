/* Shared tap-as-hover gesture module.
 *
 * A TAP — touchend within TAP_SLOP_PX of its touchstart, under TAP_MAX_MS —
 * calls handlers.show(clientX, clientY). A second tap within the slop of the
 * previous one toggles the tooltip off; a tap anywhere outside the bound
 * element also hides; travel past the slop (a drag) hides and is otherwise
 * left alone.
 *
 * Deliberately hands-off beyond that:
 *  - No preventDefault anywhere. The emulated mouse/click sequence after a
 *    tap is what makes Plotly legends, checkboxes and buttons work on touch;
 *    the plots' own mouse paths ignore the emulation via mouseSuppressed()
 *    (see the guards in cursor_tooltip.js / make_geography_plot.js /
 *    plot_qualitative_trends.js).
 *  - Plotly's dragmode stays untouched: plotly.js natively turns a touch
 *    drag over the plot area into the zoom box (verified on v3.5), and its
 *    non-passive touchstart handler claims the gesture so the page doesn't
 *    scroll from the graph — page scrolling happens from the margins, which
 *    is the intended mobile behavior.
 *
 * window.__rpTouchActive mirrors mouseSuppressed() for plot JS.
 */
(function () {
  if (window.rpTapHover) return;

  var TAP_SLOP_PX = 12;   // max travel for a touch to still count as a tap
  var TAP_MAX_MS  = 700;  // max duration of a tap (long-press is not a tap)
  var SUPPRESS_MS = 800;  // emulated-mouse ignore window after any touch

  var lastTouchTs = 0;
  var lastTap = null;     // {x, y} of the last shown tap, for toggle-off

  function noteTouch() { lastTouchTs = Date.now(); }
  function mouseSuppressed() { return Date.now() - lastTouchTs < SUPPRESS_MS; }

  function bind(el, handlers) {
    var start = null;

    el.addEventListener('touchstart', function (e) {
      noteTouch();
      if (e.touches.length !== 1) { start = null; return; }
      var t = e.touches[0];
      start = { x: t.clientX, y: t.clientY, ts: Date.now() };
    }, { passive: true });

    el.addEventListener('touchmove', function (e) {
      noteTouch();
      if (!start) return;
      var t = e.touches[0];
      if (Math.abs(t.clientX - start.x) > TAP_SLOP_PX ||
          Math.abs(t.clientY - start.y) > TAP_SLOP_PX) {
        start = null;       // a drag (Plotly zoom box or page scroll)
        handlers.hide();
      }
    }, { passive: true });

    el.addEventListener('touchend', function (e) {
      noteTouch();
      if (!start) return;
      var s = start;
      start = null;
      var t = e.changedTouches && e.changedTouches[0];
      if (!t) return;
      if (Date.now() - s.ts > TAP_MAX_MS) return;
      if (Math.abs(t.clientX - s.x) > TAP_SLOP_PX ||
          Math.abs(t.clientY - s.y) > TAP_SLOP_PX) return;
      if (lastTap && Math.abs(t.clientX - lastTap.x) <= TAP_SLOP_PX &&
          Math.abs(t.clientY - lastTap.y) <= TAP_SLOP_PX) {
        lastTap = null;     // second tap on the same spot toggles off
        handlers.hide();
        return;
      }
      lastTap = { x: t.clientX, y: t.clientY };
      handlers.show(t.clientX, t.clientY);
    }, { passive: true });

    document.addEventListener('touchend', function (e) {
      noteTouch();
      if (el.contains(e.target)) return;
      lastTap = null;
      handlers.hide();
    }, { passive: true });
  }

  window.rpTapHover = { bind: bind, mouseSuppressed: mouseSuppressed };
  try {
    Object.defineProperty(window, '__rpTouchActive', { get: mouseSuppressed });
  } catch (e) {
    window.__rpTouchActive = false;
  }
})();
