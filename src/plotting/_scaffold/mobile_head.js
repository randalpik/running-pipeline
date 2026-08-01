// Runs in <head>, before Plotly's inline newPlot in <body>.
//
// 1. Establishes `html.rp-mobile` for STANDALONE pages. Inside the shell the
//    class is pushed instead by the rp-shell-mode message (see render.py) —
//    the shell derives it from the device (pointer type + screen width),
//    which nothing on this page can perturb. Standalone has no such
//    authority, so we latch the viewport query ONCE here at load: read again
//    later and an Android URL-bar animation can flip it, which is exactly
//    the feedback loop that made the reshaped pages strobe.
// 2. Hides the first paint when a mobile reshape is going to be applied, so
//    the desktop grid doesn't flash before _scaffold/mobile.js re-renders.
(function () {
  var standalone = window.parent === window;
  var short = window.matchMedia('(max-height: 520px)').matches;
  if (standalone && short) {
    document.documentElement.classList.add('rp-mobile');
  }
  if (!window.__PLOT_MOBILE_LAYOUT) return;
  // Framed pages can't know the shell's answer yet; `short` is a good
  // enough guess for anti-flash purposes and mobile.js always clears it.
  if (short) document.documentElement.classList.add('rp-mobile-pending');
})();
