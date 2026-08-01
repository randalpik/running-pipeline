// Runs in <head>, before Plotly's inline newPlot in <body>. Establishes the
// page's mobile mode BEFORE anything is laid out.
//
// Framed: the shell already decided (device pointer type + screen width) and
// hands the answer down on the iframe URL — see srcFor() in shell.js. Waiting
// for its rp-shell-mode message instead would be too late: that message can
// only arrive after load, by which point the desktop layout has rendered and
// the mobile layout replaces it in view. Knowing here means the reshaped
// pages never show a desktop frame at all.
//
// Standalone: no such authority, so latch the viewport query ONCE. Reading it
// again later lets an Android URL-bar animation flip it, which is the
// feedback loop documented in _scaffold/mobile.js.
(function () {
  var framed = window.parent !== window;
  var mobile = framed
    ? /[?&]rpm=1/.test(window.location.search)
    : window.matchMedia('(max-height: 520px)').matches;
  if (mobile) document.documentElement.classList.add('rp-mobile');
  // Hide the first paint on pages that are about to be reshaped, so the
  // desktop grid never flashes. mobile.js clears this after its render.
  if (mobile && window.__PLOT_MOBILE_LAYOUT) {
    document.documentElement.classList.add('rp-mobile-pending');
  }
})();
