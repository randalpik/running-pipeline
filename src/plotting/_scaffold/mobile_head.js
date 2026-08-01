// Anti-flash for mobile-reshaped plots. Injected into <head> (so it runs
// before Plotly's inline newPlot in <body>): when this page ships a mobile
// layout AND the viewport is already mobile-short, hide the plot until
// _scaffold/mobile.js has re-rendered with the mobile layout — otherwise the
// desktop grid paints squished for a frame and then jumps.
// Breakpoint MUST stay in sync with MOBILE_MQ in _scaffold/mobile.js.
(function () {
  if (!window.__PLOT_MOBILE_LAYOUT) return;
  if (!window.matchMedia('(max-height: 520px)').matches) return;
  document.documentElement.classList.add('rp-mobile-pending');
})();
