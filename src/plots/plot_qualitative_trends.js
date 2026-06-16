// Misc. Trends page toggle (Weather / Other) + per-subplot inset labels.
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
// positioning runs (and the lone inset always shows).
(function () {
  var FIGS = window.__PLOT_TRENDS_FIGS || {};
  var CONFIG = window.__PLOT_TRENDS_CONFIG ||
               { responsive: true, displayModeBar: false, staticPlot: true };
  var page = window.__rpActiveTab || 'weather';
  window.__rpActiveTab = page;

  var INSET_PAD = 6;  // px gap from the subplot's top-left corner

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

  function switchTo(next, btn) {
    if (next === page) return;
    var gd = pdiv();
    var f = FIGS[next];
    document.querySelectorAll('#trends-toggle .rp-btn-pill').forEach(function (b) {
      b.classList.toggle('is-active', b === btn);
    });
    if (!gd || !f) return;
    page = next;
    window.__rpActiveTab = page;
    Plotly.newPlot(gd, f.data, f.layout, CONFIG).then(function () {
      bindRedraw(gd);
      showInsetsForPage();
      positionInsets();
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
  }

  bind();
})();
