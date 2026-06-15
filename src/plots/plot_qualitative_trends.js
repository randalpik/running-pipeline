// Misc. Trends page toggle (Weather / Other) + per-subplot inset labels.
//
// All eight panels live in one figure; each trace carries meta.page. Switching
// pages flips trace visibility by page and applies the page's precomputed
// relayout (y-axis ranges/titles/ticks + per-page image/shape visibility).
// window.__rpActiveTab tells the cursor tooltip which tab's rows to render.
//
// Inset labels are HTML overlay divs (.rp-inset) — each tagged data-rp-row /
// data-rp-page / data-rp-key — positioned at their subplot's top-left from the
// live Plotly layout (and repositioned on relayout / resize). The page toggle
// shows the active page's insets and hides the other's. The standalone
// single-panel pages load this same file: there is no #trends-toggle there, so
// only the positioning runs (and the lone inset always shows).
(function () {
  var PAGES = window.__PLOT_TRENDS_PAGES || {};
  var page = window.__rpActiveTab || 'weather';
  window.__rpActiveTab = page;

  var INSET_PAD = 6;  // px gap from the subplot's top-left corner

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  // Place each inset at the top-left of its subplot row, read from the live
  // _fullLayout axis offsets (same technique as _scaffold/overlay_anchor.js).
  function positionInsets() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var fl = gd._fullLayout;
    var rect = gd.getBoundingClientRect();
    document.querySelectorAll('.rp-inset').forEach(function (el) {
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

  function applyPage(gd) {
    if (!gd || !gd.data) return;
    // Trace visibility by meta.page.
    var vis = [];
    for (var i = 0; i < gd.data.length; i++) {
      var mp = (gd.data[i].meta || {}).page;
      vis.push(mp ? (mp === page) : true);
    }
    Plotly.restyle(gd, { visible: vis });
    var rel = (PAGES[page] || {}).relayout;
    if (rel) Plotly.relayout(gd, rel);
    showInsetsForPage();
    positionInsets();
  }

  function bind() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) { setTimeout(bind, 100); return; }
    document.querySelectorAll('#trends-toggle .rp-btn-pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var next = btn.getAttribute('data-value');
        if (next === page) return;
        page = next;
        window.__rpActiveTab = page;
        document.querySelectorAll('#trends-toggle .rp-btn-pill').forEach(function (b) {
          b.classList.toggle('is-active', b === btn);
        });
        applyPage(pdiv());
      });
    });
    // Keep insets pinned to their subplots through every layout change.
    if (typeof gd.on === 'function') {
      gd.on('plotly_afterplot', positionInsets);
      gd.on('plotly_relayout', positionInsets);
    }
    window.addEventListener('resize', positionInsets);
    positionInsets();
  }

  bind();
})();
