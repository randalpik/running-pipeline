// Misc. Trends page toggle (Weather / Other).
//
// All eight panels live in one figure; each trace carries meta.page. Switching
// pages flips trace visibility by page and applies the page's precomputed
// relayout (y-axis ranges/titles/ticks, inset-annotation visibility, and the
// Time solar-raster image visibility). window.__rpActiveTab tells the cursor
// tooltip which tab's rows to render.
(function () {
  var PAGES = window.__PLOT_TRENDS_PAGES || {};
  var page = 'weather';
  window.__rpActiveTab = page;

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

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
  }

  bind();
})();
