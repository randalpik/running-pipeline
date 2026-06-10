// Annual plot interactions: Cumulative/Average mode toggle + custom
// per-year legend (click to hide/show a year).
//
// Each trace carries both representations in meta (x_cum/y_cum dots,
// x_avg/y_avg lines); the toggle restyles x/y/mode in place so legend
// visibility state survives a mode switch. The cursor tooltip reads the
// current mode off gd.data[0].mode and per-year visibility off
// gd.data[i].visible — no shared state needed here.
(function () {
  var mode = 'cum';

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  function applyMode() {
    var gd = pdiv();
    if (!gd || !gd.data) return;
    var x = [], y = [], m = [];
    for (var i = 0; i < gd.data.length; i++) {
      var meta = gd.data[i].meta || {};
      x.push(mode === 'cum' ? meta.x_cum : meta.x_avg);
      y.push(mode === 'cum' ? meta.y_cum : meta.y_avg);
      m.push(mode === 'cum' ? 'markers' : 'lines');
    }
    Plotly.restyle(gd, { x: x, y: y, mode: m });
    Plotly.relayout(gd, {
      'yaxis.title.text': mode === 'cum' ? 'Cumulative miles'
                                         : 'Avg miles / day (28d)',
      'yaxis.autorange': true,
      'yaxis.rangemode': 'tozero',
    });
  }

  document.querySelectorAll('#annual-toggle .rp-btn-pill').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var newMode = btn.getAttribute('data-value');
      if (newMode === mode) return;
      mode = newMode;
      document.querySelectorAll('#annual-toggle .rp-btn-pill').forEach(function (b) {
        b.classList.toggle('is-active', b === btn);
      });
      applyMode();
    });
  });

  document.querySelectorAll('#annual-legend .legend-item').forEach(function (el) {
    el.addEventListener('click', function () {
      var gd = pdiv();
      if (!gd) return;
      var idx = parseInt(el.getAttribute('data-trace-idx'));
      var hidden = el.classList.contains('hidden');
      Plotly.restyle(gd, { visible: hidden ? true : 'legendonly' }, [idx]);
      el.classList.toggle('hidden', !hidden);
    });
  });
})();
