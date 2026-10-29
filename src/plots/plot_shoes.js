/* Shoes tab: rectangle hit-test, whole-pair outline, 1px minimum bars.
 *
 * window.__PLOT_SHOES = { half, pairs: [{ row, segs: [[startMs, endMs], ...],
 * html }] }. Trace i is pairs[i]; bar point i of that trace is segs[i].
 *
 * Bars are redrawn from the segment dates after every Plotly render rather
 * than parsed from Plotly's paths: a day at full zoom-out is well under a
 * pixel, and Plotly may round such a bar away.
 *
 * A pair's hover target is its whole region — first wear to last, the full
 * height of its row — not its individual segments, so dragging across seams
 * and gaps never drops the tooltip. Regions in a row never overlap (rows are
 * reused only after a year), and rows sit at fractional positions (a gap
 * separates the trainer block), so the row is found by nearest centre.
 */
(function () {
  var S = window.__PLOT_SHOES;
  if (!S) return;
  var MIN_PX = 1;
  var SLOP_PX = 2;          // mouse: forgive a near-miss beside a region's end
  var TOUCH_SLOP_PX = 10;   // tap: a fingertip is wider than a 1px region
  var OUTLINE = '#ffffff';
  var active = -1;

  var rows = [];            // distinct row centres
  var byRow = {};           // row -> [{ pair, s, e }]: each pair's whole region
  S.pairs.forEach(function (p, i) {
    if (!byRow[p.row]) { byRow[p.row] = []; rows.push(p.row); }
    byRow[p.row].push({ pair: i, s: p.segs[0][0], e: p.segs[p.segs.length - 1][1] });
  });

  function gd() { return document.querySelector('.plotly-graph-div'); }

  function shown(g, i) {
    var v = g.data[i] && g.data[i].visible;
    return v !== false && v !== 'legendonly';
  }

  // Segment's horizontal pixel extent (plot-area coordinates), >= MIN_PX wide.
  function segPx(xa, s, e) {
    var x0 = xa.c2p(s), x1 = xa.c2p(e);
    if (x1 - x0 < MIN_PX) {
      var c = (x0 + x1) / 2;
      x0 = c - MIN_PX / 2;
      x1 = c + MIN_PX / 2;
    }
    return [x0, x1];
  }

  function eachTrace(fn) {
    var g = gd();
    if (!g) return;
    g.querySelectorAll('.barlayer .trace').forEach(function (tr) {
      var cd = tr.__data__;
      if (cd && cd[0] && cd[0].trace) fn(tr, cd[0].trace.index);
    });
  }

  function paintOutline() {
    eachTrace(function (tr, i) {
      var on = i === active;
      tr.querySelectorAll('.point path').forEach(function (path) {
        path.style.stroke = on ? OUTLINE : '';
        path.style.strokeWidth = on ? '1.5px' : '0px';
      });
    });
  }

  function setActive(i) {
    if (i === active) return;
    active = i;
    paintOutline();
  }

  function redrawBars() {
    var g = gd();
    if (!g || !g._fullLayout) return;
    var xa = g._fullLayout.xaxis, ya = g._fullLayout.yaxis;
    eachTrace(function (tr, i) {
      var p = S.pairs[i];
      if (!p) return;
      var y0 = ya.c2p(p.row - S.half), y1 = ya.c2p(p.row + S.half);
      tr.querySelectorAll('.point path').forEach(function (path) {
        // .i is the point index (.p is the bar's position — its row).
        var d = path.__data__;
        var sg = d && p.segs[d.i];
        if (!sg) return;
        var x = segPx(xa, sg[0], sg[1]);
        path.setAttribute('d', 'M' + x[0] + ',' + y0 + 'V' + y1 +
                               'H' + x[1] + 'V' + y0 + 'Z');
      });
    });
    paintOutline();
  }

  // Cursor -> the pair whose region holds it: the nearest row centre within
  // half a row, then the (1px-floored) region on that row; the nearest region
  // wins when slop makes two eligible.
  window.__TT_HIT_TEST = function (pdiv, clientX, clientY, isTouch) {
    var fl = pdiv._fullLayout, xa = fl.xaxis, ya = fl.yaxis;
    var rect = pdiv.getBoundingClientRect();
    var px = clientX - rect.left - xa._offset;
    var y = ya.p2c(clientY - rect.top - ya._offset);
    var slop = isTouch ? TOUCH_SLOP_PX : SLOP_PX;
    var row = null, rowD = Infinity;
    for (var r = 0; r < rows.length; r++) {
      var dy = Math.abs(y - rows[r]);
      if (dy < rowD) { rowD = dy; row = rows[r]; }
    }
    if (row === null || rowD > 0.5) return null;
    var list = byRow[row];
    var best = -1, bestD = Infinity;
    for (var k = 0; k < list.length; k++) {
      var rg = list[k];
      if (!shown(pdiv, rg.pair)) continue;
      var x = segPx(xa, rg.s, rg.e);
      var dist = px < x[0] ? x[0] - px : (px > x[1] ? px - x[1] : 0);
      if (dist < bestD) { bestD = dist; best = rg.pair; }
    }
    if (best < 0 || bestD > slop) return null;
    setActive(best);
    return { html: S.pairs[best].html };
  };
  window.__TT_ON_HIDE = function () { setActive(-1); };

  (function bind() {
    var g = gd();
    if (!g || !g._fullLayout || !g.on) { setTimeout(bind, 100); return; }
    g.on('plotly_afterplot', redrawBars);
    redrawBars();
  })();
})();
