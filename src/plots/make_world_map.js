// Scope toggle for the world-map plot.
//
// Top-right pill bar that swaps geo.scope between world / NA / Europe / PNW.
// This toolbar is the only interaction the plot supports — drag-pan and
// scroll-zoom are disabled in the layout/config — so every visible state
// change goes through here.
//
// Every scope has its lonaxis/lataxis range computed in JS so the
// natural-earth-projected aspect of (lon range × lat range) matches the
// plot-div aspect ratio. Plotly's geolayer SVG sizes itself by fitting that
// projected box into the div; matching aspects means the geolayer rect
// equals the div rect (no internal letterbox). Same recompute fires on
// window resize so the geolayer stays div-matched as the user resizes.
// World follows the same path — it just happens to need the lat-clamp +
// lon-expand fallback because its data is enormously wider than tall.
(function () {
  var SCOPE_TRACES  = window.__PLOT_SCOPE_TRACES;
  var SCOPE_LAYOUTS = window.__PLOT_SCOPE_LAYOUTS;
  var SCOPE_BBOXES  = window.__PLOT_SCOPE_BBOXES;
  var LAT_CLAMPS    = window.__PLOT_LAT_CLAMPS;
  var currentScope  = window.__PLOT_DEFAULT_SCOPE;

  function pdiv() { return document.querySelector('.plotly-graph-div'); }
  var btns = document.querySelectorAll('#scope-toggle .rp-btn-pill');

  // d3-geo natural-earth1 polynomials. f = dx/dlon at latitude phi (radians).
  // The geolayer's projected x extent for a lon/lat box is:
  //   X = lon_span_rad × f(phi at the latitude in the box closest to the
  //                       equator — that's where meridians are widest).
  // The projected y extent is y(lat_max) - y(lat_min) where y is the
  // natural-earth lat polynomial. The geolayer's pixel aspect = X / Y.
  function ne1F(latDeg) {
    var phi = latDeg * Math.PI / 180;
    var p2 = phi * phi, p4 = p2 * p2, p10 = p4 * p4 * p2, p12 = p10 * p2;
    return 0.870700 - 0.131979 * p2 - 0.013791 * p4
         + 0.003971 * p10 - 0.001529 * p12;
  }
  function ne1Y(latDeg) {
    var phi = latDeg * Math.PI / 180;
    var p2 = phi * phi, p4 = p2 * p2, p6 = p4 * p2, p8 = p4 * p4, p10 = p8 * p2;
    var g = 1.007226 + 0.015085 * p2 - 0.044475 * p6
          + 0.028874 * p8 - 0.005916 * p10;
    return phi * g;
  }
  // d/dphi of ne1Y at latitude phi (radians). Used to invert ne1Y to
  // first order when expanding the lat range to match div aspect.
  function ne1DyDphi(latDeg) {
    var phi = latDeg * Math.PI / 180;
    var p2 = phi * phi, p4 = p2 * p2, p6 = p4 * p2, p8 = p4 * p4, p10 = p8 * p2;
    return 1.007226 + 0.045255 * p2 - 0.311325 * p6
         + 0.259866 * p8 - 0.065076 * p10;
  }
  function widestLat(lo, hi) {
    if (lo <= 0 && hi >= 0) return 0;
    return Math.abs(lo) < Math.abs(hi) ? lo : hi;
  }

  // Compute lon/lat range whose natural-earth-projected aspect == div
  // aspect, expanding from the cities' padded bbox so all data stays
  // visible. Whichever dim has shorter projected extent gets expanded.
  function aspectFitRanges(bbox, divAspect, latClamp) {
    var LAT_LO = latClamp[0], LAT_HI = latClamp[1];
    var lonMin = bbox.lon_min, lonMax = bbox.lon_max;
    var latMin = bbox.lat_min, latMax = bbox.lat_max;
    var f = ne1F(widestLat(latMin, latMax));
    var xRad = (lonMax - lonMin) * Math.PI / 180 * f;
    var yRad = ne1Y(latMax) - ne1Y(latMin);
    var dataAspect = xRad / yRad;
    if (divAspect >= dataAspect) {
      // Div is wider than data — expand lon symmetrically around lon_c.
      // Cap at 360° (full globe) so we never request more than a wraparound.
      var newXRad = yRad * divAspect;
      var newLonSpan = Math.min(360, newXRad / f * 180 / Math.PI);
      var lonC = (lonMin + lonMax) / 2;
      lonMin = lonC - newLonSpan / 2;
      lonMax = lonC + newLonSpan / 2;
    } else {
      // Div is taller than data — expand lat symmetrically around lat_c.
      // ne1Y is non-linear in phi, so the linear approximation
      //   L ≈ newYRad × 180/π / dy/dphi(lat_c)
      // is the seed; refine with a few Newton steps for the cases where
      // the lat range is wide enough that dy/dphi varies meaningfully
      // across it (e.g. NA spans ~50° of latitude).
      // Cap newLatSpan to 170° (= range [-85, 85]) so Newton never
      // evaluates the polynomial past phi=π/2 — past the pole the
      // polynomial misbehaves and dy/dphi flips sign, making Newton
      // diverge and producing a collapsed geolayer at narrow viewports.
      var latC = (latMin + latMax) / 2;
      var newYRad = xRad / divAspect;
      var newLatSpan = (newYRad / ne1DyDphi(latC)) * 180 / Math.PI;
      var maxSpan = LAT_HI - LAT_LO;
      newLatSpan = Math.min(newLatSpan, maxSpan);
      for (var iter = 0; iter < 5; iter++) {
        var lo = Math.max(LAT_LO, latC - newLatSpan / 2);
        var hi = Math.min(LAT_HI, latC + newLatSpan / 2);
        var actualY = ne1Y(hi) - ne1Y(lo);
        var err = actualY - newYRad;
        if (Math.abs(err) < 1e-7) break;
        var deriv = (ne1DyDphi(hi) + ne1DyDphi(lo)) / 2 * Math.PI / 180;
        if (deriv <= 0) break;  // safety
        newLatSpan -= err / deriv;
        newLatSpan = Math.min(newLatSpan, maxSpan);
      }
      latMin = latC - newLatSpan / 2;
      latMax = latC + newLatSpan / 2;
      // Final clamp + slide to keep within latClamp without losing the
      // requested span if possible.
      if (latMax > LAT_HI) { latMin -= (latMax - LAT_HI); latMax = LAT_HI; }
      if (latMin < LAT_LO) { latMax += (LAT_LO - latMin); latMin = LAT_LO; }
      latMin = Math.max(LAT_LO, latMin);
      latMax = Math.min(LAT_HI, latMax);

      // Secondary lon expansion: when lat hit the [-85, 85] clamp before
      // it could match div_aspect on its own (always true for world,
      // which is enormously wider than tall), the geolayer's projected
      // aspect is now smaller than div_aspect — i.e. the data is too
      // narrow horizontally for the div. Expand lon to compensate. After
      // the lat range changed, widestLat may have moved (e.g. a cross-
      // equator range puts widest at 0), so recompute f.
      var nowF = ne1F(widestLat(latMin, latMax));
      var nowY = ne1Y(latMax) - ne1Y(latMin);
      var nowX = (lonMax - lonMin) * Math.PI / 180 * nowF;
      if (nowX / nowY < divAspect) {
        var targetX = nowY * divAspect;
        var span = Math.min(360, targetX / nowF * 180 / Math.PI);
        var lonC2 = (lonMin + lonMax) / 2;
        lonMin = lonC2 - span / 2;
        lonMax = lonC2 + span / 2;
      }
    }
    return { lon: [lonMin, lonMax], lat: [latMin, latMax] };
  }

  function divAspect(gd) {
    var r = gd.getBoundingClientRect();
    return (r.width > 0 && r.height > 0) ? r.width / r.height : 1;
  }

  function buildLayoutFor(scopeId, gd) {
    var geo = JSON.parse(JSON.stringify(SCOPE_LAYOUTS[scopeId]));
    var bbox = SCOPE_BBOXES[scopeId];
    if (bbox) {
      var ranges = aspectFitRanges(bbox, divAspect(gd), LAT_CLAMPS[scopeId]);
      geo.lonaxis = { range: ranges.lon };
      geo.lataxis = { range: ranges.lat };
      // Re-center the projection on the data's lon_c. Plotly's regional
      // scopes default to a fixed rotation (e.g. NA → -100°), and when
      // the data isn't on that meridian the natural-earth meridians
      // curve asymmetrically across the box, throwing the projected
      // aspect off by a few percent. Centering rotation on lon_c makes
      // the projected box symmetric so X_span = lon_span × f(widest_lat)
      // holds exactly.
      var lonC = (ranges.lon[0] + ranges.lon[1]) / 2;
      geo.projection = Object.assign({}, geo.projection || {},
                                     { rotation: { lon: lonC, lat: 0, roll: 0 } });
    }
    return Object.assign({}, gd.layout, { geo: geo });
  }

  function applyScope(scopeId) {
    var gd = pdiv();
    if (!gd || !SCOPE_TRACES[scopeId] || !SCOPE_LAYOUTS[scopeId]) return;
    currentScope = scopeId;
    btns.forEach(function (b) {
      b.classList.toggle('is-active', b.getAttribute('data-value') === scopeId);
    });
    var traces = JSON.parse(JSON.stringify(SCOPE_TRACES[scopeId]));
    Plotly.react(gd, traces, buildLayoutFor(scopeId, gd));
  }

  function refit() {
    var gd = pdiv();
    if (!gd || !SCOPE_BBOXES[currentScope]) return;
    var ranges = aspectFitRanges(SCOPE_BBOXES[currentScope], divAspect(gd),
                                 LAT_CLAMPS[currentScope]);
    var lonC = (ranges.lon[0] + ranges.lon[1]) / 2;
    Plotly.relayout(gd, {
      'geo.lonaxis.range': ranges.lon,
      'geo.lataxis.range': ranges.lat,
      'geo.projection.rotation.lon': lonC,
    });
  }

  function bind() {
    var gd = pdiv();
    if (!gd) { setTimeout(bind, 50); return; }
    btns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyScope(btn.getAttribute('data-value'));
      });
    });
    var resizeTimer = null;
    window.addEventListener('resize', function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(refit, 100);
    });
    // First-paint refit: the figure shipped with the default scope's
    // static range (world). If default were ever a non-world scope this
    // would seed the aspect-fit; harmless no-op for world.
    if (SCOPE_BBOXES[currentScope]) refit();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
