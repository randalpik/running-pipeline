/* Manual touch-pan scrolling.
 *
 * Two Chromium behaviors strand scrolling in this app, both verified by
 * experiment rather than inferred:
 *
 *  1. Touch pans are NOT routed into a scroll container that lives under a
 *     ROTATED ancestor. The shell's hamburger drawer scrolls normally when
 *     the stage is unrotated (portrait phone held landscape) and is
 *     completely dead — both directions — once the portrait rotation is on.
 *     Taps still hit-test correctly through the transform; only the scroll
 *     gesture is lost. This strands the drawer and every scroll-mode plot
 *     page inside the rotated iframe.
 *
 *  2. Plotly claims touch drags over the plot area for its zoom box (it
 *     preventDefaults, so no native scroll happens even unrotated). On the
 *     tall mobile pages the graph covers nearly the whole document, so a
 *     swipe meant to scroll instead zooms an axis and re-renders — the
 *     "constant relayout" failure mode.
 *
 * So we recognize the pan ourselves and drive scrollTop, with a light
 * fling. Call sites decide WHICH gestures to claim, so Plotly keeps every
 * gesture we don't need:
 *
 *   rpTouchScroll.attach(target, {
 *     decide(dx, dy, startTarget) -> claim this gesture?  (delta from origin)
 *     scrollerFor(startTarget, dx, dy) -> element to scroll (null = drop it)
 *     deltaFor(sx, sy) -> scrollTop delta for this step's (dx, dy)
 *     capture          -> listen in the capture phase (needed to beat
 *                         Plotly's own handlers to the event)
 *   })
 *
 * Claimed gestures are preventDefault'd + stopPropagation'd, so they never
 * reach Plotly and never synthesize a click. Unclaimed gestures are passed
 * through untouched — the listener goes inert after the first few pixels.
 *
 * IMPORTANT for `decide`: Plotly's dragElement registers its touchmove /
 * touchend on `document` in the BUBBLE phase when a touch lands on one of
 * its drag rects. A capture-phase stopPropagation therefore also suppresses
 * its cleanup handler, leaving gd._dragging stuck true and every later
 * zoom/tap broken (observed). So never claim a gesture that STARTED on
 * Plotly's draglayer — decide on the start target, not just direction.
 */
(function () {
  if (window.rpTouchScroll) return;

  var SLOP_PX  = 8;      // travel before a gesture's direction is decided
  var FRICTION = 0.94;   // per-frame velocity decay for the fling
  var MIN_FLING_V = 0.12;  // px/ms below which a release doesn't fling
  var MIN_STEP_PX = 0.5;   // fling stops under this per-frame step

  var flingToken = 0;

  function fling(el, vPxPerMs) {
    var token = ++flingToken;
    if (Math.abs(vPxPerMs) < MIN_FLING_V) return;
    var step = vPxPerMs * 16;   // px per ~frame
    function frame() {
      if (token !== flingToken) return;   // superseded by a new gesture
      step *= FRICTION;
      if (Math.abs(step) < MIN_STEP_PX) return;
      var before = el.scrollTop;
      el.scrollTop = before + step;
      if (el.scrollTop === before) return;   // hit an edge
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function attach(target, opts) {
    var cap = !!opts.capture;
    var st = null;

    target.addEventListener('touchstart', function (e) {
      flingToken++;   // any touch cancels an in-flight fling
      if (e.touches.length !== 1) { st = null; return; }
      var t = e.touches[0];
      st = { x0: t.clientX, y0: t.clientY, x: t.clientX, y: t.clientY,
             target: e.target,
             claimed: false, dead: false, el: null, v: 0, ts: Date.now() };
    }, { capture: cap, passive: true });

    target.addEventListener('touchmove', function (e) {
      if (!st || st.dead) return;
      if (e.touches.length !== 1) { st.dead = true; return; }
      var t = e.touches[0];
      if (!st.claimed) {
        var dx = t.clientX - st.x0, dy = t.clientY - st.y0;
        if (Math.abs(dx) < SLOP_PX && Math.abs(dy) < SLOP_PX) return;
        if (!opts.decide(dx, dy, st.target)) { st.dead = true; return; }
        st.el = opts.scrollerFor(st.target, dx, dy);
        if (!st.el) { st.dead = true; return; }
        st.claimed = true;
        st.x = t.clientX; st.y = t.clientY;   // measure steps from here
      }
      var d = opts.deltaFor(t.clientX - st.x, t.clientY - st.y);
      st.x = t.clientX; st.y = t.clientY;
      st.el.scrollTop += d;
      var now = Date.now();
      st.v = d / Math.max(1, now - st.ts);
      st.ts = now;
      if (e.cancelable) e.preventDefault();
      e.stopPropagation();
    }, { capture: cap, passive: false });

    function end(e) {
      if (st && st.claimed) {
        // Stale velocity (finger paused before lifting) shouldn't fling.
        if (Date.now() - st.ts < 90) fling(st.el, st.v);
        if (e.cancelable) e.preventDefault();
        e.stopPropagation();
      }
      st = null;
    }
    target.addEventListener('touchend', end, { capture: cap, passive: false });
    target.addEventListener('touchcancel', function () { st = null; },
                            { capture: cap, passive: true });
  }

  window.rpTouchScroll = { attach: attach };
})();
