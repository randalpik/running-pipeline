// Tab shell: tabbar + iframe pool + spinner.
//
// State lives in `pool` (Map slug -> entry). Each entry has its own
// iframe; the active iframe is the one with class .active. Switching
// tabs is a CSS toggle on iframe class + tabbar button class — the
// document never re-fetches anything that's already been visited.
//
// Default slug comes from <meta name="rp-default-slug" content="...">.
// Admin reveal runs only if <body> has class has-admin.
(function () {
  var meta = document.querySelector('meta[name="rp-default-slug"]');
  var DEFAULT = meta ? meta.getAttribute('content') : '';
  var STORE_KEY = 'rp-active-tab';
  var POLL_INTERVAL_MS = 50;
  var POLL_TIMEOUT_MS = 5000;

  var bar = document.getElementById('tabbar');
  var wrap = document.getElementById('frame-wrap');
  var spinner = document.getElementById('spinner');
  var menuBtn = document.getElementById('rp-menu-btn');
  var scrim = document.getElementById('rp-menu-scrim');

  // Mobile breakpoint — MUST mirror the media query in shell.css. Plot pages
  // can't derive this themselves (their viewport is already rotated to
  // landscape), so the shell pushes it into every iframe as an
  // 'rp-shell-mode' message; the receiver in each plot page (see
  // src/plotting/render.py) mirrors it as html.rp-mobile.
  var MOBILE_MQ = window.matchMedia('(pointer: coarse) and (max-width: 940px)');
  var PORTRAIT_MQ = window.matchMedia('(orientation: portrait)');
  function postMode(win) {
    if (!win) return;
    try { win.postMessage({ type: 'rp-shell-mode', mobile: MOBILE_MQ.matches }, '*'); }
    catch (e) {}
  }
  function broadcastMode() {
    pool.forEach(function (e) { postMode(e.iframe.contentWindow); });
  }

  function setMenu(open) {
    document.body.classList.toggle('rp-menu-open', !!open);
    if (menuBtn) menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function closeMenu() { setMenu(false); }
  var buttons = bar.querySelectorAll('button.tab');
  var slugs = Array.prototype.map.call(buttons, function (b) {
    return b.dataset.slug;
  });
  var pool = new Map();   // slug -> { slug, iframe, ready }
  var activeSlug = null;

  function btnFor(slug) {
    return bar.querySelector('button.tab[data-slug="' + slug + '"]');
  }
  function isVisible(b) {
    return b && getComputedStyle(b).display !== 'none';
  }
  function visBySlug(slug) {
    return isVisible(btnFor(slug));
  }

  function pickInitial() {
    var params = new URLSearchParams(window.location.search);
    var fromUrl = params.get('tab');
    if (fromUrl && slugs.indexOf(fromUrl) >= 0 && visBySlug(fromUrl)) return fromUrl;
    var stored = null;
    try { stored = localStorage.getItem(STORE_KEY); } catch (e) {}
    if (stored && slugs.indexOf(stored) >= 0 && visBySlug(stored)) return stored;
    return DEFAULT;
  }

  function srcFor(slug, btn) {
    var href = btn && btn.dataset && btn.dataset.href;
    var src = href || (slug + '.html');
    // Hand the plot page our mode on the URL so it knows at <head> time.
    // The rp-shell-mode message below can only land after the iframe loads,
    // which is too late for the reshaped pages — they would render desktop
    // first and visibly reflow. See _scaffold/mobile_head.js.
    if (MOBILE_MQ.matches) src += (src.indexOf('?') < 0 ? '?' : '&') + 'rpm=1';
    return src;
  }

  function setSpinner(show) {
    spinner.classList.toggle('show', show);
  }

  function markReady(entry) {
    if (entry.ready) return;
    entry.ready = true;
    if (entry.slug === activeSlug) setSpinner(false);
  }

  // Poll the iframe doc for Plotly's rendered SVG. Plot HTMLs also
  // postMessage 'rp-plot-ready' once `plotly_afterplot` fires (see
  // src/plotting/render.py); polling is the fallback for HTMLs that
  // predate the postMessage hook. Whichever fires first wins.
  function pollReady(entry) {
    var start = Date.now();
    var timer = setInterval(function () {
      if (entry.ready) { clearInterval(timer); return; }
      if (Date.now() - start > POLL_TIMEOUT_MS) {
        clearInterval(timer);
        markReady(entry);
        return;
      }
      var doc;
      try { doc = entry.iframe.contentDocument; }
      catch (e) { clearInterval(timer); markReady(entry); return; }
      if (!doc) return;
      if (doc.querySelector('.plotly-graph-div .main-svg')) {
        clearInterval(timer);
        markReady(entry);
      }
    }, POLL_INTERVAL_MS);
  }

  function getOrCreate(slug, btn) {
    var entry = pool.get(slug);
    if (entry) return entry;
    var iframe = document.createElement('iframe');
    iframe.setAttribute('name', 'plot-frame-' + slug);
    iframe.dataset.slug = slug;
    iframe.src = srcFor(slug, btn);
    wrap.appendChild(iframe);
    entry = { slug: slug, iframe: iframe, ready: false };
    pool.set(slug, entry);
    iframe.addEventListener('load', function () {
      // Earliest correct moment to push the shell mode — the receiver
      // script doesn't exist inside the iframe before load.
      postMode(entry.iframe.contentWindow);
      // Non-plot pages (admin) have no .plotly-graph-div — mark ready
      // immediately so the spinner clears without waiting on the timeout.
      var doc;
      try { doc = entry.iframe.contentDocument; }
      catch (e) { markReady(entry); return; }
      if (!doc || !doc.querySelector('.plotly-graph-div')) {
        markReady(entry);
        return;
      }
      if (doc.querySelector('.plotly-graph-div .main-svg')) {
        markReady(entry);
      } else {
        pollReady(entry);
      }
    });
    return entry;
  }

  function activate(slug, opts) {
    opts = opts || {};
    var btn = btnFor(slug);
    if (!btn || !isVisible(btn)) return;
    closeMenu();  // covers tab taps, cycle() and the admin re-activate
    Array.prototype.forEach.call(buttons, function (b) {
      b.classList.toggle('active', b.dataset.slug === slug);
    });
    var entry = getOrCreate(slug, btn);
    pool.forEach(function (e) {
      e.iframe.classList.toggle('active', e.slug === slug);
    });
    activeSlug = slug;
    setSpinner(!entry.ready);
    try { localStorage.setItem(STORE_KEY, slug); } catch (e) {}
    if (!opts.skipUrl) {
      var url = new URL(window.location.href);
      url.searchParams.set('tab', slug);
      window.history.replaceState({ slug: slug }, '', url);
    }
    document.title = (function () {
      var b = bar.querySelector('button.tab.active');
      return b ? (b.textContent.trim() + " - Max's Running Data") : "Max's Running Data";
    })();
  }

  Array.prototype.forEach.call(buttons, function (b) {
    b.addEventListener('click', function () { activate(b.dataset.slug); });
  });

  // Profile switcher: navigate the whole shell to the chosen profile's root.
  var profileSelect = bar.querySelector('.profile-switch select');
  if (profileSelect) {
    profileSelect.addEventListener('change', function () {
      if (profileSelect.value) window.location.assign(profileSelect.value);
    });
  }

  // Hamburger drawer (mobile only — the button is display:none on desktop,
  // so all of this is inert there).
  if (menuBtn) {
    menuBtn.addEventListener('click', function () {
      setMenu(!document.body.classList.contains('rp-menu-open'));
    });
  }
  if (scrim) scrim.addEventListener('click', closeMenu);
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeMenu();
  });

  // Crossing the mobile breakpoint or flipping orientation: the drawer's
  // geometry assumptions changed (close it) and the iframes need the new
  // html.rp-mobile state. Rotation itself is pure CSS; Plotly relayout is
  // automatic (responsive:true uses a ResizeObserver on the graph div).
  function onModeChange() {
    closeMenu();
    broadcastMode();
  }
  function onMQ(mq, fn) {
    if (mq.addEventListener) mq.addEventListener('change', fn);
    else if (mq.addListener) mq.addListener(fn);  // legacy Safari
  }
  onMQ(MOBILE_MQ, onModeChange);
  onMQ(PORTRAIT_MQ, onModeChange);

  // Drawer scrolling. The tab list is taller than the rotated stage, and
  // Chromium does not route touch pans into a scroll container under a
  // rotated ancestor — verified: this same drawer scrolls natively when the
  // stage is unrotated and is completely dead once the rotation is on. So
  // when (and only when) we're rotated, drive the scroll ourselves.
  //
  // Gesture mapping: the stage is rotate(90deg) translateY(-100%) about
  // 0 0, so stage-local (u, v) paints at screen (W - v, u). The user's
  // "swipe up" is local -v, i.e. a screen +x drag, and scrolling down the
  // list means scrollTop += (screen dx).
  function rotated() { return MOBILE_MQ.matches && PORTRAIT_MQ.matches; }
  if (window.rpTouchScroll && bar) {
    window.rpTouchScroll.attach(bar, {
      decide: function (dx, dy) {
        if (!rotated()) return false;   // unrotated scrolls natively
        if (bar.scrollHeight <= bar.clientHeight) return false;
        return Math.abs(dx) > Math.abs(dy);   // local vertical
      },
      scrollerFor: function () { return bar; },
      deltaFor: function (stepX) { return stepX; },
    });
  }

  function cycle(direction) {
    var visible = slugs.filter(visBySlug);
    var idx = visible.indexOf(activeSlug);
    if (idx < 0) return;
    var next = idx + direction;
    if (next < 0) next = visible.length - 1;
    if (next >= visible.length) next = 0;
    activate(visible[next]);
  }

  function handleKey(e) {
    if (e.target && e.target.tagName === 'INPUT') return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (!e.altKey) return;
    cycle(e.key === 'ArrowLeft' ? -1 : 1);
    if (e.preventDefault) e.preventDefault();
  }
  window.addEventListener('keydown', handleKey);

  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d) return;
    if (d.type === 'rp-tab-key') {
      if (d.key === 'ArrowLeft')  cycle(-1);
      if (d.key === 'ArrowRight') cycle(1);
      return;
    }
    if (d.type === 'rp-plot-ready') {
      pool.forEach(function (entry) {
        if (entry.iframe.contentWindow === e.source) markReady(entry);
      });
      return;
    }
  });

  if (document.body.classList.contains('has-admin')) {
    fetch('/api/auth/me', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (me) {
        if (!me || !me.isAdmin) return;
        bar.classList.add('show-admin');
        // The admin tab is hidden until this probe resolves, so an initial
        // ?tab=admin URL is filtered out by pickInitial's visibility check.
        // Re-activate now that the tab exists.
        var params = new URLSearchParams(window.location.search);
        if (params.get('tab') === 'admin' && activeSlug !== 'admin') {
          activate('admin');
        }
      })
      .catch(function () {});
  }

  activate(pickInitial(), { skipUrl: false });

  // ---- On-device geometry readout: append ?rpdiag=1 to the site URL ----
  // Several layout failures here reproduce only on real hardware, where
  // there is no devtools. This prints both sides of the iframe boundary so
  // a screenshot is enough to diagnose. Inert without the flag.
  if (/[?&]rpdiag=1/.test(window.location.search)) {
    var stage = document.getElementById('rp-stage');
    var box = document.createElement('div');
    box.style.cssText =
      'position:absolute;left:0;top:0;z-index:9999;max-width:100%;' +
      'background:rgba(0,0,0,0.88);color:#0f0;font:10px/1.35 monospace;' +
      'padding:6px 8px;white-space:pre;pointer-events:none;' +
      'border:1px solid #0a0;overflow:hidden';
    (stage || document.body).appendChild(box);
    setInterval(function () {
      var vv = window.visualViewport;
      var st = stage ? stage.getBoundingClientRect() : null;
      var fr = document.querySelector('#frame-wrap iframe.active');
      var frr = fr ? fr.getBoundingClientRect() : null;
      var lines = [
        'SHELL win=' + window.innerWidth + 'x' + window.innerHeight +
          ' dpr=' + window.devicePixelRatio,
        '  vv=' + (vv ? Math.round(vv.width) + 'x' + Math.round(vv.height) +
          ' scale=' + vv.scale.toFixed(2) : 'n/a'),
        '  stage box=' + (stage ? stage.offsetWidth + 'x' + stage.offsetHeight : '-') +
          ' rect=' + (st ? Math.round(st.width) + 'x' + Math.round(st.height) : '-'),
        '  iframe box=' + (fr ? fr.offsetWidth + 'x' + fr.offsetHeight : '-') +
          ' rect=' + (frr ? Math.round(frr.width) + 'x' + Math.round(frr.height) : '-'),
        '  mobileMQ=' + (MOBILE_MQ.matches ? 1 : 0) +
          ' portrait=' + (PORTRAIT_MQ.matches ? 1 : 0),
      ];
      try {
        var w = fr && fr.contentWindow, d = fr && fr.contentDocument;
        if (w && d && d.documentElement) {
          var ivv = w.visualViewport;
          var gd = d.querySelector('.plotly-graph-div');
          var tb = d.querySelector('.rp-title-bar');
          lines.push('FRAME win=' + w.innerWidth + 'x' + w.innerHeight +
            ' icb=' + d.documentElement.clientWidth +
            ' body=' + d.body.clientWidth);
          lines.push('  vv=' + (ivv ? Math.round(ivv.width) + ' scale=' +
            ivv.scale.toFixed(2) : 'n/a') +
            ' titleBar=' + (tb ? Math.round(tb.getBoundingClientRect().width) : '-'));
          lines.push('  gd=' + (gd ? gd.clientWidth + 'x' + gd.clientHeight : '-') +
            ' fig=' + (gd && gd._fullLayout
              ? Math.round(gd._fullLayout.width) + 'x' +
                Math.round(gd._fullLayout.height) : '-'));
          lines.push('  rpMobile=' +
            (d.documentElement.classList.contains('rp-mobile') ? 1 : 0) +
            ' scroll=' + (d.body.classList.contains('rp-scroll') ? 1 : 0) +
            ' latched=' + (w.__rpLayoutLatched ? 1 : 0));
          var tr = w.__rpTrace || [];
          lines = lines.concat(tr.slice(-7).map(function (l) { return ' ' + l; }));
        } else {
          lines.push('FRAME <not readable>');
        }
      } catch (e) {
        lines.push('FRAME <cross-origin: ' + e.name + '>');
      }
      box.textContent = lines.join('\n');
    }, 500);
  }
})();
