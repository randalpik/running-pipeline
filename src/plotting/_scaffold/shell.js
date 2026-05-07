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
    if (href) return href;
    return slug + '.html';
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
})();
