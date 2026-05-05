"""Build the tabbed shell at output/index.html.

Used by the local dev server (scripts/dev.py) and by the production
build (scripts/run_plots.sh). The shell is a thin tab bar over an iframe
that loads each plot HTML by slug. The plot HTMLs themselves are
self-contained.

With ``--admin`` the shell renders an extra (initially hidden) "Admin"
tab; client-side JS unhides it after a /api/auth/me probe confirms the
viewer is the admin. The admin tab loads /admin.html (absolute path)
rather than the slug-relative pattern used by every other tab.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


# Tab order + human labels. The slug must match the HTML filename (without
# .html). If a slug's HTML doesn't exist yet, the tab is hidden until the
# corresponding plot script runs.
TABS = [
    ("dashboard",              "Dashboard"),
    ("race_pace_all",          "Races"),
    ("race_pace_by_distance",  "Race Distances"),
    ("cs_timeline",            "Fitness"),
    ("training_quality",       "Training"),
    ("workouts",               "Workouts"),
    ("long_runs",              "Long Runs"),
    ("recovery_pace",          "Recovery"),
    ("qualitative_trends",     "Misc. Trends"),
    ("mileage_by_geography",   "Locations"),
    ("world_map",              "World Map"),
]


def _empty_shell() -> str:
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        "<title>Max's Running Data</title>"
        '<style>body{background:#1a1a1a;color:#eee;'
        'font-family:system-ui,sans-serif;padding:2rem}</style>'
        '</head><body>'
        '<h1>No plots yet</h1>'
        '<p>Run <code>./scripts/run_plots.sh</code> or save a plot script.'
        '</p></body></html>\n'
    )


def render_shell(tabs, include_admin: bool = False) -> str:
    """Render the tab-shell HTML.

    ``tabs`` is a list of (slug, label) for plots whose HTML actually exists.
    ``include_admin`` adds an admin tab whose visibility is gated client-side
    on /api/auth/me and which loads /admin.html absolutely.
    """
    if not tabs:
        return _empty_shell()

    buttons = '\n'.join(
        f'      <button class="tab" data-slug="{slug}">{label}</button>'
        for slug, label in tabs
    )
    if include_admin:
        buttons += (
            '\n      <button class="tab admin-only" data-slug="admin" '
            'data-href="/admin.html">Admin</button>'
        )

    default_slug = tabs[0][0]
    admin_css = (
        '#tabbar .tab.admin-only { display: none; }\n'
        '  #tabbar.show-admin .tab.admin-only { display: block; }\n'
    ) if include_admin else ''
    admin_js = _ADMIN_REVEAL_JS if include_admin else ''

    return f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Max's Running Data</title>
<style>
  html, body {{
    margin: 0; padding: 0; height: 100%;
    background: #1a1a1a; color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
  }}
  #tabbar {{
    display: flex; align-items: stretch;
    background: #111; border-bottom: 1px solid #333;
    padding: 0 4px; height: 36px;
    overflow-x: auto;
    user-select: none;
  }}
  #tabbar .brand {{
    display: flex; align-items: center;
    padding: 0 18px 0 12px;
    color: #fff;
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
    border-right: 1px solid #2a2a2a;
    margin-right: 6px;
  }}
  #tabbar .tab {{
    background: transparent; color: #aaa;
    border: 0; border-bottom: 2px solid transparent;
    padding: 0 14px; margin: 0 1px;
    font: inherit; font-size: 13px;
    cursor: pointer; white-space: nowrap;
    transition: color 0.12s, background 0.12s, border-bottom-color 0.12s;
  }}
  #tabbar .tab:hover {{ color: #eee; background: #1a1a1a; }}
  #tabbar .tab.active {{
    color: #fff;
    border-bottom-color: #93f;
    background: #1a1a1a;
  }}
  {admin_css}#frame-wrap {{
    position: absolute; left: 0; right: 0;
    top: 36px; bottom: 0;
  }}
  /* Stacked iframe pool. Each visited tab gets its own iframe, kept alive
     for the page's lifetime — revisits are paint-only, no fetch or layout.
     Hide via visibility (not display:none); a Plotly figure laid out inside
     a zero-size container renders broken and never auto-recovers. */
  #frame-wrap iframe {{
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    border: 0; background: #1a1a1a;
    visibility: hidden; pointer-events: none;
    z-index: 0;
  }}
  #frame-wrap iframe.active {{
    visibility: visible; pointer-events: auto;
    z-index: 1;
  }}
  #spinner {{
    position: absolute; inset: 0; z-index: 2;
    display: none; align-items: center; justify-content: center;
    background: #1a1a1a;
    pointer-events: none;
  }}
  #spinner.show {{ display: flex; }}
  #spinner::after {{
    content: '';
    width: 32px; height: 32px;
    border: 3px solid #333;
    border-top-color: #93f;
    border-radius: 50%;
    animation: rp-spin 0.8s linear infinite;
  }}
  @keyframes rp-spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
  <div id="tabbar">
    <div class="brand">Max's Running Data</div>
{buttons}
  </div>
  <div id="frame-wrap">
    <div id="spinner"></div>
  </div>
<script>
(function () {{
  var DEFAULT = {default_slug!r};
  var STORE_KEY = 'rp-active-tab';
  var POLL_INTERVAL_MS = 50;
  var POLL_TIMEOUT_MS = 5000;

  var bar = document.getElementById('tabbar');
  var wrap = document.getElementById('frame-wrap');
  var spinner = document.getElementById('spinner');
  var buttons = bar.querySelectorAll('button.tab');
  var slugs = Array.prototype.map.call(buttons, function (b) {{
    return b.dataset.slug;
  }});
  var pool = new Map();   // slug -> {{ slug, iframe, ready }}
  var activeSlug = null;

  function btnFor(slug) {{
    return bar.querySelector('button.tab[data-slug="' + slug + '"]');
  }}
  function isVisible(b) {{
    return b && getComputedStyle(b).display !== 'none';
  }}
  function visBySlug(slug) {{
    return isVisible(btnFor(slug));
  }}

  function pickInitial() {{
    var params = new URLSearchParams(window.location.search);
    var fromUrl = params.get('tab');
    if (fromUrl && slugs.indexOf(fromUrl) >= 0 && visBySlug(fromUrl)) return fromUrl;
    var stored = null;
    try {{ stored = localStorage.getItem(STORE_KEY); }} catch (e) {{}}
    if (stored && slugs.indexOf(stored) >= 0 && visBySlug(stored)) return stored;
    return DEFAULT;
  }}

  function srcFor(slug, btn) {{
    var href = btn && btn.dataset && btn.dataset.href;
    if (href) return href;
    return slug + '.html';
  }}

  function setSpinner(show) {{
    spinner.classList.toggle('show', show);
  }}

  function markReady(entry) {{
    if (entry.ready) return;
    entry.ready = true;
    if (entry.slug === activeSlug) setSpinner(false);
  }}

  // Poll the iframe doc for Plotly's rendered SVG. Plot HTMLs also
  // postMessage 'rp-plot-ready' once `plotly_afterplot` fires (see
  // src/plotting/render.py); polling is the fallback for HTMLs that
  // predate the postMessage hook. Whichever fires first wins.
  function pollReady(entry) {{
    var start = Date.now();
    var timer = setInterval(function () {{
      if (entry.ready) {{ clearInterval(timer); return; }}
      if (Date.now() - start > POLL_TIMEOUT_MS) {{
        clearInterval(timer);
        markReady(entry);
        return;
      }}
      var doc;
      try {{ doc = entry.iframe.contentDocument; }}
      catch (e) {{ clearInterval(timer); markReady(entry); return; }}
      if (!doc) return;
      if (doc.querySelector('.plotly-graph-div .main-svg')) {{
        clearInterval(timer);
        markReady(entry);
      }}
    }}, POLL_INTERVAL_MS);
  }}

  function getOrCreate(slug, btn) {{
    var entry = pool.get(slug);
    if (entry) return entry;
    var iframe = document.createElement('iframe');
    iframe.setAttribute('name', 'plot-frame-' + slug);
    iframe.dataset.slug = slug;
    iframe.src = srcFor(slug, btn);
    wrap.appendChild(iframe);
    entry = {{ slug: slug, iframe: iframe, ready: false }};
    pool.set(slug, entry);
    iframe.addEventListener('load', function () {{
      // Non-plot pages (admin) have no .plotly-graph-div — mark ready
      // immediately so the spinner clears without waiting on the timeout.
      var doc;
      try {{ doc = entry.iframe.contentDocument; }}
      catch (e) {{ markReady(entry); return; }}
      if (!doc || !doc.querySelector('.plotly-graph-div')) {{
        markReady(entry);
        return;
      }}
      if (doc.querySelector('.plotly-graph-div .main-svg')) {{
        markReady(entry);
      }} else {{
        pollReady(entry);
      }}
    }});
    return entry;
  }}

  function activate(slug, opts) {{
    opts = opts || {{}};
    var btn = btnFor(slug);
    if (!btn || !isVisible(btn)) return;
    Array.prototype.forEach.call(buttons, function (b) {{
      b.classList.toggle('active', b.dataset.slug === slug);
    }});
    var entry = getOrCreate(slug, btn);
    pool.forEach(function (e) {{
      e.iframe.classList.toggle('active', e.slug === slug);
    }});
    activeSlug = slug;
    setSpinner(!entry.ready);
    try {{ localStorage.setItem(STORE_KEY, slug); }} catch (e) {{}}
    if (!opts.skipUrl) {{
      var url = new URL(window.location.href);
      url.searchParams.set('tab', slug);
      window.history.replaceState({{ slug: slug }}, '', url);
    }}
    document.title = (function () {{
      var b = bar.querySelector('button.tab.active');
      return b ? (b.textContent.trim() + " - Max's Running Data") : "Max's Running Data";
    }})();
  }}

  Array.prototype.forEach.call(buttons, function (b) {{
    b.addEventListener('click', function () {{ activate(b.dataset.slug); }});
  }});

  function cycle(direction) {{
    var visible = slugs.filter(visBySlug);
    var idx = visible.indexOf(activeSlug);
    if (idx < 0) return;
    var next = idx + direction;
    if (next < 0) next = visible.length - 1;
    if (next >= visible.length) next = 0;
    activate(visible[next]);
  }}

  function handleKey(e) {{
    if (e.target && e.target.tagName === 'INPUT') return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (!e.altKey) return;
    cycle(e.key === 'ArrowLeft' ? -1 : 1);
    if (e.preventDefault) e.preventDefault();
  }}
  window.addEventListener('keydown', handleKey);

  window.addEventListener('message', function (e) {{
    var d = e.data;
    if (!d) return;
    if (d.type === 'rp-tab-key') {{
      if (d.key === 'ArrowLeft')  cycle(-1);
      if (d.key === 'ArrowRight') cycle(1);
      return;
    }}
    if (d.type === 'rp-plot-ready') {{
      pool.forEach(function (entry) {{
        if (entry.iframe.contentWindow === e.source) markReady(entry);
      }});
      return;
    }}
  }});
{admin_js}
  activate(pickInitial(), {{ skipUrl: false }});
}})();
</script>
</body></html>
'''


_ADMIN_REVEAL_JS = '''
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
'''


def write_index(output_dir: Path = DEFAULT_OUTPUT_DIR,
                include_admin: bool = False) -> bool:
    """Write output_dir/index.html based on which plot HTMLs exist there.

    Returns True if the file was written, False if unchanged on disk.
    """
    output_dir.mkdir(exist_ok=True)
    index_path = output_dir / "index.html"
    existing = {p.stem for p in output_dir.glob("*.html") if p.name != "index.html"}
    tabs = [(slug, label) for slug, label in TABS if slug in existing]
    extras = sorted(existing - {slug for slug, _ in TABS})
    for slug in extras:
        tabs.append((slug, slug.replace('_', ' ')))
    new_html = render_shell(tabs, include_admin=include_admin)
    if index_path.exists() and index_path.read_text() == new_html:
        return False
    index_path.write_text(new_html)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--admin", action="store_true",
                   help="Include the (hidden-by-default) admin tab.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = p.parse_args(argv)
    wrote = write_index(args.output_dir, include_admin=args.admin)
    if wrote:
        print(f"Wrote {args.output_dir / 'index.html'}")
    else:
        print(f"{args.output_dir / 'index.html'} unchanged")


if __name__ == "__main__":
    sys.exit(main())
