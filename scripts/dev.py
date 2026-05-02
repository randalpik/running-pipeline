"""Dev server for plot iteration: livereload + per-script rerun.

Usage:
    python scripts/dev.py

Then open http://localhost:5500/ (or wait for it to open automatically).

Watches every src/plots/*.py — saving a plot script reruns just that script
and reloads any open browser tab pointed at its HTML output. Editing a
src/shared/*.py helper reruns only the plots that import it (dep map built
once at startup by parsing each plot's imports). Editing src/plotting/**/*
or any data/*.csv reruns every plot.

All rebuilds run on a background worker thread so the Tornado IOLoop stays
responsive — tab clicks keep working while plots regenerate. Saves that land
during an in-flight rebuild coalesce into a single follow-up rebuild.

Pipeline scripts (build_dataset / parse_workouts / bayes_cs_fit) are
intentionally NOT watched — those are deliberate batch operations and stay
on scripts/run_pipeline.sh.

The landing page (output/index.html) is a tab shell: a top tab bar swaps
which plot HTML is loaded into a single iframe. Each plot stays its own
self-contained document, so per-plot CSS/JS scopes don't collide and
livereload reloads only the active iframe on its source change.
"""
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from livereload import Server

REPO_ROOT  = Path(__file__).resolve().parent.parent
PLOTS_DIR  = REPO_ROOT / "src" / "plots"
OUTPUT_DIR = REPO_ROOT / "output"
INDEX_PATH = OUTPUT_DIR / "index.html"
PORT       = 5500

# livereload globs are evaluated relative to the process cwd; chdir so the
# watch patterns ("src/plots/*.py", "data/*.csv") resolve correctly regardless
# of where the user invokes the script from.
os.chdir(REPO_ROOT)


# Tab order + human labels. The slug must match the HTML filename (without
# .html). If a slug's HTML doesn't exist yet, the tab is hidden until the
# corresponding plot script runs.
TABS = [
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


def _shell_html(tabs):
    """Render the tab-shell HTML. ``tabs`` is a list of (slug, label) for
    plots whose HTML actually exists in OUTPUT_DIR."""
    if not tabs:
        return (
            '<!doctype html>\n<html><head><meta charset="utf-8">'
            '<title>plots</title>'
            '<style>body{background:#1a1a1a;color:#eee;'
            'font-family:system-ui,sans-serif;padding:2rem}</style>'
            '</head><body>'
            '<h1>No plots yet</h1>'
            '<p>Run <code>./scripts/run_plots.sh</code> or save a plot script.'
            '</p></body></html>\n'
        )

    buttons = '\n'.join(
        f'      <button class="tab" data-slug="{slug}">{label}</button>'
        for slug, label in tabs
    )
    default_slug = tabs[0][0]

    return f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<title>running-pipeline plots</title>
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
    border-bottom-color: #9cf;
    background: #1a1a1a;
  }}
  #frame-wrap {{
    position: absolute; left: 0; right: 0;
    top: 36px; bottom: 0;
  }}
  #plot-frame {{
    width: 100%; height: 100%;
    border: 0; display: block;
    background: #1a1a1a;
  }}
</style>
</head>
<body>
  <div id="tabbar">
    <div class="brand">Max's Running Data</div>
{buttons}
  </div>
  <div id="frame-wrap">
    <iframe id="plot-frame" name="plot-frame" src="about:blank"></iframe>
  </div>
<script>
(function () {{
  var DEFAULT = {default_slug!r};
  var STORE_KEY = 'rp-active-tab';
  var bar = document.getElementById('tabbar');
  var frame = document.getElementById('plot-frame');
  var buttons = bar.querySelectorAll('button.tab');
  var slugs = Array.prototype.map.call(buttons, function (b) {{
    return b.dataset.slug;
  }});

  function pickInitial() {{
    var params = new URLSearchParams(window.location.search);
    var fromUrl = params.get('tab');
    if (fromUrl && slugs.indexOf(fromUrl) >= 0) return fromUrl;
    var stored = null;
    try {{ stored = localStorage.getItem(STORE_KEY); }} catch (e) {{}}
    if (stored && slugs.indexOf(stored) >= 0) return stored;
    return DEFAULT;
  }}

  function activate(slug, opts) {{
    opts = opts || {{}};
    var found = false;
    Array.prototype.forEach.call(buttons, function (b) {{
      var on = b.dataset.slug === slug;
      b.classList.toggle('active', on);
      if (on) found = true;
    }});
    if (!found) return;
    if (frame.dataset.slug !== slug) {{
      // Cache-bust on every src assignment. Without this, the parent
      // reload triggered by livereload would re-set frame.src to the
      // same string the iframe already had, and the browser treats a
      // same-URL src assignment as a no-op (no new fetch). Appending
      // Date.now() guarantees the iframe loads the freshly-regenerated
      // HTML on every parent reload + every tab activation.
      frame.src = slug + '.html?v=' + Date.now();
      frame.dataset.slug = slug;
    }}
    try {{ localStorage.setItem(STORE_KEY, slug); }} catch (e) {{}}
    if (!opts.skipUrl) {{
      var url = new URL(window.location.href);
      url.searchParams.set('tab', slug);
      window.history.replaceState({{ slug: slug }}, '', url);
    }}
    document.title = (function () {{
      var b = bar.querySelector('button.tab.active');
      return b ? (b.textContent.trim() + ' — plots') : 'plots';
    }})();
  }}

  Array.prototype.forEach.call(buttons, function (b) {{
    b.addEventListener('click', function () {{ activate(b.dataset.slug); }});
  }});

  function cycle(direction) {{
    var idx = slugs.indexOf(frame.dataset.slug);
    if (idx < 0) return;
    var next = idx + direction;
    if (next < 0) next = slugs.length - 1;
    if (next >= slugs.length) next = 0;
    activate(slugs[next]);
  }}

  function handleKey(e) {{
    if (e.target && e.target.tagName === 'INPUT') return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (!e.altKey) return;
    cycle(e.key === 'ArrowLeft' ? -1 : 1);
    if (e.preventDefault) e.preventDefault();
  }}
  window.addEventListener('keydown', handleKey);

  // Plot iframes forward their Alt+arrow keydowns up here via postMessage,
  // since once focus is inside the iframe its window.parent never sees the
  // raw keydown event. The forwarder script lives in src/plotting/render.py
  // and runs in every plot HTML.
  window.addEventListener('message', function (e) {{
    var d = e.data;
    if (!d || d.type !== 'rp-tab-key') return;
    if (d.key === 'ArrowLeft')  cycle(-1);
    if (d.key === 'ArrowRight') cycle(1);
  }});

  activate(pickInitial(), {{ skipUrl: false }});
}})();
</script>
</body></html>
'''


def write_index():
    """Generate output/index.html: tab shell over every existing plot HTML.

    No-op when the rendered shell would be identical to what's on disk.
    The watch in main() fires on any change to ``output/*.html`` — which
    includes ``index.html`` itself — so without this guard, every write
    here re-triggers the watch and creates an infinite reload loop.
    Skipping the write when content is unchanged breaks the loop once
    the tab list stabilizes.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = {p.stem for p in OUTPUT_DIR.glob("*.html") if p.name != "index.html"}
    tabs = [(slug, label) for slug, label in TABS if slug in existing]
    # Append any HTML present in output/ that isn't in the curated TABS list,
    # so freshly-added plots show up automatically before they're labeled.
    extras = sorted(existing - {slug for slug, _ in TABS})
    for slug in extras:
        tabs.append((slug, slug.replace('_', ' ')))
    new_html = _shell_html(tabs)
    if INDEX_PATH.exists() and INDEX_PATH.read_text() == new_html:
        return
    INDEX_PATH.write_text(new_html)


def rebuild_all():
    """Regenerate every plot HTML before the server starts serving.

    Plot HTMLs on disk reflect whatever state the source was in the last
    time their script ran — which can drift arbitrarily far from the
    current source if you've edited src/plotting/, tweaked tokens, or
    just pulled new code. Doing a clean rebuild on startup means the
    first thing you see in the browser always matches the current
    source, with no "why isn't my edit reflecting" surprises.
    """
    script = REPO_ROOT / "scripts" / "run_plots.sh"
    if not script.exists():
        print(f"WARN: {script} not found; skipping startup rebuild.",
              file=sys.stderr)
        return
    print("Rebuilding all plots before starting server...")
    subprocess.run([str(script)], check=False)


# ----- Background rebuild worker --------------------------------------------
#
# livereload runs watch callbacks synchronously inside the Tornado IOLoop
# (see livereload/handlers.py:65). Running an 8-second `run_plots.sh` there
# wedges the whole server: tab clicks, websocket pings, and static-file
# requests all freeze until the subprocess returns. We solve that by
# enqueueing a job and letting a daemon worker drain the queue off-loop.
#
# A "job" is either the sentinel `_REBUILD_ALL` (rebuild every plot via
# scripts/run_plots.sh) or a frozenset of plot-script Paths to rerun
# individually. While the worker is busy, additional saves coalesce into a
# single follow-up job — no kill/restart, no parallel rebuilds.

_REBUILD_ALL = "__all__"
_jobs_lock = threading.Lock()
_jobs_cond = threading.Condition(_jobs_lock)
_pending = None  # None | _REBUILD_ALL | frozenset[Path]


def _merge(current, incoming):
    if current is None:
        return incoming
    if current == _REBUILD_ALL or incoming == _REBUILD_ALL:
        return _REBUILD_ALL
    return current | incoming


def enqueue(job):
    """Schedule a rebuild. Coalesces with any pending job.

    `job` is either `_REBUILD_ALL` or a frozenset of plot-script Paths.
    Safe to call from any thread (in practice: the IOLoop thread, via
    livereload watch callbacks).
    """
    global _pending
    with _jobs_cond:
        _pending = _merge(_pending, job)
        _jobs_cond.notify()


def _run_subset(paths):
    """Run a specific set of plot scripts sequentially."""
    label = ", ".join(sorted(p.stem for p in paths))
    print(f"==> rebuilding {len(paths)} plot(s): {label}")
    t0 = time.monotonic()
    for path in sorted(paths):
        rel = str(path.relative_to(REPO_ROOT))
        r = subprocess.run([sys.executable, rel], cwd=REPO_ROOT)
        if r.returncode != 0:
            print(f"FAILED: {rel} (exit {r.returncode})", file=sys.stderr)
    print(f"    done in {time.monotonic() - t0:.1f}s")


def _run_all():
    """Run scripts/run_plots.sh — the same path rebuild_all() uses."""
    script = REPO_ROOT / "scripts" / "run_plots.sh"
    if not script.exists():
        print(f"WARN: {script} not found; skipping rebuild.", file=sys.stderr)
        return
    print("==> rebuilding all plots")
    t0 = time.monotonic()
    subprocess.run([str(script)], check=False, cwd=REPO_ROOT)
    print(f"    done in {time.monotonic() - t0:.1f}s")


def _worker():
    global _pending
    while True:
        with _jobs_cond:
            while _pending is None:
                _jobs_cond.wait()
            job, _pending = _pending, None
        if job == _REBUILD_ALL:
            _run_all()
        else:
            _run_subset(job)


# ----- Dependency map -------------------------------------------------------
#
# Built once at startup. Maps each src/shared/*.py file to the set of
# src/plots/*.py files that import from it. Lets a single helper edit
# rebuild only its dependents instead of all 8 plots.

_SHARED_IMPORT_RX = re.compile(r"^\s*from\s+src\.shared\.(\w+)\s+import\b",
                               re.MULTILINE)


def build_shared_dep_map(plot_scripts, shared_dir):
    """Return {shared_path: frozenset(plot_paths)} by parsing imports."""
    by_module = {}
    for plot in plot_scripts:
        text = plot.read_text()
        for mod in _SHARED_IMPORT_RX.findall(text):
            by_module.setdefault(mod, set()).add(plot)
    dep_map = {}
    for mod, plots in by_module.items():
        shared_path = shared_dir / f"{mod}.py"
        if shared_path.exists():
            dep_map[shared_path] = frozenset(plots)
    return dep_map


def main():
    rebuild_all()
    write_index()

    threading.Thread(target=_worker, daemon=True).start()

    server = Server()

    plot_scripts = sorted(PLOTS_DIR.glob("*.py"))
    shared_dir = REPO_ROOT / "src" / "shared"
    dep_map = build_shared_dep_map(plot_scripts, shared_dir)

    print("Shared-helper dependency map:")
    for shared_path in sorted(dep_map):
        plots = sorted(p.stem for p in dep_map[shared_path])
        print(f"  {shared_path.name} -> {len(plots)} plots: {', '.join(plots)}")
    untracked = sorted(
        p for p in shared_dir.glob("*.py")
        if p.name != "__init__.py" and p not in dep_map
    )
    for p in untracked:
        print(f"  {p.name} -> (no plot imports — will rebuild all on change)")

    def _named(label, fn):
        fn.name = label  # surfaces in livereload's "Running task: <name>" log
        return fn

    # Per-plot watches: editing src/plots/foo.py reruns foo.py only.
    for script in plot_scripts:
        rel = str(script.relative_to(REPO_ROOT))
        server.watch(rel, _named(
            f"plot:{script.stem}", lambda s=script: enqueue(frozenset({s}))))

    # Per-shared-module watches: rerun only the plots that import this module.
    # Watch each shared file directly (not the glob) so the per-file callback
    # carries the right dependent set.
    for shared_path, dependents in dep_map.items():
        rel = str(shared_path.relative_to(REPO_ROOT))
        server.watch(rel, _named(
            f"shared:{shared_path.stem}",
            lambda d=dependents: enqueue(d)))

    # Any other src/shared/*.py file (no known importers) falls back to
    # rebuild-all — safer than silently doing nothing.
    for p in untracked:
        rel = str(p.relative_to(REPO_ROOT))
        server.watch(rel, _named(
            f"shared-untracked:{p.stem}",
            lambda: enqueue(_REBUILD_ALL)))

    # These genuinely affect every plot. Watch the directory rather than a
    # `src/plotting/**/*` glob: pyinotify's add_watch uses non-recursive glob,
    # so `**` is treated as a literal directory name and the per-file inotify
    # watches for src/plotting/*.py are never created. A directory path with
    # rec=True (which pyinotify defaults to) avoids that bug.
    server.watch("src/plotting", _named(
        "plotting:*", lambda: enqueue(_REBUILD_ALL)))

    # data/*.csv changes rerun all plots — except plot-derived CSVs that some
    # plot scripts emit into data/ (training_quality_track.csv,
    # training_quality_offsets.csv). Those would feed an infinite rebuild
    # loop: rebuild writes the CSV → CSV watch fires → rebuild again.
    server.watch(
        "data/*.csv",
        _named("data:*.csv", lambda: enqueue(_REBUILD_ALL)),
        ignore=lambda p: os.path.basename(p).startswith("training_quality_"),
    )

    # Index regenerates whenever a plot HTML appears or disappears.
    server.watch("output/*.html", write_index)

    print(f"Serving {OUTPUT_DIR} at http://localhost:{PORT}/")
    print(f"Watching {len(plot_scripts)} plot scripts + "
          f"{len(dep_map) + len(untracked)} shared modules + "
          "src/plotting/**/* + data/*.csv")
    print("Ctrl-C to stop.")
    server.serve(root=str(OUTPUT_DIR), port=PORT)


if __name__ == "__main__":
    main()
