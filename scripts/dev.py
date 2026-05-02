"""Dev server for plot iteration: livereload + per-script rerun.

Usage:
    python scripts/dev.py

Then open http://localhost:5500/ (or wait for it to open automatically).

Watches every src/plots/*.py — saving a plot script reruns just that script
and reloads any open browser tab pointed at its HTML output. Editing
src/shared/*.py, src/plotting/**/* or any data/*.csv reruns every plot via
scripts/run_plots.sh (dumb default; smarter per-CSV mapping can come later).

Pipeline scripts (build_dataset / parse_workouts / bayes_cs_fit) are
intentionally NOT watched — those are deliberate batch operations and stay
on scripts/run_pipeline.sh.

The landing page (output/index.html) is a tab shell: a top tab bar swaps
which plot HTML is loaded into a single iframe. Each plot stays its own
self-contained document, so per-plot CSS/JS scopes don't collide and
livereload reloads only the active iframe on its source change.
"""
import os
import subprocess
import sys
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


def main():
    rebuild_all()
    write_index()

    server = Server()

    # One watch per plot script — editing src/plots/foo.py reruns foo.py only.
    # Passing the command as a string (rather than livereload.shell(...)) lets
    # livereload tag the task name with the actual command, which shows up in
    # the dev-server log.
    plot_scripts = sorted(PLOTS_DIR.glob("*.py"))
    for script in plot_scripts:
        rel = str(script.relative_to(REPO_ROOT))
        server.watch(rel, f"python {rel}")

    # Shared module changes touch every plot — rerun all.
    server.watch("src/shared/*.py", "./scripts/run_plots.sh")
    server.watch("src/plotting/**/*", "./scripts/run_plots.sh")

    # Data changes also rerun all (dumb default).
    server.watch("data/*.csv", "./scripts/run_plots.sh")

    # Index regenerates whenever a plot HTML appears or disappears.
    server.watch("output/*.html", write_index)

    print(f"Serving {OUTPUT_DIR} at http://localhost:{PORT}/")
    print(f"Watching {len(plot_scripts)} plot scripts + src/shared/*.py "
          "+ src/plotting/**/* + data/*.csv")
    print("Ctrl-C to stop.")
    server.serve(root=str(OUTPUT_DIR), port=PORT)


if __name__ == "__main__":
    main()
