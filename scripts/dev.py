"""Dev server for plot iteration: livereload + per-script rerun.

Usage:
    python scripts/dev.py

Then open http://localhost:5500/ (or wait for it to open automatically).

Watches every src/plots/*.py — saving a plot script reruns just that script
and reloads any open browser tab pointed at its HTML output. Editing
src/shared/*.py or any data/*.csv reruns every plot via scripts/run_plots.sh
(dumb default; smarter per-CSV mapping can come later).

Pipeline scripts (build_dataset / parse_workouts / bayes_cs_fit) are
intentionally NOT watched — those are deliberate batch operations and stay
on scripts/run_pipeline.sh.
"""
import os
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


def write_index():
    """Generate output/index.html listing every other *.html in output/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    htmls = sorted(
        p.name for p in OUTPUT_DIR.glob("*.html") if p.name != "index.html"
    )
    if not htmls:
        body = ('<p>No plots yet. Run <code>./scripts/run_plots.sh</code> '
                'or save a plot script to generate one.</p>')
    else:
        items = "\n".join(f'    <li><a href="{h}">{h}</a></li>' for h in htmls)
        body = f"<ul>\n{items}\n</ul>"
    INDEX_PATH.write_text(
        '<!doctype html>\n'
        '<html><head><meta charset="utf-8"><title>plots</title>\n'
        '<style>\n'
        '  body { font-family: system-ui, sans-serif; padding: 2rem;\n'
        '         background: #1a1a1a; color: #eee; }\n'
        '  a { color: #9cf; text-decoration: none; }\n'
        '  a:hover { text-decoration: underline; }\n'
        '  h1 { margin-top: 0; }\n'
        '  li { margin: 0.3em 0; font-size: 1.05em; }\n'
        '</style>\n'
        '</head><body>\n'
        '<h1>plots</h1>\n'
        f'{body}\n'
        '</body></html>\n'
    )


def main():
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

    # Data changes also rerun all (dumb default).
    server.watch("data/*.csv", "./scripts/run_plots.sh")

    print(f"Serving {OUTPUT_DIR} at http://localhost:{PORT}/")
    print(f"Watching {len(plot_scripts)} plot scripts + src/shared/*.py + data/*.csv")
    print("Ctrl-C to stop.")
    server.serve(root=str(OUTPUT_DIR), port=PORT, open_url_delay=1)


if __name__ == "__main__":
    main()
