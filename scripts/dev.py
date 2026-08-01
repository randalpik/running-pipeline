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
from livereload.handlers import LiveReloadHandler
from tornado import ioloop

REPO_ROOT  = Path(__file__).resolve().parent.parent
PLOTS_DIR  = REPO_ROOT / "src" / "plots"
OUTPUT_DIR = REPO_ROOT / "output"
INDEX_PATH = OUTPUT_DIR / "index.html"
PORT       = 5500

sys.path.insert(0, str(REPO_ROOT))
from src.plots.build_shell import write_index as _build_shell_write_index

# livereload globs are evaluated relative to the process cwd; chdir so the
# watch patterns ("src/plots/*.py", "data/*.csv") resolve correctly regardless
# of where the user invokes the script from.
os.chdir(REPO_ROOT)


# Tab list and shell rendering live in src/plots/build_shell.py so they can
# be reused by the production build (run_plots.sh -> build_shell.py --admin).


def write_index():
    """Regenerate output/index.html if its contents would differ.

    Delegates to src.plots.build_shell. No-op when unchanged so that
    livereload's watch on output/*.html doesn't bounce in a loop.

    The dev server roots at output/, so it also serves any sub-profile built
    under output/profiles/<id>/. We list those (plus the default) in the
    profile switcher so the dropdown shows up and navigates in dev exactly as
    in prod — url_base (/profiles/<id>/) matches the on-disk layout.
    """
    from src.profiles import PROFILES, default_profile
    dflt = default_profile()
    switcher = [(dflt.id, dflt.label, dflt.url_base)]
    for p in PROFILES:
        if not p.default and (p.output_dir / "index.html").exists():
            switcher.append((p.id, p.label, p.url_base))
    _build_shell_write_index(OUTPUT_DIR, include_admin=False,
                             profiles=switcher, current_id=dflt.id)


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

# Captured in main() before server.serve() so the worker thread can schedule
# browser reloads onto the Tornado IOLoop running on the main thread.
_ioloop = None


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
        try:
            if job == _REBUILD_ALL:
                _run_all()
            else:
                _run_subset(job)
        finally:
            with _jobs_cond:
                drained = _pending is None
            if drained and _ioloop is not None:
                try:
                    _ioloop.add_callback(LiveReloadHandler.reload_waiters)
                except RuntimeError:
                    pass  # IOLoop closed during shutdown


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

    # Suppress livereload's auto-reload for this watch: run the callback, then
    # null out watcher.filepath so poll_tasks() early-returns at handlers.py:70
    # without sending a reload. We drive reload manually from the rebuild
    # worker once the queue drains, so the browser refreshes exactly once
    # AFTER the new HTML is on disk — never against stale content, never
    # throttled by livereload's 3-second global rate limit.
    #
    # `wrapped` is intentionally zero-arg: livereload inspects the callback
    # signature (watcher.py:106) and passes the list of changed files to any
    # callback that declares parameters. None of our callbacks use that list,
    # and write_index() in particular takes zero args.
    def _silent(fn):
        def wrapped():
            try:
                return fn()
            finally:
                if server.watcher is not None:
                    server.watcher.filepath = None
        wrapped.name = getattr(fn, 'name',
                               getattr(fn, '__name__', 'silent'))
        return wrapped

    # Per-plot watches: editing src/plots/foo.py reruns foo.py only.
    for script in plot_scripts:
        rel = str(script.relative_to(REPO_ROOT))
        server.watch(rel, _silent(_named(
            f"plot:{script.stem}", lambda s=script: enqueue(frozenset({s})))))

    # Per-shared-module watches: rerun only the plots that import this module.
    # Watch each shared file directly (not the glob) so the per-file callback
    # carries the right dependent set.
    for shared_path, dependents in dep_map.items():
        rel = str(shared_path.relative_to(REPO_ROOT))
        server.watch(rel, _silent(_named(
            f"shared:{shared_path.stem}",
            lambda d=dependents: enqueue(d))))

    # Any other src/shared/*.py file (no known importers) falls back to
    # rebuild-all — safer than silently doing nothing.
    for p in untracked:
        rel = str(p.relative_to(REPO_ROOT))
        server.watch(rel, _silent(_named(
            f"shared-untracked:{p.stem}",
            lambda: enqueue(_REBUILD_ALL))))

    # These genuinely affect every plot. Watch the directory rather than a
    # `src/plotting/**/*` glob: pyinotify's add_watch uses non-recursive glob,
    # so `**` is treated as a literal directory name and the per-file inotify
    # watches for src/plotting/*.py are never created. A directory path with
    # rec=True (which pyinotify defaults to) avoids that bug.
    # shell.css/shell.js are excluded: they only feed index.html, so they get
    # a dedicated write_index watch below instead of a full rebuild.
    _SHELL_FILES = {"shell.css", "shell.js"}
    server.watch("src/plotting", _silent(_named(
        "plotting:*", lambda: enqueue(_REBUILD_ALL))),
        ignore=lambda p: os.path.basename(p) in _SHELL_FILES)
    server.watch("src/plotting/_scaffold/shell.css", _silent(write_index))
    server.watch("src/plotting/_scaffold/shell.js", _silent(write_index))

    # data/*.csv changes rerun all plots — except plot-derived CSVs that some
    # plot scripts emit into data/ (training_quality_track.csv,
    # training_quality_exclusions.csv, hill_model.csv). Those would feed an
    # infinite rebuild loop: rebuild writes the CSV → CSV watch fires →
    # rebuild again.
    server.watch(
        "data/*.csv",
        _silent(_named("data:*.csv", lambda: enqueue(_REBUILD_ALL))),
        ignore=lambda p: (os.path.basename(p).startswith("training_quality_")
                          or os.path.basename(p) == "hill_model.csv"),
    )

    # Index regenerates whenever a plot HTML appears or disappears.
    server.watch("output/*.html", _silent(write_index))

    # Capture the main-thread IOLoop so the rebuild worker can schedule
    # reload_waiters onto it. Server.serve() will use the same singleton.
    global _ioloop
    _ioloop = ioloop.IOLoop.current()

    print(f"Serving {OUTPUT_DIR} at http://localhost:{PORT}/")
    print(f"Watching {len(plot_scripts)} plot scripts + "
          f"{len(dep_map) + len(untracked)} shared modules + "
          "src/plotting/**/* + data/*.csv")
    print("Ctrl-C to stop.")
    server.serve(root=str(OUTPUT_DIR), port=PORT)


if __name__ == "__main__":
    main()
