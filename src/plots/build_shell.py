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
        '<link rel="icon" href="data:,">'
        '<style>body{background:#1a1a1a;color:#eee;'
        'font-family:system-ui,sans-serif;padding:2rem}</style>'
        '</head><body>'
        '<h1>No plots yet</h1>'
        '<p>Run <code>./scripts/run_plots.sh</code> or save a plot script.'
        '</p></body></html>\n'
    )


_SCAFFOLD_DIR = REPO_ROOT / 'src' / 'plotting' / '_scaffold'
_SHELL_CSS = (_SCAFFOLD_DIR / 'shell.css').read_text()
_SHELL_JS = (_SCAFFOLD_DIR / 'shell.js').read_text()


def render_shell(tabs, include_admin: bool = False) -> str:
    """Render the tab-shell HTML.

    ``tabs`` is a list of (slug, label) for plots whose HTML actually exists.
    ``include_admin`` adds an admin tab whose visibility is gated client-side
    on /api/auth/me and which loads /admin.html absolutely. The shell's
    style + behavior live in src/plotting/_scaffold/shell.css and shell.js;
    this function only emits the structural HTML and the default-slug meta.
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
    body_class = ' class="has-admin"' if include_admin else ''

    return f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="darkreader-lock">
<title>Max's Running Data</title>
<link rel="icon" href="data:,">
<meta name="rp-default-slug" content="{default_slug}">
<style>
{_SHELL_CSS}
</style>
</head>
<body{body_class}>
  <div id="tabbar">
    <div class="brand">Max's Running Data</div>
{buttons}
  </div>
  <div id="frame-wrap">
    <div id="spinner"></div>
  </div>
<script>
{_SHELL_JS}
</script>
</body></html>
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
