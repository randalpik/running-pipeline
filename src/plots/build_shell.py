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
    ("annual",                 "Annual"),
    ("mileage_by_geography",   "Locations"),
    ("world_map",              "World Map"),
]


def _empty_shell() -> str:
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
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


# Read lazily, not at import time: scripts/dev.py imports write_index once at
# startup and re-invokes it from its file watcher — a module-level read would
# pin whatever shell.css/js said when the dev server booted, silently ignoring
# every later edit until a restart.
def _shell_css() -> str:
    return (_SCAFFOLD_DIR / 'shell.css').read_text()


def _shell_js() -> str:
    # touch_scroll.js first — shell.js attaches the drawer's pan handler to it
    # at startup (Chromium won't scroll anything inside the rotated stage).
    return ((_SCAFFOLD_DIR / 'touch_scroll.js').read_text() + '\n'
            + (_SCAFFOLD_DIR / 'shell.js').read_text())


def _profile_switch_html(profiles, current_id) -> str:
    """Render the right-aligned profile <select>, or '' for a single profile.

    ``profiles`` is a list of (id, label, url_base); ``current_id`` marks the
    selected option. shell.js navigates to the chosen option's value (the
    profile's site root) on change.
    """
    if not profiles or len(profiles) < 2:
        return ''
    options = '\n'.join(
        f'        <option value="{url_base}"'
        f'{" selected" if pid == current_id else ""}>{label}</option>'
        for pid, label, url_base in profiles
    )
    return ('\n    <div class="profile-switch">\n'
            '      <select aria-label="Profile">\n'
            f'{options}\n'
            '      </select>\n'
            '    </div>')


def render_shell(tabs, include_admin: bool = False,
                 profiles=None, current_id=None) -> str:
    """Render the tab-shell HTML.

    ``tabs`` is a list of (slug, label) for plots whose HTML actually exists.
    ``include_admin`` adds an admin tab whose visibility is gated client-side
    on /api/auth/me and which loads /admin.html absolutely. ``profiles`` (a
    list of (id, label, url_base)) + ``current_id`` add a profile switcher when
    there's more than one profile. The shell's style + behavior live in
    src/plotting/_scaffold/shell.css and shell.js; this function only emits the
    structural HTML and the default-slug meta.
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
    profile_switch = _profile_switch_html(profiles, current_id)

    return f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="darkreader-lock">
<title>Max's Running Data</title>
<link rel="icon" href="data:,">
<meta name="rp-default-slug" content="{default_slug}">
<style>
{_shell_css()}
</style>
</head>
<body{body_class}>
  <div id="rp-stage">
    <div id="tabbar">
      <div class="brand">Max's Running Data</div>
{buttons}{profile_switch}
    </div>
    <button id="rp-menu-btn" type="button" aria-label="Menu"
            aria-expanded="false" aria-controls="tabbar"
    ><span></span><span></span><span></span></button>
    <div id="rp-menu-scrim"></div>
    <div id="frame-wrap">
      <div id="spinner"></div>
    </div>
  </div>
<script>
{_shell_js()}
</script>
</body></html>
'''


def write_index(output_dir: Path = DEFAULT_OUTPUT_DIR,
                include_admin: bool = False,
                profiles=None, current_id=None) -> bool:
    """Write output_dir/index.html based on which plot HTMLs exist there.

    Tabs whose plot HTML doesn't exist in ``output_dir`` are omitted — that's
    how a profile lacking a dataset (e.g. no workouts) auto-hides that tab.
    ``profiles``/``current_id`` add the profile switcher (see render_shell).
    Returns True if the file was written, False if unchanged on disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    existing = {p.stem for p in output_dir.glob("*.html") if p.name != "index.html"}
    tab_slugs = {slug for slug, _ in TABS}
    tabs = [(slug, label) for slug, label in TABS if slug in existing]
    # Auto-promote any *other* HTML to a tab — EXCEPT a known plot's child pages
    # (``<parent_slug>_<child>``, e.g. qualitative_trends_time), which are
    # link targets opened in their own tab, not shell tabs of their own.
    extras = sorted(s for s in (existing - tab_slugs)
                    if not any(s.startswith(t + "_") for t in tab_slugs))
    for slug in extras:
        tabs.append((slug, slug.replace('_', ' ')))
    new_html = render_shell(tabs, include_admin=include_admin,
                            profiles=profiles, current_id=current_id)
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
