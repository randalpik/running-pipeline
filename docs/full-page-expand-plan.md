# Plan: per-graph full-page expand for Misc. Trends

**Status:** scoped, not started. Handoff for a fresh thread.

## Goal

Let a viewer open any single Misc. Trends panel as its own dedicated full-page
view — "just that trend graph, no top bar or anything else." Triggered by a small
**↗ button next to each panel's inset title** that opens the expanded view in a
**new browser tab**.

There are 8 expandable panels (7 envelopes + the Conditions scatter):
`conditions, temperature, humidity, wind` (Weather) and
`volume, altitude, time, weight` (Other).

## Current architecture (what you need to know)

All in `src/plots/plot_qualitative_trends.py` + sibling `.js`, written by the
single authority `src/plotting/render.py::render_plot`.

- **One figure, 8 panels** in a 4-row `make_subplots` (`shared_xaxes=True`). Each
  row holds one Weather panel and one Other panel sharing that row's axes
  (`x{r}`/`y{r}`); the page toggle swaps which is visible.
- **Every envelope is a gradient RASTER** (`render_gradient_raster`, layout image
  `layer='above'`) + **white vector trend path shape(s)** (`layer='above'`).
  Conditions is the only real data **trace** (a scatter). Plus 4 invisible
  per-row **anchor traces** that keep the subplots alive.
- **Inset titles are Plotly annotations** (8 of them, indices 0–7), positioned
  `xref='x{n} domain'`, `x=0.008, y=0.97`. They are NOT clickable — this is the
  first thing that must change.
- **Page toggle** = `widgets.toggle_bar('trends-toggle', …)` in `overlay_html` +
  the sibling `plot_qualitative_trends.js`, which on click sets
  `window.__rpActiveTab`, flips trace visibility by `meta.page`, and applies
  `PAGES[page].relayout` — a dict of `yaxis{r}.range/.title/.tick*`,
  `annotations[i].visible`, `images[i].visible`, `shapes[i].visible`.
- `images` = list of `(image_index, page)`; `shapes_list` = list of
  `(shape_index, page)`; both built in the panel loop and consumed when building
  `pages`.
- **Tooltip** = `CursorTooltip(payload=…, build_js=_BUILD_JS)`. `_BUILD_JS`'s
  `buildTooltip(day)` reads `window.__rpActiveTab` and renders that tab's rows.
- Plot is `staticPlot: True` (no native zoom/pan); `Plotly.relayout` still works.
- **Hosting:** every view is a gated-by-default HTML at the root of `site/dist`.
  The shell (`build_shell.py`) embeds `qualitative_trends.html` as an iframe tab.
  A new standalone HTML (or the same one hit directly) is auto-gated — no gate
  change needed (it 302s to login only if the session is invalid).

## Recommended approach: separate per-panel standalone HTMLs (Approach B)

Generate one standalone single-panel page per panel —
`qualitative_trends_<key>.html` (e.g. `qualitative_trends_time.html`) — each a
genuine full-height, chrome-free plot. The inset ↗ links to that file with
`target="_blank"`.

**Why B over the query-param alternative (A, below):** it produces *exactly* what
the user asked for (a real standalone page, no hidden top bar, no other panels'
images downloaded), it matches this project's existing "one HTML per view"
pattern (10+ plots are already generated this way), and it avoids the fragile
runtime DOM surgery A requires (collapsing 3 subplot domains, hiding chrome,
re-anchoring the tooltip). Cost: ~8 small extra HTMLs (~0.4–0.8 MB each — one
raster + maybe one trend shape) and 8 extra build invocations. Acceptable.

### Step 1 — Extract a reusable panel renderer (refactor, no behavior change)

The panel loop body in `main()` already branches scatter vs envelope. Pull it
into a function:

```
def render_panel(fig, pn, row, full, dates, short, *, weather_cf, cond_cf,
                 weather_syn, wcolor) -> (images_added, shapes_added, tab_payload)
```

so the same code builds a panel into row `r` of the multi-panel figure AND into
row 1 of a single-panel figure. Keep the existing multi-panel `main()` calling it
in the loop (verify byte-identical output — same traces/images/shapes/payload).

### Step 2 — Single-panel build mode

Add `--panel <key>` to `main()` (and a `build_single(key)` path). When set:
- Build a **1-row** `make_subplots` with just that panel via `render_panel`.
- Full-height domain; no page toggle, no `meta.page` visibility games (only one
  panel, always visible). Image/shape `visible=True`.
- Y-axis range/title/ticks = that panel's config (reuse the per-page axis kwargs
  for the panel's own page). For `time`, the clock tickvals/ticktext.
- Title: render the metric label as the plot `title` (render_plot's title bar) OR
  keep the inset; user said "no top bar," so prefer a minimal title or none —
  decide with the inset overlay reused as the only label.
- **Tooltip:** reuse `_BUILD_JS` but constrain it to the single metric. Simplest:
  pass a `window.__rpSingle = '<key>'` global; in `buildTooltip`, when set, render
  only the date + Location + that metric's day/avg rows (skip the tab loop). The
  payload for a single page need only carry that metric's arrays + `city`.
- Output path `qualitative_trends_<key>.html`. `staticPlot: True`, same scaffolds.

### Step 3 — Clickable inset overlay (replaces the annotation insets) — COMMON

Plotly annotations can't hold a clickable element, so convert the 8 inset titles
to **HTML overlay divs** positioned per-subplot:
- Emit 8 small divs (label text + an `<a class="rp-inset-open" target="_blank"
  href="qualitative_trends_<key>.html">↗</a>`) via `overlay_html`, each tagged
  with its panel key + page + row.
- Position them at each subplot's top-left using a new sibling-JS routine that
  reads `_fullLayout['yaxis'+n]._offset/_length` and `xaxis` offsets (same
  technique as `_scaffold/overlay_anchor.js`), re-running on resize.
- The page toggle (`plot_qualitative_trends.js`) shows the active page's inset
  divs and hides the other page's (replacing the current
  `annotations[i].visible` toggle). Drop the Plotly annotations (or keep them for
  non-interactive fallback and overlay only the ↗ — but cleaner to fully move to
  HTML).
- Style `.rp-inset` / `.rp-inset-open` in `_scaffold/base.css` (small, muted, hover
  brighten — matches `.rp-*` system).

The ↗ `href` is relative, so inside the shell iframe it resolves against
`qualitative_trends.html`'s URL and opens the sibling file in a new top-level tab.

### Step 4 — Build wiring

In `scripts/run_plots.sh`, after the main `plot_qualitative_trends.py`, loop the 8
keys: `python src/plots/plot_qualitative_trends.py --panel <key>`. They land in
`output/` → staged to `site/dist/` root → auto-gated. No `build_shell` change
(they are NOT tabs; only linked from the main plot). No `gate.ts` change.

## Files to modify

- `src/plots/plot_qualitative_trends.py` — extract `render_panel`, add `--panel`
  single-panel mode, HTML inset overlays + globals, single-metric tooltip branch.
- `src/plots/plot_qualitative_trends.js` — position inset overlays per subplot;
  toggle inset divs per page (replace annotation-visibility toggle).
- `src/plotting/_scaffold/base.css` — `.rp-inset`, `.rp-inset-open` styles.
- `scripts/run_plots.sh` — generate the 8 single-panel HTMLs.
- (No change to `build_shell.py`, `gate.ts`, or `render.py`.)

## Gotchas / risks

- **Inset positioning** is the fiddliest part — get the subplot pixel rect from
  `_fullLayout` and reposition on `resize` + after `plotly_afterplot`. Reuse the
  `overlay_anchor.js` pattern; don't hand-roll.
- **iframe new-tab:** confirm `target="_blank"` from inside the shell iframe opens
  a top-level tab at the resolved sibling URL (it should; if not, use
  `window.open(href, '_blank')`).
- **Single-panel tooltip:** the spike + `findSubplotAt` already handle one
  subplot; just make `buildTooltip` render one metric (via `__rpSingle`).
- **Standalone load:** `render.py`'s tab-key / plot-ready forwarders no-op when
  `window.parent === window`, so a directly-opened page is fine.
- **Output size / build time:** 8 extra small HTMLs; each shares the directory
  `plotly.min.js`. Confirm total `site/dist` size stays reasonable.
- **Keep them out of the shell tab list** — they're link targets, not tabs.

## Fallback approach A (query-param, same file) — only if B's build cost is unwanted

Inset ↗ opens `qualitative_trends.html?panel=<key>` in a new tab. On load the JS
reads the param and: hides the title bar + toggle (CSS), selects the panel's page,
relayouts the 4 `yaxis{r}.domain` so the target = `[0.04, 1]` and the other three
collapse (`[0, 0]` — **test this; may need `[0, 1e-6]`**), and hides the other
panels' images/shapes/annotations/anchor traces. Pros: zero new files, no build
change. Cons: fragile domain-collapse, downloads all 7 rasters, chrome hidden not
absent. Documented here only as a fallback; prefer B.

## Verification

1. `python src/plots/plot_qualitative_trends.py` then `--panel time`,
   `--panel temperature`, `--panel conditions` → 3 standalone HTMLs render full
   height, chrome-free, correct gradient + (for time) no trend, correct y-axis.
2. Playwright: open the main plot, click a panel's ↗ → new page loads the right
   single panel; hover shows date + Location + that metric's day/avg rows only;
   no JS errors.
3. Toggle Weather/Other on the main plot → inset overlays (with ↗) swap with the
   page; positions hold on window resize.
4. `bash scripts/run_plots.sh` → all 8 single-panel files generated; main plot
   unchanged (diff traces/images/shapes counts).
5. Confirm the single-panel pages are gated (no `gate.ts` exemption added).
