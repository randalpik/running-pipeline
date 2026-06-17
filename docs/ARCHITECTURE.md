# Architecture & Reference Map

The bird's-eye view of this repo: data sources, the pipeline that processes them, the three independent analysis layers, and which reference doc to open for which concern. Read this before diving into a specific reference doc — it's the routing layer.

## Data flow

```
Google Drive (Running Log xlsx, Lifetime Miles, Routes, Max's Running Data)
        │
        ▼
   drive_snapshot.csv  +  historical_daily.csv (frozen pre-current-year)
        │
        ▼
   build_dataset.py
        │
        ├──▶ daily.csv      (running-only, joined location metadata)
        │
        └──▶ races.csv      (post-adjustment truth)
                │
                ▼
        bayes_cs_fit.py  ──▶ bayes_cs_summary.csv  +  params  +  posterior .nc
                │
                ▼
            plot scripts
                │
                ▼
        output/*.html
```

`daily.csv` is running-only as of April 2026 (zero-mile days pruned). `races.csv` back-propagates `city_state` and `surface` into the corresponding daily race rows for `race_seq=1`.

## Truth sources

The repo's truth sources are checked-in CSVs in `data/` (flat — no raw/processed split). Drive is fetched via `drive_fetch.py` only when those are missing or stale. The `Max's Running Data` workbook (id `1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w`) carries the manual-overlay sheets:

| sheet | purpose |
|---|---|
| `changes` | per-row corrections — `date | race_seq | field | value | note` |
| `additions` | races not in the daily log (mostly pre-2016) — `date | distance_m | time_sec | surface | location | event`. Also produces stub daily rows for race dates not already in daily |
| `locations` | `log_location → city_state / display_name / elev_per_mile / altitude / terrain_type` lookup |
| `hills` | hill-loop metadata used by parser and TQ — `abbrev | location | type | distance_m | elev_gain_up | elev_gain_down | elev_net | elev_per_min` |
| `coordinates` | `city_state → (latitude, longitude)` overrides on top of the Nominatim cache for the world map. The cache (`data/city_coords.csv`) also stores an IANA `tz` per city — auto-resolved from lat/lon via Open-Meteo at geocode time — which drives the Misc. Trends Time panel's canonical-timezone solar gradient (see `qualitative-trends-reference.md`) |
| `historical` | `city_state | min_hist | max_hist | log_location` — surfaces remembered-but-unlogged cities on the world map AND date-range-overrides non-race daily rows. Replaces the legacy 2016-17 `infer_2016_2017_location` date branches |

2018+ is converged: across eight years there's only one location change and four surface changes. New work focuses on pre-2018 rough edges and on 2026+ ongoing data.

## Three analysis layers

The layers are intentionally **not combined**. Each answers a different question, and a combined model was tested and rejected.

### Layer 1 — Critical Speed (CS)

Bayesian estimate of fitness on a single continuous scale, derived from race results only. Replaces earlier VDOT-based approaches. Workouts and long runs do **not** feed back into CS; a combined model was tested in April 2026 and underperformed CS-only on the 2018+ HM+Track5K subset (RMSE 13.4 vs 10.6).

CS-implied 5K is treated as ground truth for fitness. The other layers sit beside it, not inside it.

→ See `cs-model-reference.md`

### Layer 2 — Training Quality (TQ)

A residual layer on top of CS. Translates each workout, long run, and hill effort into a 5K-equivalent pace, then asks how it compares to what CS predicts for that date. Captures fitness building between races, training running ahead of (or behind) realized fitness, and training composition during breakthrough periods.

→ See `training-quality-reference.md`

### Layer 3 — Recovery

Era-windowed regression on recovery-run pace, with a small set of OLS features (apparent/"feels-like" temperature — humidity folded in via the heat index, recent race exposure, time-of-day, pinned physical footing/altitude/elevation, and a pinned wind cost on watch-measured days). Volume normalizers and bodyweight features were explored and rejected. Captures the slow drift in baseline aerobic fitness independent of race performance.

→ See `recovery-runs-reference.md`

### Cross-cutting — route normalization

Locations propagate location-specific pace costs into the recovery model, TQ long-run residuals, **and race CS**. Lives between the layers and informs all three. The locations sheet (in `Max's Running Data`) seeds `log_location → city_state / display_name / elev_per_mile / altitude / terrain_type`; since the per-second watch enrichment shipped (June 2026), this doc also owns the **physical elevation engine** — the grade cost model (`elevation_cost.py`), the science-pinned altitude curve, the pinned footing/altitude betas (`physical_route_betas`), the per-run features, and the race DEM — that all three layers share.

→ See `route-normalization-reference.md`

### Diagnostic — qualitative trends

The volume / temperature / weight panel: three stacked subplots with moving-average trendlines on top of vertical-gradient min-max envelopes. Pure visualization, not a model — used for context when interpreting the three analytical layers.

→ See `qualitative-trends-reference.md`

## Reference doc map

| Doc | Layer | Covers |
|-----|-------|--------|
| `cs-model-reference.md` | 1 | PyMC HSGP fit, race exclusion logic, why the combined model was rejected, run flow (`bayes_cs_fit.py --tag vN` then `bayes_cs_plot.py --tag vN`, ~8min). |
| `training-quality-reference.md` | 2 | tau decay (210), distance thresholds, long-run binning, ONE shared CS predictor for quality workouts (per-category offsets removed June 2026 — era effort policy is signal), hill correction (pinned Minetti net cost + fitted trail term, `hill_model.csv`), continuous-fartlek 500/300 reconstruction (float:hard pinned 1.25), watch-measured hill-block time (GPS loop-point detection in `reps.py`, surveyed distance authoritative), long-run watch reconciliation (June 2026 — log/watch distance calibration in `long_runs.py`, watch moving times, pause-aware D_eff, Nashville mislogged-route rules), track-relative iterative outlier prune, rep/hill scatter weights, smoother bandwidth, ESS gates, gap-break logic. |
| `recovery-runs-reference.md` | 3 | Era window (±182d), OLS features, route inclusion threshold, rejected normalizers (volume, bodyweight). |
| `route-normalization-reference.md` | cross | Locations-sheet schema/dataflow + the **physical elevation engine** (June 2026, watch enrichment): grade cost model (`elevation_cost.py`, terrain×effort refund), threshold+linear altitude curve, pinned footing/altitude betas (`physical_route_betas`, the single source shared by recovery/long/race), per-run features (`per_run_elevation`/`per_run_altitude`), and the race DEM (`dem_elevation.py`). |
| `qualitative-trends-reference.md` | diagnostic | Window sizes (56/28/56-day MA, 14d min-max with 7d smoothing), gradient-strip rendering, weight gap-align before+after smoothing. |
| `watch-derived-cache-spec.md` | cross | Data-layer spec for the watch-derived cache (corrected mileage, per-run elevation/altitude, weather) consumed by the elevation engine and the recovery/long-run/race models. |

## Pre-2016 race data

Daily logs go back to 2016. Race history before that came from a one-time enrichment effort:

- `Lifetime Miles` "5K Record Progression" table (sole pre-2016 source in Drive).
- `Routes` "Half Marathon Record Progression" (2016–17 era).
- Athlinks API (76 races, 2008–2025).
- athletic.net HTML (86 entries: track + relay + XC).

Merged into `merged_races.csv` (161 rows, distance-snapped to standard distances) which drove 55 manual additions. The scrapers (`parse_athlinks_api.py`, `parse_athletic_net.py`) live in outputs if persisted; they aren't part of the regular pipeline.

## Drive file IDs

| File | ID |
|------|-----|
| 2026 Running Log | `1zvjx4RUzdZ11lsbyrzibGQJyLXUOR7gRUKwIwIILTuA` |
| Lifetime Miles | `10l629w-jChPdnwpVYQ2Lgj9JiSZH3_mNwTVeP0KusEM` |
| Max's Running Routes | `1tWPI9j8JCJidrOyu8Gw5lJ4aS-8__51gompGjArzUWU` |
| Max's Running Data | `1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w` |
| Running Data folder | `1b5yUJBkQA7FZfQX4STHBoFOnQBdlsQMv` |

## Plotting layer

Each plot script in `src/plots/` reads CSVs from `data/`, builds a Plotly figure, and writes a self-contained HTML to `output/` via the single rendering authority `render_plot()` in `src/plotting/render.py`. The rendering layer is deliberately split into four concerns; see `CLAUDE.md` § "Plot conventions" for the working rules.

```
src/plotting/
  tokens.py             design tokens (colors, sizes)
  layout.py axes.py     Plotly helpers
  markers.py formatters.py smoothing.py
  render.py             render_plot() — writes HTML, injects CSS/JS
  widgets.py            HTML primitives (sidebar, button_row, …)
  _scaffold/            shared CSS/JS loaded into every plot
    base.css            dark-theme chrome + .rp-* design system
    cursor_tooltip.js   smart spikeline + smooth/snap tooltip
    overlay_anchor.js   positions overlays below the legend
    shell.css shell.js  tab shell — loaded by build_shell.py

src/plots/
  <plot>.py             data prep + figure construction
  <plot>.js             plot-specific JS (loaded via overlay_js_files)
  <plot>.css            plot-specific CSS (loaded via extra_head_css_files)
```

Two outliers bypass `render_plot()`: `dashboard.py` (text-only stats — no Plotly bundle needed) and `build_shell.py` (the tab shell itself, not a plot). Both still consume the shared scaffold (`_scaffold/base.css`, `_scaffold/shell.{css,js}`, `_TAB_KEY_FORWARDER_JS`).

**Output inventory:** ~12 HTML files + `plotly.min.js` (shared across all plots, ~4.7 MB), all written to `output/` and copied verbatim to `site/dist/` at deploy time.

## Hosting layer

The site at **running.maxrandalmusic.com** lives in `site/`, deployed to Netlify by the `build-and-deploy.yml` GitHub Actions workflow. The pipeline runs in CI, generates plots into `output/`, and the deploy step copies `output/*` directly into `site/dist/` (root, no subdirectory).

### Pipeline triggers

The workflow accepts three trigger modes, all flowing into the same job. The "Resolve inputs" step normalizes `fit` / `historical` flags from whichever mode fired:

| Trigger | How | When |
|---|---|---|
| `workflow_dispatch` | Manual via GitHub UI or admin button on the site | Ad-hoc rebuilds, refit on demand |
| `repository_dispatch` (`pipeline-run`) | Apps Script in the Running Log workbook posts to the GitHub API | Automatic — every Workout-column edit, after a 60s settle window |
| `repository_dispatch` (`pipeline-run-fit`) | Same Apps Script, additionally sent when the workout text contains `race@` | Automatic refit on race days |

The Sheets trigger is `scripts/sheets_trigger.gs` (Apps Script bound to the workbook). It uses an `onEdit` installable trigger to queue edits in `ScriptProperties` and a 1-minute time-driven trigger to drain the queue. Multiple edits to the same row reset the timer (debounce). On race days the regular run goes first; the refit run queues behind it via the workflow's `concurrency: build-and-deploy` group.

### URL layout — flat at root

Plot HTMLs and the tabbed shell sit at the root of the deployed site, NOT under a `/plots/` namespace. Examples:

```
running.maxrandalmusic.com/                  — tabbed shell (output/index.html)
running.maxrandalmusic.com/dashboard.html    — Dashboard tab content
running.maxrandalmusic.com/world_map.html    — World Map tab content
running.maxrandalmusic.com/admin.html        — admin UI (loaded into iframe by the shell's Admin tab)
running.maxrandalmusic.com/login.html        — login + access-request page
running.maxrandalmusic.com/api/*             — Netlify Functions (auth, admin endpoints)
```

### Auth gate is allow-list-of-public-paths

The Edge Function `site/netlify/edge-functions/gate.ts` fires on **every path** (`/**`) and is the source of truth for the auth model:

- The gate's `EXEMPT_PATHS` set + `EXEMPT_PREFIXES` list name the **public** routes (login.html, plotly.min.js, /api/, /.netlify/, etc.).
- Every other path requires a valid session JWT cookie whose email is on the allowlist (or is `ADMIN_EMAIL`).
- Unauthorized requests get a 302 to `/login.html` (with a `?next=` round-trip back to the original URL after sign-in).

This is deliberately **safer-by-default**: a new file or route added anywhere in the deployed site is gated unless someone explicitly adds it to the exempt list. Forgetting to exempt a public page just means it gets login-walled until fixed; forgetting to gate a private page would leak data, and this design prevents that failure mode.

### Admin gating

Admin endpoints (`/api/admin/*`) verify both the session cookie and that the email matches `ADMIN_EMAIL` server-side via `requireAdmin` in `site/netlify/functions/_shared/admin-guard.ts`. The admin tab in the tabbed shell is hidden via CSS by default and unhidden client-side only when `/api/auth/me` returns `isAdmin: true`. Visibility is purely cosmetic — security is at the API layer, not the UI.
