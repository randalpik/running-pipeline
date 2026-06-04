# CLAUDE.md

Operational guidance for Claude Code working in this repo. For project-level context — what each reference doc covers, how the analysis layers fit together, where the data comes from — see `docs/ARCHITECTURE.md`.

This is a personal running-data pipeline: parsers and transformations turning daily training logs into analysis-ready CSVs and HTML plots. Owner: Max.

> **People — pronouns (non-negotiable):** **Maddy uses they/them.** Maddy is Max's partner and the second dashboard runner (the `maddy` profile).
>
> The rule is a **bright line, not a substitution exercise**: emit **zero gendered third-person pronouns for Maddy** — never she/her or he/him — in prose, code comments, and commit messages, **including when discussing this rule itself**. Default to the name ("Maddy", "Maddy's") or they/them. In dense technical passages — where the lapse reliably happens, because attention is on the substance and the pronoun fires reflexively — prefer the proper noun and restructure sentences to avoid third-person pronouns for people entirely. Treat any pronoun token next to "Maddy" as a stop-and-check before emitting it. This is binary: a gendered pronoun for Maddy is a clean failure, full stop. (There is no longer a Stop-hook guard for this — it was removed as ineffective; correctness is on the generation, not a backstop.)

## Working agreements

**Zero tolerance for unsupported facts.** Before stating any date, number, count, or quote, verify it against current tool output or a current-context file. Memory of earlier-in-session content is not a valid source. Voice uncertainty explicitly ("I think X, let me check") rather than asserting. When the request is ambiguous (numbered lists, references like "do 1 and 3"), ask before guessing — the cost of being wrong is higher than the cost of one clarifying exchange.

**Absolute dates beat relative.** When estimating dates from input, prefer absolute ("2026-04-20", "late March 2026") over relative ("28 days ago"). Documents persist; relative dates rot.

**Plan first.** For any non-trivial change, lay out the plan and wait for confirmation before executing. Plan mode handles much of this, but the discipline applies even outside plan mode.

**Output discipline.** Batch expensive operations (Drive fetches especially) at the start of a turn. Synthesize in focused passes rather than fetch-then-chart-then-write-then-artifact in one go. If a task would require many sequential reads, propose splitting across turns.

**Rule-addition threshold.** Worth adding a hardcoded or event-normalization rule if it would save 10+ manual adjustment entries. Not worth it for ≤3. Don't propose low-volume rules; suggest manual adjustments for those cases.

**"Continue" protocol.** If a message is strictly the single word "Continue", it was auto-sent by a tool-use limit warning. Finish whatever was in progress and present any files; don't treat it as a fresh prompt.

## Data conventions

- **Sleep:** 1.5-hr cycles since 2018; raw hours in 2017.
- **Temperature:** Celsius.
- **Paces:** per-mile, always — both recovery and quality.
- **Race distances:** meters, always (21097 = HM, 42195 = marathon).
- **Asterisk on shoe name:** disambiguates two physical pairs of the same model in the same year.
- **Workout jog segments (`Nj`):** N is **minutes**, not miles.
- **Quality distances (`Nt@`/`Ni@`/`Nr@`/`Nf@`):** N is **meters**.
- **Daily Miles column** is always the authoritative total — don't recompute from segments.

## Workout string coding

Quality letters: `t`=tempo, `i`=intervals, `r`=repetitions, `f`=fartlek.

| Pattern | Format |
|---|---|
| Quality block | `xj, xn@t:tt, xj` (e.g. `20j, 10000i@4:56, 21j`) |
| Recovery pace | `n@t:tt[/x<st/sp>]` where n is rec / long / trail |
| Continuous hills | `xj, tthc-yx location[/xsp], xj` |
| Hill reps | `xj, t:tthr-yx location[/xsp], xj` |
| Race | `xj, xrec/t, x race@t:tt, xj` |

Suffixes: `j`=jog, `st`=strides, `sp`=sprints, `hc`=hill continuous, `hr`=hill reps.

## Workout classification rules

1. Unclassifiable entries: 0 miles → rest; positive mileage → recovery.
2. Quality shorthand without `@pace` is **not** a quality workout (e.g. bare `24f` = recovery).
3. Multi-race days: races after the first get `race_seq >= 2` and `fatigued = True`. Filter or caveat when analyzing.
4. Pre-2018 data is less consistent — don't preserve 2016 quirks beyond race-data needs.

## `daily.csv` is running-only

As of April 2026, zero-mile days are pruned and `is_rest` is dropped. Schema:

`date, year, month, day_of_year, dow, miles, minutes, pace_sec_per_mi, temp_c, sleep_cycles, weather, conditions, wind, time_of_day, shoes, location, weight_lbs, surface, partners, workout_raw, run_type, recovery_pace_sec_per_mi, quality_distance_m, quality_pace_sec_per_mi, quality_segment_type, num_races, schema_year_era, source_file` + join cols `display_name, city_state, elev_per_mile, altitude, terrain_type`.

Pre-2016 race additions also produce stub daily rows (`source_file=snapshot:additions`, mileage from `distance_m / 1609.344`) so cities Max only ever raced at — Maple Valley, Carnation 2008-2015 — surface on the world map.

Plot date floors are **data-derived per profile**, not hardcoded (see `src/shared/plot_window.py`): daily-centric plots start at the first non-race daily entry (`daily_floor()`), race/fitness plots ~1 month before the first race (`race_floor()`). For Max this lands daily plots at 2016 (his only pre-2016 rows are race stubs) and race plots at ~2008; for a watch profile starting 2020 the axes begin in 2020. The world map is the only consumer of pre-2016 data. Two **Max-specific carve-outs stay hardcoded** because they encode a hand-estimated early-career fitness curve, not data: the Fitness/`cs_timeline` 2013-06 cutoff, and the race plots' `handdrawn_start=2008-04-01` dotted CS curve (both harmlessly clip off-screen for later-starting profiles).

`races.csv` is the post-adjustment truth for race rows. `race_seq=1` rows back-propagate `city_state` and `surface` into the corresponding daily row.

## `build_dataset.py` pipeline order

```
load + combine
  → build_race_segments
  → apply_race_rules
  → append_additions
  → apply_adjustments
  → apply_autopop
  → surface refresh        (re-call surface_from_location after autopop)
  → join_location_metadata
  → apply_historical       (city_state always; location only where blank)
  → race city_state/surface back-prop into daily race rows
  → synthesize_daily_from_additions  (pre-2016 race-only daily stubs)
```

Autopop runs **after** adjustments so event-normalization can match events that were set via the changes sheet. Adjustment-sourced surfaces are preserved during the surface refresh. The historical override fills `location` only where it's currently blank so freeze-time hill-loop synthesis (`powerline west`, `rollercoaster`, …) survives the catch-all entry. Synthesize runs last so its rows pick up the same canonical city_state/surface as `races.csv`.

## Race classification rules

- 5K within the XC date window → XC.
- 5K otherwise → Road.
- Distance < 3218m → Track.

## Location and event normalization

**Three hardcoded location substrings** (case-insensitive, on raw log location):

| substring | event | city_state | chains? |
|---|---|---|---|
| `carnation` | `Run for the Pies` | `Carnation, WA` | — |
| `shoreline` | `Club Northwest All Comers` | `Shoreline, WA` | → `surface=Track` |
| `boston` | `Boston Marathon` | `Boston, MA` | — |

`city_state` always overwrites; event is set only when blank.

**Six event-normalization rules** (case-insensitive):

| match | type | target |
|---|---|---|
| `All-Comers` / `All Comers` | contains | `infer_location=shoreline` (chains) |
| `Run for the Pies` / `Pies and Pints` | contains | `infer_location=carnation` (chains) |
| `Winter Grand Prix` | contains | `city_state=Seattle, WA` (direct) |
| `\bRedmond\b(?!\s*@)` | regex | `city_state=Redmond, WA` (direct) |

The Redmond regex catches `Eastlake @ Redmond` and `Redmond Invite` but **not** `Redmond @ Eastlake` (Max's HS was Redmond — these were home meets, location handled differently).

Matchers: `event_contains`, `event_endswith`, `event_regex`. Targets: `infer_location` (re-runs the location chain) or `city_state` (sets directly).

## Snapshot bundle format

Single CSV with `# section: NAME key=val` markers between seven blocks: `current_log` (year=YYYY), `changes`, `additions`, `locations`, `hills`, `coordinates`, `historical`. Read with `snapshot.read_snapshot()`.

`historical` schema: `city_state, min_hist, max_hist, log_location` (last column optional). Two roles: (a) seed cities on the world map that have no daily rows (Sapporo, JP); (b) override `city_state` (always) and `location` (only when blank) on non-race daily rows whose dates fall in the range. Multiple rows per city express disjoint visit windows; entries are applied in row order with last-wins on overlapping ranges. This replaces the legacy date-range branch of `infer_2016_2017_location`.

`coordinates` schema: `city_state, latitude, longitude` — overrides applied on top of the Nominatim cache so geocoding fixes are reproducible from source.

`build_dataset` resolution order: `--snapshot` flag → `data/drive_snapshot.csv` → Drive auto-fetch (last resort). Flags: `--refresh-snapshot`, `--no-fetch`.

The 2016 schema special-case: `split_2016_notes` peels trailing `[Event]` markers before doing the `w/` / `solo` split, then reattaches. Keep the peel-then-reattach pattern.

## Plot conventions

The plotting layer separates four concerns. **Never embed `<style>` or `<script>` blocks in Python f-strings** — every shape below has a home for its concern.

```
src/plotting/
  tokens.py            design tokens (colors, sizes) — single re-skin point
  layout.py axes.py markers.py formatters.py smoothing.py
                       Plotly helpers
  render.py            single rendering authority — render_plot()
  widgets.py           HTML primitives (sidebar, button_row, …)
  _scaffold/           shared CSS/JS loaded by every plot
    base.css           dark-theme chrome + .rp-* design system
    cursor_tooltip.js  smart spikeline + smooth/snap tooltip
    overlay_anchor.js  positions overlays below the legend
    shell.css shell.js tab shell — loaded by build_shell.py

src/plots/
  <plot>.py            data prep + Plotly figure construction
  <plot>.js            plot-specific JS (event handlers, restyle logic)
  <plot>.css           plot-specific CSS (only when rules don't generalize)
```

**`render_plot()` is the only thing that writes a plot HTML.** Each plot calls it with the figure plus, optionally, `overlay_html` (structural HTML), `overlay_js_files` (sibling `.js` paths), `extra_head_css_files` (sibling `.css` paths), and `cursor_tooltip` (the smart spikeline scaffold).

**Python ↔ JS handoff:** plots pass values to their sibling JS via `widgets.js_globals({KEY: value})`, which serializes as `window.__PLOT_<KEY>`. The `.js` file reads those globals at startup. Don't interpolate values into JS source — keep `.js` files static.

**Shared CSS classes (`_scaffold/base.css`):** `.rp-sidebar`, `.rp-sidebar-title`, `.rp-sidebar-sub`, `.rp-sidebar-stats`, `.rp-sidebar-noteworthy`, `.rp-sidebar-divider`, `.rp-detail-row`, `.rp-toggle-bar`, `.rp-btn-row`, `.rp-btn`, `.rp-btn-pill`, `.rp-row`, `.rp-row-meta`, `.rp-table` (`.num` for right-align). Active state is the `is-active` modifier. Use `widgets.*` helpers to compose these — don't hand-write the markup.

**When to use plot-specific `.css` vs. extending `_scaffold/base.css`:** generalize into `base.css` if two or more plots would share the rule; keep co-located when the rules are inherently plot-specific (geography's nested legend tree, long-runs' gradient strip).

**Plot-domain knobs:**
- `pr_marker(base_size)` — white-ringed diamonds for PR markers (line=1.5px, ring=base+1 tight hug).
- `PR_EXCLUDED_SURFACES = {'Downhill'}` — plotted but PR-ineligible.
- `GAP_BREAK_DAYS` — 90 for the TQ smoother; preserves the 2020–21 labrum gap.

**Plotly numeric-array gotcha:** numeric arrays in figure JSON serialize as `{dtype, bdata, _inputArray}`. `_inputArray` is a `Float64Array`, so `Array.isArray(...)` returns false. Use length checks. PR recompute via the `plotly_restyle` listener; skip when `indices == [prIdx]`.

**Plotly hover suppression:** plots with `cursor_tooltip` should set `hoverinfo='skip'` on every trace so Plotly's native hover label never renders. If a plot has `hoverlabel=` on any trace (e.g. recovery), pair it with `extra_head_css='.hovertext { display: none !important; }'`. The rule is **not** in `base.css` because `make_world_map.py` relies on Plotly's native `hovertemplate` hover.

## Hosting / auth model

The site at `running.maxrandalmusic.com` is **gated by default**. The Edge Function at `site/netlify/edge-functions/gate.ts` fires on every path (`/**`); only paths in its `EXEMPT_PATHS` set or `EXEMPT_PREFIXES` list bypass the gate. Plot HTMLs, the shell, and the admin page all live at the **root** of `site/dist/` — there is no `/plots/` namespace. Visitors hit `running.maxrandalmusic.com` and the gate either lets them through (if their session cookie is valid and their email is on the allowlist) or 302s them to `/login.html`.

**When adding a new file or route, the safe failure mode is "stays gated."** Adding a new HTML to `site/dist/`, a new plot script, or a new function endpoint requires no extra step to be auth-protected — that's the default. **To make something public, you must add it to the gate's exempt list.** Forgetting to gate something is what this design prevents; forgetting to exempt something just means a public page becomes login-walled until you fix it.

The exempt list (in `gate.ts`) currently covers: `/login.html`, `/plotly.min.js`, `/favicon.ico`, `/robots.txt`, plus the prefixes `/api/` and `/.netlify/` (each `/api/*` function does its own auth check; `auth-config` and `auth-exchange` are intentionally public, the rest verify a session or admin status server-side).
