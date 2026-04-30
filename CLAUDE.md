# CLAUDE.md

Operational guidance for Claude Code working in this repo. For project-level context — what each reference doc covers, how the analysis layers fit together, where the data comes from — see `docs/ARCHITECTURE.md`.

This is a personal running-data pipeline: parsers and transformations turning daily training logs into analysis-ready CSVs and HTML plots. Owner: Max.

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
  → race city_state/surface back-prop into daily race rows
```

Autopop runs **after** adjustments so event-normalization can match events that were set via the changes sheet. Adjustment-sourced surfaces are preserved during the surface refresh.

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

Single CSV with `# section: NAME key=val` markers between five blocks: `current_log` (year=YYYY), `changes`, `additions`, `locations`, `hills`. Read with `snapshot.read_snapshot()`.

`build_dataset` resolution order: `--snapshot` flag → `data/drive_snapshot.csv` → Drive auto-fetch (last resort). Flags: `--refresh-snapshot`, `--no-fetch`.

The 2016 schema special-case: `split_2016_notes` peels trailing `[Event]` markers before doing the `w/` / `solo` split, then reattaches. Keep the peel-then-reattach pattern.

## Plot conventions

Shared utilities live in `src/plotting/` and should be imported by every plot script — not copy-pasted.

- `pr_marker(base_size)` — white-ringed diamonds for PR markers (line=1.5px, ring=base+1 tight hug).
- `PR_EXCLUDED_SURFACES = {'Downhill'}` — plotted but PR-ineligible.
- `GAP_BREAK_DAYS` — 90 for the TQ smoother; preserves the 2020–21 labrum gap.
- Color and font tokens — single source.
- A standard plotly layout helper.

**Plotly numeric-array gotcha:** numeric arrays in figure JSON serialize as `{dtype, bdata, _inputArray}`. `_inputArray` is a `Float64Array`, so `Array.isArray(...)` returns false. Use length checks. PR recompute via the `plotly_restyle` listener; skip when `indices == [prIdx]`.
