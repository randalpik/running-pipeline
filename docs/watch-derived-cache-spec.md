# Watch-derived data: unified per-day cache (spec)

Status: **proposed** (design approved in discussion; not yet implemented).
Owner: Max. Author: Claude.

## 1. Goal

Collect every watch-derived per-day quantity the pipeline consumes into one
compact, date-keyed source of truth, built **incrementally by presence**: if a
row is populated, its watch-derived data is trusted and never regenerated unless
we explicitly direct it. The only automatic trigger is a **new run** (a logged
run-day with no derived row yet). This makes the regular daily build — the one
Max runs after logging, and wants fast — derive only the new day('s) and skip
the rest, while the expensive full sweep happens only on schema changes,
historical-log edits, or the CS refit (all rare, or on GHA where time is free).

The animating principle is the one we already applied to the slim cache:
**derive once into the compact form everything references; re-derive only when
what we extract changes.** The per-second stream is immutable, so re-parsing it
every build is pure waste.

## 2. The problem this solves

The Coros detail cache is **~339 MB**: the elevation backfill turned 2033 of
2180 activities into *rich* records (per-second `[t, dist, heart, gpsLat,
gpsLon, altitude, speed]`). Today every watch-derived producer re-reads and
`json.loads` that full 339 MB on **every** pipeline run, even when nothing
changed:

| Producer | Output(s) | In daily pipeline? | Cost |
|---|---|---|---|
| `weather_measured.py` | `weather_measured.csv` | yes (pre-`build_dataset`) | ~5 s (now mtime-guarded) |
| `long_runs.py` | `long_run_measured.csv`, `recovery_measured.csv`, `long_run_calibration.csv` | yes (post-`build_dataset`) | ~10 s |
| `reps.py` | `workout_measured.csv` | yes (post-`build_dataset`, gated on CS) | (parse heavy) |
| `backfill_elevation.py` | `elevation_measured.csv`, `elevation_splits.csv` | **no** — manual backfill, append-mode | — |

Measured field usage across consumers (verified): every per-second field is
read by *someone* (`long_runs`: t,dist; `elevation`: t,dist,altitude; `reps`:
t,dist,heart,gps) **except `speed`**, which is dead. So the stream can't be
shrunk much by dropping fields, and it's our *only* copy (raw ~1.5 MB Coros
detail isn't cached and old activities age out of the API). The win is therefore
not compression of the stream but **not re-parsing immutable history**.

## 3. Core principles

1. **Presence, not mtime.** A populated row ⇒ accurate. `daily.csv` is rewritten
   wholesale every build, so its mtime is meaningless as a freshness signal —
   we never use file timestamps to decide staleness.
2. **Cache the watch scalars; join the classification fresh.** The derived row
   stores only quantities that are pure functions of the immutable stream.
   `run_type` (long / recovery / race / workout) is **not** stored — it is taken
   from final `daily.csv` every build via a cheap join. This is what makes
   race-adds and reclassifications free (see §7).
3. **Automatic regen only on a new run.** A logged run-day with no derived row
   gets derived. Nothing else auto-regenerates.
4. **Explicit / event regen for everything else.** Schema change, historical-log
   edit, or `--force` → full sweep. The CS refit is treated as such an event
   (the **safety valve**, §6): when CS refits, *all* derived tables fully
   regenerate from the immutable streams, drift-proofing the incremental caches.
   This is cheap relative to where it happens (GHA, ~15 min job).

## 4. The unified table

`data/watch_daily.csv` — one row per **watch-activity day**, keyed by ISO
`date`. Holds the scalar per-day watch derivations all consumers need:

```
date,
n_acts, status,                 # status = rich | slim (best stream available)
# --- weather (from the day's rep activity) ---
temp_c, weather_bin, wind_ms, humidity_pct, time_of_day,
# --- distance / structure (whole-day, watch-measured) ---
watch_miles, watch_moving_s, watch_total_s, pause_s, stall_s,
n_segs, d_eff_frac, longest_seg_mi,
# --- elevation ---
gain_ft, loss_ft, corr_miles,
# --- provenance ---
label_ids,                      # sorted labelId set consumed for this day
derived_schema_version          # bumps to force a global regen on schema change
```

- **`run_type` is deliberately absent.** Consumers join it from `daily.csv`.
- **`weather_bin`** is carried for completeness but the daily pipeline still
  *holds* the hand-logged `weather` (per the accuracy comparison); it's here so
  the table is the single source if we ever revisit that decision.
- **Variable-length artifacts stay as date-keyed satellites**, under the same
  presence/regen rules — they can't flatten into one row:
  - `workout_measured.csv` — per-rep structure (reps).
  - `elevation_splits.csv` — per-mile gain/loss.
  These reference `watch_daily.date`; the scalar roll-ups (`gain_ft`,
  `d_eff_frac`, …) live in `watch_daily` so most consumers never open a
  satellite.

### Watch-less log-days
A logged run with no watch activity (pre-2021, watch not worn, synth race stub)
gets a row with `n_acts=0` and null watch fields. This is a **populated**
state — it is never retried (except the recent-window rule, §5) — so absence of
watch data doesn't masquerade as "not yet derived."

## 5. Build algorithm

```
build_watch_daily(force=False, full_regen=False):
    if force or full_regen or schema_version_changed():
        candidates = all label files                 # full sweep
        existing   = {}                              # rebuild from scratch
    else:
        existing   = load(watch_daily.csv)           # trusted rows
        # labelIds are time-monotonic -> new files by FILENAME, no parse:
        new_files  = [f for f in label files if labelId(f) > max_processed_labelId(existing)]
        # plus recent-window empties to catch late-arriving syncs:
        retry_dates = {d for d in existing.empty_rows if d >= today - RECENT_WINDOW}
        candidates = new_files âˆª files_for(retry_dates)

    days = group_by_local_date(parse(candidates))    # parse ONLY candidates
    for d, acts in days:
        existing[d] = derive_day(d, acts)            # weather+dist+elev scalars
    # log-days with no activity at all -> ensure a null row exists, once:
    for d in daily_run_days not in existing:
        existing[d] = null_row(d)
    write(watch_daily.csv, existing)
```

Key properties:
- **Common case (one new run):** parse a handful of new files, derive one row,
  trust the rest. The 339 MB is never fully read.
- **`max_processed_labelId`** is read from the cached table's `label_ids` — no
  stream parse needed to know what's new.
- **`RECENT_WINDOW`** (≈30 days) re-attempts recent null rows so a watch
  activity that synced *after* the log row (ordering in `build_profiles`: max
  builds before the coros sync) is picked up next build. Old nulls stay trusted.
- **`derived_schema_version`** is a constant in code; bumping it (when we change
  what we extract) makes `schema_version_changed()` true → global regen, no
  flag needed.

## 6. The CS safety valve

`reps` (and only reps among the watch derivations) reconciles against the logged
`workout_raw` **and** the CS cutoff, so a CS refit can change an
already-derived recent workout day. CS refits automatically on GHA whenever a
race is added.

Rule: **the producers run in `full_regen` mode iff the CS fit ran this build.**
The orchestrator already knows this (it decided to fit), so it passes the signal
— structural, not timestamp-based. Concretely:
- Local daily build: `run_pipeline.sh` reuses cached `bayes_cs_*` (no `--fit`) →
  `full_regen=False` → everything incremental → fast.
- GHA race-add: CS refits → orchestrator passes `full_regen=True` → all derived
  tables rebuild from the immutable streams (~tens of seconds against a ~15-min
  job), refreshing reps against new CS and drift-proofing the rest.

So the only thing that pays the full-parse cost is an event that already lives
where cost doesn't matter.

## 7. Interaction with races.csv

Adding race data touches `daily.csv` three ways (`build_dataset.py`): retype the
day to `run_type='race'` + clear quality/recovery; back-prop city_state/surface;
synth a stub row for race dates with no daily row. Mapped onto this design:

- **Reclassification is free** because `run_type` is *not* cached — it's joined
  fresh from final `daily.csv` every build. A day that flips `long → race`
  drops out of the long/recovery/workout consumer views automatically; the
  watch scalars for that day stay valid and are never re-parsed.
- **Synth stubs / pre-watch race days** → `n_acts=0` null rows (§4). One write
  when first added; old, so the recent-window leaves them alone.
- **Modern race days (2021+)** raced with the watch → real activity → weather /
  elevation / `watch_miles` derive normally; the row just carries
  `run_type='race'` from the fresh join. (`races.csv` keeps its hand-logged
  temp — built before the watch-weather join, unchanged from what shipped.)
- **reps** is the one race-sensitive derivation (race retype clears quality);
  it's covered by the CS-fit safety valve, since a race-add is exactly what
  refits CS on GHA. Locally, the fresh `run_type` join still excludes the race
  day from the workout view, so nothing stale is consumed.

Ordering requirement: producers that need `run_type` read **post-`build_dataset`
daily** (after retype/back-prop/synth). `long_runs`/`reps` already do; weather is
watch-only and indifferent.

## 8. Migration of existing producers

`watch_daily.csv` is produced by one new orchestrating builder
(`src/coros/watch_daily.py`) that calls the existing derivation logic and writes
the unified row + satellites. Each current producer is refactored so its
*derivation* is a callable the builder invokes per-day; the producer's CLI stays
for standalone/back-compat but delegates to the builder.

- **weather** — fold `weather_measured.py` derivation into `watch_daily`
  (`temp_c, weather_bin, wind_ms, humidity_pct, time_of_day`). `build_dataset`'s
  `_apply_weather_measured` reads `watch_daily.csv` instead of
  `weather_measured.csv` (or we keep emitting `weather_measured.csv` as a
  projection for zero consumer churn — TBD §11).
- **long_runs** — `measure_runs` per-day output → `watch_daily` scalars
  (`watch_miles…d_eff_frac`). `fit_calibration` stays a **parse-free** step that
  recomputes every build off `watch_daily` rows joined to logged miles +
  `run_type` from daily; writes `long_run_calibration.csv` unchanged.
- **elevation** — promote `backfill_elevation.py` into the regular flow as a
  `watch_daily` contributor (`gain_ft, loss_ft, corr_miles`) + `elevation_splits`
  satellite. It already appends incrementally; the presence rule generalizes that.
- **reps** — `build_workout_measured` becomes presence-based + CS-safety-valve;
  `workout_measured.csv` stays the satellite, keyed by `watch_daily.date`.

Consumers to repoint (read the unified table / satellites; mechanical):
`recovery_model.py` (recovery_measured, elevation_measured), `workouts.py` +
`plot_workouts.py` (long_run_measured, workout_measured), `build_dataset.py`
(weather), `parse_workouts.py` (workout_measured). Decision in §11 on whether to
repoint or keep legacy CSVs as projections.

## 9. Pipeline wiring

- `run_pipeline.sh`: replace the separate `weather_measured` / `long_runs` /
  `reps` steps with one `watch_daily` build before `build_dataset`'s consumers
  need it. `full_regen` flag wired to whether the CS fit ran this build.
- `build_profiles.py`: same, per profile (coros/maddy build their own
  `watch_daily` from their own cache; the max profile's builder reads the
  `coros` sibling cache, as `weather_measured` does today).
- Reconciliation steps that need `run_type` continue to run **after**
  `build_dataset`.

## 10. Explicit regen surface

- `--force` on the builder: full sweep regardless of presence.
- `derived_schema_version` bump: auto global regen on next build.
- CS-fit event: `full_regen=True` (safety valve).
- Deleting `watch_daily.csv`: cold rebuild.

Document: "after editing historical log data or changing a derivation, run with
`--force` (or bump the schema version)." This is the conscious trade Max wants —
pay the full parse on deliberate changes, never on a routine daily log.

## 11. Open questions

1. **Legacy CSVs: repoint vs. project.** Keep emitting
   `recovery_measured.csv` / `long_run_measured.csv` / `workout_measured.csv` /
   `weather_measured.csv` as thin projections of `watch_daily` (zero consumer
   churn, slight duplication) **or** repoint every consumer to `watch_daily`
   (cleaner, more edits). Lean: project first (smaller diff), repoint later.
2. **RECENT_WINDOW length.** 30 days proposed; pick from how late watch syncs
   realistically arrive.
3. **Per-profile vs shared builder.** coros/maddy derive natively; max derives
   from the coros sibling cache. Confirm the builder takes the cache dir as a
   param (it does today via `get_profile('coros')`).
4. **Phasing.** Suggest: (A) build `watch_daily` + presence/labelId/recent-window
   + safety-valve, projecting legacy CSVs; verify byte-identical outputs vs. a
   full rebuild; (B) repoint consumers; (C) fold elevation into the regular flow.

## 12. Verification

- **Equivalence:** a `--force` full build of `watch_daily` (and the projected
  legacy CSVs) must be byte-identical to today's outputs on the current cache.
- **Incrementality:** touch/sync one new activity → only that day's row changes;
  assert no other rows differ and the 339 MB isn't fully parsed (instrument the
  parse count).
- **Race-add:** add a race on a former long-run day → with no `--fit`, the long
  view drops it and watch scalars are unchanged (no re-parse); with `--fit`,
  reps regenerates and matches a from-scratch build.
- **Late sync:** simulate a log-day whose activity appears a build later → the
  recent-window retry fills it; outside the window it stays null.
- **Timing:** regular daily build (one new run, no fit) parses only new files;
  target sub-second watch-derivation overhead.
