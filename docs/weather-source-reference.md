# Weather source reference

Where the watch-era `temp_c` / `wind_mph` / `humidity_pct` come from, why, and
what was ruled out. Read before touching `src/coros/weather_hrrr.py`, the
per-day weather rule in `build_current_log._day_weather`, or any model term that
consumes these columns.

## What the pipeline does (Oct 2026)

Every outdoor Coros activity gets NOAA **HRRR** (3 km hourly model, via
Open-Meteo's historical-forecast API, `models=gfs_hrrr`) temperature, relative
humidity and 10 m wind at its first GPS fix, **averaged over the run's elapsed
window** (start → start + total time, minute-sampled, linear between hours).
The day's value is the run-time-weighted mean over its outdoor runs. Where HRRR
is unavailable (indoor, no GPS, outside HRRR's North-American domain — 31 of
Max's 2,291 watch runs, all abroad) the run falls back to the Coros snapshot.
`watch_daily.weather_src` records `hrrr` / `coros` / `mixed` / `pending`.

Pre-2021 days keep the hand log. The weather *bin* (`weather`) and the
qualitative `wind` bin are unchanged: the hand-logged bin is still held, and
the watch-only profiles' `wind` bin stays on the Coros snapshot because its
thresholds (`mappings.WIND_BINS_MPH`) are Coros-scale.

## Source semantics

| Source | What it is | Time it describes |
|---|---|---|
| Coros `weather` | AccuWeather snapshot attached to the activity (no timestamp) | **Run start** (see below) |
| Hand log (2016 – 2025) | Wunderground, or a real outdoor thermometer (most Redmond logs before 2026) | `(start + end) / 2`, rounded to integer °C (half up) |
| HRRR (Open-Meteo) | 3 km model; Open-Meteo shifts T and dewpoint from grid-cell height to the requested point's 90 m-DEM elevation at 6.5 °C/km (RH and wind untouched) | Any hour; we average the run window |

**Coros is a start-of-run snapshot.** Against an hourly reference, regressing
`Coros − ref@start` on `ref@end − ref@start` gives slope −0.02 (95 % CI −0.14 to
+0.11) on runs that changed ≥ 2 °C — 0 means start, 0.5 average, 1 end — and
the best-matching lag sits at −30 … +10 min. Cost: run-average minus start is
+1.85 °C on long runs starting before 10:00 (+2.29 °C for ≥ 2 h), about −0.5 °C
for evening runs. The hand log behaves like the run average: `hand − Coros`
tracks `avg − start` with slope +0.74 [0.61, 0.88]. So the June 2026 watch-temp
adoption had silently swapped a run-average-like series (2016 – 2020) for a
start-of-run one (2021+); the "watch reads 0.43 °C cold vs hand" found then was
mostly this.

## Why HRRR — the evaluation

No source is ground truth, so the ranking uses **extended triple collocation**:
with three sources whose errors are independent, pairwise covariances identify
each source's error variance; with more sources the system is overdetermined
and pairs that share an upstream (agreeing too well) show up. ERA5, ERA5-Land
and ECMWF IFS share errors (same model family), so only triplets drawn from
distinct families were used. 10 main location cells, single-activity days,
2021 – 2025.

**Temperature level** (day-to-day error SD after removing each source's
per-city mean, runs where the temperature barely changed, n = 373):

| HRRR | hand | Coros | nearest airport (ASOS) | ERA5 / ERA5-Land / IFS |
|---|---|---|---|---|
| **0.52** [0.43, 0.60] | 0.65 | 0.75 | 0.85 | 1.4 – 1.6 °C |

HRRR is best or tied in each of Boulder, Chicago, Nashville and Redmond, and
stable across triplets and the flatness threshold.

**Within-run change** (error SD on half the run's change, n = 1,416; `hand −
Coros` is an independent observation of it): HRRR 0.17 °C vs 0.36 – 0.39 for
every other hourly source. HRRR's change explains `hand − Coros` with slope 0.91.

**Static per-city offsets** are invisible to collocation and were checked
against physical references:

- *Redmond* (hand = real thermometer on the hill): Coros reads warm by ~0.4 –
  0.65 °C regardless of wind or time of day (so not a lapse-rate effect, which
  needs a mixed boundary layer — mechanism unknown); HRRR is within ~0.1 – 0.2 °C,
  its residual gap appearing only on calm days (a local calm-air effect a 3 km
  grid can't resolve). The ranges span the hand log's rounding (0 to +0.25 °C
  mean, depending on whether the start/end readings were whole degrees).
- *Chicago*: Midway reads +2.36 °C above HRRR-at-the-lakefront in Apr – Jul with
  onshore wind, +0.3 – 0.6 otherwise — the lake-breeze signature, which HRRR
  resolves; Wunderground matches HRRR within ~0.3 °C in every stratum.

**Wind** (correlation with the common signal — scale-free, since sources differ
in height/averaging): HRRR 0.94 > airport 0.85 > Coros 0.80 > ERA5 0.76 > hand
bin 0.65. HRRR is best or tied in every city. Coros wind runs ≈ 1 / 1.7 of
HRRR's level (Chicago 5.0 vs 9.2 mph). The "truth" here is the shared 10 m
regional wind, not what a runner feels on a sheltered street.

**Humidity** is a three-way tie (dewpoint: airport 0.991, Coros 0.990, HRRR
0.987; RH: HRRR 0.968 best), ERA5 clearly worst. HRRR is used for consistency —
the same series as the temperature, over the same window.

## Model consequences

- **Temperature** (pooled recovery + long-run heat hinge): picked up
  automatically. Refit on the shipped data: slope +0.340 → +0.328 s/mi per °C
  above 6 °C (projected +0.326); time-of-day −7.38 → −6.99; altitude
  +1.73 → +1.63 s/mi per 1000 ft; R² (detrended) 0.299 → 0.298; other terms
  < 1 %. Watch-era long runs: heat credit +0.30 s/mi on average (p90 +1.04,
  max +1.9); morning long runs read ~1.7 °C warmer.
- Watch days with no Coros weather (2020-12-25 … 31, the watch's first week)
  previously kept the hand temp; HRRR now fills them.
- **Wind term REMOVED from the recovery model.** On Coros wind it had decayed to
  +0.16 s/mi/mph (t = 1.5); on HRRR wind it fit +0.02 (t = 0.3); within city,
  windier days were *faster* in Nashville and Boulder for both sources
  (t −2.2 … −3.4) — the pooled positive slope came only from mixing cities, and
  Coros's small signal sat in the part of its wind HRRR doesn't explain.
  `wind_mph` remains displayed data (tooltips, colour-by, Misc. Trends).

## Implementation

- `src/coros/weather_hrrr.py` — `ensure(details_dir)` fetches a sidecar for
  each activity lacking a final one; `attach()` puts it on the in-memory
  record; `derive()` is the per-activity computation.
- **Sidecar** `details/wx/<labelId>.json`: `src` (`hrrr` | `none` + `reason`),
  `temp_c` (window mean), `temp_start_c`, `temp_max_c`, `rh_pct`, `wind_mph`,
  `lat`/`lon`/`elev_m`, `t0`/`t1`, `fetched_utc`, `final`. It lives inside the
  cached details directory so CI persists it **without adding a path to the
  actions/cache list** (the list is the cache's identity); every details reader
  globs non-recursive `*.json`, so the subfolder is invisible to them.
- **Final vs provisional**: a value fetched within 48 h of the run's end is
  provisional and re-fetched once after that; a transient failure (network,
  429, timeout, hours not yet available) writes nothing and retries next
  build; a run of ≥ 5 transient failures with no success between them stops
  the step (offline / API down).
- **Wiring**: `sync_current_log` calls `ensure` (all Coros profiles, also on
  `--rebuild-coros`); `run_pipeline.sh` runs it before `watch_daily`
  (normally a no-op). `watch_daily` (schema 7) re-derives a `pending` day once
  every sidecar for it is final — small-file reads, no stream parse.
- **Backfill cost**: one request per activity, 4 workers (~4 req/s): Max's
  2,316 + Maddy's 150 activities once per environment (locally, then the first
  CI run).

## Gotchas

- Open-Meteo **downscales temperature to the requested point's elevation**.
  Fetching at a cell centre puts the value at the centre's elevation (ERA5 at
  the Boulder cell centre sat at 1,993 m → −2.9 °C). Always request at the
  run's own coordinates.
- Hand temps are integers; rounding noise (SD ≈ 0.29 °C) is part of the hand
  log's measured error.
- `temp_c` stays air temperature. Humidity / heat index were dropped from the
  recovery model in June 2026 and this change does not revisit that.

## Don't re-propose

- **Anchoring to Coros + an ERA5 shape** (the first idea): Coros is noisier
  than HRRR at level, and ERA5 is the worst source on both level and shape.
- **A scalar wind-speed pace term** on HRRR or Coros wind — not identified (see
  above). A future wind attempt would need a different form (e.g. route-relative
  headwind exposure), not a re-fit of the same scalar.
