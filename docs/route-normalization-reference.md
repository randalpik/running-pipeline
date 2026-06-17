# Route normalization — reference

Cross-cutting analysis on how location-specific pace costs propagate from
the locations sheet and the per-second watch stream through to recovery,
long-run, and race residuals. This is the **canonical home for the physical
elevation engine** (the cost model, the altitude curve, the pinned betas, the
per-run features, the DEM) — the recovery, training-quality, and CS-model
references describe how each layer *applies* it and point here for the engine.

## Schema and dataflow

The `locations` sheet of `Max's Running Data` (id
`1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w`) carries one row per
known log_location with columns:

| column         | type    | notes                                              |
|----------------|---------|----------------------------------------------------|
| log_location   | string  | match key — must equal raw `location` in daily.csv |
| city_state     | string  | "City, ST/Country" — also used by race autopop     |
| display_name   | string  | optional pretty name; blank if same as city        |
| elev_per_mile  | int     | feet per mile, route-intrinsic gain                |
| altitude       | int     | feet above sea level (0 for sea-level routes)      |
| terrain_type   | string  | "paved", "mixed", or "trail"                       |

Apr 2026 state: ~117 locations have at minimum log_location +
city_state + display_name; a subset (currently ~23) have
elev_per_mile/altitude/terrain_type populated. Max enriches over time.

`build_dataset.py` left-joins these columns onto `daily.csv` via the
`_join_location_metadata` helper just before writing. Downstream tools
read route metadata as daily.csv columns — no separate snapshot or
locations.csv lookup needed. Daily rows whose location isn't in the
sheet (blank, typos) get NaN in the metadata columns; downstream code
must handle NaN.

Since the per-second elevation enrichment shipped (below), `elev_per_mile`
and `altitude` are no longer the primary route-cost inputs — they are the
**fallbacks** the per-run features use when watch data is absent (pre-watch
runs, CI without the details cache). `terrain_type` is still load-bearing: it
buckets every run (paved / mixed / trail) for the cost model.

The 2016 and 2017 xlsx schemas don't have a location column, so the
parser synthesizes one at freeze time via `infer_2016_2017_location`.
As of 2026 the function does only **hill-loop synthesis**: hill
workouts parse the loop abbrev from `workout_raw` against the hills
tab (`hill_lookup`) to land precise route names (`powerline west`,
`rollercoaster`, …) — load-bearing for recovery route betas. Non-hill
2016-17 rows leave freeze with `location=None`.

The date-range rules that used to live here (Geneva, Butte, Kadoka,
Kansas City, Nashville windows; `education hill` default) were
migrated to the snapshot's `historical` section in 2026. Each window
is now a row with `city_state, min_hist, max_hist, log_location`, and
`build_dataset.py` applies them after `_join_location_metadata`. The
override fills `location` only where it's currently blank, so the
hill-loop route names survive the broad Redmond catch-all. The hills
tab still lives in Max's Running Data xlsx and rides through the
snapshot's `hills` section.

Because `apply_historical` can set a row's `location` *after* the metadata
join, `build_dataset._backfill_location_metadata` runs a second pass that
re-attaches `terrain_type`/`elev_per_mile`/`altitude` to those
historically-located routes (e.g. `education hill` — 524 rows that were
otherwise NaN and would have failed the paved gate and the route betas).

## Effort era thesis (April 2026)

Recovery effort has stayed remarkably consistent across the log;
long-run effort has shifted dramatically by era. This is why route-cost
estimation can't naively pool recovery + long — era-specific long-run
effort would leak into route-specific terms (the physical engine below
controls for it with an `is_long` level dummy + an era backfit).

Year-by-year median pace headroom over CS pace (sec/mi; positive =
slower than CS):

| year | recovery | long | gap (rec − long) |
|-----:|---------:|-----:|-----------------:|
| 2017 |     +56  | +49  |        +7        |
| 2018 |     +44  | +57  |       −13        |
| 2019 |     +51  | +55  |        −4        |
| 2020 |     +39  | +29  |       +10        |
| 2021 |     +35  | +14  |       +21        |
| 2022 |     +29  |  +6  |       +23        |
| 2023 |     +33  | +12  |       +21        |
| 2024 |     +35  | +37  |        −2        |
| 2025 |     +39  | +38  |        +1        |
| 2026 |     +37  | +37  |         0        |

Three eras visible:

- **2017–19**: long ≈ recovery effort (slightly slower if anything in
  2018; "longer easy days" mentality)
- **2020–23**: long is dramatically faster than recovery (+10 to +23
  sec/mi gap). This is the "quality, sub-marathon effort long run"
  prescription Max introduced when training in Nashville — Belle Meade,
  Greenway, North Greenway are the iconic routes here, all logged at
  long-run effort that would be sub-MP territory.
- **2024–26**: long ≈ recovery again (~0 gap). Long runs returned to
  honest-effort high-volume style.

This era structure is also why **footing and altitude are era-correlated**
(off-road/sea-level routes cluster early, paved/Boulder-altitude late) and
must be isolated by a backfit rather than a sequential era-detrend (which
would absorb them).

## The physical elevation engine (SHIPPED, June 2026)

Per-route empirical dummies and the hand-set `elev_per_mile × constant`
route cost were replaced by a physical, era-free engine driven by the
per-second watch altitude/GPS stream. One engine feeds all three layers — recovery, long-run, race
— through a single set of constants, so a route's intrinsic cost is computed
the same way everywhere.

### The cost model — `src/shared/elevation_cost.py`

Per-mile grade cost (s/mi) from per-mile gain/loss (ft/mi) on a terrain:

```
cost = c_up · gain  −  refund · c_up · loss
```

- **`c_up`** (`CLIMB_COST`): paved 0.19, mixed/trail 0.26 s/mi per ft/mi
  climbed (rough terrain costs more to climb).
- **`refund`** = the fraction of the climb cost returned on the descent —
  **terrain × EFFORT dependent**, the key hard-won finding. Paved descents
  refund nearly fully at easy effort (`REFUND_RECOVERY` paved 1.0 — you bank
  them) and less as you approach race pace (`REFUND_PAVED_BY_EFFORT` /
  `paved_refund(effort)` falls to ~0.85 at race effort — near v_max you can't
  speed up to cash the descent). Mixed/trail refund only ~0.34 at **all**
  efforts (rough footing caps the descent). So **paved rolling is cheap even
  racing (~0.3%); the textbook 1–2% rolling cost holds only for mixed/trail.**
- **Effort** = frontier pace at the run's *own* distance / run pace (cap ≤ 1;
  a marathon at marathon pace is at its ceiling ~1.0). Races sit at ~1.0;
  recovery and long runs both center ~0.85 (their effort ranges coincide, so
  the transfer between them is constant-effort, no scaling).
- The cost is **pinned, not fitted**: gain and loss are collinear within loops
  at run level, so the climb/refund constants are imported from the per-mile
  data (where gain and loss separate), not estimated from run-level residuals.

### Altitude (hypoxia) — threshold + linear curve

`recovery_model.altitude_regressor(alt_kft) = max(0, alt_kft − 3.0)`
(constant `ALTITUDE_THRESHOLD_KFT = 3.0`). The **shape is pinned from
physiology, the slope is fit to Max's data** — because his altitude data is
bimodal (≈1800 sea-level runs + ≈515 Boulder ≈5400 ft + only **3** Magnolia
runs at 8–9k ft), so it can identify a scale but not a shape.

- Shape: VO₂max is ~flat below ~914 m (3000 ft), then declines ~linearly with
  altitude (Wehrlin & Hallén 2006 — linear from 300→2800 m, which covers Max's
  2569 m max; the clinical heuristic is ~8–11% VO₂max per 1000 m *above*
  3000 ft; Péronnet et al. 1991 for the distance-dependence of the running-
  performance loss).
- Slope (fitted on the pooled corpus, below): ≈ **+2.28 s/mi per 1000 ft above
  the 3000 ft threshold** → Boulder ≈ +5.5 s/mi (the bulk of the data, which
  anchors the fit and is ~unchanged from the old model), **Magnolia ≈ +12.3
  (+37%)**, everything < 3000 ft → **exactly 0**.
- A line through the origin (the prior model) was wrong both ways: it invented
  a phantom hypoxia effect at low altitude (an ~11 s "correction" on a pre-
  watch Nashville 400 ft marathon) and under-sloped the high end. The threshold
  fixes both; it also removed a bug where the phantom low-altitude term flipped
  a race's "has measured data" flag and wrongly admitted a pre-watch downhill
  time trial to the CS fit.

### The pinned betas — `recovery_model.physical_route_betas()`

THE single source of truth for the two **fitted** physical constants:
off-road **footing** (`is_offroad` = mixed+trail binary, the flat-surface
penalty ≈ **+4.78 s/mi**) and the **altitude slope** (above). One constant per
channel applies in the recovery model, the long-run 5K conversion, and the
race correction.

- Pools recovery + in-slice long runs on a shared `pace − cs_pace` scale with
  an `is_long` level dummy and the recovery **era-backfit** (era smoother ↔
  parametric factors iterated). Pooling lets the large recovery corpus + the
  shared era control discipline the era-confounded long-run off-road rows
  (which alone read a spurious +12.7); the betas are corpus-stable.
- This **supersedes the earlier recovery-only principle** and the old between-
  route regression `β = −13.7 + 0.17·elev_per_mile + 6.6·is_mixed`. The earlier
  worry — that pooling long runs lets the 2020–23 quality-long-run era pull
  Nashville route betas too negative — is now handled by the `is_long` dummy +
  era backfit, so pooling is safe and adds the trail/altitude contrast recovery
  alone lacks (watershed, the lone trail route, has no recovery runs).
- Cached per data dir. Degrades gracefully: recovery-only when there are no
  long-run rows, zeros (= no correction) when the recovery fit is unavailable
  (sparse profiles, CI without a details cache).

### Per-run features

- **`per_run_elevation(df)`** → gain/loss (ft/mi). Fallback chain: per-run
  watch-MEASURED (`elevation_measured.csv`) → route-median measured (pre-watch
  runs on a watch-covered route) → the route's `elev_per_mile` constant
  (balanced gain≈loss) → 0. Anomaly guard: extreme watch failures
  (`gain/mi > max(100, 3× route median)`) revert to the route median.
- **`per_run_altitude(df)`** → thousands of feet. The run's MEASURED mean
  elevation — midpoint of the watch's smoothed daily min/max
  (`altitude_daily.csv`) — then the location base-elevation constant
  (`altitude`) as the pre-watch fallback, then 0. (The **race** path takes
  altitude from the DEM mean / `altitude_daily` only, never the per-city
  constant, so a pre-watch low-altitude race gets no phantom hypoxia term.)
  Always passed through `altitude_regressor` before it enters a cost.

### Data layer & the race DEM (see `watch-derived-cache-spec.md`)

`scripts/backfill_elevation.py` writes `elevation_measured.csv` (per-day
gain/loss, Minetti factor, + per race-row `dem_*` columns), `altitude_daily.csv`
(daily min/max elevation), and `elevation_splits.csv` (per-mile). `src/coros/
elevation.py` does the gridding/smoothing (10 m grid, 120 m window, matching
the device's own `elevGain` within 1–3% on hilly runs).

**Races use a DEM, not the barometric stream.** The watch's per-race *net* is
noise (the same Bolder Boulder course read −19 ft/mi net one year and +5 the
next) while its horizontal GPS track is reliable and reproducible. So
`src/coros/dem_elevation.py` resamples elevation from a DEM (OpenTopoData USGS
NED 10 m, SRTM 30 m fallback; point cache `data/dem_cache.json`) along the
cached GPS track, written into the race rows of `elevation_measured.csv` as
`dem_gain_ft`/`dem_loss_ft`/`dem_net_ft`/`dem_mean_elev_ft`. The DEM net
subsumes the earlier loop heuristic (a loop reads net ≈ 0 naturally). Recovery
and long runs stay on the barometric stream — they average out over thousands
of runs, and the pinned betas are fit on them.

### Key principles (honor these)

1. **One watch-validity decision per day.** If the watch failed to record the
   run (distance grossly off the hand log), NOTHING uses its data; the distance
   correction layers extra criteria on top, but elevation uses every *valid*
   day (it's the only grade source, more reliable than GPS distance).
2. **Pin only what's unidentifiable at the grain; fit everything else.** Grade
   constants are pinned (gain/loss collinear within loops). Footing and the
   altitude *slope* are fitted (identifiable from the terrain/altitude
   contrast). The altitude *shape* is pinned (the bimodal data can't find it) —
   from science.
3. **Backfit for era-correlated factors** (footing/altitude vary on the era
   timescale; a sequential era-detrend would absorb them).
4. **Refund is terrain × effort dependent.** Energy transfers across effort;
   the downhill *pace* refund does not.
5. **Bucket by LOCATION terrain, not race surface** (e.g. 7 "Road" races are
   run on mixed-terrain locations — Run for the Pies, etc.).
6. **Channels stay separate** (grade / footing / altitude) — no double-count.
7. **For races, pin the curve, don't refit it** from the races being corrected
   (races feed CS and the correction is against CS-derived effort — circular).

### How each layer applies the engine

- **Recovery** (`recovery-runs-reference.md`): the physical era-free route
  model replaces per-route dummies (temp/fatigue/TOD + footing + altitude +
  pinned grade `elev_cost`, all on corrected pace).
- **Long-run 5K conversion** (`training-quality-reference.md`,
  `workouts.project_long_runs`): grade + footing + altitude credited as flat /
  sea-level-equivalent TIME corrections *before* the World Athletics
  down-conversion to 5K-equivalent (June 2026 — replaced the β_long un-bias).
  Replaced the old route-constant `0.17·elev_per_mile`.
- **Race CS** (`cs-model-reference.md`, `recovery_model.race_physical_
  correction`): corrects each watch-covered race's time to its flat / sea-level
  / smooth-equivalent before it informs CS, replacing the categorical XC ×1.08
  / Downhill-exclusion where watch data exists (categorical = pre-watch
  fallback). Track races gate grade off (by surface) but keep altitude.

## Powerline name disambiguation

Four distinct routes on the PSE Trail in Redmond, with confusingly
overlapping naming between the location field and workout shorthand:

| log token / location          | display          | character                        |
|------------------------------|------------------|----------------------------------|
| `pwr1`                       | powerline west   | short steep hill loop            |
| `powerline 2`                | powerline mid    | flat gravel (intervals/tempo)    |
| `pwr3`/`pw3`/`powerline 3`   | powerline east   | long rolling hill (cont. loop)   |
| `pwr2` (workout shorthand)   | (no loc)         | steepest segment of east, hill reps |

Nashville-era occurrences of `powerline 2`/`powerline 3` are visit
days; the routes are physical features in Redmond, not duplicated in
Nashville.

The clash between `pwr2` (workout shorthand for steepest segment of
east) and `powerline 2` (location for the flat gravel section) is a
naming legacy, not a contradiction. These are workout-specific entries
and don't carry elev_per_mile (continuous-hills elevation is handled
separately in the workout decomposer).

## Open items

- **Trail terrain underidentified.** The mixed/trail split leans on one trail
  route (~16 rows); the trail channel uses mixed as a proxy. Adding 2–3 Boulder
  mountain trail routes (Bear Peak, Flagstaff, Magnolia, etc.) with
  `terrain_type` populated would let a trail-specific footing/refund stabilize.
- **GHA/production parity.** Local builds use the watch elevation (and the race
  DEM); CI currently lacks the details cache, so deployed numbers differ from
  local. By design for now — local correctness is the goal; the fix at
  production go-live is to fetch the same watch data / commit the `*_measured`
  + `altitude_daily` + `dem_cache` artifacts, not a modeling change.
- **One-pass beta lag.** `physical_route_betas` depends on CS, which the race
  correction shifts — so a refit nudges the betas, used at the next refit. Max
  chose to let them evolve naturally (verified to converge geometrically;
  contraction ~0.05 footing / ~0.19 altitude) rather than pin them.
- **Locations-sheet enrichment.** `elev_per_mile`/`altitude`/`terrain_type` are
  the pre-watch/CI fallbacks; populating more of the ~117 locations improves the
  fallback path for runs the watch never recorded.
</content>
