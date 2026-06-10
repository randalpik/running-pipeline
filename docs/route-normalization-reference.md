# Route normalization — reference

Cross-cutting analysis on how location-specific pace costs propagate from
the locations sheet through to recovery and long-run residuals. Captures
working assumptions, decisions, and findings that don't sit cleanly
within either the recovery model (`recovery-runs-reference.md`) or the
training-quality framework (`training-quality-reference.md`) but inform
both.

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

## Effort era thesis (April 2026)

Recovery effort has stayed remarkably consistent across the log;
long-run effort has shifted dramatically by era. This matters because
naively pooling recovery + long for route-beta estimation lets
era-specific long-run effort leak into route-specific terms.

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

## Recovery-only design principle

Empirical route betas in the recovery plotter are fit on
**recovery-only** data, not recovery+long. This is intentional and
load-bearing.

If long runs are pooled in, the 2020-23 quality-long-run era pulls
route betas for long-run-dominant Nashville routes too negative (they
look "fast" — but it's the era's effort, not a property of the route).
Confirmed empirically — when refitting at MIN_ROUTE_N=13 with recovery+
long pooled, Belle Meade lands at β=−20 and North Greenway at −19,
~16-21 sec/mi more negative than predicted by their elev_per_mile +
terrain. Watershed (logged exclusively as long runs in the post-Boulder
era) lands at +17 instead of the +8 predicted.

Recovery-only refit is clean: every route's beta is consistent with the
elev+terrain-predicted value to within the noise floor.

**Working principle**: the route's intrinsic cost (terrain, elevation,
surface) shouldn't depend on what effort you're running it at.
Recovery is the cleanest baseline because effort is most consistent
there.

## Cross-route findings (recovery-only, n=11 routes with elev populated)

Single-feature regression of route beta against elev_per_mile:

```
β_route = −13.7 + 0.22 · elev_per_mile      R² = 0.64, RMSE 6.6
```

Two-feature with terrain dummy:

```
β_route = −13.7 + 0.17 · elev_per_mile + 6.6 · is_mixed     R² = 0.70
```

Three-feature with altitude dummy added:

```
β_route = −13 + 0.16 · elev_per_mile + 6.6 · is_mixed + ε · is_altitude
                                                              R² = 0.72
```

Altitude as a feature adds essentially nothing once elev_per_mile is in
the model — the 4 altitude routes lie on the same elev line as the
sea-level routes. **Don't add altitude correction** until there's a
genuine outlier, or substantially more altitude+long-run coverage.

Terrain matters: at the same elev_per_mile, mixed routes cost
~6-7 sec/mi more than paved at recovery effort. Trail penalty currently
unidentifiable from recovery alone (watershed, the only trail route in
the data, has zero recovery runs — all long).

## Long-run TQ corrections (June 2026: physical terms)

Long-run residuals in the training-quality framework are corrected by an
in-plot OLS fit on the in-slice set (`miles ∈ [15.1, 25.3]`):

```
raw_resid ~ bin + elev_pm(pinned at +0.17) + altitude(fitted)
            + temp + race-fatigue
```

See `training-quality-reference.md` Stage 5b for parameters.

### History: empirical per-route betas (May 2026) and why they were removed

Stage 5b originally fit `raw_resid ~ bin + route` with empirical
per-route dummies, accepting era confounding as tolerable for plot
purposes ("the smoother track interpretation shifts to within-route
within-era trend"). June 2026 showed the confounding corrupts
*point-level* readings, which the graph visually invites: every named
long-run route lives in one contiguous era with its own effort policy,
so betas measured "typical effort that year", not terrain. Physically
near-identical flat sea-level routes (south lakefront, north greenway)
carried betas 31 s/mi apart, and a moderate 2026 effort out-ranked the
all-time-best 2023 long run after correction.

The fix is the **predicted-β idea this doc anticipated** (the cross-route
fit `β = −13.7 + 0.17·elev_per_mile + 6.6·is_mixed`): the TQ model now
pins the elevation slope at the recovery-derived **+0.17 s/mi per ft/mi**
— effort-uncontaminated because recovery effort is uniform and recovery
routes span eras; transferable because elevation cost is mechanical
work, not a physiological state response — and fits altitude (≈ +3 s/mi
per 1000 ft, identified by the within-Boulder-era sea-level contrast).
The empirically-fit elevation slope on long runs comes out wrong-signed
(−0.13) from the same era confounding, which is also why the earlier
"altitude has no statistical effect" null was unfalsifiable: all 42
in-slice altitude long runs fall inside the 2023–25 Boulder window, so
altitude was unidentifiable from long-run data without an external
anchor.

The recovery plot stays untouched: recovery's own empirical route betas
remain valid (uniform effort, cross-era routes) and its
`route_betas_{tag}.csv` continues to represent intrinsic route cost.

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

## Open questions

### Trail route expansion

Adding 2-3 more trail routes with elev_per_mile populated would let the
trail-penalty coefficient stabilize. Currently the +9 trail offset
rests on watershed alone. Useful for Boulder mountain runs (Bear Peak,
Flagstaff, Longs Peak, Magnolia, etc.) and Banff trails.

### Altitude as feature

For **recovery** route betas, altitude currently doesn't help (3-4
altitude routes, all lie on the same elev_per_mile line as sea-level) —
plausible, since recovery effort at altitude is only mildly suppressed.
Worth revisiting if (a) Max accumulates substantially more altitude data
or (b) a genuinely outlying altitude route appears that elev_per_mile
alone can't explain.

For **long runs** the earlier null was an identification failure, not a
finding: all in-slice altitude long runs sit inside the 2023–25 Boulder
high-effort era, so altitude was perfectly confounded with effort
policy. The June 2026 TQ physical model fits altitude at ≈ +3 s/mi per
1000 ft via the within-era sea-level contrast (see "Long-run TQ
corrections" above).

### When to rebuild the elev+terrain coefficients

The `−13.7 + 0.17·elev + 6.6·is_mixed` coefficients above are a
snapshot. They should be rerun whenever:
- A new route crosses MIN_ROUTE_N (currently 13) for recovery
- elev_per_mile/terrain_type values are revised in the locations sheet
- Trail or altitude data expands enough to flip the conclusion that
  altitude doesn't contribute (for recovery)

(June 2026) The TQ long-run model pins `LR_ELEV_SLOPE = 0.17` in
`src/shared/long_run_model.py` from this fit — when these coefficients
are rebuilt, update the pinned constant to match.

The recovery plotter writes `route_betas_{tag}.csv` on every run, which
provides the per-route empirical betas. The cross-route fit (regressing
those betas against elev_per_mile/terrain_type) hasn't been added to
any production script — it's a one-off analysis. Suitable for a
notebook or ad-hoc script when revisited.

### Race route corrections

Not yet considered. Race times use the race's own surface tag (Road/
Track/XC/Downhill) for surface_source, but elev_per_mile / terrain_type
on the *race location* could potentially refine race-pace expectations
in the same way they refine recovery pace. Whether this matters in
practice depends on how much race elevation varies — most road races
are on rolling courses with elev_per_mile in the 30-80 range, which
under the current model would predict only a few seconds per mile of
adjustment.
