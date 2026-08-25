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

## The physical elevation engine (Aug 2026: two-channel hill model)

Per-route empirical dummies and the hand-set `elev_per_mile × constant`
route cost were replaced by a physical, era-free engine driven by the
per-second watch altitude/GPS stream. One engine feeds all three layers —
recovery, long-run, race — through a single set of constants, so a route's
intrinsic cost is computed the same way everywhere.

### The cost model — `src/shared/elevation_cost.py`

Hill cost is a **fraction of pace**, priced from each run's validated hill
segments — two independent channels, no refund ratio:

```
cost_fraction = c(g_up)·hill_gain_pm − b(g_dn)·hill_loss_pm
c(g) = c0 + c1·g            climb cost per ft/mi of HILL vertical
b(g) = b0 + b1·g            descent benefit per ft/mi (b1 < 0)
cost_s_per_mi = cost_fraction · pace / (1 + cost_fraction)
```

Inputs are pure course geometry: vertical inside hill segments (ft/mi) and the
vertical-weighted mean grade of each direction's segments (the weighting under
which a linear model is exact). Sub-threshold undulation is the flat baseline,
priced at zero. Nothing about execution enters, so a correction cannot be moved
by running differently.

- **Why two channels, not a refund ratio.** The ratio form forces one
  parameter to carry two physically separate slopes; every unconstrained fit
  pushed it unphysical (>1 — rolling courses reading net-fast) whenever the
  data wanted a small climb cost against a large descent benefit. Independent
  channels also drop the consistent-effort assumption — which is what kept
  hill workouts on a separate Minetti model.
- **Engine boundary — hill workouts stay on Minetti (Aug 2026,
  investigated and rejected).** The channel split made a unification
  *possible* in principle; the empirical check killed it in both directions.
  Hill workouts live at/beyond the calibration's support (loop blocks
  150–267 ft/mi at 5.7–10.1% vs mile-gain p99 ~176 ft/mi, climb-grade p99
  ~8%; hill reps 543–571 ft/mi at 10.3–10.8%, past the 12% clamp), and they
  invert the engine's perturbative design — full-run corrections are 2–8%
  fractions, hc blocks 7–23%, hill reps 54–59%, so the correction *is* the
  measurement and parameter error amplifies 3–10×. On hc the bridge needs
  the descent channel, which principle 4 below forbids transferring across
  effort (workout descents are attacked; b(g) was measured on cautious easy
  descents) — it over-corrects ~1.7× Minetti, breaks cross-loop effort
  coherence, and reads the documented 2021-12-19 anchor at 4:47 vs the
  intuition-pinned 4:55. On reps the climb-only bridge is principled but
  the linear c(g) extrapolates below Minetti's in-domain convex curve at
  >10%, slowing rep 5K-equivalents ~31–33 s/mi into no effort class.
  Full numbers: training-quality-reference.md "Considered and rejected".
- **Climb cost is near grade-invariant** (c1 small, Minetti-consistent). The
  steepness effect is a **descent** phenomenon: b(g) declines hard with grade
  and **crosses zero near 9% — the braking regime — measured inside the data**
  (hill-grade p99 ~9.6% once mixed/trail miles joined the calibration), not
  extrapolated. Beyond `GRADE_DOMAIN_PCT` (12%) grades are clamped.
- **Climb cost is pace-PROPORTIONAL**, not a fixed s/mi. Minetti 2002 derives
  1.026e-3 per ft/mi as the textbook anchor (C′(0)/C(0)/5280); the fitted c(g)
  sits below it with position-matched sustained-climb natural experiments in
  between.
- **One constant set for all terrains and efforts.** Terrain is day-constant,
  so the mile-grain day FE absorbs footing during calibration; terrain
  survives only as the flat `trail_frac` footing fitted separately. The old
  effort schedule stays retired.
- `pace/(1+frac)` applies the fraction to the FLAT-equivalent pace, so a run
  slowed by its own hills cannot inflate its own correction.
- **Magnitude is EXCESS gross vertical**: gross fused gain/loss minus the
  measured flat-ground floor (`floor_g`/`floor_l` ≈ 15 ft/mi — TV noise +
  micro-undulation). The day-demeaned calibration absorbs the floor and never
  priced it; charging it anyway gave every run a phantom ~2 s/mi (Berlin read
  +52 s). Steepness enters as `g_eff = Σ_hills(vert·grade)/E_gain` — the
  statistic under which the run-level closed form exactly equals the
  calibration's window sums (hill-less vertical at grade 0; a hills-only mean
  grade over-charges the interaction on flat feet — it inflated Boston ~2×).
- **Live**: `scripts/calibrate_climb.py` refits c0/c1/b0/b1 + the floor each
  build (`data/elevation_calibration.csv`), with sign / natural-experiment
  validation warnings; committed defaults cover thin corpora.

Calibration spec — **continuous sliding-window regression**: one observation
per 30 m of every eligible run, window length **1.5 mi**. No window placement
exists (fixed mile bins moved coefficients 20-35% under a half-mile boundary
shift); the scale is bounded both ways by the Aug 2026 sweep — ≥ ~1.25 mi so a
window contains a hill plus its smeared pace response (below that the
steepness slopes attenuate into the base rates), ≤ ~2 mi before identification
thins (few windows per run; short runs leave the corpus; NOT gain/loss
collinearity, measured flat at −0.51 across scales). Priced rates at
corpus-typical grades are scale-invariant within ~2% from 0.25-2.0 mi.
Controls: day demeaning (FE); **quintile drift dummies** (flat pace is not
linear through a run: −3.4% at the second decile, +3.0% at the fifth; climbs
sit late in Max's runs and descents early, so under-absorbed drift lands on
the two channels with opposite signs); 4σ MAD prune at window grain only;
recovery + long, paved + mixed + trail.

### Hill segmentation — `src/coros/elevation.py`

A hill is a contiguous same-sign stretch of the grade curve (Aug 2026 rebuild;
every choice below was validated against before/after profile panels and the
mile-grain fit):

- **No elevation smoothing.** The device stream is already filtered (flat
  ground holds a line to 0.74 ft RMS/100 m). The old 80 m rolling mean rounded
  short pitches under the hill floor and deflated grades ~2× (a real 13.3%
  descent read 3.9%) — it inflated the fitted descent slope ~80%.
- **Grade sign from a centred 40 m lag** (`SEG_LAG_M`): the stream is ~1 ft
  quantized, so one quantum over a 10 m step reads 3% (phantom pitch); over
  40 m it reads 0.75%, under threshold.
- **>45° steps zeroed** (barometric resets — a level shift; the totals'
  `SPIKE_GRADE_CAP` clip would leave a 32.8 ft residue that clears the floor
  as a phantom hill).
- **±2% threshold** (`SEG_MIN_GRADE`) — pitched vs flat; this line doubles as
  the steepness selection. **≥12 ft vertical** (`SEG_MIN_VERT_FT`, the
  momentum floor). Same-sign pitches merge across flat gaps ≤60 m
  (`SEG_GAP_M`) with absorbed flat capped at 25% of the hill's extent
  (`SEG_FLAT_FRAC`) — the old iterative gap-closing was unbounded (bridged up
  to 2,920 m of flat, halving real grades).
- **Grade = vertical / PITCHED distance** — absorbed flat never dilutes it.
  Ceiling 100% (45°), not the old 20% clip, which censored ordinary steep
  hills (20% ≈ 11°, treadmill range).

### The fused substrate — `elevation.fuse_altitude` + `dem_elevation.activity_dem_profile`

Everything (segments, grades, gain/loss totals, mile splits) is measured on a
**complementary-filter fusion** of the two altitude sources:

```
drift(d) = rolling_median( baro(d) − DEM(d), 1.5 km )
fused(d) = baro(d) − drift(d)
```

Baro is right at high frequency (structures, pitches; wrong slowly via ambient
pressure) and DEM is right at low frequency (net trend, loop closure; wrong
locally — bare-earth lidar strips bridges/decks/post-lidar construction, and
GPS wander reads the contour beside the path). The 1.5 km median cannot be
moved by a 100-400 m structure, so structures pass through intact while
pressure drift is removed. Proof battery (the first two never consult DEM):
out-and-back self-mismatch 6.6→3.0 ft median; cross-visit SD at repeated spots
12.8→8.9 ft; known structures preserved within ~2 ft of baro (vs ~0 under
DEM-trust); loop |net| 9.8→4.1 ft. **This retires the old two-scale rule**
(DEM totals / baro shape) and the totals rescale: one substrate, and structure
climbs now land in totals too. No-DEM days (Canada, `track_ok` failures, thin
cache) keep pure baro + spike guards.

### The persistence-aware hill veto — backfill `apply_hill_veto`

DEM **refutes, never confirms**. A hill is vetoed only when BOTH:

1. DEM shows near-flat ground under it after ±80 m alignment
   (`dem_elevation.aligned_net` — axis misregistration accounted for 70% of
   naive veto hits): same-direction net < 25% of the claim AND < 12 ft;
2. the disagreement is a **one-off**: veto candidates cluster by location
   (55 m cells, 8-neighbour union-find so GPS drift can't fragment one
   structure into several "locations"); recurrence on ≥2 dates = structure
   bare-earth lidar cannot see (arch bridges, boardwalks, post-lidar
   construction — all confirmed cases) → baro trusted, never vetoed.

Demanding DEM *confirmation* instead was tested and rejected: fit R² fell
monotonically with vertical removed and the coefficients went unphysical — a
large share of "DEM-missing" hills are real. Final scope ~0.4% of hill
vertical: one-off pressure blips (median ~14 ft), including phantom hills
inside races (Run for the Pies 2021, Craft Classic 2024 mile 12.9). The veto
runs over the full `data/elevation_hills.csv` every build, so a structure's
first visit is retroactively un-vetoed when its second visit lands.

### The hills artifact — `data/elevation_hills.csv`

One row per hill: `date, act, d0, d1, vert_ft, grade_pct, kind, lat, lon,
dem_net_ft, vetoed`. The backfill's post-pass rebuilds the per-run
(`elevation_measured.csv`: `seg_up_ft/seg_dn_ft/g_up_pct/g_dn_pct`) and
per-mile (`elevation_splits.csv`: `seg_up_ft/seg_dn_ft/g_up/g_down`) segment
quantities from the surviving hills, so fit and application always see one
statistic. Race rows are measured on the race activity alone and carry
race-scoped values.

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
- Slope (fitted on the pooled corpus, below; live value ≈ +2.0 as of Aug 2026): ≈ **+2 s/mi per 1000 ft above
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
**footing** (`trail_frac` — continuous 0 paved / ½ mixed / 1 trail, so trail
carries ~2× the mixed penalty with one parameter; the flat-surface cost at
full trail share, ≈ +16 s/mi as of Aug 2026, i.e. ≈ +8 on mixed) and the
**altitude slope** (above). One constant per channel applies in the recovery
model, the long-run 5K conversion, and the race correction. (Live values
evolve with each refit — read them from the fit, not this doc.)

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

**DEM policy: 10 m lidar only (Aug 2026).** `ned10m` (the whole US including
Hawaii) is the only DEM in the chain; anywhere it does not reach, the run keeps
its **barometric** profile and there is deliberately no coarser fallback.
Measured on identical GPS tracks (`scratch/rundle_comparison.html`): against
baro, ned10m sits +5 ft/mi on a forested trail while srtm30m sits +40, aster30m
+81, and NRCan CDEM swings +16 on one route and −30 on another. The coarse
sources each fail differently in different terrain, and their fine structure
correlates better with EACH OTHER than with the barometer, so there is no
calibration to borrow. Gross gain is a total-variation statistic, so any
per-sample noise adds to it and never subtracts — SRTM's surcharge measured a
flat +30.4 ft/mi on two Banff routes whose real relief differed 5×. Baro's own
error is a mild under-read plus occasional spikes (capped at 45° per step in
`elevation.py`), which errs toward under-correcting: the conservative direction.

**Misses are cached too (Aug 2026).** `dem_cache.json` stores a `None` for any
cell NED answered for but does not cover, so it is never queried again
(`dem_elevation.MISS_NOTE`). Without this the NED-only policy above had a nasty
side effect: only *hits* were cached, so every build re-asked the public API for
the same ~18k cells under Paris / Calais / Berlin / Seoul / Tokyo / Osaka /
Banff — 5–10 minutes per run at the 1 req/s courtesy limit, fetching nothing
(the pipeline log's `+0 fetched`). Foreign runs still keep their barometric
profile exactly as designed; only the re-asking stopped. A cell is
negative-cached **only on a successful response** — a network failure returns no
list at all, so one flaky minute can't blacklist a route. `DEM.point_count()` is
the honest "real points" count for logging, since `len(cache)` now counts misses.

**Races use a DEM, not the barometric stream.** The watch's per-race *net* is
noise (the same Bolder Boulder course read −19 ft/mi net one year and +5 the
next) while its horizontal GPS track is reliable and reproducible. So
`src/coros/dem_elevation.py` resamples elevation from a DEM (OpenTopoData USGS
NED 10 m only — see the DEM policy above; point cache
`data/dem_cache.json`) along the
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
