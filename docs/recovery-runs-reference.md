# Recovery-runs analysis — reference

This doc captures the design and decisions behind `make_recovery_plots.py`
so future work can pick up without re-deriving choices. Read this before
reconsidering the model or extending it.

## Purpose

Surface real signal from daily recovery runs by stripping out confounds
that would otherwise obscure cross-sectional and longitudinal patterns.
Recovery pace is the noisiest single fitness signal in the log (route
choice, weather, fatigue from quality work, etc. all push it around) but
it's also the densest — ~250 points per training year vs. a handful of
races. Done right, normalized recovery pace is a high-resolution view of
day-to-day state that races can't provide.

The output is an interactive 2-panel HTML chart with checkbox-driven
normalization toggles. Left panel shows absolute pace vs the CS gold
curve; right panel shows residual (pace − CS) vs flat zero.

## Files

- `make_recovery_plots.py` — single script. Reads `daily.csv`, `races.csv`,
  `bayes_cs_summary_{tag}.csv`. Writes `recovery_pace.html`. Default
  invocation: `python make_recovery_plots.py --tag v11`. Runtime: seconds.
  The model itself (the physical route channels + the fitted confounds) lives
  in `recovery_model.py`; `physical_route_betas()` there is the single source
  of truth for the footing/altitude constants shared with the long-run and
  race conversions (see `route-normalization-reference.md`).

Location metadata (display_name, city_state, elev_per_mile, altitude,
terrain_type) is read directly from columns on `daily.csv` — populated
during the build by `build_dataset._join_location_metadata` from the
locations sheet. No snapshot dependency in the plotter.

## Decisions locked in (April 2026)

### Era trend is special

Era trend is a centered ±182-day rolling mean of recovery residual,
computed against the non-pruned data pool. It's the model's longitudinal
component — what was your typical residual back then?

The era-trend toggle applies **only to the residual panel**. The absolute
pace panel always shows actual recorded pace minus only the cross-sectional
factor adjustments. This keeps the CS gold curve as a stable reference and
lets the user ask "how does my pace compare to CS over time, controlling
for known confounds?" without the chart's baseline shifting.

When toggled on, the residual subtracts `(era_trend − global_mean_residual)`
rather than `era_trend` itself. This centers points around the global mean
(~+44 sec/mi) instead of collapsing them to zero, preserving the visual
"how slow is recovery vs CS on average" reference while removing
period-to-period drift.

±182d was chosen to control for seasonal variation cleanly. **Don't change
it year-to-year**, even when sub-period fluctuations leak through (1-3
month cycle effects do — see "Findings" below). Era trend is doing its
job to within its window.

### Cross-sectional features (OLS on era-detrended residual)

```
residual_detrended ~ β_temp     · temp_centered
                   + β_foot     · is_offroad          (pinned, physical)
                   + β_alt      · altitude_regressor  (pinned, physical)
                   +              elev_cost            (pinned, grade-aware)
                   + β_marathon · fat_marathon        (exp(−(t−1)/2), day-1 anchored)
                   + β_race     · fat_race_short      (exp(−(t−1)/1))
                   + β_long     · fat_long            ((mi/20)³ · exp(−(t−1)/1))
                   + β_tod      · tod_is_pm
```

The route handling is a **physical, era-free model** (June 2026 — replaced
the per-route dummies; see the "Physical route model" subsection below). The
temp / fatigue / TOD terms are the fast-varying, era-immune confounds fitted
on the era-detrended residual; the three physical channels (footing,
altitude, grade) are pinned from `recovery_model.physical_route_betas()` and
consumed via `fit_recovery_model(pin_physical=True)`.

- **Temperature**: one-sided heat hinge `max(0, T − 6 °C)`, slope pinned from
  the pooled fit (≈ +0.34 s/mi per °C, Sept 2026). Real but small; most of the
  seasonal pace pattern is summer-training-intensity confound that the era
  trend absorbs.

- **Route handling — physical, era-free model** (June 2026): the per-route
  dummies were **removed** and replaced by three physical channels that
  re-explain the route variance without any per-route knob. See the
  "Physical route model" subsection below for the full decomposition and why
  the fit is a backfit. The three channels are footing (`is_offroad`),
  altitude, and a grade-aware `elev_cost`; the first two are pinned from
  `recovery_model.physical_route_betas()` (the single source of truth shared
  with the long-run and race conversions), the third is imported from
  per-mile elevation data.

- **Recent effort** (Sept 2026; was "Recent race"): three terms, each
  **anchored at day 1** — contribution `β · dose · exp(−(t − 1)/τ)` on day
  t ≥ 1 — so every β reads as the next-day effect. Fades are pinned
  (`FATIGUE_TAU_DAYS = {marathon: 2, race_short: 1, long: 1}`); sizes are fit
  live by the pooled estimator (`physical_route_betas`). Evidence came from
  **clean-chain** event studies: only recovery runs with no other hard session
  between the event and the run, against a σ = 28 d trend fit on ordinary days
  (previous day recovery/rest, no race in 7 d, no marathon in 21 d).
  - **Marathon, 2-day fade.** Per marathon (each counted once): day 1
    +22.3 ± 4.3 (14), day 2 +15.9 ± 3.4 (16), day 3 +6.7 ± 2.5 (17), days
    3–8 +4.5 ± 1.8. Best fade 2 d (ΔAIC: 1 d +11.8, 6 d +13.4); a cluster
    bootstrap over marathons picked 1.5–4 d every time, never 1 or 6. Not a
    baseline artifact (dropping post-marathon weeks and taper: days 3–8
    +4.6 ± 1.5). Marathons are categorically a different effort: 24-mile
    training runs run +11.1 ± 3.1 on day 1 and +3.3 ± 2.0 on day 2, about what
    the long-run term predicts, while the long-run term at 26.2 mi predicts
    only +17.3 / +6.4.
  - **Short race, 1-day fade.** Every race below marathon distance (track,
    road, XC, half marathon, time trials) in one term. Pooled day 1
    +8.9 ± 1.5 (106 races), day 2 +3.0 ± 1.6. Track vs road/XC not
    distinguishable (day 1 Welch p = 0.18, rank p = 0.09; day 2 p = 0.15 /
    0.37), and a split didn't beat one term on AIC. The old 5-day tail was
    never fit and had absorbed runs with intervening sessions; the day-1 size
    (~+9) was already right. Smoothed over (within noise): mile-and-under
    race days +14.6 ± 3.3, 5Ks +5.4 ± 2.5, two-race days +14.0 ± 2.6.
  - **Long run, 1-day fade, dosed `(miles / 20)³`.** The most recent
    `run_type == 'long'` of any kind (no slice/partner/snow gate), at its
    canonical distance — watch-corrected where available
    (`workouts.long_run_miles`). Clean-chain day 1 by length: ≤12 mi +0.3,
    12–14 +1.8, 14–16 +1.7, 16–18 +3.9, 18–20 +9.6, 20–22 +8.8, 22–24 +15.4.
    Smooth by design (Max: a 14.9 vs 15.0 mi cliff is unacceptable): the cube
    fit within noise of the best hinge (ΔAIC 0.9) with no knot, duration was
    ~6.6 AIC worse, linear 12.5 worse, a flat per-run term 26.8 worse.
  - **Identification guard:** a term enters the pooled fit only with at least
    `FATIGUE_MIN_EXPOSED = 15` fit rows at ≥ `FATIGUE_EXPOSED_LEVEL = 0.2`,
    else its β is 0. As of Sept 2026 maddy holds all three at 0 (no marathons,
    one long run, 13 exposed short-race runs); the short-race term returns
    once the corpus reaches 15.
  - **Snapshot (max, Sept 2026):** marathon +23.4, short race +9.9, long run
    +7.9 at 20 mi. Adopting the three terms moved the pooled physical betas:
    footing 15.66 → 17.36 (days 2–5 after short races were 59% off-road vs 19%
    otherwise, so the retired race tail had been soaking up footing) and
    altitude 2.01 → 1.79 (27% of days after 18+ mi long runs were above
    3000 ft vs 19%); detrended R² 0.278 → 0.299.

- **Time of day** (`tod_is_pm`): binary indicator, 1 if afternoon/late, 0
  if early/morning. β ≈ −7.4 sec/mi (afternoon runs are faster; Sept 2026).
  100% of recovery rows have a TOD value. In April 2026 adding it collapsed
  the old 14-day linear `fat_long`; that collapse was the feature's shape, not
  a real absence (see Recent effort).

### Physical route model (June 2026 — replaced per-route dummies)

The watch-stream enrichment work retired the per-route dummies in favour of
a **physical, era-free** route model. The route's pace cost is no longer a hand-set
`elev_per_mile × constant` or a fitted per-location offset — it decomposes
into three physical channels that apply identically across eras (off-road
routes cluster early/sea-level, paved/altitude in the Boulder era, so the
dummies were partly encoding the era trend itself).

**Fit via a backfit.** The era trend is a temporal smoother, and footing /
altitude vary on the same era timescale, so a one-pass era-detrend *absorbs*
them (footing collapsed to 0). The model is fit by **iterating the era
smoother against the parametric factors** (era ↔ factors) until they
stabilize, which isolates the physical terms from the overlap years. The
fast-varying confounds (temp / fatigue / TOD) are era-immune and fitted
freely; the physical channels are pinned.

**Channels:**

- **Off-road footing** (`is_offroad`, the mixed+trail binary — every training
  RUN buckets by its LOCATION `terrain_type`; races bucket by SURFACE, see
  route-normalization-reference.md): the flat-surface penalty for running on
  non-paved ground (trail_frac coding: mixed = half the full-trail value;
  live ≈ +16 at full trail as of Aug 2026). Pinned, not fitted in this model.
- **Altitude** (hypoxia): ≈ **+2 sec/mi per 1000 ft above a ~3000 ft
  threshold** (VO2max is ~flat below the threshold, then declines roughly
  linearly — the shape is science-pinned, only the slope is data-fit).
  Below ~3000 ft the term is exactly 0. Pinned, not fitted here. (Threshold-
  curve derivation lives with the engine — → see `route-normalization-reference.md`.)
- **Grade-aware `elev_cost`** (pinned): scales with each route's *per-run
  measured* hill segments, so a hilly-mixed route costs far more than a flat
  one. Constants come from the per-mile calibration (where climb and descent
  separate), not chosen — the two verticals are collinear within loops at
  run-level. The two-channel `elevation_cost` formula, the hill segmentation,
  the fused baro+DEM substrate, and the altitude threshold-curve derivation
  are not duplicated here → see `route-normalization-reference.md`
  (elevation engine).

**`physical_route_betas()` is the single source of truth** for the footing
and altitude constants. It pools recovery + in-slice long runs on a shared
`pace − cs_pace` scale (with an `is_long` level dummy and the recovery
era-backfit), so one constant per channel applies in both the recovery model
and the long-run / race conversions. `fit_recovery_model(pin_physical=True)`
consumes those pinned betas. (Recovery-only would give footing +4.09 /
altitude +0.87; both shift <1 se when the long runs join.)

**Per-run features (the physical channels read these, not per-location
constants):**

- `per_run_elevation` — gain/loss in ft/mi, with the fallback chain
  per-run watch-measured → route-median → the route's `elev_per_mile`
  constant → 0.
- `per_run_altitude` — the midpoint of the watch's smoothed daily min/max
  (from `altitude_daily.csv`, the layer feeding the Altitude trend), with the
  location base-elevation constant as the pre-watch fallback, then 0. This is
  per-run MEASURED, not a per-city estimate, so it adds within-location
  resolution and fixes hand-set constant errors.

**Elevation source — barometric, not DEM.** Recovery runs use the watch's
barometric elevation directly: across a large recovery corpus the per-run
barometric noise averages out. Only RACES throw away the barometric vertical
and resample from a DEM along the GPS track (the per-race net is too noisy to
trust on a single course) — see the cache/route references for that path.

**Fit quality:** R²_detrended ≈ 0.298, raw ≈ 0.622 — essentially unchanged
from the old per-route-dummy model. The physical terms re-explain the route
variance era-free, so nothing was lost by dropping the dummies.

### Pruning: three classes, can overlap

Three independent flags. A row can be in any combination of them. All
flags exclude the row from the OLS fit but the points remain plotted.

1. **Bad conditions** (`is_bad_cond`): `conditions_clean ∈ {snow, icy}`
   OR `workout_raw` matches the regex `\bsnow\b` (catches `[2" snow]`
   bracket annotations the conditions field missed). 2017's untagged
   snow days come in via the workout regex. **`inside` was removed from
   the bad-cond set** in April 2026 — those are treadmill/indoor-track
   runs at stable surface and pace, valid data. That holds for
   HAND-LOGGED indoor days only: **watch-derived indoor runs are dropped
   from the corpus entirely** (Aug 2026, `is_watch_indoor`) before any
   flag or axis is computed, because on a watch-import profile the
   treadmill distance is the watch's uncalibrated stride estimate rather
   than a number Max chose. maddy's 2026-07-05 (Coros sportType 101,
   0.92 mi / 10.9 min = 11:50/mi) sat 114 s/mi outside a 447–596 s/mi
   corpus and, with the axis taken from `quantile(0.999)` over ~100 rows,
   stretched the whole plot from 6:30–10:00 to 6:00–12:00 by itself.
   Dropped, not flagged — a flagged row is still plotted and would still
   set the axis. The test is `weather == 'indoors'` **and** a
   `snapshot:current_log` source: both halves matter, because
   `build_current_log` writes exactly `indoors` for a watch indoor run
   but Max's 2017-07-19/20 *hand-log* rows read `indoors` too (his other
   15 use `inside`). Mileage is untouched either way.

2. **Partner runs** (`is_partner_run`): any `partners` entry outside
   `ADMITTED_PARTNERS` = blank/solo/none/**varsity**. Varsity was
   admitted June 2026 (Max): in the 2016-17 era the varsity group's
   recovery pace WAS Max's own pace strategy — those 90 runs are his
   effort policy, not someone else's — so they belong in the fit pool
   (partner-pruned 169 → 79; R² on raw residual 0.54 → 0.68 because the
   2016-17 era trend now tracks the era's true level; the recovery
   marathon/short fatigue ratio that pins the TQ long-run model moved
   1.82 → ~1.5). Individual named partners remain pruned — a
   fundamentally different population (pace targets and route choices
   aren't Max's own).

3. **Outliers** (`is_outlier_loo`): `|residual from leave-one-out 28-day
   local mean| > 45 sec/mi`. The local mean is computed against the
   *clean* neighbor pool (rows that are neither bad-cond nor partner-run),
   so the baseline reflects typical solo recovery state. The LOO residual
   is computed for **every** recovery row, so a bad-cond or partner-run
   day can also receive the outlier flag — the three classes overlap.
   ±45 was chosen as conservative denoising (~3σ on the LOO residual
   distribution); catches truly anomalous days (travel/jet-lag, illness,
   extreme post-marathon fatigue) without biting into normal variance.

`is_pruned = is_bad_cond | is_partner_run | is_outlier_loo`.

### Watch / route-rule distance corrections (June 2026)

The fit runs on the **corrected** pace where one exists
(`pace_for_fit`); the logged columns are never rewritten and the plot
still displays logged values. Machinery in
`recovery_model.add_watch_corrections`, mirroring the long-run
treatment (`workouts._lr_watch_corrections`); measurement artifact
`recovery_measured.csv` written by `src/coros/long_runs.py` alongside
the long-run one; the **calibration curve is shared**
(`long_run_calibration.csv` — it was already fit on the pooled paved
recovery+long corpus).

- **GPS track gate** (`gps_ok`, Aug 2026): a day whose watch recorded it
  fully in TIME but not in SPACE keeps its logged distance and pace, and
  leaves the calibration corpus. Two axes, both borrowed from the
  elevation layer rather than re-derived: a first GPS fix later than
  `dem_elevation.LAG_CEIL_S = 30 s`, or GPS coverage below
  `COVERAGE_FLOOR = 0.80`. The lag arm is the one that bites (142 of Max's
  recovery days): the watch dead-reckons the un-fixed opening stretch, so
  moving time stays whole and the shortfall hides inside `WATCH_FAIL_DEV`
  below. Validated against **loop closure** — on clean-lock activities the
  median start-to-end gap is 22 m and 86% close within 100 m (per route:
  centennial 99.6%, east boulder 99.2%, mccabe 99.0%, lakefront 98.0%),
  while closure collapses to 43% in the 20–30 s lag band, 22% at 30–45 s
  and 0–4% beyond 45 s, with the median gap tracking `pace × lag` at a
  ratio of 0.56 (the chord of the missing arc). The gate ships on **lag,
  not on the gap**: ~14% of the corpus is genuine point-to-point running
  that a gap test would wrongly reject, whereas only 2 of 108 lag>30 s
  days close the loop. Worked case — 2026-08-24, `river trail`, logged
  10.0 mi: 518 s to first fix, 1782 m open loop, watch reads 9.289 mi
  (1.3 mi untracked), and the old path shaved it to 9.87 mi at +8.1 s/mi.
  Coverage alone would have missed it (0.88 > the 0.80 floor); it is
  centred at ~1.02 because GPS jitter inflates the point-to-point sum, so
  it masks a missing arc.
  **Late-lock recovery (Sept 2026, long AND recovery runs).** The lag is
  wall-clock, but standing paused costs no distance, and the watch
  dead-reckons the stretch it runs without a fix. So the lag is re-measured
  in MOVING seconds before the fix (`gps_prefix_moving_s`). A late-lock day
  is admitted on its own watch distance, with nothing added, when both hold:
  - every late activity ran at most `PREFIX_MOVING_CEIL_S = 60 s` before its
    fix;
  - its loop closes, meaning the start-to-finish gap exceeds the unfixed arc
    by no more than `LOOP_CLOSE_TOL_M = 100 m` (`gps_loop_excess_m`).

  The verdict is `gps_recovered`. `gps_ok` still gates the calibration fit,
  so the curve stays clean-lock only.

  Validation used route peers: each lag-rejected day's watch distance was
  compared with the median of clean-lock days on the same route at the same
  logged distance (50 days had ≥ 5 peers; peer IQR ~140–190 m).
  | run before the fix | n | watch vs peers |
  |---|---|---|
  | 30–45 s | 27 | median 30 m long (matches) |
  | 45–60 s | 5 | ~100 m short |
  | 60–120 s | 8 | 73 m short |
  | > 2 min | 10 | 285 m short |
  Extrapolating the opening instead (moving s × early pace) over-added 50–70
  m on short openings, so it was dropped.

  The loop check still catches real problems, e.g. 2026-03-14 mccabe with
  925 m unexplained, and point-to-point runs like kona. Recovery-run
  late-lock days differ from the long-run case. Pauses are a median 9% of
  the lag (vs over half on the seven Aug–Nov 2025 South Lakefront long runs,
  which ran 34–54 s before the fix), and 29 of 107 ran over 2 min before it.
  Those stay rejected. Admitted: all 7 late-lock long runs, and 61 of the 143
  gps-rejected recovery days.
- **Watch correction** (paved + time-complete days): corrected distance
  = `watch_mi·(1+slope) + intercept`, corrected time = watch moving
  seconds, corrected pace = their ratio. ~1,190 of ~2,550 recovery days
  qualify; 788 actually move (median +7.4 s/mi, p99 +28).
- **Paved gate**: trail/mixed/un-typed terrain keeps logged values —
  same rationale as the long-run gate (the curve is a paved fit; trail
  GPS corner-cutting is a route property).
- **Watch-failure guard** (`WATCH_FAIL_DEV = 0.06`): the time gate
  can't catch GPS that loses *distance* with intact time (sporadic
  lakefront days read 20-30% short → implied "corrections" of +50-550
  s/mi). Disambiguation from genuine route mislogging is route
  structure: real mislogs are systematic (the route's median inflation
  moves), GPS failures are one-day spikes against an honest route
  median. A day is corrected only when its logged/calibrated inflation
  sits within 0.06 of its route median (95th pct of the deviation
  distribution is 0.0585; 57 days skipped; the ±45 LOO prune backstops
  the rest).
- **Trailing strides/sprints** (`STRIDE_SUFFIX_RX` — `…/6x100st`,
  `…/400st`, `…/6sp`): **Max pauses the watch for the strides**, so the
  watch records the recovery portion only — it's *short* of the logged
  run, not long. Verified June 2026 against the 22 watch-covered paved
  stride days vs 1,227 normal: watch moving time runs −1.3% of logged
  (vs +1.0% normal), recorded pause time is 4.5 min (vs 1.75), and
  logged miles sit **+0.32 mi above** the calibrated watch distance (vs
  −0.01 normal) — that 0.32 mi is the stride distance the paused watch
  never recorded. Two consequences: (a) **excluded from the
  calibration-fit corpus** — the +0.32 mi excess would bias the slope/
  intercept up and over-correct every day; (b) **no watch correction
  applied** — `recovery_pace_sec_per_mi` is the explicit logged
  `rec@M:SS` (`extract_recovery_pace`), already the clean recovery-segment
  pace, and a recovery-only watch measurement run through the
  whole-run-fit calibration would only reproduce it, so the logged pace
  is kept. A strides day on a mislogged ROUTE still takes the route-rule
  pace ×factor — that's a distance-estimate error biasing the logged
  @pace itself, independent of strides — but carries no corrected
  distance (corr_miles NaN).
- **Route-era rules**: `MISLOGGED_ROUTES` (belle meade & greenway both
  **1.05** flat, 2018 → 2022-04-15, pinned 2026-06-17 — watch-confirmed
  ~5% real over-logging once the watch GPS-undercount artifact is removed;
  the constant lives here, not in workouts.py) also deflates those routes'
  ~9 recovery rows. No OTHER recovery
  route warrants a rule: every watch-covered route's median inflation
  sits within 0.97–1.011 (the Nashville long-route inflation was a
  property of those route-distance estimates, not of Max's logging).
- **Distance bracket** (same as long runs): true distance is bracketed
  `watch_mi <= true <= hand-logged`. The watch under-reads, so
  `(watch + calibration error)` is the estimate; the hand log is a hard
  ceiling Max never under-estimates, so `corr_mi = min(estimate, logged)`.
  **Time is the anchor** — it IS the watch time (the hand log just rounds
  it to the minute, ~30 s), so `corr_time` stays the watch time and
  `corr_pace = corr_time / corr_mi` is a pure derivative, never clamped on
  its own. On routes Max logs tightly — centennial 0.991/0.986 (pre/post
  2022-04-15), mccabe 0.985, boulder creek 0.970, hopewell junction 0.970
  logged/honest — the estimate overshoots the log, so they clamp to the
  logged distance (effectively uncorrected, never longer). This replaced
  the old pace-floor "overread clamp" (June 2026), which governed pace (a
  derivative) instead of distance: it let `corr_mi` exceed the log on some
  days and fabricated a phantom distance (`moving_s / logged_pace`) on the
  days it bound — visible only as a distance/time incoherence the
  pace-axis plot couldn't show.

### Features tested and rejected

These were tested empirically and produced near-zero, sub-noise, or
period-confounded coefficients. **Do not re-add without strong new
evidence.** Each entry below documents the test and rationale.

- **Sleep cycles** — no detectable next-day pace effect. β collapsed to
  near-zero in multivariate OLS. Re-tested in April 2026 with TOD added,
  still null.

- **Recovery distance** — daily mileage doesn't predict daily pace
  independently of other factors. β ≈ 0. **Re-tested in depth Sept 2026 with a
  rolling local leave-one-out estimator** — the null holds on the full corpus
  (−0.028 ± 0.154) but becomes a small positive (+0.429 ± 0.184) on
  watch-corrected rows alone, and the `maddy` profile carries a large real effect
  (+5.10 ± 1.25). See "Run length, chronic load and pauses" under Findings before
  re-opening this.

- **Hill workouts** (continuous + reps) — no term. Sept 2026 re-check: the
  overall null hides a timing split — an afternoon session followed by a
  next-morning run is +8.3 ± 1.5 (50), morning → next morning −5.2 ± 1.1 (49).
  The effect is turnaround, not the session; not modeled.

- **Tempo / interval / fartlek workout fatigue** — still null (Sept 2026
  re-check against an away-from-workout baseline: day 1 +0.7 ± 1.0, day 2
  +1.3 ± 1.2, same in both eras; only 6% of workouts are followed by a rest
  day, so selection doesn't hide it). The one real pattern is turnaround:
  across all hard sessions, afternoon/evening → next-morning runs are
  +9.8 ± 1.0 vs +2.3 ± 0.7 morning → next morning (track intervals +6.4 vs
  −3.8; the 500/300 track fartlek +12.6 ± 1.3). On ordinary days a
  previous-day afternoon run costs only ~2.8 either way. Not modeled; a
  turnaround term would have to be fit jointly with the short-race term
  (most track races are evening → next morning, yet races stay slow
  evening → next evening, +10.2 ± 1.9).

- **Heavy rain weather** — tested April 2026. n=36 days, mean
  fully-normalized residual −5.3, p=0.002 vs clear weather. Real but
  small mean shift in the *fast* direction (running harder when miserable
  out, possibly cooler/firmer routes), with **no extra variance**
  (std 12.6 vs 12.5 reference). Pruning would remove valid data with a
  known bias rather than excess noise. Not added as a feature because
  the effect is already swept up by the existing outlier prune for any
  truly anomalous days, and adding a binary feature for a small
  structural shift wasn't worth the knob.

- **Wind — ADOPTED June 2026.** The April 2026 qualitative-bin test
  (n=23, Δmean +0.27 vs clear, p=0.93) failed on power, not effect: hand
  logging only captured ~15% of runs. With continuous watch `wind_mph`
  now on ~60% of recovery days (n=1446 unpruned), wind comes in at
  **+0.29 s/mi per mph (t=2.6, p=0.009)** — ~+3.3 s/mi at 15 mph. It's
  near-orthogonal to temp (r=−0.04) so it leaves the temp beta untouched,
  and it pulls ~0.3 s/mi out of the PM (tod) effect (afternoons run
  windier). Wired as a **pooled, pinned per-mph cost** (`wind_beta`),
  estimated on the watch subset but applied as a fixed offset only where a
  watch reading exists — so the main fit keeps the full corpus instead of
  collapsing to watch-era rows. Calm (0 mph) is the zero-contribution
  baseline. Degrades to 0 (no-op) without watch data.

- **Temperature is a ONE-SIDED HEAT HINGE — `temp_centered = max(0, air_temp − 6°C)`
  (SUPERSEDES the symmetric/heat-index form, June 2026).** The earlier
  apparent-temp-centered-at-12 term was *bidirectional*: every sub-12°C day was
  credited a phantom cold speedup mirroring the heat penalty (it predicted
  ~−8 s/mi at −19°C where the data is a flat ~−2). The true recovery shape,
  pinned from 2356 well-normalized runs, is a flat cold plateau and a monotonic
  heat rise from ~6°C, fit only above the onset (β ≈ +0.31 s/mi per °C above 6,
  the slope unchanged by the form). The contribution is re-referenced to the
  **median clean-day hinge** so normalizing temperature moves hot days faster
  and cool days slower around a typical day (like wind), not the whole cloud one
  direction. **Humidity / the heat index were DROPPED:** a separate humidity
  term is weak (~+0.04 s/mi/%, t=2.6) and the heat index — humidity's physical
  encoding — never beat plain air temp in the corrected comparison, so this is
  air temperature, not feels-like. Long runs reuse the same hinge SHAPE with a
  freely-fit (steeper) slope in the long-run model's temperature covariate.
  (The long-run pause handling is a pause-uncertainty *erosion* with **no heat
  term** — heat lives only in this temperature covariate; see
  `long-run-pause-uncertainty-reference.md`.)

- **Shoe age** (cumulative miles in this physical pair before the run) —
  tested April 2026. n=1671 runs across 18 pairs with ≥30 runs each,
  span up to 1200 mi within a pair. **Global Pearson r=+0.004, p=0.88;
  median within-pair slope +0.10 sec/mi per 100 mi; signs split 9 vs 8
  across pairs.** No detectable wear effect. Plausible explanation: Max
  retires shoes at 700-1000 mi before they noticeably degrade.

- **Volume-based normalizers (any form)** — tested April 2026 as a
  candidate for flattening the post-2022 biannual cycle in the
  trendline. 11 features tried (raw 56DMA, era-normalized vol_norm,
  rate-of-change at 14/28d, lagged versions at 30/60/89/120d). Best
  single feature dvdt_14: r=+0.118, R²=0.014. Best multi-feature combo
  (all 11): R²=0.036. Oracle ceiling (sin+cos@178d): R²=0.027. The
  per-day variance is ~97% noise so any cyclic predictor is capped at
  R² ≈ 0.03. See "Biannual bumps in trendline" finding for the full
  analysis. **Don't re-test until 2-3 more cycles of clean post-2022
  data accumulate** (i.e., 2027-2028).
  (Identifying "physical pairs" required splitting same-label shoes on
  inactivity gaps ≥180 days, since some shoe labels were reused across
  multiple pairs in different years without disambiguation.)

- **Per-shoe-pair offsets** — tested April 2026. Spread across 18 pairs
  with n ≥ 30 is 15.7 sec/mi (ANOVA F=7.22, p<0.0001), so between-pair
  variance is statistically real. **But same-model pair-to-pair
  consistency is poor** (Hoka Clifton 8 #1 vs #2 differ by 3.84;
  Endorphin Speed 2 #1 vs #2 by 2.14; Endorphin Speed 4 #1 vs #2 by
  1.49). Consecutive Brooks Ghost 11 → Ghost 12 (literally adjacent
  3-month windows in 2020) differ by ~8 sec/mi across the COVID
  lockdown boundary. Conclusion: **the apparent shoe-pair signal is
  almost entirely sub-era-trend period drift** (fitness rebuilds,
  training-cycle phases, geographic transitions). Not shoe-intrinsic.
  Adding shoe-pair as a feature would double-count period effects under
  a shoe label.

- **Training-quality residual** (the smoothed track from
  `plot_training_quality.py`) — tested as a feature in V6. Univariate
  correlation with normalized recovery pace was meaningful (+0.33
  overall, +0.46 in 2016-17 specifically), but multivariate β collapsed
  to +0.08 once era trend was in the model. The two signals are highly
  collinear — both are smoothed long-term tracks of fitness vs CS,
  and era trend (data-driven from recovery itself) wins. R² didn't
  move when TQ was added. Removed in V7.

- **Bodyweight as a feature** — tested April 2026 after the realization
  that weight was the one logged metric never analyzed. Distance-adjusted
  using the recovery-only within-month miles slope α = −0.267 lbs/mi
  (acute dehydration channel; the all-runs slope of −0.203 is biased
  shallow by saturation on long runs and races). Imputed for missing
  days: linear interpolation in time within the observed range
  (2022-01-01 to 2026-04-27, 76% raw coverage) and flat fill at the 2022
  mean (159.79 lbs distance-adj) for the pre-2022 1008-row prefix.

  **Joint OLS on full pool: β_weight = −0.006**, t ≈ 0, ΔR² ≈ 0. After
  era detrending the residual has r = +0.006 with weight_da_c — no
  remaining signal to fit. A subset-only fit on the 916 weighed days
  gave β = −0.358 with p = 0.04, but that's a sample artifact: era_trend
  is computed on the full pool and its relative contribution shifts when
  the OLS sample is restricted, allowing weight to absorb variance the
  full-data fit attributes elsewhere. Not generalizable.

  **Reason it can't separate from era_trend**: 78% of post-2022 weight
  variation is on >1yr timescales (long-term decline 159.8 → 153.6 lbs
  across 2022-2026), exactly the band era_trend's ±182d centered window
  captures. Only ~1.7 lbs of short-term SD remains, and it doesn't
  predict short-term pace residuals.

  **Pre-imposition test** (subtract β_weight × weight_da_c from
  residual_raw *before* computing era_trend, then measure how much
  era_trend's variance in 2022+ shrinks): peak reduction at β = −0.30
  is 5% on range and 2% on SD. The 2022+ era_trend trajectory is
  non-monotonic year-by-year (≈+37, +44, +39, +43, +45 across
  2022-2026) while weight declined nearly monotonically; they don't
  share a trajectory, so imposing one cannot meaningfully reduce the
  other's load. 2024-to-2025 is the killer: weight kept dropping
  ~1.5 lbs but era_trend went *slower* by ~4 sec/mi.

  Weight loss is real (~6 lbs over 4 years, confirmed via hydration-
  adjusted yearly means) and physiologically should affect pace, but
  in this analysis structure the bodyweight effect is inseparable
  from generic "fitness era" drift. **Do not re-add as a feature
  without restructuring how era_trend works** (e.g., shorter window,
  or removing era_trend entirely — both would change the chart's
  longitudinal abstraction).

  Side note from the same investigation: weight is **always** measured
  post-run (zero rest-day weighings out of 1,202; median 7.1 mi). PM
  weighings run −0.36 lbs vs AM after controlling for miles+temp,
  likely reflecting cumulative day-long dehydration. These observations
  are why distance-adjustment is mandatory before using weight in any
  model — the raw weight signal is dominated by acute hydration loss,
  not bodyweight.

- **Long-run fatigue** (`fat_long`) — **ADOPTED Sept 2026** (see Recent
  effort). The April 2026 rejection was a feature-shape artifact: a 14-day
  linear decay on weekly long runs reads 0.93 on day 1 and 0.86 on day 2
  while the real effect goes +4.3 → −0.3, so the regressor barely varied and
  TOD absorbed it. The clean effect survives inside each time-of-day stratum
  (AM +3.6 ± 0.9, PM +5.2 ± 1.4), on every weekday, and scales with length.

- **Super-compensation period** (days 14-50 post-marathon, where pace is
  ~2-3 sec/mi *faster* than baseline — see findings below) — empirically
  real but signal magnitude is small and sample density per individual
  marathon is sparse. Adding a separate "super-comp" feature would add
  a knob for ~5 sec/mi peak-to-trough swing across only ~5 weeks per
  marathon cycle. Not worth the complexity; left as visible structure
  in the trendline.

### Visibility section — three independent toggles

Three checkboxes below the normalization section, each with its own
count. A shared All/None pair flips just the visibility toggles
(`data-mode="filter"`, separate from the normalize section's
`data-mode="norm"`).

1. **Hide bad conditions** — sets opacity 0 and y=null for `is_bad_cond`
   rows.
2. **Hide non-solo** — same for `is_partner_run`.
3. **Hide outliers** — same for `is_outlier_loo`.

Implementation detail: hidden points get y=null on the trace
(not just opacity=0), which suppresses **both** rendering and Plotly's
hover hit-testing. Pure opacity=0 leaves the points hoverable
underneath. Stored per-point in residual trace `meta`; JS `update()`
computes `visibleMask` from the OR of (toggle ∧ flag) across the three
classes, passes it into `rollingTrend()`.

Because flags overlap, the three counts shown next to each checkbox can
sum to more than the unique pruned total. The stats line reads
`n=X in fit; Y excluded (classes overlap)` to make this explicit.

### Trend line — Gaussian σ=28d, daily step

Switched in April 2026 from uniform-window rolling mean (±28d, weekly
step) to **Gaussian kernel σ=28d truncated at ±4σ, daily step**.
Reasoning:
- Uniform window with weekly step had visible stair-stepping at each
  weekly tick.
- Switching to daily step alone helped, but uniform window's hard cutoff
  caused spike artifacts when individual outlier-ish points entered/exited.
- Gaussian smooth with same effective span (±28d ≈ 95% of mass within ±56d
  for σ=28) gives continuous weight decay, much smoother visually with
  similar responsiveness to multi-week features.

Earlier short-lived intermediate (σ=14) was actually *more* responsive
than the old uniform window. σ=28 was selected after observation that
σ=14 left visible 2-3 week excursions; doubling smooths these out while
still capturing genuine multi-month structure.

### Tooltip — smart spikeline scaffold

Recovery uses the shared cursor-tooltip scaffold (`_scaffold/cursor_tooltip.js`, opt-in via `cursor_tooltip=CursorTooltip(...)` on `render_plot()`); the tooltip and spike are rendered by `.rp-tooltip` and `.rp-spike` from `_scaffold/base.css`. Plotly's native hover label is suppressed via `extra_head_css='.hovertext { display: none !important; }'` because recovery has `hoverlabel=` on its scatter trace that would otherwise double up. (The rule isn't global in `base.css` because `make_world_map.py` relies on Plotly's native `hovertemplate`.)

The scaffold runs in two modes:

- **Smooth** — cursor isn't near any data point. Spikeline tracks the cursor's date; tooltip is built from the smooth-mode `buildTooltip(day)` defined in `smooth_build_js`. The "Nearest run [±Nd]" section binary-searches the per-run `sessions` payload and surfaces a run within ±60 days (`nearest_window_days`).
- **Snap** — cursor is within `snap_px` (default 30px) of a marker on a `meta.snap_eligible=True` trace. Spikeline jumps to the marker's x; tooltip uses `customdata[i]` (the per-point HTML pre-rendered in Python) for its run-detail section, but the date header + CS-pace/Trend-pace + residual rows are still drawn by `buildTooltip`.

Per-run HTML (used by both modes) is built once at fig construction by `build_hover(row)`:
- Header: `YYYY-MM-DD (DOW)` — DOW abbreviation matches the TQ tooltip format.
- Pace + miles, temp, location label
- Conditions / partners / outlier flags rendered as `<i>... (excluded from fit)</i>` when the run is pruned from the fit
- Most recent race while its fatigue term is still ≥ 5% of its next-day size (`recovery_model.fatigue_window_days`: marathon 7 d, short race 4 d): `Recent marathon: Nd ago` or `Recent race: Nd ago` (whichever category is closest)
- `Recent long run: N mi, Xd ago` while the dosed long-run term is ≥ 5% of a 20-miler's next-day effect (`hover.LONG_RUN_ROW_MIN`): a 22-miler through day 4, a 15-miler through day 3, a 10-miler the day after only, nothing under ~7.4 mi
- Time of day (raw label)

The scaffold itself owns the date header, the CS-pace / Trend-pace section, and the CS-residual / Trend-residual rows — those come from the `payload` dict (`cs_pace`, `trend_pace`, `trend_resid`, `sessions`, `first_day`, `nearest_window_days`) serialized into `window.__TT_DATA`. `build_hover()`'s output is the run-specific tail appended after that.

### Sidebar layout

- Fixed top-right: `right: 12px; top: 48px; width: 240px`
- `top: 48px` clears Plotly's modebar
- 5 normalization checkboxes with their own All/None
- Divider, then 3 visibility checkboxes with their own All/None
- Collapsed coefficient details below; expandable `<details>`
- Stats line at bottom: `n=X in fit; Y excluded (classes overlap)`,
  R² values, trend type
- Colorbar moved to `y=0.45 yanchor='top', len=0.30` (lower portion of
  right margin, below toolbar's expanded footprint)
- Legend (CS pace + Trend) at `y=0.10 yanchor='top'`, just below the
  colorbar. Layout stays consistent whether coefficient details are
  expanded or collapsed.

## Pipeline

```
daily.csv ──┐
races.csv ──┼─→ make_recovery_plots.py ─→ recovery_pace.html
bayes_cs_   │
summary.csv ┘
```

Single-pass:
1. Load daily.csv, filter to `run_type == 'recovery'`
2. Compute days-since each race category (marathon, race_short) and the most
   recent long run with its canonical miles
3. Compute `is_bad_cond`, `is_partner_run`, `tod_is_pm`
4. Compute `is_outlier_loo` against clean neighbor pool (LOO 28d)
5. Combine: `is_pruned = is_bad_cond | is_partner_run | is_outlier_loo`
6. Pin the physical route channels from `physical_route_betas()` (footing,
   altitude); attach the per-run `per_run_elevation` / `per_run_altitude`
   features for the grade-aware `elev_cost`
7. Compute era trend (centered rolling mean over non-pruned pool), backfit
   against the physical channels
8. Compute the day-1-anchored fatigue features (per-category τ; long runs
   dosed by (miles/20)³; terms below the exposure guard held at 0)
9. Fit the era-immune confounds (temp/fatigue/TOD) on the era-detrended
   residual using non-pruned, non-NaN rows, with the physical channels pinned
10. Build per-point contribution channels (temp, route [grade+footing+
    altitude], recent_effort, tod, era — order matches JS `FACTOR_ORDER`)
11. Render plot with embedded JS for normalization toggles, visibility
    toggles, custom tooltip, and trend recomputation

## Coefficient reference (snapshot April 2026)

```
intercept            +2.53
temp_centered        +0.31      sec/mi per °C above 6°C (one-sided heat hinge)
fat_marathon        +17.0       exp(−t/6) decay from marathon       [superseded Sept 2026]
fat_race_short       +8.7       exp(−t/5) decay from short race     [superseded Sept 2026]
  Sept 2026 (day-1 anchored, next-day sizes): fat_marathon +23.4 (τ 2 d),
  fat_race_short +9.9 (τ 1 d), fat_long +7.9 at 20 mi (τ 1 d)
tod_is_pm            −4.7       sec/mi for afternoon/late vs early/morning

Physical route channels (pinned, not fitted here — from physical_route_betas):
  trail_frac          ~+16      sec/mi flat-surface footing at full trail share (mixed = half)
  altitude            ~+2       sec/mi per 1000 ft above the ~3000 ft threshold
  elev_cost           (per-run) grade-aware, scales with measured gain/loss

R² on detrended:    0.298
R² on raw residual: 0.622
```

Note: numbers shift slightly between rebuilds as the locations sheet
evolves and additional days log into the high-volume routes. Don't
treat the table as authoritative — re-run the plotter for current
values. The footing/altitude constants are pinned from
`physical_route_betas()`, so they move only when that pooled fit is re-run.

## Findings

### Marathon-fatigue decay shape (April 2026)

*(Superseded Sept 2026: the per-marathon clean-chain analysis puts the fade at
about 2 days, not 6.2 — see Recent effort. This curve was smoothed with a
~5-day Gaussian, which by itself spreads a sharp two-day spike into a slower
tail, and its windows didn't exclude runs with other hard sessions.)*

Empirical analysis (refit OLS without `fat_marathon`, plot residual vs
days-since-most-recent-marathon, fit decay shapes):

```
day    smoothed mean residual (sec/mi)
  0    +6.83  ← peak (with ~5d Gaussian smooth)
  3    +5.14
  7    +3.13
 14    +0.93  ← essentially zero by here
 21    −0.57
 35    −1.87  ← super-compensation begins
 42    −3.10  ← deepest trough
 56    −1.96
 70    +0.36  ← back to baseline
```

Curve fits on days 0-90:
- Linear (clipped): amplitude 6.35, decay-to-zero at day 15.5
- Exponential: amplitude 7.81, **τ = 6.2 days**, 95% decayed by day 18.6

The 14-day cutoff (used in the old linear feature) is empirically
defensible — fatigue does effectively die by day ~15. The shape was
wrong though: linear underestimates the early sharpness. Switched to
exponential `exp(−t/6)` for marathon, `exp(−t/5)` for short race.
No more arbitrary cutoff, shape matches biology (every tissue-repair
process is exponential).

The negative period (days 14-70) is **post-marathon
super-compensation** — peak fitness window after the body finishes
acute repair. Real signal but small; not modeled (see "Features tested
and rejected").

### Biannual bumps in trendline (April 2026)

Visible bumps in the fully-normalized trend appear roughly twice yearly
since 2022. Investigated extensively April 2026 — outcome below.

**Cycle is real and post-2022 specific.** Phase-aligned average around 7
local minima of the JS-matched trend (full-norm + all hide-toggles, σ=28d
smoother) since 2022-06:

- Volume MIN at +44d post-fitness-peak (taper + post-race recovery week)
- Volume MAX at +100d post-fitness-peak (mid-build for next race)
- Cycle period ≈178d (FFT top peak in BOTH the recovery trend and
  era-normalized 56DMA volume — strongest evidence of coupling)
- Pre-2022 cohort (n=4 cycles, including Boston 2022) did NOT follow this
  pattern: inter-peak spacing was irregular (195-326d). The phase-locked
  oscillation appears to have emerged from the stable twice-yearly racing
  structure that started in 2022-06+.

**Mechanism: coupled training-fitness oscillation.** Cross-correlation on
the smoothed trend vs era-normalized 56DMA volume, 2022-06+:

```
lag (days)   r     interpretation
  −60       −0.29  vol leads fitness — standard training effect
    0       +0.08  near-zero synchronous
  +30       +0.38  taper/post-race: fit peaks, vol bottoms
  +120      −0.40  rebuild + next-cycle build: fit peak → vol peak ~4mo later
```

The structure is a phase-shifted cosine — positive at +30d, negative at
+120d, separated by ~90d ≈ half-period of the 178d cycle.

**Volume tested as OLS normalizer (Apr 2026) — REJECTED.** The hope was
that adding a volume-based feature could flatten the trendline and
account for the biannual bumps without leaning on race-day data (already
in CS).

```
candidate                         r        R²
vol_56 (raw level)              +0.051    0.003
vol_norm (era-normalized 56DMA) +0.064    0.004
dvdt_14 (14d Δ on smoothed)     +0.118    0.014
dvdt_28 (28d Δ on smoothed)     +0.118    0.014
vol_norm_lag30                  −0.063    0.004
vol_norm_lag60                  −0.049    0.002
vol_norm_lag90                  −0.016    0.000

Multi-feature combos:
  best 2-feature (dvdt_14 + dv_lag120):       R² = 0.020
  best 3-feature:                              R² = 0.022
  all 11 volume features combined:             R² = 0.036

Oracle ceiling (perfect cyclic predictor):
  sin+cos at 178d period:                      R² = 0.027
  sin+cos at 158d:                             R² = 0.004
  sin+cos at 200d:                             R² = 0.000
```

**Why the per-day analysis disagrees with the smoothed cross-correlation
(r=±0.40 → R²=0.16 expected).** The cross-correlation operates on σ=28d
smoothed signals, where noise is averaged out and the cyclic signal
dominates. The OLS operates per-day where noise dominates. With
signal_var ≈ 0.03 × total_var, the per-day correlation ceiling is about
√0.03 ≈ 0.17 even for a "perfect" cyclic predictor. The volume features
already capture about as much as theoretically possible at the per-day
level; there's nothing more to extract.

**Conclusion.** The biannual cycle is a real feature of post-2022
training but ~97% of the per-day OLS-residual variance is irreducible
noise. Adding any volume feature (or all 11 combined) reduces the
trendline oscillation by a few percent at best — not visually noticeable.
The cycle remains visible as the residual oscillation in the trendline
and that's an honest representation of what the model can and cannot
explain. Re-test in 2-3 years: with 12-15 cycles instead of 7, the
cyclic signal might separate cleanly enough to become OLS-fittable.

### Time-of-day pulled long-run signal apart

Adding `tod_is_pm` cleaned up an earlier confound: long runs are
predominantly morning, so `fat_long` had been carrying the
morning-slowness signal. Once TOD pulled out the +4.6 sec/mi
morning-vs-afternoon shift, `fat_long` collapsed from +0.78 (kept) to
+0.20 (rejection-threshold-equivalent). Removed. *(Superseded Sept 2026: the
collapse came from the feature's 14-day linear shape, not from a real
absence — `fat_long` returned with a 1-day fade and a distance dose; see
Recent effort. The converse lesson: a badly shaped feature can look null.)*

This pattern is worth remembering: **a feature can look meaningful in
isolation but be a proxy for an unmeasured covariate**. Always test
in joint OLS, not just bivariate.

### Run length, chronic load and pauses (Sept 2026 investigation)

Prompted by the Sept 2026 trend level: normalized recovery pace sat slower than
at any point since 2017 while CS held within a few s/mi of its all-time best.
**Nothing was added to the model.** This section records what was tested, what
the numbers were, and the framing that should govern any future attempt.

#### The governing frame (read this before re-opening the question)

Max's position, Sept 2026, and the reason this investigation closed rather than
shipped a term:

> The standing hypothesis is that the slowdown is a combination of consistently
> longer recovery runs, more quality days and races overall (slower recovery
> before and after more hard days), and decreased pauses. What blocks
> quantification is that a large portion of the effect — particularly from
> pauses — is **mental rather than physiological**. By definition that belongs to
> "my recovery scheme", which is what the trend line is *capturing*, not a
> per-run effect to be normalized away. No recovery run's pace is limited by
> capability over its distance; pace is determined indirectly by fitness, the
> normalization factors, the current scheme, and non-running physical and mental
> state.

The design consequence: a factor that varies because the *scheme* changed is not
a normalization candidate. It is the signal the era trend exists to show. Only
confounds that vary run-to-run *within* a scheme belong in the cross-sectional
block. Every candidate below failed on exactly this line — they are scheme-level,
not run-level.

#### The diagnostic (this part is solid and worth keeping)

Late 2024 (Sep 1 – Dec 31) → summer 2026 (Jun 1 – Aug 28), raw residual +25.5
s/mi. What the existing normalizers absorbed:

```
temperature   +2.42     grade       -1.71
race fatigue  +2.33     footing     +1.70
time of day   -0.27     altitude    -4.36
wind          -0.09     ----------------
                        TOTAL       +0.03
```

They cancel. The model explains **0.03 of 25.5 s/mi**, and the route channel is a
−4.4 credit (Max left altitude; per-mile vertical roughly halved). Ruled out
along the way:

- **Not the heat hinge.** Predicted +2.84 / +4.04 / +5.41 against observed
  +2.42 / +4.14 / +5.12 in the 22–26 / 26–30 / 30+ °C bins. Well calibrated at
  the top end.
- **Not the watch corrections.** They moved the wrong way: +10.5 s/mi of added
  slowness in late 2024 vs +4.3 in summer 2026.
- **Not CS drift.** Season-controlled (Jun 1 – Aug 28), CS is flat 2024 → 2026
  (302.9 → 302.4) while the normalized residual moves +47.4 → +68.3.

Season-controlled summers, normalized residual vs mean recovery distance:

```
2021 +37.4 / 6.13mi   2022 +37.9 / 5.96   2024 +47.4 / 7.04   2019 +47.5 / 7.39
2020 +48.7 / 7.76     2023 +54.7 / 8.54   2025 +58.2 / 8.38   2026 +68.3 / 8.32
```

Sorted by residual, the distance column sorts itself. That co-movement is real
and is what motivated everything below — and nothing below survived it.

#### Per-run distance — the rolling local leave-one-out estimator

The right tool for "is today's longer run slower", and it did not exist in the
repo before this. For each run, neighbours = clean recovery runs within ±W days
with itself excluded; regress (its residual − neighbour mean) on (its distance −
neighbour mean) or its distance percentile in that set. Both sides local, so a
trend inside the window cannot pose as a distance effect. Prefer it over block
fixed effects: 28-day blocks put short runs at a block edge and manufacture
shape (a spurious U appeared in Q2–Q3 2026 under blocking and vanished here).

Max, ±14d, n=2360, decile means of residual-vs-local:

```
pct    0-10   10-25   25-50   50-75   75-90   90-100
       +0.15   -0.68   -0.25   +0.30   +1.07   +0.08
se      0.60    0.55    0.48    0.50    0.77    1.05
```

β = **−0.028 ± 0.154** s/mi per mile. Flat at ±7/14/21/28d and in 2023+ alone.
Top tail cut finer: 90–95 +3.86 ±1.66, 95–99 −0.32 ±1.97, 99–100 −3.75 ±1.67 —
no tail effect. **Yesterday's** distance is also null (+0.018 ± 0.153), jointly
with own-distance both stay at zero.

**Caveat that matters (Sept 2026):** restricted to watch-corrected rows — the
only era with honest per-run distances — raw distance is **+0.429 ± 0.184**
(t=+2.3) locally and +0.399 ± 0.223 within route×28d. The full-corpus null is
diluted by pre-watch rows, where logged distance is a route constant and carries
no genuine per-run variation. So the honest statement is *small but positive in
the watch era* (~+3 s/mi across a 4→11 mi range), not zero.

#### The maddy contrast — a real per-run effect on another profile

maddy, ±14d, n=103, same estimator, percentile bins:
`−9.09, −2.82, −3.49, +1.23, +6.33, +12.01`. β = **+5.10 ± 1.25**, and +5.18 /
+5.10 / +5.00 / +5.10 at ±7/14/21/28d — timescale-invariant, which is the
signature of a genuine effect (Max's grows monotonically with the contrast
window, the signature of confounding). Within-location +5.45 (t=+4.6); survives
trimming either tail. Roughly +25 s/mi across a 2→7.5 mi range.

Not a pool-boundary artifact: both profiles' recovery pools span ~0.65× to ~2×
their own median (Max 6.20 median / 11.20 p99; maddy 3.76 / 7.60), so it is not
that maddy's long runs are hiding in the recovery pool while Max's are not. The
plain reading is that 4→11 mi is inside Max's aerobic steady state and the same
relative range is not inside maddy's.

**A per-profile β is therefore the correct shape if this is ever revisited** —
it fits ≈0 on Max and ≈+5 on maddy. It was not shipped because a box that is a
visible no-op on the default profile wasn't worth the knob.

#### Watch-derived profiles: the pace–distance slope is measurement-dependent

The `coros` profile (Max's own days, watch-derived pace) returns **−2.4** s/mi
per mile — opposite sign, same runner, same days. On 1456 shared days the
recorded *times* agree (minutes gap mean 0.34, slope 0.000/mi) but the miles do
not: `coros_miles = −0.203 + 0.9518 × max_miles`. A **fixed ~0.2 mi per-run
shortfall** amortized over the run predicts −pace·c/D² ≈ −2.0 against −2.45
observed. The `max` profile's GPS gate and `min(estimate, logged)` clamp suppress
this; a raw watch profile has no such protection. **Any distance-vs-pace result
on a watch profile must be checked against a fixed per-run distance offset
before it is believed** — the bias is negative, so it makes maddy's +5.10 if
anything conservative.

#### Chronic recovery-run length — works 2019+, zero over the whole history

The strongest candidate found, and the one that failed most instructively.
56-day trailing mean recovery-run length against the normalized residual
(era NOT removed): univariate +4.78 (t=+12.1, R²=0.057); with weekly recovery
volume and year fixed effects **+3.544 (t=+6.7)**. Survives hard-day frequency
(+3.596, t=+6.3, with `n_hard` at −0.189, t=−0.3 — they correlate only −0.275,
so they genuinely separate), total volume, and volume by type.

Applied to the whole history it does nothing:

```
window          corr(trend, length track)   trend SD before -> after
whole history            +0.002                14.98 -> 15.20
2016-2018                -0.083                21.15 -> 21.19
2019+                    +0.758                 8.63 ->  7.01
```

Zero to three decimals over the full span, and subtracting it makes the
trendline marginally *worse* on both SD and range. Not a beta-choice artifact —
the within-year (+3.544) and global univariate (+3.893) betas are close and both
do nothing. 2016–2018 kills it: the residual falls 54 s/mi across three years at
flat recovery length (5.6–6.0 mi), and the term has nothing to say about any of
it. Everything it buys is in 2019+ (19% off SD, 21% off range), i.e. it is a knob
fitted to the window it was found in.

Two further weaknesses, both standing:

- **Forward placebo loads.** Trailing 56d +3.26 (t=+6.1) vs the *next* 56 days
  +2.32 (t=+5.1), jointly. A future block should not predict today's pace. The
  effect is regime-level ("which block am I in"), not a lagged response.
- **It is algebraically `v_rec / n_rec`** (exact identity). "Longer runs" and
  "fewer recovery days" are the same coefficient — +3.544 per mile of length is
  −4.861 per recovery day per week — and `n_rec` is 97.7% determined by
  `n_days − n_hard`. The data cannot tell the two readings apart.

**Divergence search** (the one test that could break the tie): of 42 quarters,
only 6 have chronic length departing from its volume-implied value by >0.5 mi
(2016Q1, 2019Q3, 2020Q3, 2021Q2, 2021Q4, 2023Q3). All six move the residual in
the same direction as length; corr(len_r, y_r) = +0.407 across all quarters.
Suggestive, but six episodes, max divergence 1.36 mi, and the two largest are the
COVID volume collapse and rebuild. Per-day, chronic length and total volume
correlate **0.789** — the log has essentially never varied one without the other,
and a deliberate divergence is off the table (Max does not alter training to run
experiments on the graphs).

#### Chronic volume by type, and hard-day frequency — both null

Weekly volume split by run type, year FE: recovery +0.093 (t=+1.3), long −0.052
(−0.6), quality +0.088 (+0.8), race +0.262 (+2.3). Only race volume is marginal
and recent races already carry a beta. **Without** year FE these look large
(v_long −1.09, v_qual +0.64) but flip sign — era structure, not load. Quality
volume does not hit harder than recovery volume in this data.

Hard *days* per week: +1.211 (t=+1.8) alone, and −0.189 (t=−0.3) once chronic
recovery length is in. Tested because between late 2024 and summer 2026 hard days
went 0.52 → 1.65/week (direct counts: 122 running days / 113 recovery days /
42.7 rec mi per week / 6.58 mi mean run → 89 / 68 / 44.7 / 8.36; Max ran every
day in both windows). The frequency change is real and large; it does not
explain the residual.

#### Pauses and stops — real, correctly signed, too small

The one candidate that came back with the hypothesized sign in the local
estimator. Stop density collapsed, and it is not a firmware artifact — `stall_s`
stays ≈0 throughout, so auto-pause is still firing and the stops genuinely are
not there:

```
year      2021   2022   2023   2024   2025   2026
stops/mi  1.35   1.71   0.94   0.40   0.35   0.20
pause s    263    257    200    110     80     35
```

Local ±14d, watch-corrected rows: **stops/mi −1.71 ± 0.75** (t=−2.3). Total pause
*seconds* is null (−0.0018 ± 0.0026): it is the **count** of interruptions that
carries signal, not their duration — consistent with a catch-your-breath
mechanism. Within route × 28d block: **−2.39 ± 1.26** (t=−1.9); per route
mccabe −7.75 (t=−2.7, the route with real day-to-day traffic variation),
east boulder −2.99 (−1.1), centennial −0.46 (−0.2), lakefront +1.24 (+0.2).
Within-route spread in stops/mi is only 0.25, hence the weak power.

**Magnitude and timing both rule it out as the explanation.** At β=−2.39 the
2021→2026 stop decline predicts +2.75 s/mi against an actual +22.24 (~12%), and
+0.47 of the 2024→2026 move. The stop density had already collapsed years before
the June 2026 move — Boulder city routes 2021–22 ran 1.55 stops/mi, the Chicago
lakefront era was already 0.27, and the move took it 0.27 → 0.07 (worth +0.48
s/mi against +9.3 actual across that boundary, n=27 post-move). The mechanism
fired in 2023–24, not 2026.

Per the governing frame above, this one is squarely a *scheme* effect and partly
mental, so it belongs in the trend line regardless of its size.

#### Pause-adjusted distance — the long-run algorithm does not apply here

`durability.eroded_deff` **cannot run on recovery days**: its segment store holds
142 dates, all long-run days, zero of 2633 recovery days, so every recovery date
falls through to `return d_m`. Run at gate=1 across 1134 watch-corrected recovery
rows the erosion fraction is **1.000 at mean, p10, p90 and min**. Applying it
would require building the recovery-side segment store first.

The pause-adjusted distance that *does* exist for recovery is the
connected-fatigue `d_eff_frac` in `recovery_measured.csv` (same `measure_day`
machinery). It moves the way the hypothesis predicts — by year 0.803, 0.810,
0.856, 0.906, 0.929, **0.969**, i.e. from ~20% pause-eroded to ~3% — but it adds
nothing over raw distance:

```
regressor                    local ±14d        within route x 28d
raw distance                 +0.429 ± 0.184    +0.399 ± 0.223
pause-adjusted (connected)   +0.475 ± 0.210    +0.402 ± 0.249
adjustment fraction alone    -2.53  ± 4.78     --
longest unbroken segment     +0.537 ± 0.206    +0.556 ± 0.242
```

**Why it can't work, and it's a genuine design mismatch:** the connected measure
is duration-weighted through `RECON_TAU_S`, and `eroded_deff` is explicit that
"pause LENGTH drives erosion, not count — a 20s stop is ~nil… it can't be gamed
by resuming and re-pausing." That assumption is correct for its own job (a
credibility penalty on demonstrated long-run capability *should* be ungameable),
and it is exactly backwards for easy-run physiology, where the data says count
carries the signal and duration does not. Don't reach for the long-run pause
machinery on recovery questions.

`longest_seg_mi` (longest unbroken stretch) is the best distance-family
regressor in the set and is the stoplight hypothesis restated as a distance — an
unbroken mile costs more than a punctuated one. Still only a couple of s/mi
across the range.

#### Correction to the April 2026 volume rejection

The "Volume-based normalizers (any form)" entry above tested 11 volume features
against the **era-detrended** residual. Era trend is a ±182d smoother, so it had
already absorbed anything moving on the chronic-load timescale — that test could
not have found or excluded a scheme-level load effect, and its R² ceiling of
~0.03 is a statement about per-day noise, not about chronic load. The Sept 2026
work above tested against the era-**inclusive** residual, which is the right
target for that question. The conclusion (nothing shippable) is unchanged; the
reasoning in that entry should not be cited as having settled the chronic case.

#### Outcome

Roughly 19–21 s/mi of the Sept 2026 level remains unattributed to anything the
model measures. Per the governing frame, that is the expected result rather than
a gap: the residual is scheme plus non-running state, and the trend line is the
honest place for it. Max's expectation is that the trend over the following
months answers the question directly. **Re-open only with a genuine divergence in
the log** (chronic recovery length moving against volume, or a scheme change that
isolates one channel), not with another functional form.

## Open questions / future work

### Potential additions (low priority)

- **Cold-end temperature term** — temp is currently linear; pace
  probably degrades below 0°C beyond linear, but data is sparse there.
  Quadratic or piecewise term could help if cold-running data grows.
- **2016-17 blank-tag location backfill** (~180 untagged rows
  post-prune) — would unlock more route betas in that era and tighten
  the baseline.
- **Trail-route data expansion** — currently watershed is the only
  trail route, leaving the trail penalty under-constrained.

### Things to NOT pursue without strong new evidence

- Adding shoe-pair as a feature (period-confounded, see rejected list)
- Adding bodyweight as a feature (collinear with era_trend, see
  rejected list — would require restructuring era_trend's role)
- Workout fatigue terms (tempo/interval/fartlek/hill) — null once timing is
  accounted for (Sept 2026); the turnaround effect is the lead if revisited
- A thresholded long-run dose (Max, Sept 2026: the long-run term must scale
  smoothly with distance)
- Tighter era window (would absorb cross-sectional signal)
- Linear or hard-cutoff fatigue decay (replaced with exponential —
  empirically better-shaped, no arbitrary cutoff)
- Per-run or chronic run-length normalizers on the Max profile (Sept 2026 —
  chronic length is zero over the whole history and per-run is sub-noise; a
  per-profile β remains the right shape if it is ever revisited for `maddy`)
- The long-run pause-erosion machinery (`durability.eroded_deff`) on recovery
  questions — its segment store is long-run-only and it is duration-weighted,
  which is backwards for easy running (Sept 2026)
- Rejected sleep features without a concrete new hypothesis or data
  improvement (wind and humidity were re-tested with richer watch data
  and adopted June 2026 — see rejected/adopted list above)

## Naming notes

- "Recent effort" was renamed to "Recent race" in April 2026 after
  `fat_long` was dropped, and back to "Recent effort" in Sept 2026 when
  `fat_long` returned.
- Earlier name "Recent quality" (V6 and earlier) was changed to avoid
  confusion with the training-quality framework.
- "Training quality residual" feature was briefly added in V6 then
  removed in V7 (see "Features tested and rejected").
- `FATIGUE_DECAY_DAYS` (single 14d constant for linear feature) was
  replaced by `FATIGUE_TAU_DAYS` (per-category τ dict) +
  `FATIGUE_HOVER_DAYS` (cosmetic threshold for showing recent-race
  line in tooltip; does not affect the model). Sept 2026: the tooltip windows
  now follow the model (`fatigue_window_days`, `hover.LONG_RUN_ROW_MIN`) and
  `FATIGUE_HOVER_DAYS` is gone.

## Plot: colour-by dropdown (Sept 2026)

The Recovery scatter's marker colour is selectable from a sidebar dropdown
rather than hardwired to temperature. Eight encodings, shared with the Long
Runs tab and specified once in `src/plotting/color_by.py`: distance, weekly
volume, temperature, wind, elevation gain, altitude, time of day, recent effort.
**Distance is the default** — the Sept 2026 rolling-LOO re-test put its
recovery β at ~0 (see "Run length, chronic load and pauses"), but the null is a
statement about *conditional* predictive power, not about whether the variable
organises the cloud. It is worth looking at.

Three consequences for this plot specifically:

- **The vertical temperature colorbar is gone.** The colour key is now a
  horizontal gradient strip at the top of `#norm-filter`. That reclaimed ~45 px
  of right margin, which went into widening the sidebar 240 → 284 px. The
  sidebar also gained a `max-height: calc(90vh - 85px)` cap so it clears the
  legend (parked at paper `y=0.10` in the same margin) on short viewports; the
  pre-existing layout had only 2 px of clearance at 700 px tall.
- **`marker.color` is a per-point CSS-string array, not a `coloraxis`.** Time of
  day can't be expressed as a 1-D colorscale — its gradient anchors are each
  day's solar times. The hide-filters' per-point `marker.opacity` array is a
  separate key and is untouched by a colour change.
- **Temperature is the Misc. Trends ramp verbatim** — `-10 #0E7BAA → 22 #5A9E3D
  → 40 #C82020`, the log's CF colours darkened ~30%. This tab briefly carried
  the undarkened originals on the argument that markers have no white trendline
  over them, but two cheaper levers gave the same legibility without a second
  temperature palette to keep in sync: marker opacity 0.6 → 0.9
  (`MARKER_OPACITY`, shipped to the sibling JS via `js_globals` because the
  hide-filter restyles `marker.opacity` per point and must restore it), and
  `NO_DATA_GRAY` dropped from `#a0a0a0` to `#555555` so unmeasured points
  recede toward the background instead of glowing brighter than the data.

### Adding the dropdown to a third plot

`src/plotting/color_by.py` owns the whole spec — order, value extraction,
gradient, legend ticks — so Recovery and Long Runs cannot drift, and the four
environmental encodings match the Misc. Trends panels via `shared_scales()`.
The canonical order drives the dropdown **and** `hover.context_lines`:
`distance, weekly_volume, temp, wind, elev_gain, altitude, time_of_day,
recent_race` (the key predates the "Recent effort" label). Both plots default
to **distance**.

The contract is opt-in. A plot joins by shipping `window.__PLOT_COLOR_BY`,
rendering `widgets.gradient_strip('cb-strip', …)` plus
`widgets.select('cb-select', …, data_attr='colorby')`, tagging its marker
traces `meta.color_by = True`, and listing `color_by.SCAFFOLD_JS` first in
`overlay_js_files`. The handler resolves traces by that meta flag, never by
index, and restyles **only** `marker.color` — leaving each plot's own
handlers (y, `marker.opacity`) alone.

**Scale provenance is deliberate.** Distance and elevation gain are
*per-plot*: they scale with the run type, and a shared scale would park every
recovery run in the dark third of a ramp topped by a 22-mile long run. Weekly
volume, wind and altitude are *shared*, computed off the full `daily.csv`
calendar, because they are properties of the day rather than the run.
Temperature, time of day and recent effort are *fixed*.

**Recent effort superimposes three ramps (Sept 2026).** One per recovery-model
fatigue term — short race `#FF3B30` (the old recent-race red), marathon the
Road race blue (`tokens.SURFACES['Road']`), long run the frontier purple
(`tokens.FRONTIER_LINE`) — each full colour the day after the session and
fading linearly to gray at `recovery_model.fatigue_window_days` (1 + 3τ:
short race and long run 4 d, marathon 7 d), where the term falls below 5% of
its next-day size and the tooltip's recent-race / recent-long-run rows drop
off. Each run takes the ramp of whichever term has the **largest modelled
effect that day** (pooled β × feature, the sizes the recovery fit pins), not
simply the most recent session: a marathon 2 days back outranks a 10-mile long
run yesterday. The legend swaps the single bar for three stacked strips, each
with its own day scale (`widgets.gradient_rows`, the `cb-strip-stack`
container, mirrored by `renderStack` in `_scaffold/color_by.js`); any variable
that ships `strips` gets the stack. A profile whose fatigue terms are all held
at 0 by the identification guard (maddy, Sept 2026) renders every run gray.
Replaced a single red ramp on days since any race, gray at 14 d.

**Gray means "0 or no data"** — `gradients.NO_DATA_GRAY` (`#555555`), the same
value `gradient_at` returns for NaN, so negligible and unknown render alike on
the one-sided ramps. It sits deliberately close to the `#1a1a1a` background
(contrast 2.3) so those ramps read dark → bright rather than light-gray →
mid-colour. It never reaches Misc. Trends: that plot's ramps only ever sample
finite y values, verified by instrumenting `gradient_at` for NaN hits (0).

### Tooltip

The per-run rows are now built by two shared builders in `src/plotting/hover.py`
so this tab and Long Runs stay in sync on both content and order:

1. `location_line(r)` — italic route/city (was buried mid-tooltip here).
2. `long_run_lines(r, show_logged=False, show_elevation=False, source_tag=…)` —
   the Long Runs `{dist} mi @ {pace}/mi` form, replacing this tab's old
   `Pace: 7:30/mi  (6.2 mi)`. The `[watch-measured]` / `[corrected route]` tag
   and its `≥ 1 s/mi` gate are unchanged, passed in as `source_tag`.
3. `context_lines(r, elev_gate_value=row['contrib_elevation'])` — the seven
   context rows in the same canonical order as the colour-by dropdown.

Gates that did **not** change: elevation still fires on
`|contrib_elevation| ≥ 1.0` (recovery rows carry no `grade_cost_s_per_mi`, hence
the `gate_value` override), and recent race still only shows inside its
fatigue window (`fatigue_window_days` since Sept 2026). New rows: weekly volume (always) and altitude (≥ 1000 ft,
which keeps Chicago and the lower Nashville runs from carrying a dead row).
Provenance/exclusion italics — conditions, partners, outlier — stay below the
context block.
