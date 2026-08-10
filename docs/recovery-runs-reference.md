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
                   + β_marathon · fatigue_marathon    (exp(−t/τ_mar))
                   + β_race     · fatigue_race_short  (exp(−t/τ_rs))
                   + β_tod      · tod_is_pm
```

The route handling is a **physical, era-free model** (June 2026 — replaced
the per-route dummies; see the "Physical route model" subsection below). The
temp / fatigue / TOD terms are the fast-varying, era-immune confounds fitted
on the era-detrended residual; the three physical channels (footing,
altitude, grade) are pinned from `recovery_model.physical_route_betas()` and
consumed via `fit_recovery_model(pin_physical=True)`.

- **Temperature**: linear, reference 12°C. β ≈ +0.28 sec/mi per °C from
  reference. Real but small — captures roughly 25% of the seasonal pace
  pattern; the rest is summer-training-intensity confound that era trend
  absorbs.

- **Route handling — physical, era-free model** (June 2026): the per-route
  dummies were **removed** and replaced by three physical channels that
  re-explain the route variance without any per-route knob. See the
  "Physical route model" subsection below for the full decomposition and why
  the fit is a backfit. The three channels are footing (`is_offroad`),
  altitude, and a grade-aware `elev_cost`; the first two are pinned from
  `recovery_model.physical_route_betas()` (the single source of truth shared
  with the long-run and race conversions), the third is imported from
  per-mile elevation data.

- **Recent race** (renamed from "Recent effort" in April 2026): two
  categories — marathon and race_short — using **exponential decay
  exp(−t/τ)** with τ values empirically derived from the data:
  `FATIGUE_TAU_DAYS = {marathon: 6, race_short: 5}`. β values around
  marathon +17, race_short +9 sec/mi for the day-0 contribution. See the
  "Marathon-fatigue decay shape" finding below for the empirical curve
  this replaced the old 14d-linear assumption.

- **Time of day** (`tod_is_pm`): binary indicator, 1 if afternoon/late, 0
  if early/morning. β ≈ −4.6 sec/mi (afternoon runs are faster). 100% of
  recovery rows have a TOD value. Adding this feature also collapsed
  `fat_long` to noise (long runs are predominantly morning, so the
  morning-slowness signal had been wrongly absorbed by long-run fatigue
  before TOD was in the model). `fat_long` removed in the same iteration.

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

- **Off-road footing** (`is_offroad`, the mixed+trail binary — bucket every
  run by its LOCATION `terrain_type`, never by surface): the flat-surface
  penalty for running on non-paved ground (trail_frac coding: mixed = half
  the full-trail value; live ≈ +16 at full trail as of Aug 2026). Pinned, not
  fitted in this model.
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
   runs at stable surface and pace, valid data.

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
  independently of other factors. β ≈ 0.

- **Hill workouts** (continuous + reps) — sub-noise effect on next-day
  recovery pace. The legs may feel it but the residuals don't show it
  beyond what `fat_race_short` already captures (some hill sessions get
  classified as race-equivalent if they involved hard sustained efforts).

- **Tempo / interval / fartlek workout fatigue** — only races and long
  runs leave detectable next-day signature; standard quality workouts
  don't. (Long runs themselves were removed from the model in April
  2026 — see TOD feature note above.)

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

- **Long-run fatigue** (`fat_long`) — kept until TOD was added in April
  2026. Original linear-decay β ≈ +2 sec/mi peak; after TOD pulled out
  the morning-slowness component, `fat_long` β collapsed to +0.20
  (essentially zero — same magnitude as already-rejected features).
  Long runs are predominantly morning, so prior versions had been
  wrongly attributing morning-slowness to long-run fatigue. Removed.

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
- Most recent race within `FATIGUE_HOVER_DAYS = 14`: `Recent marathon: Nd ago` or `Recent race: Nd ago` (whichever category is closest)
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
2. Compute days-since each quality category (marathon, race_short)
3. Compute `is_bad_cond`, `is_partner_run`, `tod_is_pm`
4. Compute `is_outlier_loo` against clean neighbor pool (LOO 28d)
5. Combine: `is_pruned = is_bad_cond | is_partner_run | is_outlier_loo`
6. Pin the physical route channels from `physical_route_betas()` (footing,
   altitude); attach the per-run `per_run_elevation` / `per_run_altitude`
   features for the grade-aware `elev_cost`
7. Compute era trend (centered rolling mean over non-pruned pool), backfit
   against the physical channels
8. Compute exponential fatigue features (per-category τ)
9. Fit the era-immune confounds (temp/fatigue/TOD) on the era-detrended
   residual using non-pruned, non-NaN rows, with the physical channels pinned
10. Build per-point contribution channels (temp, route [grade+footing+
    altitude], recent_race, tod, era — order matches JS `FACTOR_ORDER`)
11. Render plot with embedded JS for normalization toggles, visibility
    toggles, custom tooltip, and trend recomputation

## Coefficient reference (snapshot April 2026)

```
intercept            +2.53
temp_centered        +0.31      sec/mi per °C above 6°C (one-sided heat hinge)
fat_marathon        +17.0       exp(−t/6) decay from marathon
fat_race_short       +8.7       exp(−t/5) decay from short race
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
+0.20 (rejection-threshold-equivalent). Removed.

This pattern is worth remembering: **a feature can look meaningful in
isolation but be a proxy for an unmeasured covariate**. Always test
in joint OLS, not just bivariate.

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
- Re-introducing `fat_long` (collapsed to noise after TOD)
- Tighter era window (would absorb cross-sectional signal)
- Linear or hard-cutoff fatigue decay (replaced with exponential —
  empirically better-shaped, no arbitrary cutoff)
- Rejected sleep features without a concrete new hypothesis or data
  improvement (wind and humidity were re-tested with richer watch data
  and adopted June 2026 — see rejected/adopted list above)

## Naming notes

- "Recent effort" was renamed to "Recent race" in April 2026 after
  `fat_long` was dropped — only marathon and short-race fatigue remain.
- Earlier name "Recent quality" (V6 and earlier) was changed to avoid
  confusion with the training-quality framework.
- "Training quality residual" feature was briefly added in V6 then
  removed in V7 (see "Features tested and rejected").
- `FATIGUE_DECAY_DAYS` (single 14d constant for linear feature) was
  replaced by `FATIGUE_TAU_DAYS` (per-category τ dict) +
  `FATIGUE_HOVER_DAYS` (cosmetic threshold for showing recent-race
  line in tooltip; does not affect the model).
