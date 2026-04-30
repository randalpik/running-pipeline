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
  `bayes_cs_summary_{tag}.csv`. Writes `recovery_pace.html` and
  `route_betas_{tag}.csv` (per-route empirical betas, recovery-only,
  effort-uncontaminated — consumable by downstream long-run TQ work, see
  `route-normalization-reference.md`). Default invocation: `python
  make_recovery_plots.py --tag v11`. Runtime: seconds.

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
                   + Σ β_r      · route_dummy_r       (n ≥ MIN_ROUTE_N)
                   + β_marathon · fatigue_marathon    (exp(−t/τ_mar))
                   + β_race     · fatigue_race_short  (exp(−t/τ_rs))
                   + β_tod      · tod_is_pm
```

- **Temperature**: linear, reference 12°C. β ≈ +0.28 sec/mi per °C from
  reference. Real but small — captures roughly 25% of the seasonal pace
  pattern; the rest is summer-training-intensity confound that era trend
  absorbs.

- **Route dummies**: one per location with n ≥ 13 recovery runs. 12
  qualifying routes typically. Coefficients range from −10 (centennial,
  fast flat Nashville) to +27 (banff, altitude+terrain) sec/mi. Recovery-
  only by design — see `route-normalization-reference.md` for why long
  runs are excluded from the route-beta fit.

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

### Pruning: three classes, can overlap

Three independent flags. A row can be in any combination of them. All
flags exclude the row from the OLS fit but the points remain plotted.

1. **Bad conditions** (`is_bad_cond`): `conditions_clean ∈ {snow, icy}`
   OR `workout_raw` matches the regex `\bsnow\b` (catches `[2" snow]`
   bracket annotations the conditions field missed). 2017's untagged
   snow days come in via the workout regex. **`inside` was removed from
   the bad-cond set** in April 2026 — those are treadmill/indoor-track
   runs at stable surface and pace, valid data.

2. **Partner runs** (`is_partner_run`): any `partners` entry that isn't
   blank/solo/none. Concentrated in 2016-17 HS varsity team easy days.
   This is a fundamentally different population: in 2016-17, partner
   runs averaged 442 sec/mi vs. solo 400 sec/mi (42 sec/mi gap). Pruning
   these markedly improves R² on detrended residual.

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

- **Wind (high)** — tested April 2026. n=23 days, Δmean +0.27 vs clear,
  p=0.93. Effectively zero. Even the single "extreme" wind row sits
  just under the ±45 outlier threshold. Wind is logged sparsely (only
  ~15% of recent runs have a value), so the test is power-limited, but
  the effect — if any — is well below what we'd care about.

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

### Custom tooltip with arrow

Plotly's built-in hover label is suppressed via CSS
(`.hovertext { display: none !important; }`) and replaced with a
custom `<div id="rec-tooltip">` rendered on `plotly_hover` events.
Reasons:
- Plotly's modebar is at top-right of the chart; default-positioned
  hover labels would extend into the sidebar area when hovering points
  near the right edge of the plot.
- Plotly's auto-flip only kicks in at the paper edge, not the plot-area
  edge, so the sidebar-overlap problem isn't solved natively.

The custom tooltip:
- Snaps to the actual data point's pixel position via
  `point.xaxis.c2p(...)` + `_offset` + plot bounding rect.
- Has a CSS arrow (`::before` border + `::after` fill) indicating the
  data point's location, with `data-side="right"` or `"left"` toggling
  which edge the arrow renders on.
- Uses **fixed-pixel flip threshold** `FLIP_LEFT_PX = 360`: if point's
  screen X is within 360px of the viewport right edge, tooltip flips to
  the left side of the point. Decision is point-position-based (not
  tooltip-width-based), so no jitter when the cursor moves around within
  one point's hover zone.

Hover content (built once at fig construction by `build_hover()`):
- Header: `YYYY-MM-DD (DOW)` — DOW abbreviation matches the TQ tooltip
  format.
- Pace + miles, CS pace, residual, temp, location label
- Most-recent race only (within 14d): `Recent marathon: Nd ago` or
  `Recent race: Nd ago` (whichever category is closest)
- Pruning flags rendered as `<i>... (excluded from fit)</i>` with
  simplified outlier text (just `Outlier (excluded from fit)`, no
  redundant LOO-residual display)
- Time of day (raw label)

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
bayes_cs_   │                            └→ route_betas_{tag}.csv
summary.csv ┘
```

Single-pass:
1. Load daily.csv, filter to `run_type == 'recovery'`
2. Compute days-since each quality category (marathon, race_short)
3. Compute `is_bad_cond`, `is_partner_run`, `tod_is_pm`
4. Compute `is_outlier_loo` against clean neighbor pool (LOO 28d)
5. Combine: `is_pruned = is_bad_cond | is_partner_run | is_outlier_loo`
6. Determine qualifying routes (n ≥ MIN_ROUTE_N on non-pruned subset)
7. Compute era trend (centered rolling mean over non-pruned pool)
8. Compute exponential fatigue features (per-category τ)
9. Fit OLS on era-detrended residual using non-pruned, non-NaN rows
10. Build per-point contribution channels (5: temp, route, recent_race,
    tod, era — order matches JS `FACTOR_ORDER`)
11. Render plot with embedded JS for normalization toggles, visibility
    toggles, custom tooltip, and trend recomputation

## Coefficient reference (snapshot April 2026)

```
intercept            +2.53
temp_centered        +0.28      sec/mi per °C from 12°C reference
fat_marathon        +17.0       exp(−t/6) decay from marathon
fat_race_short       +8.7       exp(−t/5) decay from short race
tod_is_pm            −4.7       sec/mi for afternoon/late vs early/morning

Route offsets (vs unspecified-location baseline):
  centennial         −10.1     n≈519 — Nashville flat/road
  baton rouge         −9.6     n≈20  — Nashville-era flat-paved
  east boulder        −7.1     n≈394 — Boulder flat
  mccabe              −6.8     n≈400 — Nashville rolling
  lakefront           −3.7     n≈216 — Chicago lakefront
  nike>nature         +4.6     n≈51  — Redmond rolling
  english hill        +6.8     n≈140 — Redmond hilly
  river trail         +8.0     n≈32  — Sammamish River Trail
  nike>powerline     +11.2     n≈82  — Redmond hilly
  suburbia           +12.5     n≈108 — Redmond very hilly
  boulder creek      +13.1     n=14  — Boulder creek path
  banff              +26.8     n≈25  — altitude + terrain

R² on detrended:    0.323
R² on raw residual: 0.542
fit n ≈ 2,231
unique pruned: 281 (96 bad-cond, 170 partner, 32 outlier; classes overlap)
```

Note: numbers shift slightly between rebuilds as the locations sheet
evolves and additional days log into the high-volume routes. Don't
treat the table as authoritative — re-run the plotter for current
values.

Side-output: `route_betas_{tag}.csv` writes the per-route betas in a
3-column flat file (`route, n, beta_sec_per_mi`) for consumption by
downstream tooling.

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
- Rejected weather/wind/sleep features without a concrete new
  hypothesis or data improvement

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
