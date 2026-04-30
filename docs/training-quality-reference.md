# Training-quality framework — reference

Companion analysis layer to the CS model. Translates workouts and long runs
into 5K-equivalent paces, then computes residuals against the CS-implied
fitness curve to extract a training-quality signal.

## Purpose

CS is a race-only fitness measure. Many useful signals — fitness building
between races, training that's running ahead of (or behind) realized fitness,
training composition during breakthrough periods — aren't visible from CS
alone. This framework adds a parallel layer derived from training data that
sits next to CS without feeding back into it.

## Decisions locked in (April 2026)

- **CS is the ground truth.** Workouts and long runs do NOT feed back into
  the CS fit. Backtest of combined model showed RMSE 13.4 vs CS-only 10.6 on
  focused 2018+ HM+Track5K subset; per-race prediction is dominated by
  execution noise that CS already captures and training residuals don't
  improve. Framework value is retrospective, not prospective.
- **Projection: CS-hyperbolic for everything.** Same `t_5K = (5000 − D')·t /
  (d − D')` projection the CS model uses. Replaces the earlier Riegel-based
  approach. Theoretically grounded; doesn't require an unmotivated exponent.
- **No β_long anywhere.** β_long encodes max-effort long-distance fade,
  doesn't apply to sub-max efforts. Including it would inflate fitness
  implications beyond what the data demonstrates.
- **τ = 210 sec/mi for workout D_eff decay.** Calibrated so 6×1600 @ 3:00/mi
  rest gives D_eff ≈ 5000m (anchor preserved from earlier work).
- **Reps excluded from the smoother.** Anaerobic top-end work, doesn't
  reflect aerobic CS-frame fitness. Decomposer still emits them; plot
  pipeline filters them out.
- **Hill continuous workouts included as a TQ category (April 2026).**
  Per-loop offsets for `lc`, `rc`, `pwr1` (n>7 cutoff). No grade
  adjustment is applied: the per-loop offset absorbs any uniform per-
  loop transformation, so adjustments like Minetti's energy-cost curve
  are mathematically redundant. Other loops (pwr3, hj, 106th, fm1) lack
  enough data for stable offsets and are dropped from TQ; they will be
  surfaced on the future all-workouts qualitative plot using loop
  distance/elevation for display only. Hill repeats are categorically
  excluded (no per-rep distance is logged, so no pace can be recovered).

## Pipeline (April 2026)

### Stage 1 — Workout decomposer (`parse_workouts.py`)

Reads `daily.csv` and decomposes quality workout strings into a structured
table: `workout_decomposed_v7.csv` (currently 259 rows). Same schema as the
older `workout_vdot_v6.csv`, with these rule changes:

- **Continuous-fartlek classification**: fartleks 6400–10000m with zero or
  no rest annotation are classified as continuous_fartlek (not interval).
  Boundary at 6400m fixes 2024-03-09 (6800m) and 2024-07-07 (8000m).
- **Implicit decomposition**: workouts written as bare `Nt` / `Ni` / `Nf`
  with no `Nx` reps decompose to standard rep distances (interval ≥ 4800m
  → 1600m reps; 3200–4799 → 800m; rep → 400m; tempo < 7000 → 1000m,
  ≥ 7000 → 1600m).
- **Default rests**: tempo 60 s/mi, interval 140 s/mi (800m) or 180 s/mi
  (other), rep 420 s/mi, continuous_fartlek 0.

Decomposer-level prunes (24 rows go to `workout_pruned_v7.csv`):
2016-07-11 anomaly; tempos with paces over 10:00/mi; `qd < 100`; continuous
tempos with explicit `0:00` rest; sub-4000m fartleks with no/zero rest.

### Stage 2 — Workout projection

Per workout (in the plot pipeline):
1. `decay = exp(−rest_per_mile / 210)`
2. `D_eff = rep_dist · (1 + (rep_count − 1) · decay)`
3. `t_eff = pace_per_mile · D_eff / 1609.344` (seconds at workout pace)
4. `t_5K = (5000 − D'_t) · t_eff / (D_eff − D'_t)` where `D'_t` is the CS
   posterior median D' interpolated to the workout date
5. `P5K = t_5K · 1609.344 / 5000` (sec/mi)
6. `raw_resid = P5K − CS_implied_5K_pace_at_t` (sec/mi)

### Stage 3 — Long run projection

Filter: `run_type == 'long'` AND `miles ≥ 20`. No upper bound (was
previously `< 26`; removed because peak-volume runs at 26+ are legitimate
training data).

Per long run:
1. `t_run = recovery_pace_sec_per_mi · miles` (seconds)
2. `D_eff = miles · 1609.344` (continuous, no decay)
3. `t_5K = (5000 − D'_t) · t_run / (D_eff − D'_t)`
4. `P5K = t_5K · 1609.344 / 5000`
5. `raw_resid = P5K − CS_implied_5K_pace`
6. Bin: `lr_20-22.9` if miles < 23, else `lr_23+`.

### Stage 3.5 — Hill continuous projection

Filter: `run_type == 'hill_cont'` AND `loop ∈ {lc, rc, pwr1}`. The n>7
cutoff matches the aggressive scoping used elsewhere (analogous to the
`miles ≥ 20` long-run cutoff). Loop is parsed from the workout string
(`hc-Nx <loop>`); 4 sessions in May–June 2018 missing the inline loop
token are recovered from the `location` column ("rollercoaster" → rc,
"powerline west" → pwr1). Two `hc/rep` hybrid sessions (Sept 2016) are
dropped at parse time.

Loop lookup table (source: "hills" tab in *Max's Running Data*) — only
`distance_m` is used by TQ. The full elevation data (up/down/net) lives
in the source for use by the future all-workouts qualitative plot, not
here.

| Loop  | Distance (m, full loop) |
|-------|------------------------:|
| lc    | 1290                    |
| rc    | 850                     |
| pwr1  | 620                     |

Per session:
1. `total_dist_m = nreps · loop_distance_m`
2. `actual_pace = (session_min · 60) / (total_dist_m / 1609.344)`
3. CS-hyp projection on `(total_dist_m, actual_pace · total_dist_m / 1609.344)`
   → `t_5K`, `P5K`, `raw_resid` as in Stage 2.
4. Category: `hill_<loop>`.

**No grade or terrain adjustment is applied.** The per-loop offset
(Stage 5) absorbs all loop-specific systematics — grade, surface,
terrain — by construction: any uniform per-loop shift collapses into
the offset and is subtracted back out before the smoother sees it.
A Minetti-style energy-cost adjustment was tried and rejected for this
reason; the only signal it could have added was the within-loop pace-
dependent spread, which empirically is < 4 s/mi against per-loop
residual SDs of 5–22 s/mi.

### Stage 4 — Corrections

Applied before offset computation:

- **Tempo → Interval reclassification**: any tempo with `rep_dist ≥ 1600m`
  AND no `Nx` in the raw log → interval. Catches 2024-05-04 (`10000t@4:56`,
  written as 10K continuous tempo but functionally an interval). Preserves
  2016 `4×1600t` / `3×1600t` legitimate tempos.
- **Snow filter**: drops sessions with `snow` in `workout_raw` or
  `conditions`. Removes 4 workouts and 1 long run; surface conditions
  invalidate pace projection. (Note: 2026-01-19 and 2026-01-22 were on
  ice, not snow — to be reclassified in next data freeze.)
- **XC −6% pace correction**: applied to (a) all sessions in the HS XC
  season window 2016-07-01 through 2016-10-31, and (b) any tempo logged
  with `quality_distance == 5000m` (HS 5K course as 5×segments, three
  summer 2017 entries). Pace divided by 1.06 before projection. Tooltip
  marks corrected sessions with `[XC-corrected −6%]` in light blue.

### Stage 5 — Per-category offsets and outlier prune

For each category, the median raw residual becomes the offset; the
corrected residual is `raw_resid − offset`.

Outliers are pruned iteratively: any session with corrected residual
> +23.3 s/mi after offset application is dropped, offsets are recomputed
on the surviving set, and the cycle repeats until no point exceeds the
threshold. Converges in 3–4 passes. Currently prunes 5 workouts, 11 long
runs, and 25 hills as outliers (in addition to the ~35 reps that are
excluded categorically). Hills have a higher prune rate (~20%) than
workouts/long runs because the long right tail of "easy hill days" sits
just above the +23.3 cutoff.

### Per-category offsets (final, April 2026)

| Category            |   n | Offset (sec/mi) | Interpretation                             |
|---------------------|----:|----------------:|--------------------------------------------|
| interval            | 145 |          −3.06  | At ~5K capability                          |
| continuous_fartlek  |  35 |         +19.68  | Sub-threshold continuous                   |
| tempo               |  36 |         +15.75  | Sub-threshold by design                    |
| hill_lc             |  72 |         +31.01  | Paved 3.5% loop, tempo+hill effort         |
| hill_pwr1           |   7 |         +95.68  | Steep gravel 9.8% loop                     |
| hill_rc             |  19 |         +74.03  | Rocky-trail 5.8% loop                      |
| lr_20-22.9          |  62 |         +28.17  | Committed long, ~28s/mi off CS pace        |
| lr_23+              |  30 |         +47.48  | Peak volume, more conservative             |

Reps (~35) are excluded from the smoother.

Total kept after all filters and prunes: 216 workouts + 92 long runs +
98 hills = 406. Hill offsets are large because they encode actual
on-loop pace (no grade adjustment); steeper loops have proportionally
larger offsets.

## Stage 6 — Adaptive Gaussian smoother

Corrected residuals are smoothed over time on a 7-day grid:

- **Base bandwidth**: 30 days
- **Effective sample size (ESS) target**: 12 observations
- **Bandwidth**: smallest value such that ESS ≥ 12, found by **bisection**
  to ~0.5-day precision (continuous, not snapped to 1.25× steps — earlier
  step-based growth produced 7 distinct bandwidth values across the
  timeline with 62 transitions, each visible as a small jog in the line)
- **Maximum bandwidth**: 400 days; if ESS still < 12 at max bandwidth, the
  grid point is NaN
- **Gap break**: if any consecutive pair of training points (workouts,
  long runs, or hill_cont sessions after all filters) is more than 90
  days apart, the smoother output is NaN inside that gap. Currently the
  only qualifying gap is 2020-12-23 → 2021-04-19 (117 days, labrum
  injury). Threshold was lowered from 120 to 90 days when hills were
  added; the 2020-12-23 hill workout would otherwise have closed the
  gap to under 120 days and reconnected the line through the labrum
  recovery period.

Rendered as a near-white line on top of the CS-implied 5K curve at
position `CS_implied + smoothed_resid / 60`. Above CS = training reflects
fitness races haven't yet shown; below = lagging behind.

## Visualization (`plot_training_quality.py`)

- Y-axis: 4:20–6:00 min/mi, gridlines every 10 seconds
- X-axis: yearly gridlines anchored at January 1
- Each session positioned at `CS_implied + corrected_resid / 60` so
  vertical distance from the CS line equals offset-corrected fitness
  signal
- Custom vertical-hover tooltip: at any cursor x, shows the date, race
  fitness (CS), training quality (smoother), diff in s/mi, and the
  nearest session with day-difference label and full session detail
- Plotly's per-trace hover is fully disabled; the hover system is a
  precomputed daily payload (CS array, smoother array with NaN in gaps,
  sessions list) consumed by a JS mousemove handler that converts pixel
  x → day → tooltip

## Validated retrospective signals

Findings are robust to the offset shifts between pipeline iterations
(qualitative direction stable; specific medians shift by a few s/mi).

**2022 breakthrough block (Nov 2021 – Nov 2022):** Sustained negative
corrected residuals across both intervals and 20–22.9mi long runs. Both
training types tracked together → high-confidence forecast. Realized in
Nov 2022 Nashville Marathon (2:28:51, 1st).

**2017 fall block (Sep – Dec 2017):** Pure intervals + tempos, sustained
negative residuals. HS-era quality block.

**2024 lagging period:** Median residual drifted positive through Q1.
Realized in Boston 2024 (2:46:34). Framework correctly flagged training
wasn't building.

## Why the framework doesn't predict per-race times

Backtest result on focused 2018+ HM+Track5K subset (n=24): combined model
RMSE 13.4 vs CS-only 10.6 sec/mi. Pearson r between training residual and
race residual = −0.16. Three reasons:

1. **CS-only is already very tight on Track 5K** (RMSE 4.7 sec/mi).
   Race-day execution noise is most of the remaining error; training
   residual can't resolve below it.
2. **CS-implied HM is miscalibrated.** 2022–2024 HMs ran +7 to +27 sec/mi
   slower than CS predicted while track 5Ks ran on prediction. β_long was
   fit primarily on marathons (sparse HM data); the HM projection is
   underconstrained. Adding a (correct) training residual to a
   (miscalibrated) HM prediction makes it worse.
3. **Per-category offsets are time-invariant but prescription varied by
   era.** Continuous fartleks happened only 2019–2020; the pooled offset
   under-corrects 2019 specifically. Tempos in HS era were prescribed
   differently than modern era. This adds noise to per-race predictions
   but doesn't change retrospective block-level patterns.

## Considered and rejected

- **Riegel projection.** Used initially; mathematically reasonable but
  theoretically unfounded. Replaced by CS-hyperbolic to unify with the
  rest of the pipeline.
- **Workouts feeding into CS as inequality observations.** Considered as
  a way to fill 2020 race gaps. Rejected: training residual SD and
  per-category offset structure introduce era-specific bias that
  contaminates the CS curve.
- **β_long correction on long runs.** Would inflate sub-max efforts
  toward max-effort projections. Wrong tool for sub-max physiology.
- **Three-bin long-run split (16-19.9 / 20-22.9 / 23+).** Considered but
  rejected. 16-19.9 was too heterogeneous (mix of recovery-style and
  semi-honest efforts) — the `miles ≥ 20` filter is cleaner than
  inventing a third bin around a noisy boundary.
- **Long-run upper bound at 26mi.** Removed. Peak-volume runs at 26+ are
  legitimate training data; the previous reasoning (race-related) was
  actually selecting *for* committed efforts.
- **Race-inclusive gap detection.** Considered using {workouts, long
  runs, races} for the gap-break threshold so summer 2018 (heavy racing,
  light workouts) wouldn't break. Unnecessary: the 120-day threshold
  catches only the 2020 labrum gap, summer 2018 is at 108 days.
- **Per-block (per-era) offset fits.** Considered after the 2020-2021
  gap detection split the data into two contiguous blocks. Rejected:
  splitting into 2 blocks would create instability in offset estimates
  with no meaningful benefit; offsets reflect what's typical for that
  workout type, not period-specific.
- **Per-race CS leave-one-out refit for backtest.** Would address CS
  self-leakage in prediction errors. Cost: ~17 hours of fits. Skipped
  because the result was decisive enough without it.
- **Terrain-grouped hill offsets** (paved/gravel/trail). Considered to
  rescue low-n loops (pwr3, 106th). IQRs showed only marginal
  improvement over per-loop. Replaced with the n>7 cutoff: cleaner, no
  exclusions to maintain, consistent with `miles ≥ 20` for long runs.
- **Minetti grade adjustment for hill_continuous.** Initially included.
  Removed once it became clear that per-loop offsets absorb any uniform
  per-loop transformation by construction; the only signal Minetti could
  add was the within-loop pace-dependent spread, which empirically is
  < 4 s/mi against per-loop residual SDs of 5–22 s/mi. Loop elevation
  data is retained in the source ("hills" tab) for the future
  all-workouts qualitative plot, where it will display total estimated
  vertical without contributing to TQ.
- **Hill repeats in TQ.** Distance per rep isn't recorded, only nreps
  and rep duration. Pace can't be recovered from logged data; back-
  deriving it via assumed effort would import the answer. Repeats will
  appear on the future all-workouts qualitative plot using nreps ×
  rep_dur × loop elev_per_min for an estimated total vertical, but
  contribute nothing to TQ.
- **Perceived-intensity (1–10) signal for hill repeats.** Logged in
  2016–17, dropped during data freeze. Pre-check showed values cluster
  at 6–7 for nearly all hill rep sessions; after regressing out
  workload, residual variance is too low for useful signal. Refreezing
  would be cheap but the signal isn't there.

## Working assumptions

- Per-category offsets are constants pooled across all years. Era-specific
  offsets would likely improve per-race prediction but adds complexity
  and isn't needed for retrospective block analysis.
- Adaptive Gaussian smoother bandwidth/ESS parameters are tuned visually;
  reasonable to revisit if data density changes substantially.
- The framework remains a parallel layer to CS, not a feedback input.

## Files

### Pipeline scripts (run in order)

- `parse_workouts.py` — daily.csv → workout_decomposed_v7.csv
- `plot_training_quality.py` — workout_decomposed_v7.csv + daily.csv +
  bayes_cs_summary_v11.csv → training_quality.html

Path resolution: both scripts try `./output/`, then current dir, then
`/mnt/project/`. Outputs go to `./output/` if it exists, else
`/mnt/user-data/outputs/`, else current dir.

### Inputs

- `daily.csv` — daily log, source for both workouts and long runs
- `bayes_cs_summary_v11.csv` — CS posterior; framework uses `cs_mps_med`
  and `dp_med` to compute CS-implied 5K and CS-hyp projections
- `workout_decomposed_v7.csv` — output of `parse_workouts.py`, parsed
  workout table

### Companion docs

- `cs-model-reference.md` — CS model reference (race-only fitness curve
  this framework is layered on top of)

### Earlier (superseded) scripts in project

- `training_unified_pipeline.py` — earlier monolithic pipeline; logic
  now split between `parse_workouts.py` and `plot_training_quality.py`
- `training_quality_track.py` — earlier renderer with stepwise bandwidth
  growth and no custom hover; replaced by `plot_training_quality.py`
- `training_backtest.py` — past-only smoother evaluation against actual
  race performance; useful if revisiting prediction question
