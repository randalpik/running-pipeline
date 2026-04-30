# Critical Speed (CS) model — reference

This doc captures the design and decisions behind the Bayesian CS pipeline
(`bayes_cs_fit.py` + `bayes_cs_plot.py`) so future work can pick up without
re-deriving choices. Read this before reconsidering the model.

## Purpose

Estimate Max's fitness over time on a single continuous scale by inferring
**Critical Speed (CS)** — the asymptotic sustainable pace — from race
results. CS is intended for workout pace prediction and longitudinal fitness
tracking, replacing earlier VDOT-based approaches.

## Files

- `bayes_cs_fit.py` — fits the model with PyMC HSGP. Reads `races.csv`,
  derives auto-exclusions internally, writes summary, residuals, params,
  posterior `.nc`, diagnostics, and an exclusions audit. Default invocation:
  `python bayes_cs_fit.py --tag v10`. Runtime ~8 min.
- `bayes_cs_plot.py` — renders the timeline as interactive HTML. Reads
  sibling files matching `--tag` and writes `cs_timeline{tag}.html`. Runtime
  seconds.
- `bayes_cs_auto_exclusions{tag}.csv` — written by each fit. Audit trail
  with one row per pruned race: date, distance_m, event, surface, tier,
  metric, value, threshold, n_neighbors, sigma_global. Inspect after a
  refit to see which races the rule pruned and why.

## Data flow

```
races.csv ──→ bayes_cs_fit.py ─→ bayes_cs_summary{tag}.csv
                              ├─→ bayes_cs_residuals{tag}.csv
                              ├─→ bayes_cs_params{tag}.csv
                              ├─→ bayes_cs_auto_exclusions{tag}.csv
                              ├─→ bayes_cs_posterior{tag}.nc
                              └─→ bayes_cs_diagnostics{tag}.txt
                                          │
                                          ↓
                              bayes_cs_plot.py ─→ cs_timeline{tag}.html
```

Plot is split from fit so visualization can iterate without refitting (~8min
saved per tweak). Both scripts default to sibling files relative to the
script location.

## Model architecture

### Hierarchical CS GP

```
log_cs(t) = mu_cs + log_cs_trend(t) + log_cs_dev(t)
```

- `log_cs_trend`: long-scale Matérn-5/2 GP, ℓ_long ~ LogNormal(log 5y, 0.5).
  Captures multi-year fitness drift. Posterior amplitude ~0.29.
- `log_cs_dev`: short-scale Matérn-5/2 GP, ℓ_dev ~ LogNormal(log 0.25y, 0.4).
  Captures training-cycle wiggles within seasons. Posterior amplitude ~0.06.
- HSGP basis sizes: m=100 for dev, m=33 for long.

**Why hierarchical**: at boundaries (past last race, before first race), the
short GP decays to zero quickly while the long trend persists, anchoring
extrapolations to a smooth fitness trajectory rather than letting them
follow the most recent dip.

### D' (anaerobic capacity) GP

`log_dp(t) = mu_dp + GP_dp(t)` with Matérn-5/2 kernel, ℓ_dp ~ LogN(log 0.5y).
Posterior median ~174m, **near-constant** across the timeline (range 169-177m,
std 1.5m). 95% CI is ~73m wide — the data lacks leverage to pin D' down
because Max has few sub-1500m races. D' is hidden from hover; it's a
necessary model parameter but not informative for the user.

### Likelihood

```
expected_t = (d - D') / CS · bias_factor
log τ ~ Normal(log(expected_t), σ_per_race)
```

with:

- `σ_per_race = σ_base · (d/5000)^α` — distance-dependent noise.
  Posteriors: σ_base ≈ 0.028, α ≈ 0.03.
- `bias_factor = 1 + β_long · max(0, log(d/d_thresh))` for d_thresh = 10K.
  β_long posterior ≈ 0.071. Marathon inflation ≈ +10.3%, HM ≈ +5.3%.

### XC pre-correction

XC race times are divided by `(1 + xc_correction)` BEFORE entering the model
(`--xc-correction 0.06` by default — 6% terrain penalty). This is exogenous,
not learned, because XC races cluster in fall seasons with no concurrent
non-XC races for the model to compare against — so the model can't
empirically separate terrain from fitness loss. We encode terrain
correction as a fact and let the model fit pre-corrected times naturally.

The 6% value was calibrated visually: at 4% Max's HS XC seasons still
appeared as fitness troughs vs. spring track; 6% aligns them with surrounding
track seasons consistent with his felt experience.

## Inference

- Grid: 7 days. ~940 points over the 18-year span.
- Sampler: NUTS, 4 chains × 1000 draws + 1000 tune, target_accept=0.95.
- Divergences typically <20/4000 (well under 1%). R-hat 1.00 across params.

## Plot conventions

### Anchor distance: 5K

Every race diamond projects to its 5K-equivalent pace. 5K chosen because
~92 of Max's races are 5Ks vs 7 at 10K — natural reference for the bulk
of the data.

### Projection method: hyperbolic CS, not Riegel

For each race at (d_race, t_race), un-bias the time first if d > 10K:

```
t_unbiased  = t_race / (1 + β_long · log(d_race/10K))   if d_race > 10K
            = t_race                                     otherwise
t_5K_equiv  = (5000 - D'_date) · t_unbiased / (d_race - D'_date)
```

- Race time IS the data — diamonds preserve race-to-race deviations.
- D' anchors the short-distance physiology, so 800m and mile diamonds
  project honestly without piecewise Riegel exponents.
- Un-biasing for d > 10K projects "the underlying fitness this race
  implies" rather than the raw race-execution pace (which would be
  near-asymptotic CS for marathons, making the projection useless).

### CS line interpretation

The line plots **asymptotic CS pace** (1609.344/CS_mps/60), the slowest
sustainable pace approached at infinite distance. Race diamonds at
finite distance project to **faster** pace than the line by the kinematic
D'/t boost — so most diamonds sit above the line by ~12s/mi at 5K-anchor
(below = faster on the chart since y-axis is inverted pace).

Sub-line diamonds are races that ran ~12+s/mi faster than the model
expected. Most such cases are short summer/winter road races during
off-seasons or unusual conditions. Pruning candidates are inspected
individually using the rule: "did Max have to stop/walk, or was there
significant terrain slowdown? If yes, prune; if no, keep."

### Gridlines: yearly. Y-axis range: 4:30 to 8:00 min/mi

### Tooltip format

- Date header
- CS median, 50% interval, 95% interval (no D')
- Nearest race section: distance + actual time (subsecond precision via
  `sec_to_mss_full`) + raw pace
- Equivalent line — four cases:
  - Non-XC 5K: omitted (would duplicate race time)
  - XC 5K: `XC-corrected: <time> (<pace>/mi)`
  - XC non-5K: `5K-equiv (XC-corrected): <time> (<pace>/mi)`
  - Non-XC non-5K: `5K-equiv: <time> (<pace>/mi)`

## Auto-exclusion rule (replaces manual `cs_exclusions_v7.csv`)

Implemented in `derive_exclusions()` in `bayes_cs_fit.py`. Two-tier rule
chosen empirically by inspecting the actual residual distributions of long
vs short races; long and short need different treatments because their
residual structures are different.

### Tier 1 — Marathon (≥25K) and HM (15K–24.999K)

Symmetric ±2yr same-band median residual in seconds. Prune if absolute
residual exceeds the band-specific threshold:

- **Marathon: > +115s** — kept marathons cluster from −925s up to +113s
  (Nashville 2018 debut, sparse n=2). First excluded marathon is Berlin
  2023 at +118s. Threshold of +115s sits in the +5s gap.
- **HM: > +500s** — kept HMs span −159s to +177s (Craft Classic 2024).
  Only excluded HM is Deception Pass at +1754s (extreme XC course).
  Threshold of +500s gives wide safety margin for future summer-HM
  variability.

Why symmetric (not past-only): marathons have bonk-polluted past medians
(2021 Boston/Nashville bonks pull subsequent past-medians upward, making
later 2022 bonks look "fast" by past-only). Symmetric pulls in fast races
from both sides of the date and gives a stable reference.

### Tier 2 — Sub-marathon (<15K)

Same-band past-only ±2yr median of log-pace, with `K_min ≥ 5` past races
required. Past-only protects against fitness-rise contamination — early-HS
races aren't measured against future faster races.

Global σ across all sub-marathon races via iterated trimmed MAD (3
iterations at 2% trim). Uniform scale across bands and eras avoids the
local-cluster artifacts that per-race MAD-z would produce. Empirically
σ_global ≈ 0.048 (~4.8% pace) — represents the natural spread of race-to-
race variation against own-history.

Threshold: `z = (log_resid - global_median) / σ_global > 2.5`.

### Important design note

The Tier 2 rule is **intentionally less aggressive** than the prior manual
list. Statistically, the sub-marathon residual distribution does not have a
fat tail of bonks: 95th percentile z ≈ +2.08, 99th ≈ +2.45, max ≈ +4.45
(Tahoma 2014, +20% slow). Several races that "look like" bonks visually
(WGP 2016 XC at z=2.42, WGP 2019 XC at z=1.75) sit at the same z as
currently-kept races (Frank Shorter 2025 z=2.42, Derby Days 2018 z=2.31)
and cannot be cleanly separated by data alone.

These are statistically indistinguishable from normal race-day variance and
are absorbed by the model's likelihood (σ_per_race ≈ 0.028 in log time)
without structural distortion of the CS curve. Only Tahoma 2014 is a
genuine sub-marathon outlier.

This means the auto-rule produces ~11 exclusions vs the prior ~16. The
five not auto-pruned (Nike 2008, Tahoma 2013, WGP 2016 XC, Color Run 2016,
WGP 2019 XC) stay in the fit. If a future race is a clear bonk it will
land at z > 2.5 and auto-prune; otherwise the model handles it.

### Tunable parameters

`derive_exclusions()` exposes thresholds as keyword args:
`tier1_marathon_thresh=115.0`, `tier1_hm_thresh=500.0`, `tier2_z_thresh=2.5`,
`tier2_kmin=5`, `window_years=2.0`, `trim_pct=0.02`, `sigma_iters=3`. Edit
defaults in the function signature if recalibration is needed.



- **Hardcoded marathon correction** (e.g., "all marathons +6%"). Rejected
  because Max's marathons are legitimately bad due to mental/logistical
  factors, not a fixed physiological law. A learnable β_long (centered at
  0) lets the model update if a future well-executed marathon arrives.
- **Learnable β_xc inside the model**. Tried; posterior settled at
  β_xc ≈ 0.0075 because the model's CS GP absorbs XC slowness as fitness
  loss instead of as terrain penalty (no concurrent non-XC data to
  contrast against). Replaced with exogenous pre-correction.
- **Sub-7d grid resolution**. Considered 3d/1d. Skipped — the marathon-
  bias issue dominates over resolution; finer grid amplifies noise more
  than it reveals signal. β_long fix made the existing 7d grid work.
- **Adding 400m races back to the fit**. Would help pin D' down but
  imports sprint physiology that the hyperbolic model doesn't capture
  cleanly. Net negative for CS estimates.
- **Trend-only line displayed separately** on the chart. Removed —
  the total median is the actionable signal; trend underneath is internal
  scaffolding.
- **Visualizing diamond confidence by interval width or fading line at
  boundaries**. Considered as alternatives to fixing the boundary trough
  via hierarchical GP. Rejected; hierarchical fix was structural and
  symmetric (handles both pre-data and post-data boundaries).
- **Per-band MAD-z for auto-exclusions** (z = resid / local-band MAD).
  Tried during the auto-exclusion design pass. Rejected because tightly-
  clustered modern eras (2016-2018 5Ks, post-2022 track) had small local
  MADs that flagged ordinary slow days as 5+ MADs out (e.g. Derby Days
  2018 at per-band z=6.49 on hilly summer course). Replaced by global σ
  (iterated trimmed MAD over the full sub-marathon pool) which is uniform
  across bands and eras.
- **Riegel-merged 5K-equivalent for sub-marathon outlier detection**.
  Tried merging 1500m-10K via Riegel exponent 1.06 to give 3K bonks like
  Tahoma 2013 a same-band reference (since per-band 1500-3499m had no
  past data for it). Caught Tahoma 2013 at z=4.20 but introduced multiple
  false positives (Resolution Run 2025 at z=7.39, Pies/Derby Days/Turkey
  Trot kept races at z=2.5-4.3) because course-difficulty variance got
  averaged across all 5K-eq projections. Rejected; the Riegel-merged
  metric's signal-to-noise is worse than per-band metric's despite
  higher data density.
- **Asymmetric Tier 1 thresholds for HM and Marathon**. Marathon and HM
  needed split thresholds (+115s vs +500s) because their kept-residual
  spreads differ structurally: marathons cluster tightly at good days
  (kept_max +113), HMs span wider (kept_max +177). A single threshold
  would either miss Berlin 2023 (+118 marathon) or prune Craft Classic
  HMs (+151, +177). Cleanest fix was split thresholds, accepted as a
  permanent design choice.
- **Single Tier 2 z-threshold catching all "obvious" XC bonks** (e.g.
  WGP 2016 XC, WGP 2019 XC, Tahoma 2013). Rejected because at any z
  catching them, the rule also prunes equivalently-residualized kept
  races (Frank Shorter 2025, WGP 2019 Road, etc.). The data does not
  separate these races. The rule errs toward keeping them in the fit;
  σ_per_race ≈ 0.028 absorbs ~3σ events without distortion.

## Working assumptions

- D' is stable over a competitive runner's career — the model's flat
  posterior matches physiology; we don't need to extract D' trends.
- CS is the right fitness scale because it correlates with race
  performance across all common distances when bias-corrected, and it's
  what training paces should be calibrated against.
- Future races coming in will update CS via continued fits. Tag
  convention: `--tag YYYYMMDD` or `--tag vN` per refit.

## Next thread

Workout-pace overlay: project quality workout segments
(`quality_distance_m`, `quality_pace_sec_per_mi` from `daily.csv`) onto
the CS line using the same hyperbolic+5K projection. Replaces the older
VDOT-based workout chart (`make_vdot_plots.py`). Workouts will sit above
the CS line by a workout-vs-race conversion gap — visualizing that gap
is the goal.

Open questions when starting: (1) overlay on existing chart vs separate
plot, (2) smoothing of workout points, (3) which `quality_segment_type`
values to include.
