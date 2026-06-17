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
expected_t = (d - D') / CS          (pure CP2; no fade term)
log τ ~ Normal(log(expected_t), σ_per_race)
```

with `σ_per_race = σ_base · (d/5000)^α` — distance-dependent noise
(σ_base ≈ 0.028, α ≈ 0.03).

**`β_long` is RETIRED (June 2026 — the IAAF hybrid, see "Cross-distance
equivalence" below).** The fit no longer carries a long-distance fade term.
Instead, every race **above 5K is down-converted to its 5K-equivalent via the
World Athletics scoring tables BEFORE the fit** (in the race-ingestion block of
`bayes_cs_fit.py`), so the model only ever sees efforts ≤ 5K and `bias_factor`
is identically 1. `bayes_cs_params.csv` still writes `beta_long_med = 0.0`
(no-op) for the handful of consumers that still unpack it from
`load_cs_outputs`; that plumbing is vestigial.

### XC pre-correction

XC race times are divided by `(1 + xc_correction)` BEFORE entering the model
(`--xc-correction 0.08` by default — 8% terrain penalty). This is exogenous,
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

### Projection method: 3-parameter critical speed (CP3), not Riegel

**June 2026 — CP3 replaced the plain hyperbola in the projection layer**
(the fit itself stays 2-parameter; it excludes < 1500 m where the models
differ). Each race is read as a sample of its own Morton curve
`v(t) = CS + D′/(t + τ)`, `τ = D′/(v_max − CS)` — the hyperbola bent at
short durations by a finite top speed v_max. **v_max is an uncertainty
interval, two conservative edges per profile** (cs_projection registry;
Max: evidence 9.5 / prediction 8.3, derived by scripts/calibrate_vmax.py
from his own corpus — the 400-corpus sprint-credit envelope and the
PR-sweep bounds respectively). CP3 replaced two disconnected short-effort
corrections (the race-side β_short knob and the workout-side g(d)) with one
effort-aware anaerobic term, so a 400m race and a 400m rep now analyze
identically. Reading a race as evidence
uses the HIGH edge (short diamonds conservative — every 400 ever raced
sits at/behind the frontier by construction); forward solves (dashboard
predictions, lines-at-anchor) use the LOW edge (400/800 predictions never
beat the lifetime PRs at any date in the sweep).

**CP3 + v_max is the UP-conversion only (effective distance ≤ 5K).** For races
at/below 5K, solve the race's own implied CS (closed-form quadratic —
SELF-consistent, the fit's CS never enters) and forward-solve the 5K time on
that curve. The fit's D′ is bridged per date as the effective anaerobic
distance at 5K (`D′₃ = D′₂/(1 − D′₂/((v_max−CS)·t5K))`, per edge), which
preserves the fit's 5K prediction exactly and keeps the models within ~1 s/mi
over 3000m–10K; miles read ~2.5–4 s/mi fitter than under CP2; sub-800m efforts
are where the bend really acts.

**Above 5K, the DOWN-conversion is World Athletics, not CP3** (see "Cross-
distance equivalence"). The old `β_long` un-bias is gone.

- Race time IS the data — diamonds preserve race-to-race deviations.
- The bend makes 400m/800m diamonds and predictions honest with ONE
  physiological parameter — the former β_short display knob (two outcome-
  pinned constants below 875 m) is retired, as is the workout-side g(d).

### Cross-distance equivalence: the World Athletics hybrid (June 2026)

The single rule that homogenises every effort (race, long run, workout) to a
5K-equivalent — replacing the fitted `β_long` fade entirely:

- **Effective distance > 5K → World Athletics down-conversion.** Score the
  (corrected) time on the WA 2025 men's tables, then read the equivalent 5K
  time at the same score. Aerobic-to-aerobic, empirical, no fitted parameter.
- **Effective distance ≤ 5K → CP3 + v_max up-conversion** (above).
- **5K is the shared anchor** — both regimes give 5K = 5K, so they meet
  continuously; no kink at the boundary.

**Why the split, not all-IAAF.** WA tables equate *specialists across events*
(a world-class 400 and a world-class 5K both score ~1300). Using them to
convert one athlete's 400 to their 5K assumes a population-average
anaerobic↔aerobic balance, which a distance specialist doesn't have — it under-
rates his short races by ~1–2 min of 5K-equivalent (his 57s 400 → 16:58, not a
fitness statement). The equivalence is only valid where both efforts tax the
*same* system. So WA handles the aerobic (>5K) side it's good at; CP3 + v_max —
calibrated on his own corpus — handles the short side IAAF can't.

This fixed three things at once that the single-`β_long`-log couldn't:
over-credited HMs, a 10K "cliff" (10K projecting slower than both its 5K and HM
neighbours), and the marathon-only calibration leaking onto every other
distance. The CS refit on the WA-grounded races came out ~5 s/mi faster in
recent years (HMs now credited as the strength they are) and smooth.

Implementation: `src/shared/wa_scoring.py` (coefficients, `wa_points`,
`wa_5k_equiv_time`, distance interpolation on the dense road ladder
5K/10K/15K/20K/HM/25K/30K/marathon, and the long-run pause penalty below).

### Long-run pause penalty (durability-gated drift)

Long runs (all > 5K → WA) carry one extra adjustment a continuous race doesn't:
a paused run's **moving** pace overstates the pace sustainable continuously,
because stoplight pauses reset the fast (HR / W′) component of cardiovascular
drift. The penalty (added to the time before the WA score):

```
penalty_frac = k · (1 + temp_amp) · max(0, i − i₀(T)) · φ_fast · (T/2) · r_eff
```
- `i = run speed / CS speed` (intensity).
- `i₀(T)` is **duration-dependent** (durability): the sustainable fraction of
  CS collapses with time-on-feet — ~0.90 for short efforts descending to ~0.82
  at marathon duration (marathoners race 82–88% CS and fade from 26–33 km; CP
  itself drops ~10% after 120 min). This is what makes a 2–3 h, ~0.91-CS long
  run register as the near-limit effort it is, rather than "barely heavy."
- `temp_amp = 0.025·max(0, T_C − 12)` — heat amplifies drift (Tucker/Wingo),
  and is *designed* to net against the warm-day temperature credit the long-run
  model separately applies.
- Upper/conservative band (`k=0.5, φ_fast=0.6, r_eff=0.8`), per the standing
  rule to be conservative making fitness claims from non-races.

Only the FAST component is penalised (the slow component — glycogen, core temp,
plasma volume — both paused and continuous runs pay, so it cancels). Easy long
runs (i below their duration-onset) get zero penalty. Net effect on the
frontier: exactly one long run (a peak 2023 effort) tops the best HM race, by
~2 s/mi; the rest cluster at or below it.

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

### Performance frontier (purple line, June 2026)

The Fitness plot carries a second line: the **performance frontier** — the
demonstrated-5K-capability envelope (`src/shared/performance_frontier.py`).
Semantics: every kept TQ point and eligible race 5K-equivalent is PROOF of
5K capability at its date; the frontier answers "how fast could I have run
a 5K that day, given surrounding performances" — a race-PREDICTION line,
deliberately more responsive than CS, with no CI (an envelope over
evidence, not a posterior; accuracy is owned by upstream point selection).

**Formulation: floor = the CS-implied 5K prediction** ("I'm at least as
fast as CS predicts" — CS's decay structure is the rigorously verified
part, so the frontier inherits it). Demonstrations contribute only their
EXCESS above that floor, weighted by a super-Gaussian shoulder
`exp(−(|Δt|/τ)^4)` (τ ≈ 38d forward / 32d backward): excess holds near
peak level for ~a month on both sides of a peak, then dies decisively by
~8 weeks — calibrated June 2026 on the corpus's own top-decile-excess
peaks, and consistent with the physiology end-to-end (build plateaus into
a peak; taper-hold after; decisive detraining on the ~6-week scale). The
line rides the CS-5K curve through quiet stretches and bulges faster where
something super-CS was demonstrated. No gap machinery: stale proofs die by
shape, and the floor itself carries long absences (the 2020-21 labrum
crash is in the race fit). Two earlier arm designs are dead — linear cones
(zigzag; no physiological process is linear) and bounded-loss exponential
cones (saturated "immortal tails": a 2017 peak clamped all of 2018 flat).
A point that binds the envelope DEFINES it locally — suspicious bulges are
audited by deep-diving their binding demonstration. EVERY non-race
demonstration above the floor is rendered (legend group "Frontier
workouts"), styled exactly as Training renders its session markers
(per-category colors, small dots beside the larger race diamonds); binding
points carry a "sets the frontier" hover note. A faint gold "CS 5K
prediction" line (the floor itself, same treatment as the Long Runs HM
line) makes frontier-vs-floor divergence readable; the per-day tooltip
gains a "5K frontier" row.

The frontier renders **vibrant purple** (`FRONTIER_LINE = #9933ff`, the
tab-shell accent; it was briefly red). On Fitness the asymptotic CS line is
DEMOTED to a faint gold reference — its CrI ribbons are gone — and the
**frontier carries the 95% band instead**: the frontier swept across the CS
CrI (purple fill, so band and line read as one object; collapses where a
demonstration binds, equals the CS CrI on the floor). The tooltip leads
with the frontier + band; CS median is a context row.

**Rollout (June 2026), per Max's per-tab decisions.** `standard_demos()`
is the shared canonical demo set; every consumer computes the identical
line. Fitness: home tab (above). **Races + Race distances: the frontier is
the ONLY line** — gold CS lines removed entirely; the hand-drawn pre-2013
cubic survives invisibly as the frontier's floor in that era (the GP isn't
really estimating CS there); per-panel lines via `frontier_at_anchor()`
(CP3 forward solve for anchors ≤5K, World Athletics up-conversion for anchors
>5K — both β_short and β_long retired, June 2026); hover Δs are vs the frontier. Training: purple line added (normalized mode shows
frontier−CS-5K excess at/below zero). Workouts: purple 5K line. Long runs:
frontier marathon (bright purple) + HM (faint purple) — the gold CS pair
is REMOVED (frontier lines only; tooltip rows are frontier values).
Recovery: deliberately untouched (recovery calibrates against sustainable
physiology, not peak capability). **Demo eligibility: time ≥ 120 s — 800s
bind by design** (June 2026; a ≥ 1500 m cutoff was tried and reverted the
same day). Max's 800s bind in 5 of 7 non-fatigued cases; the
investigation showed this is the fit's own small D′ speaking (relative to
HIS anaerobic capacity a 2:04.1 really is peak-level — the IAAF "weak at
short" comparison is population-relative and already encoded in D′), and
Max verified the binding cases are coherent peaks (the 2017-03-30 800 was
his genuine lifetime best to that point, superseded weeks later by
1600s). Only sub-120 s sprints stay display-only, read via the
conservative evidence-edge v_max. **Dashboard: predictions are direct
frontier projections** — "the fastest I could physically run this, given
the current frontier", no empirical residual offsets (the old
recency-weighted long-run residual machinery is gone) — evaluated AS OF
TODAY, not the fit grid's extrapolated end (frontier excess decays on the
~6-week scale, unlike CS). The prediction band is the **frontier swept
across the CS 95% CrI**: where a recent demonstration binds, the sweeps
collapse onto it (proof pins the prediction — e.g. ±11s on the 5K twelve
days after a binding HM); on the floor the band equals the CS CrI. Inputs: race diamonds (post-exclusion, XC-corrected, β-unbiased) +
`data/training_quality_corpus.csv` (kept TQ corpus, persisted by
plot_training_quality.py — which therefore runs BEFORE bayes_cs_plot in
run_plots.sh). Frontier lives in 5K-equivalent pace space (diamond space),
not asymptotic-CS space.

**Why the frontier and not workout-enrichment of the fit (guardrail).** An
earlier line of work tried feeding near-race training observations into the
CS *likelihood* itself. It was halted at validation: the in-band efforts
carry a one-sided sub-max effort bias (~+2.7 s/mi, era-modulated) that a
symmetric selection band cannot remove, so on a hold-out (2019-06→2020-12
races removed, workouts filling the gap) enrichment made held-out race
predictions *worse* (mean |err| 2.49% → 3.26%). Probabilistic one-sided
variants (ExGaussian / Tobit / latent-slack / quantile delta-fits) all
collapsed to "wide symmetric noise" whenever the noise was estimable. The
reframing that shipped: workouts are PROOF of capability, not noisy
observations of it — hence the deterministic frontier above (no posterior,
no CI). `bayes_cs_fit.py --workout-obs` survives as a flag-gated, default-off
relic. **Do not re-open likelihood enrichment without addressing the
level-bias finding.**

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



- **Hardcoded marathon correction** (e.g., "all marathons +6%"). Rejected,
  and the learnable `β_long` that replaced it is now ALSO retired (June 2026):
  the marathon-only single-log fade over-credited HMs, created a 10K cliff, and
  leaked the marathon calibration onto every distance. Replaced by the World
  Athletics hybrid — marathons (and all >5K efforts) down-convert to their
  empirical 5K-equivalent before the fit, so no in-model marathon correction is
  needed at all. See "Cross-distance equivalence."
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

## Race-covariate spike (June 2026) — checked, mostly null

Question: can race INPUTS to the CS fit defensibly carry covariate
adjustments? Method: residuals = race 5K-equivalent (fit conventions: XC
correction + β_long) minus the CS posterior at that date; distance-bin
intercepts; MAD prune; ≥1500 m. Structural caveat first: races DEFINE the
CS curve, so any covariate that varies smoothly in time (season/temp
cycles, training-block fatigue) is partially absorbed by the fit and its
beta attenuated — only transients sharp relative to the CS smoothing
timescale are cleanly visible. Findings:

- **Fatigue:** the recovery-style days-since decay is null beyond
  same-day (β −0.2 ± 2.9, LOO straddles zero — n is sufficient; there's
  just nothing there at 2+ days). **Same-day seconds** (race_seq ≥ 2,
  CS-excluded so out-of-sample) show a real penalty: mean **+7.2 s/mi,
  sd 9.0, n = 10** (range −5.5…+22.5; all-comers/HS doubles). Group
  effect ≈ 2.5σ, but per-race correction would be indefensible at sd 9 —
  and the within-day gap/first-race distance aren't logged. The
  exclusion policy stands; the number is now quantified.
- **Temperature, 5K bin** (the most promising slice a priori: n = 56
  with temp): β **−0.02 ± 0.17** — null, and since adjacent 5Ks weeks
  apart see real temp swings the CS curve can't track, this is only
  mildly attenuated: the 2σ band [−0.36, +0.32] sits below the long-run
  beta (+0.37). 5K race temp sensitivity is genuinely smaller than
  long-run sensitivity — effort compensates at race intensity. 3200:
  −0.25 ± 0.15 (warm-trending-FASTER — summer-track season confound).
  1600: fit degenerates (MAD prune keeps 12/26 near-identical track
  miles; its "significant" beta is an artifact, not a finding).
  10k+HM pooled: +0.08 ± 0.25, n = 18. No usable temp beta anywhere.
- **Elevation gain: categorical regression CLOSED (June 2026) — but a
  per-run measured correction now SHIPS** (see "Race physical correction"
  below). The *categorical/whole-corpus* approach was never fittable
  (26/245 race rows carry elev_per_mile, and it's location-level, not
  course-level; only ~28 road races are even fillable by log-location
  key, nearly all flat), and its acid test failed: Deception Pass HM
  (2016-04-02, 1:51:32, ~275 ft/mi by Max's recollection) stays excluded
  under every defensible correction. The full calculation: Minetti net
  factor for 3600 ft up+down over the HM = 1.138 (a 13.8% time penalty —
  bigger than the 8% XC correction); stacking terrain × climb (1.08 ×
  1.138 = 1.229) brings the residual from +114 to +61 s/mi at the 5K
  anchor — ~+866s on the exclusion metric, still well over the +500s HM
  threshold. Landing at normal race scatter (+15 s/mi) would need a
  total factor of 1.40 (~6,500 ft at the Minetti curve): the remainder
  is stairs/bottlenecked singletrack/sand that no grade-or-footing
  model prices in. That race is excluded because it genuinely isn't
  CS-informative, not because the model lacks a term. **What changed:**
  the categorical regression stays closed, but for races WITH watch
  coverage a per-run DEM-based physical correction now ships and feeds
  CS directly — superseding the "not fittable" conclusion *for the
  watch-covered subset*. The categorical rules (XC ×1.08, Downhill
  hard-exclusion) survive as the pre-watch fallback.

One durable insight from the Deception Pass decomposition: the **XC
correction is FOOTING, not elevation** — the hill model's trail term
(+27 s/mi ≈ 8% at those paces) independently matches the 8% XC factor,
so terrain × Minetti stacking is a correct decomposition, not
double-counting (XC courses' own rolling terrain is mild enough that
the 8% absorbs it).

Bottom line: **race times stay unadjusted, all threads closed** —
fatigue (null beyond same-day; same-day quantified at +7.2 s/mi and
already excluded), temperature (null everywhere, 5K hypothesis
rejected), elevation (the one motivating race is unsalvageable; the
fillable remainder is flat). Re-affirms the earlier no-beta-adjustments
decision, now with numbers.

## Race physical correction (June 2026, SHIPPED)

Supersedes the "elevation not fittable" conclusion above **for races with
watch coverage**: the categorical regression stays closed, but a per-run
measured correction now corrects each watch-covered race's TIME for grade +
off-road footing + altitude to its flat / sea-level / smooth-equivalent
BEFORE it informs CS, so the demonstrated-capability frontier measures
fitness, not the course. (Engine: the `elevation_cost` cost model,
`physical_route_betas`, the altitude threshold curve, the DEM mechanics —
→ see route-normalization-reference.md (elevation engine).)

- **Single source of truth: `recovery_model.race_physical_correction(races)`**.
  Applied IDENTICALLY in two places that must stay consistent: **(a)** the CS
  fit (`bayes_cs_fit.py`) — subtracted from `race_times` before the β_long
  un-bias and the likelihood; `build_eligible` now ADMITS a watch-covered
  Downhill race (the measured grade discounts its assisted time) where it
  used to hard-exclude all Downhill; `derive_exclusions` applies the same
  correction so exclusion residuals match. **(b)**
  `cs_projection.project_races_to_5k_pace` — the displayed race diamonds and
  the performance frontier they feed, gated by an `apply_physical_correction`
  flag (default on; OFF for the actual-pace race plots in `make_race_plots`).
  If the fit and the projection disagreed, the plotted race position wouldn't
  match what fed CS.
- **Replaces the categorical ONLY WHERE WATCH DATA EXISTS.** It supersedes
  the categorical XC ×1.08 pre-correction and the Downhill hard-exclusion
  only for watch-covered races; the categorical rules remain the pre-watch
  fallback. Today no XC or Downhill race has watch coverage, so the
  replacement paths are armed but no-op — all current XC still get ×1.08, and
  the one pre-watch Downhill TT stays excluded.
- **Sign convention.** A net-DOWNHILL race (e.g. Boston) gets time ADDED, so
  it projects SLOWER and correctly discounts the assisted time for CS.
  Net-uphill / altitude races are credited faster.
- **Track races: grade gated OFF** (by surface — flat, barometric noise) but
  the **altitude (hypoxia) term still applies** (a Boulder track race is
  altitude-suppressed).
- **Race effort ≈ 1.0 by construction**, so paved descents refund at the race
  edge (`paved_refund(1.0) ≈ 0.85`) — grade bites hardest at race effort.
- **Inputs.** Per-race grade/altitude come from a DEM resampled along the
  watch GPS track (the barometric per-race net is noise). Footing + altitude
  betas are pinned from `physical_route_betas` (the same constants recovery
  and long runs use); altitude uses a threshold curve (zero below ~3000 ft).
- **Refit results.** ~96% 95%-coverage, residual bands tight; the dominant CS
  change was 2020 getting ~5.5 s/mi slower (a previously-mis-admitted
  downhill-assisted mile removed). The per-race correction is a PyMC
  model-input change.

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
