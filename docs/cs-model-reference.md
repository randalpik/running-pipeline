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

### Hierarchical latent-fitness GP

The fit is a **single latent process on log 5K-equivalent TIME** (June 2026
redesign — it was two GPs, CS(t) and D'(t), with a hyperbolic likelihood):

```
log_t5k(t) = mu_fit + log_fit_trend(t) + log_fit_dev(t)
```

- `log_fit_trend`: long-scale Matérn-5/2 GP, ℓ_long ~ LogNormal(log 5y, 0.5).
  Multi-year fitness arc. Posterior amplitude ~0.3.
- `log_fit_dev`: short-scale Matérn-5/2 GP, ℓ_dev ~ LogNormal(log 0.25y, 0.4).
  Training-cycle wiggles within seasons. Posterior amplitude ~0.04.
- HSGP basis sizes: m=100 for dev, m=33 for long.

**Why hierarchical**: at boundaries (past last race, before first race) the
short GP decays to zero while the long trend persists, anchoring extrapolation
to a smooth trajectory rather than the most recent dip.

### Why a single 5K-equiv fitness curve (D' retired as a fitted parameter)

The model used to fit two GPs — CS(t) and D'(t) — through the CP2 hyperbola
`t=(d−D')/CS`. Two findings retired that:

1. **D' time-variation was never identified.** Its GP amplitude posterior hugged
   zero; fitted D'(t) was flat (~149±3 m over 13 years). Pinning a time-varying
   D' needs several distances clustered in time, which Max's racing rarely gives.
2. **A single hyperbola can't match IAAF's empirical shape across 800 m–5K**, so
   it over-rated short races (the mile read ~10 s/mi fitter than IAAF, the 800
   ~22). Once every aerobic race is IAAF-homogenized to a 5K-equivalent (below),
   all observations sit at one distance and the hyperbola is degenerate — only
   `(5000−D')/CS` is identified, not CS and D' separately.

So the fit is now a single latent 5K-equiv fitness curve. **D' survives only as
a fixed constant** (`cs_projection.dprime_fixed`, 150 m for Max — the historical
fitted median) doing two jobs: backing out a nominal bare-CS = `(5000−D')/t5k`
for the recovery/long-run baselines, and feeding the CP3 sub-1500 sprint leg.

### Likelihood

```
log t5k_i ~ Normal(log T5K(t_i), σ_base)
```

where `t5k_i` is each AEROBIC race (≥1500 m) down-converted to its
5K-equivalent via the World Athletics tables (identity at 5K). σ is **uniform**
now — every observation is a 5K-equiv at one distance, so the old
distance-scaling term α is retired (σ_base ≈ 0.037).

**Sub-1500 m sprints (400/800) are EXCLUDED from the fit** — IAAF's
population-average equivalence distorts a distance specialist there, and they're
frontier demos, not aerobic-fitness anchors. **`β_long` is also retired** (every
race >5K already down-converts to 5K-equiv); `bayes_cs_params.csv` still writes
`beta_long_med = 0.0` (no-op) for vestigial consumers.

### Output schema (legacy compatibility)

The summary keeps the old column names so downstream code is unchanged:
`cs_mps_med = (5000−D'_fixed)/t5k`, `dp_med = D'_fixed` (constant), and
`load_cs_outputs` derives `p5k_implied =` the latent 5K-equiv pace.

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
short durations by a finite top speed v_max. CP3 replaced two disconnected
short-effort corrections (the race-side β_short knob and the workout-side
g(d)) with one effort-aware anaerobic term, so a 400m race and a 400m rep
now analyze identically.

**v_max is not estimated — it's bracketed.** It's under-identified from the
data (a runner's per-race implied v_max scatters wildly — execution noise, not
a trend — and top-end speed is trained and decays at an unknown rate), so the
model never claims a value. Instead v_max is tied to the CS we *are* confident
in, as **two conservative CS-MULTIPLES per profile** (`v_max = k · CS(t)`),
each the **binding extreme of a constraint against demonstrated performance**.
For Max: **k_evid 1.97 / k_pred 1.53** (run `python scripts/calibrate_vmax.py`
to see the binding races). A multiple — not a flat m/s — transfers across
profiles and avoids a flat v_max sitting too close to an athlete's
(era-stable) sprint speed whenever CS dips, which over-credits short races (a
59.3 400 → a 15:32 5K, a time Max wouldn't reach for four years).

- **k_evid (evidence / reading a race DOWN as 5K proof — HIGH).** The
  *smallest* multiple that keeps every short (< 700 m) race behind the
  **aerobic (≥ 1500 m) race frontier** — the races reliably WA-convertible to a
  5K. So a sprint never converts to a 5K faster than demonstrated aerobic
  fitness, and **"a 400 never defines the frontier" is true by construction**
  (no special-case rule). The 400 is held against the aerobic *races*, never
  against its own 800 — and because the ≥ 1500 m frontier is pure World
  Athletics (v_max-independent), the derivation is a clean monotonic root-find
  with no fixed-point. 800s are neither the reference (a model-projected short
  race isn't a yardstick) nor constrained (they still bind the displayed
  frontier as genuine demos), so a same-session 400 and 800 separate naturally
  rather than being pinned together.
- **k_pred (prediction / forward-solving a short time UP — LOW).** The
  *largest* multiple whose 400/800 predictions never beat the lifetime PRs
  (binding at peak fitness). The gap to k_evid is the irreducible v_max
  uncertainty we lack the data to close.

The ratios are **derived automatically each build** by
`scripts/calibrate_vmax.py` (after the CS fit, before the workout/plot steps)
and written to `data/vmax_ratios.csv`, which cs_projection reads — no manual
tuning. The per-profile `VMAX_*_CS_RATIO_BY_PROFILE` dicts and the conservative
defaults (**k_evid 2.00 / k_pred 1.50**, for a profile with no qualifying short
race yet) are the fallback when that artifact is absent. One genuine short race
pins both edges; the defaults bracket wider than any derived pair, so the first
real race only tightens the bracket inward.

**CP3 + v_max is the UP-conversion for the SPRINT leg only (< 1500 m).** The
IAAF↔CP3 boundary moved from 5K to **1500 m** (June 2026 redesign): IAAF beats a
single CP2 curve across 1500 m–5K — it puts Max's mile just behind his 3200,
matching self-knowledge, whereas one hyperbola over-rated the mile by ~10 s/mi —
so CP3 + v_max is now reserved for the specialist-distorted sprint end. A
sub-1500 race converts in **two legs**: solve the race's own implied CS
(closed-form quadratic — SELF-consistent, the fitness curve never enters),
forward-solve to 1500 m on that curve, then hand off to the WA tables
1500 m → 5K. Continuous at 1500 m. The reservoir uses the FIXED D′
(`dprime_fixed`, 150 m), bridged per edge (`D′₃ = D′₂/(1 − D′₂/((v_max−CS)·t5K))`).

**At and above 1500 m, the conversion is World Athletics** (see "Cross-distance
equivalence"). The old `β_long` un-bias is gone.

- Race time IS the data — diamonds preserve race-to-race deviations.
- The bend makes 400m/800m diamonds and predictions honest with ONE
  physiological parameter — the former β_short display knob (two outcome-
  pinned constants below 875 m) is retired, as is the workout-side g(d).

### Cross-distance equivalence: the World Athletics hybrid (June 2026)

The single rule that homogenises every effort (race, long run, workout) to a
5K-equivalent — replacing the fitted `β_long` fade entirely:

- **Distance ≥ 1500 m → World Athletics.** Score the (corrected) time on the WA
  2025 men's tables, read the equivalent 5K time at the same score. Identity at
  5K; **Riegel power-law fallback** for performances off the bottom of the table
  (the WA points parabola vertexes at ~24:00 for a 5K — slower early-career
  races would otherwise clamp to a ~7:00/mi floor). Aerobic-to-aerobic,
  empirical, no fitted parameter.
- **Distance < 1500 m → CP3 + v_max → 1500 m, then WA → 5K** (above).
- **1500 m is the boundary** — a 1500 m race converts identically either way, so
  the two regimes meet continuously; no kink.

**Why the split, and why the boundary is 1500 m (not 5K).** WA tables equate
*specialists across events* (a world-class 400 and a world-class 5K both score
~1300). Using them to convert a distance specialist's true SPRINT (400/800) to
5K assumes a population-average anaerobic↔aerobic balance he doesn't have — it
under-rates those by ~1–2 min of 5K-equivalent (his 57 s 400 → 16:58, not a
fitness statement), so CP3 + v_max — with the per-profile evidence multiple —
owns the sprint leg. But that distortion is a *sprint* phenomenon: across 1500 m–5K both
efforts are aerobic and IAAF is validated against his own event hierarchy
(mile↔3200↔5K), where it beats a single CP2 curve. The earlier 5K boundary
applied sprint logic to the mile and over-rated it; 1500 m is where IAAF becomes
trustworthy.

This fixed three things at once that the single-`β_long`-log couldn't:
over-credited HMs, a 10K "cliff" (10K projecting slower than both its 5K and HM
neighbours), and the marathon-only calibration leaking onto every other
distance. The CS refit on the WA-grounded races came out ~5 s/mi faster in
recent years (HMs now credited as the strength they are) and smooth.

Implementation: `src/shared/wa_scoring.py` (coefficients, `wa_points`,
`wa_5k_equiv_time`). Non-anchor distances interpolate on a **monotone cubic
(PCHIP) through the real race anchors — 5K/10K/HM/marathon — in log-distance/
log-time**; the intermediate WA road tabs (15/20/25/30 k) were dropped as a
track/road mix that made the curve non-monotone at the HM tab (a spike that could
let the long-run erosion below backfire). The smooth curve is exact at the four
anchors, so no race conversion moved. The long-run pause handling (below) lives
separately in `src/shared/durability.py`.

### Direction matters: down-convert evidence (1500 m boundary), up-convert predictions (5 K boundary)

The hybrid above is the **down-conversion**: an actual effort → its 5K-equivalent,
for the fit and for placing demonstrations on the frontier. **Predictions run the
other way** — the frontier's 5K *capability* → a predicted time at a target
distance — and the two directions deliberately use **different WA/CP3 boundaries**.
This asymmetry is intentional; do not "unify" the two onto one boundary.

| Direction | Used by | Boundary | Below boundary | At/above boundary |
|---|---|---|---|---|
| **Down** (effort → 5K-equiv) | the fit; race-diamond placement | **1500 m** | CP3 + v_max (evidence edge) → 1500 m, then WA → 5K | World Athletics |
| **Up** (5K capability → target) | dashboard predictions; `frontier_at_anchor` / `pace5k_series_to_anchor` lines-at-anchor | **5 K** | CP3 + v_max (prediction edge) forward solve from 5K | World Athletics |

**Why the up-conversion keeps CP3 all the way to 5 K** (not 1500 m). WA up-conversion
to a *short* distance assumes the athlete has population-typical leg speed for his
aerobic level. He doesn't — projecting his (aerobically-set) 5K frontier up to
800 / 1500 / mile via WA over-predicts, beating his own flat track PRs by several
seconds (mile −7 s, 1500 −5 s, 800 −1 s in the June 2026 check). CP3 + v_max instead
caps every short prediction at the conservative LOW edge (`k_pred · CS`, the
largest multiple whose predictions don't beat his PRs), so a forward solve never
claims sprint speed he hasn't shown. Down-conversion
has no such problem: a real 1500 m race already encodes his actual 1500 ability, so
WA reads it honestly from 1500 m up. (So the diamond at a short anchor — a
down-converted race — and the prediction line through it — an up-converted
capability — are *legitimately different objects*; they need not coincide.)

**At and above 5 K both directions are WA** — the aerobic fade is symmetric and
validated, so a marathon down-converts and a marathon prediction up-converts through
the same tables. This is why the **long-run pace prediction is WA**: a multi-hour
effort is a > 5K up-conversion. (The retired `β_long` had left a `cp3_time × β_long`
path on this prediction that, with `β_long = 0`, was bare CP3 — which asymptotes to
CS *from above* and so predicted a 2-hour pace *faster than CS*. Routing it through
the up-conversion fixed it.)

Implementation: down-conversion is `t5k_to_anchor_time` / `project_races_to_5k_pace`
(1500 m); up-conversion is `pace5k_series_to_anchor` and `cs_line_at_anchor` (5 K).
The shared `t5k_to_anchor_time` is the 1500 m (evidence) convention — used for race
placement and for the long-run (> 5K, where it is pure WA either way); it must **not**
be used for ≤ 5K predictions.

### Long-run pause handling (pause-uncertainty erosion)

Long runs are usually **paused** (stoplights, water, regroups), and a paused run
cannot be trusted as proof of continuous capability the way an unbroken one can.
The projection therefore **erodes the demonstrated distance** before the WA score
(`durability.eroded_deff`): each pause shrinks all *subsequent* confirmed distance
by `exp(−gate · RATE · pause_sec · lateness)` — driven by pause **LENGTH** (not
count) and **LATENESS**, scaled by an **uncapped effort gate** so only
near/over-race-pace runs are touched and the easy cloud rides on pace. Watch runs
use measured stops; pre-watch runs impute the global **P90** stop structure (the
uncertainty of an unobserved run).

This is an **uncertainty model, not a physical recovery model.** A prior
"physical" pause penalty — a durability + W′-balance marginal-stop-value
simulation — was **retired**: priced at the W′-limited redline it over-credited
*every* long run, violated conservation, and was non-monotone. (It also
superseded an even earlier fixed-cadence cardiovascular-drift formula, since
removed.)

> **Full model, the rejected physical attempts, the knobs, the monotone-WA fix,
> and the 2020-11-06 hard case: `docs/long-run-pause-uncertainty-reference.md`.
> Read it before changing long-run pause handling.**

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
physiology, not peak capability). **Demo eligibility: time ≥ 120 s** (800s
included). **800s CAN bind the frontier** as genuine aerobic-ish demonstrations
— currently the 2016-03-31 800 does. This is by design: the evidence multiple
`k_evid` is calibrated so only sub-700 m races (the 400s) are forced behind the
aerobic (≥ 1500 m) frontier and never bind it (see the v_max section above);
800s are *not* held against that constraint, so a strong one defines the line,
while a same-session 400 sits behind it rather than tied to it. (History: under
the earlier flat-edge model a ≥ 1500 m demo cutoff was tried and reverted; the
per-profile evidence multiple supersedes it — "a 400 never defines the frontier"
now falls out of the calibration instead of being a hard rule.) Only sub-120 s
sprints (the 400s) stay display-only — never demos, read down via the evidence
multiple. **Dashboard: predictions are direct
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

### Tier 1 — long races (≥15K: HM + Marathon)

Retuned **July 2026** to match the WA switch: residuals are measured in
**log WA-5K-equivalent time** — the same space the fit's likelihood
consumes — instead of raw same-band seconds. For each ≥15K race, residual =
own `log(t5k)` minus the median `log(t5k)` of **all eligible aerobic races
(≥1500m, any distance)** within a symmetric ±2yr window (self excluded,
n≥2 guard; pools run 20–95 races vs the old same-band 2–5). Prune if the
residual exceeds **+6%** (`tier1_thresh=0.06`, log-space fraction).

One threshold covers both bands with wide margins: kept long races top out
at +3.40% (Boston 2025) vs the first excluded at +7.51% (Nashville RnR
2021); on the HM side kept-max is +1.31% (Craft Classic 2024) vs Deception
Pass 2016 at +24.14%. Excluded: Boston 2021 (+18.18%), Boston 2024
(+8.01%), Boulderthon 2024 (+7.74%), Nashville RnR 2021 (+7.51%),
Deception Pass 2016 (+24.14%). The retune **readmitted five marathons**
the old raw-seconds rule pruned — Nashville 2019 (+1.19%), Boston 2022
(+1.92%), Berlin 2023 (+2.32%), Boston 2025 (+3.40%), Seattle 2025
(+1.53%) — races far from the marathon-band median in raw seconds but
entirely consistent with concurrent short-race fitness once WA-converted
(Seattle 2025's 5K-equiv, 15:33 corrected, was *faster* than a 5K five
weeks earlier).

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
`tier1_thresh=0.06` (log WA-5K-equiv fraction, ≈ +6%), `tier2_z_thresh=2.5`,
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
- **Asymmetric Tier 1 thresholds for HM and Marathon** (+115s vs +500s,
  raw same-band seconds). SUPERSEDED July 2026: the raw-seconds metric
  predated the WA switch — it judged marathons only against other
  marathons, so the threshold had to thread a +5s knife-edge (kept_max
  +113s vs excl_min +118s) and still excluded races whose 5K-equivalents
  were consistent with concurrent fitness (Seattle 2025). In WA 5K-equiv
  space the kept/bonk gap is wide (+3.4% vs +7.5%) and one 6% threshold
  covers both bands; the split is no longer needed.
- **A single percentage threshold for ALL distances** (extending the
  Tier 1 WA-space rule below 15K to retire Tier 2). Investigated July
  2026, rejected: symmetric 6% flips 24 kept races out — the 2013–16
  rise era (symmetric medians are biased wherever fitness changed fast),
  hot-course summer 5Ks (Pies/Derby Days/Turkey Trot at +7–10%), and the
  2021 comeback TTs — because short races have ~3× the percent spread of
  long ones in 5K-equiv terms. A past-only ~11% single rule avoids the
  flips but readmits Boston 2024/Boulderthon (kept 5Ks sit at +9.7%,
  above those bonks) with a thin +9.7%/+12.0% margin. Tier 2 isn't a
  special case for two races — it's what protects early-career and
  short-race data from a threshold calibrated on long-race physiology.
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
  anchor — ~+866s on the then-current raw-seconds exclusion metric, still
  well over its +500s HM threshold (and +24% on the July 2026 WA-space
  rule vs its 6% cutoff — the conclusion survives the retune).
  Landing at normal race scatter (+15 s/mi) would need a
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
- **No effort schedule** (retired Aug 2026): the two-channel hill model —
  climb cost + steepness-dependent descent benefit — prices races and easy
  runs identically; see route-normalization-reference.md.
- **Inputs.** Per-race hill quantities come from the race activity measured
  alone on the fused baro+DEM substrate (veto-cleaned hill segments); mean
  elevation stays DEM. Footing + altitude
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
