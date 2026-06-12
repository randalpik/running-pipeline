# CS Workout-Enrichment Plan

**Status (June 2026): spike executed — HALTED at validation. The fit stays race-only.**
The spike ran this plan with a near-race band selection (Max's June 2026 variant: every TQ
point within ±14 s/mi of CS, the band set by the fastest corrected residual). Gate 1 *passed*
— near-race efforts are race-grade precise (pair-repeatability σ 0.027 vs races 0.029) — but
the hold-out test failed decisively: with the 2019-06→2020-12 races removed, enrichment made
held-out race predictions worse (mean |err| 2.49% → 3.26%, bias −1.50% → −3.09%) because the
in-band population keeps a one-sided sub-max effort bias (+2.7 s/mi mean, era-modulated 0…+6)
that the symmetric band cannot remove. Full numbers, machinery, and the one surviving idea
(level-free shape-only observations) in
[cs-workout-enrichment-spike-report.md](cs-workout-enrichment-spike-report.md).
`bayes_cs_fit.py --workout-obs` remains as the flag-gated, default-off implementation.

This doc is the plan-of-record the spike executed; kept for the gate rationale. Do not re-open
enrichment without addressing the level-bias finding in the report.

See also: [cs-model-reference.md](cs-model-reference.md) (the fit itself),
[training-quality-reference.md](training-quality-reference.md) (the workout projection that would
feed it), and [ARCHITECTURE.md](ARCHITECTURE.md).

## Why this is high-stakes

1. **CS threads through nearly every chart.** `bayes_cs_summary.csv` (CS + D′ per day) drives the
   Fitness/CS-timeline plot, the Races plot's projections, the Workouts/Training 5K-equivalent
   axis, the recovery and long-run projections, and the dashboard cards. A bad CS curve corrupts
   all of them at once. The bar for changing its *inputs* is therefore much higher than for any
   single plot.
2. **The fit is expensive.** It's a PyMC HSGP model (minutes per run, posterior `.nc` artifact).
   Iteration is slow, so changes must be designed carefully up front, not tuned by trial-and-error.
3. **Race-only was a deliberate choice.** Races are clean maximal efforts; training data is messy
   and was historically excluded for good reason. Enriched workouts are a *new* kind of input
   (real per-rep structure, CS-free projection) — but the burden is on proving they help without
   distorting the established race-anchored curve.

## What's already built (the prerequisite)

The connected-fatigue workout projection (`parse_workouts._connected_core`, `shared/workouts.py`)
produces a **CS-free** effective effort `(D_eff, t_eff)` per quality day. The implied critical
speed it reads off — `CS_workout = (D_eff − D′) / t_eff` — uses only the race-fit **D′** (the
hyperbola intercept), never the race CS. That independence is exactly what makes adding workouts
to the fit non-circular for CS. Calibration so far:

- `RECON_TAU_S = 540 s` was fit by minimizing workout-implied-CS vs race-CS SSE; lands in the W′
  reconstitution literature band (corroboration, not circularity).
- At that fit, **interval-day workout-implied CS was unbiased (mean ≈ 0) with ≈ 8.4 s/mi scatter**.
  ⚠️ That number predates the `g(d)` anaerobic correction and the era/type rest policy. It mostly
  affected short reps (≤800 m), not the ≥800 m intervals this plan concerns, so it should still
  hold — **re-confirm it on the current pipeline before using it as the weight denominator.**
- A residual-vs-longest-rep correlation (~+0.25) remains: the **workout-vs-race effort gap**
  (worse fueling/fatigue/shoes/adrenaline; workouts are sub-max). This is the basis for a
  downweight, not a bias to "correct away."

## The gates (work through in order; each can halt the project)

### Gate 1 — Weight benchmark against *real* race residuals
Set the per-workout observation weight as `(σ_race / σ_workout)²`, where both are measured the
**same way** against the same reference.
- `σ_workout`: scatter of workout-implied CS vs the fitted CS curve, on the **current** pipeline
  (re-measure; expect ~8 s/mi for intervals).
- `σ_race`: per-race residual scatter **from the actual bayes model** — i.e. the Fitness-graph
  race points vs the CS-predicted line, *with* the model's surface/XC/distance corrections applied.
  **Do NOT use a crude 2-parameter curve** — it inflated race scatter to ~21 s/mi (a bogus result
  that made workouts look 6–8× better than races). Extract residuals from the posterior instead.
- **Halt condition:** if workouts are much noisier than races (σ_workout ≫ σ_race), the weight is
  tiny and enrichment isn't worth the risk — stop here.

### Gate 2 — D′ feedback / circularity
The workout projection consumes race-fit **D′**. If workouts also inform D′ in the fit, that's a
feedback loop.
- **Resolution:** workouts constrain the **CS trend only**; races remain the sole anchor for D′
  (and for the long-distance `beta_long` term). Implement either as (a) a model where workout
  observations enter the CS likelihood but not the D′ one, or (b) an iterative two-pass fit
  (race-only D′ → project workouts → CS-trend fit with both). Prefer (a) if expressible in PyMC.
- Workouts are mid-distance (~5–13 min efforts), so they carry little D′ information anyway —
  this is a guardrail, not a tuning lever.

### Gate 3 — Effort-gap handling
Workouts are sub-max (Gate-0 residual). Decide whether to (a) fold the gap into the downweight
only, or (b) add a small fixed "workout effort offset" term the fit estimates (analogous to the
XC pre-correction). Lean toward (a) unless the gap proves strongly structured.

## Validation protocol (before anything ships)

1. **Shadow run.** Produce `bayes_cs_summary` both ways (race-only vs +workouts) and diff the CS &
   D′ curves over the whole timeline. Quantify: median and max CS shift, and specifically whether
   any *race-anchored* region moves (it should barely).
2. **Downstream sweep.** Regenerate every CS-dependent plot from the shadow summary and eyeball
   the Fitness, Races, Workouts/Training, recovery, and long-run charts for regressions.
3. **Leave-one-out sanity.** Confirm workout observations sharpen the CS timeline *between* races
   (their intended value: filling the sparse-race gaps) without overriding race evidence at race
   dates.
4. **Reversibility.** Gate the whole thing behind a flag (e.g. `--include-workouts`) so race-only
   remains the one-command fallback. Never make enriched workouts the only path.

## Open decisions for the dedicated thread

- Which workouts qualify as observations: intervals only, or tempos too (continuous tempos are
  single-block efforts near CS — plausibly clean)? Start intervals-only.
- Whether to use only watch-**measured** days (highest quality) or also the recorded/estimated-rest
  reconstructions. Start watch-measured-only to minimize input noise.
- Time-window: only the watch era (2020+) has measured data; the pre-watch reconstructions are
  noisier and the early race calendar is dense, so likely not worth including.

## One-line summary

The math is de-circularized and the projection is shipped; the remaining work is a careful,
flag-gated, shadow-validated addition of *watch-measured interval* observations to the CS
likelihood (trend only, races anchor D′), at an empirical downweight set by Gate 1 — pursued only
if that benchmark shows workouts are competitive with race observations.
