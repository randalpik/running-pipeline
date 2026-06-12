# CS workout-enrichment spike — report (June 2026)

> **Postscript (June 2026, same thread):** the HALT below stands for the
> original question — feeding workouts into the CS *likelihood*. The thread
> then iterated through probabilistic one-sided variants (ExGaussian /
> Tobit / latent-slack / quantile delta-fits over a frozen race anchor; all
> failed in characterized ways — whenever observation noise was estimable,
> the sampler chose "wide symmetric noise" and the asymmetry vanished) and
> landed on Max's reframing: workouts are PROOF of capability, not noisy
> observations of it. That ships as the **performance frontier** — a
> deterministic demonstrated-capability envelope on the Fitness tab,
> semantically separate from CS (no posterior, no CI). See
> [cs-model-reference.md](cs-model-reference.md) § "Performance frontier"
> and `src/shared/performance_frontier.py`. The pair-repeatability and
> structure-function machinery developed here supplied the frontier's cone
> slopes (gain 1.74 / decay 1.04 s/mi/wk).

**Verdict: HALT — do not feed near-race training observations into the CS fit.**
The enrichment is precise but biased: it sharpens the CS curve dramatically
(95% CI −43%) around the wrong center wherever races are absent. The hold-out
test is unambiguous — removing the 2019-06 → 2020-12 races and letting
near-race workouts fill the gap made predictions of the held-out races *worse*
(mean |error| 2.49% → 3.26%, bias −1.50% → −3.09%) under every variant tested.
Race-only remains the production fit. The `--workout-obs` flag in
`bayes_cs_fit.py` stays (default off, fully reversible) as the record of the
implementation.

This executes the plan in [cs-workout-enrichment-plan.md](cs-workout-enrichment-plan.md)
with one design change from Max (June 2026): instead of watch-measured
intervals only, candidate observations were **near-race efforts** — every kept
TQ point (workout / long run / hill) whose corrected residual lies within
±B of the CS curve, where B = |fastest corrected residual in the corpus|.

All artifacts: `output/debug/spike_cs_enrichment/` (fit summaries, obs CSVs,
logs, shadow timelines). Spike scripts: `scripts/spike_*.py` (one-off,
deletable). Fits run June 12, 2026 on races through 2026-05-31 (209 eligible
after auto-exclusion), TQ corpus of 627 kept points (236 workouts, 265 long
runs, 126 hills).

## The near-race band

- B = **13.97 s/mi** — the fastest corrected residual (2020-11-06,
  16.8 mi belle meade long run, route-rule-corrected). Matches the −14
  Max quoted.
- **292 / 627 points (47%) fall in band**: 178 workouts, 89 long runs,
  25 hills.
- The predicted **era-type shift is real and sharp**. Long runs in band:
  0/14 (2017) → 2/33 (2018) → 15/41 (2020) → **22/27 (2022)**, 17/28 (2023).
  Pre-2019 the band is almost entirely intervals; 2022-23 it is mostly long
  runs. The band tracks what Max's training prioritized per era, exactly as
  hypothesized.
- In-band yearly mean residual (s/mi, + = slower than CS): 2016 +0.0,
  2017 +1.3, 2018 +4.2, **2019 +6.1, 2020 +4.6**, 2021 +1.0, **2022 −0.0**,
  2023 +1.8, 2024 +1.4, 2025 +3.8, 2026 +3.0. This is the pull pattern the
  enriched fit then exhibits, almost verbatim.

## Gate 1 — noise benchmark: PASSES, with the risk inverted

The naive measurement (in-band scatter vs the CS curve, fractional log-time
scale) gives σ = 0.0221 — but it's biased low by construction: selecting
|resid| ≤ B truncates the very distribution being measured. The honest,
selection-free estimator added in this spike is **pair repeatability**:
sd(difference)/√2 over same-source pairs ≤ 21 days apart — true fitness
barely moves inside the window and the CS curve cancels out of the
difference. Same estimator applied to races:

| population | pair σ (fractional) | n pairs |
|---|---|---|
| races (kept, vs race-only fit) | 0.0290 | 341 |
| workouts (full corpus) | **0.0272** | 444 |
| long runs (full corpus) | 0.0325 | 474 |
| hills (full corpus) | 0.0336 | 212 |
| watch-verified workouts | **0.0127** | 37 |

Two findings:

1. **Near-race training efforts are race-grade precise.** The plan's halt
   condition (σ_workout ≫ σ_race → weights too small to matter) does not
   trigger — it inverts. With per-source σ as observation noise, 292 obs
   carry leverage comparable to the whole race corpus. (The fitted race
   σ_base on current data is 0.038, so a workout obs at 0.0272 actually
   out-weighs a 5K race.)
2. The full-corpus scatter **vs the CS curve** is 0.0534 — twice the pair σ.
   Most of the workout-residual variance is *slow-moving era effort policy*,
   not day-to-day noise. The band selection removes the far-from-CS
   population, but the in-band mean is still **+0.0084 fractional
   (≈ +2.7 s/mi)** — the sub-max effort gap survives selection, because the
   band is symmetric while the population is one-sided.

That residual bias, not noise, is what ultimately kills the design.

## Gate 2 — circularity: implemented and verified

`bayes_cs_fit.py --workout-obs <csv>` adds each observation as a
5K-equivalent effort: `log(t5k) ~ N(log((5000 − dp_fixed)/CS(t)), σ_obs)`
with **D′ frozen at the race-fit median** for that date — the gradient flows
into the CS GPs only; races remain the sole anchor for D′ and β_long (which
doesn't apply at 5000 m). σ_obs is the per-source pair σ above, a constant,
not fitted. Race-only path is byte-identical when the flag is absent.

Verified on the shadow fit (enriched vs race-only, same race set):

- D′ grid median 165 m → 171 m, max shift 11 m — well inside the ~73 m-wide
  95% CI. No feedback.
- σ_base 0.038 → 0.037: races are not being explained away.
- β_long 0.0761 → 0.0706 (CS slightly slower near marathons → less fade
  needed). Modest but real coupling; would need watching if this ever shipped.
- Sampling healthy everywhere: R-hat 1.00, divergences 6–21 / 4000 across
  all six spike fits.

## Shadow comparison (full race set)

`spike_raceonly` vs `spike_nearrace` (per-source σ), with `spike_nearrace_x15`
(σ × 1.5) as the weight-sensitivity arm:

- **CS shift**: median |Δ| 1.90 s/mi, mean +1.73 (slower). Concentrated
  exactly where the in-band means predict: **2018 +3.5, 2019 +5.6, 2020
  +4.3**, 2022 −0.9, 2026 +5.3 (max +9.2 s/mi in the late-2026 extrapolation
  tail). 2007–2015 wobbles ±2-3 s/mi purely through hyperparameter leakage
  (no obs exist there).
- **Sharpening is huge**: median 95% CI width 21.9 → 12.5 s/mi (−43%);
  −18.5 s/mi at race-sparse grid points. This is the intended value — *if*
  the center were trustworthy.
- **Race-anchored regions move too**: median |Δ| 1.80 s/mi within 30 d of a
  race (vs 2.19 beyond 60 d). Aggregate race residuals barely change
  (mean |resid| 2.63% → 2.66%), but the obs-dense 2019-20 block degrades
  coherently: the All-Comers/time-trial cluster goes ~1.4–1.6% further fast
  of the curve (e.g. 2020-02-03 TT −2.75% → −4.32%).
- **The headline number moves**: current CS (2026-06-06 grid point)
  5:13.3 → **5:18.6 /mi** (95% width 30 → 17 s/mi) — pulled by 7 in-band
  2026 obs (mean +3.0) against the most recent race, the 2026-05-31 North
  Shore HM, which beat the race-only curve by 2.71% and ends up −3.85% under
  the enriched one. Every CS consumer (dashboard cards, race projections,
  workout-pace predictions) would read ~5 s/mi pessimistic today.
- ×1.5 downweight: same shape, ~20% smaller amplitude (2019 +4.7, 2026 +4.1,
  CI −6.3). The result is data-driven, not weight-driven.

## The decisive test — hold-out of 2019-06 → 2020-12

The one extended race gap in the corpus where enrichment claims the most
value. All 21 race rows in the window were removed (193 eligible remain);
17 kept races inside it are the evaluation set. Hygiene: the band was
**re-selected against the hold-out race-only curve** (arm A), so no held-out
race information enters arm B even through selection. Two band variants
because the re-derived fastest residual balloons to −26.01 s/mi against the
sagging gap curve (a finding in itself — see below): fixed B = 13.97
(343 obs) and self-derived B = 26.01 (470 obs).

Prediction error on held-out races (+ = ran slower than predicted):

| arm | mean abs err | RMSE | mean err |
|---|---|---|---|
| A: race-only, gap interpolated | 2.49% | 3.00% | −1.50% |
| B1: + near-race obs, B=14 | 3.26% | 3.85% | −3.09% |
| B2: + near-race obs, B=26 | 3.36% | 3.94% | −3.19% |
| reference: production fit (saw the races) | 1.75% | 2.07% | −0.29% |

**Enrichment moved every one of the 17 held-out races ~1.6–1.9% in the wrong
direction.** Max's actual fitness in the gap — demonstrated by the held-out
races, including the 2020-07-03 5K TT (15:26) and the 2020-11-21 marathon TT
(2:30:57) — was *faster* than even the race-only interpolation, while the
in-band training data said *slower* (+4.6 to +6.1 s/mi). The near-race subset
was reporting era effort policy, not latent fitness.

This directly answers the question the band selection was designed to ask.
Of the two readings —

1. *"this workout data suggests fitness was better than races could predict"*
2. *"this workout data is just indicative of overall effort in that era;
   the mismatch from races is a real mismatch"*

— the hold-out picks (2), at least for the one gap that can be tested, and
the 2026 head-to-head (workouts vs the North Shore HM) points the same way.
The band cannot isolate (1), because the selection is symmetric around CS
while the effort gap is one-sided: the slow half of the band is always denser
than the fast half, so the selected population inherits a +2.7 s/mi class
bias that era-modulates between 0 and +6 s/mi. Sharpening the posterior
around that biased center makes the curve *more confidently wrong* exactly
where there are no races to push back.

Caveats stated honestly: n = 17, one era, time-trial-heavy — but this is
precisely the regime enrichment exists for, the era is the only extended gap
available, and solo time trials would if anything run *slow* of true race
fitness, which makes the conclusion conservative. The plan set the burden of
proof on enrichment ("proving they help without distorting"); it is not met.

## Secondary findings

- **Band-width instability.** The "fastest corrected residual" definition is
  itself curve-dependent: against the hold-out curve it doubled (−13.97 →
  −26.01, same physical run). In production the curve always has all races, so
  the definition is stable there — but any future selection rule should use a
  fixed constant or a robust quantile, not a single extreme point. (The
  defining point is also a route-rule-corrected long run, not watch-measured.)
- **Boundary extrapolation amplifies the bias.** Beyond the last race the dev
  GP decays but the obs keep pulling: the late-2026 tail shifts +8 to +9 s/mi.
  Any enrichment variant must be evaluated on its boundary behavior, since
  that's the dashboard's "current fitness".
- **15% of the corpus sits within 3 s/mi of a band edge** — one-pass
  selection keeps the feedback bounded, but iterated refits (select → fit →
  re-select) would churn membership. If anything ever ships, selection must
  stay pinned to the race-only curve, i.e. production runs two fits per
  refresh.
- **2019-20 in-band points are mostly workouts** (26–31/yr vs 2–5 long runs),
  so the known 2018-19 long-run route-rule softness is not what drove the
  hold-out failure.

## What survives this spike

- **The precision result stands.** Near-race training efforts repeat at
  race-grade precision (pair σ 0.0272 vs races 0.0290), and watch-verified
  intervals at 0.0127 are *twice* as precise as races. The information
  content is real; only the level is untrustworthy.
- **A level-free formulation is the one idea left open**: let workout
  observations inform the *within-era shape* of the CS curve while a free,
  slowly-varying offset absorbs the era effort level (races stay the sole
  level anchor). That targets exactly what the pair-σ shows workouts know
  (day-to-day relative fitness) while discarding what they don't (absolute
  level). Substantially more model complexity for between-race wiggle
  refinement — only worth opening if a concrete downstream consumer needs
  it.
- **Watch-verified-intervals-only** (the plan's original candidate set) was
  not separately shadow-fit: the corpus is 2020-12+ only (58 days), where
  races are already dense, so the value-add is small by construction — and
  the hold-out failure mode (level bias) applies to any absolute-level
  observation, just with a smaller bias. Not pursued.
- The `--workout-obs` flag and the spike scripts document the machinery; the
  pair-repeatability estimator (`scripts/spike_repeatability.py`) is worth
  reusing whenever a new observation class is proposed for any fit.

## Run inventory

| tag | races | obs | purpose |
|---|---|---|---|
| `spike_raceonly` | 209 | — | fresh baseline, current data |
| `spike_nearrace` | 209 | 292 @ pair σ | primary enriched |
| `spike_nearrace_x15` | 209 | 292 @ σ×1.5 | weight sensitivity |
| `spike_loo_raceonly` | 193 | — | hold-out arm A |
| `spike_loo_nearrace` | 193 | 343 @ B=14 | hold-out arm B1 |
| `spike_loo_nearrace_b26` | 193 | 470 @ B=26 | hold-out arm B2 |

All six: 7 d grid, m=100 basis, 4×1000 draws, seed 42, XC correction 0.08.
Shadow timelines: `cs_timeline_{raceonly,nearrace}.html` in the spike dir.
