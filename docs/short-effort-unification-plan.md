# Short-effort unification — plan

**Status (June 2026): proposed, not started.** Born from the performance-
frontier thread. This doc defines the problem and the candidate designs so a
dedicated thread can pick it up. See [cs-model-reference.md](cs-model-reference.md)
(frontier + projection conventions) and
[training-quality-reference.md](training-quality-reference.md) (g(d), CF
reconstruction).

## The problem

Short maximal efforts (≲ 800 m) are handled by **two disconnected
corrections for one phenomenon** — the CP hyperbola over-credits short
efforts:

- **Races:** `β_short` (make_race_plots / dashboard) — a time-inflation
  knob applied below `d_thresh_short`, currently 0.363 / 875 m, calibrated
  June 2026 so today's 400 m and 800 m frontier predictions land exactly on
  the lifetime PRs. It is a *knob pinned to outcomes*, not physiology, and
  it is overloaded with three jobs: diamond display position, demo
  conversion, and prediction conversion.
- **Workout reps:** `g_anaerobic(d)` (workouts.py) — a distance-only pace
  penalty (+14.3 s/mi at 400 m), calibrated on rep-PACE 400s (~72 s).

Observed failures (June 2026):

1. **Chart positions are wrong at the top.** Fast 400s project among the
   career-best 5K-equivalents ("impossible — superseded only slightly by
   the championship meet and only equaled 5 years later"). Excluding
   fatigued races from PR pools hid the rings but is a **band-aid with a
   wrong rationale for short events** — aerobic fatigue from an hour
   earlier doesn't meaningfully affect a single 400 — and the points still
   sit visibly too high. The projection itself over-credits them.
2. **The race/rep invariant is violated.** A 400 m race should analyze
   exactly like a single 400 m rep. Today their corrections differ ~4×
   (β_short ≈ +29% time at race pace vs g(400) = +14.3 s/mi at rep pace) —
   because a 57 s race 400 sits far deeper in the anaerobic regime than a
   72 s rep 400. The codebase already states the truth twice: **the effect
   scales with speed-above-CS, not distance.**
3. **The 800 m floor can't be priced by this knob.** Honest 800s sit
   +7–11 s/mi behind the 5K frontier (real sustained-effort physiology);
   boosting them 7% pushed 12/20 short races past the frontier, while not
   boosting left predictions optimistic. The two-anchor calibration holds
   *today*; it is not invariant as the frontier moves.
4. **CF structural compromise.** g(d) charges +10.3 s/mi to every CF hard
   500 run at ~5K effort — essentially all of CF's +11.6 median vs
   intervals' +0.1. Removing g for CF only would be per-category logic,
   which is exactly what we're trying to eliminate.
5. **Jog vs pause (secondary).** The watch corpus splits `rest_jog_s` /
   `rest_stand_s`; jog-heavy interval days read 2.6 s/mi slower-corrected
   (direction consistent with jogs carrying aerobic load) but weak
   (ρ=+0.17, p=0.23, n=55 — jog fraction barely varies, IQR 0.35–0.45).
   May be absorbed by an effort-aware correction, or need its own τ split
   in the connected accumulator.

## Candidate designs

**A. Effort-aware anaerobic correction (recommended first spike).** Replace
both β_short and g(d) with one function of **speed above CS at the effort's
date**, applied per effort segment — races and reps go through the identical
path. The invariant holds by construction; the correction becomes era-aware
(a 57 s 400 in 2017 vs 2026 corrects differently, which is right). Cheap,
local to the projection layer.

Calibration data already in hand: the watch rep corpus (per-rep paces); the
2023-08-02 400 ≡ 2023-07-26 4:30-mile pair; the 800 population floor; the
CF↔interval alignment target; the PR prediction anchors.

**B. 3-parameter critical power (the principled endgame).** Add a v_max
term to the hyperbola (Morton-style), bending it correctly at short
durations — replaces β_short and possibly g(d) wholesale in cs_projection
and the connected accumulator. One new personal parameter (v_max, from
sprint data or fixed). More invasive; the race-only CS *fit* is untouched
(it already excludes < 1500 m). If A's fitted shape comes out looking like
the CP3 form, promote to B.

## Validation gates

1. **Invariant test:** a race 400 and a rep 400 at the same speed produce
   identical 5K-equivalents by construction; verify on 2023-08-02 vs
   nearest watch-measured rep 400s.
2. **Population audit:** zero non-fatigued shorts past the frontier; no
   short race reads as a cross-distance career high in any era; the
   fatigued-PR-pool exclusion can then be reverted (it's a band-aid).
3. **CF convergence:** CF median residual within ~2 s/mi of intervals with
   NO per-category logic.
4. **Prediction invariance:** predictions ≤ PRs at every distance under a
   frontier *sweep* (not just today's value).
5. **Frontier feedback:** de-pessimizing CF lifts the 2019–20 frontier
   (the race gap) — rerun the era readings and the binding audit.
6. **Rep re-validation:** the enrichment corpus checks (41/66 exact, τ
   anchor) hold or improve.

## Open questions

- v_max sourcing (lifetime sprint data is thin: 12 fatigued-heavy 400s).
- Does the jog/pause τ split survive once effort-awareness lands, or is
  the CF float credit it would provide already covered?
- Retire the 875 m knob entirely, or keep as a display-only fallback for
  profiles with no short-effort corpus?
