# Short-effort unification — plan

**Status (June 2026): IMPLEMENTED — design B (CP3), per Max's call.** Design
A was rejected before implementation (Max): pinning short races to the
current-fitness CS is conceptually circular even though short races don't
feed the CS fit — the projection must read each effort on its own curve.
Design B does exactly that: the implied-CS solve is self-consistent (τ
depends on the CS being solved for), so the fit's CS never enters a race's
projection. See the **Outcome** section at the bottom for what shipped,
gate results, and the v_max calibration trade-off. This doc defines the
problem and the candidate designs. See
[cs-model-reference.md](cs-model-reference.md) (frontier + projection
conventions) and [training-quality-reference.md](training-quality-reference.md)
(the retired g(d), CF reconstruction).

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
  *(Resolved: calibrated on the equivalence pair — see Outcome.)*
- Does the jog/pause τ split survive once effort-awareness lands, or is
  the CF float credit it would provide already covered? *(Still open — not
  addressed in this pass; the signal was weak, ρ=+0.17, p=0.23.)*
- Retire the 875 m knob entirely, or keep as a display-only fallback for
  profiles with no short-effort corpus? *(Retired entirely. Profiles with
  no sprint corpus get a fixed default v_max = 8.5 m/s; below-1500m is the
  only regime where the choice matters.)*

## Outcome (June 2026, design B)

**What shipped.** The Morton 3-parameter model `v(t) = CS + D′/(t + τ)`,
`τ = D′/(v_max − CS)`, replaces both corrections wholesale:

- `cs_projection` projects every race on its own CP3 curve (closed-form
  implied-CS quadratic; forward time solve for predictions and
  lines-at-anchor). β_short/d_thresh_short deleted everywhere
  (make_race_plots, dashboard, frontier_at_anchor). β_long is untouched —
  race-execution fade above 10K is a separate phenomenon.
- The CP2 fit stays the fit. Its D′ is bridged per date as the *effective
  anaerobic distance at 5K*: `D′₃ = D′₂/(1 − D′₂/((v_max−CS)·t5K))`, which
  preserves the fit's 5K prediction exactly (frontier floor, CS-implied-5K
  line, TQ residual frame unchanged) and confines changes to where the fit
  has no data. Mile/1500 diamonds shift ≈ −2.5…−4 s/mi vs CP2 depending on
  the edge (the bend can't be invisible at the mile while real at 400).
- g(d) is deleted. The connected accumulator deflates each rep's supra-CS
  speed by `t/(t+τ)` (CP3's anaerobic-availability ratio at the rep's
  duration), with CS the workout's OWN implied CS solved as a fixed point
  (no CS-fit input) and the FIRST rep exempt (its W′ deployment is the D′
  the projection prices). At rep paces this reproduces the retired g(d)
  (≈+12 vs +14.3 s/mi at 400); at race paces it scales ~4×, as failure #2
  demanded.
- The fatigued-PR-pool band-aid is reverted (gate 2): fatigued shorts now
  project at ordinary-scatter positions, no longer "impossible career
  highs".
- **Frontier demos: time ≥ 120 s stands — 800s bind by design.** A
  distance ≥ 1500 m cutoff was tried and reverted the same day. The
  investigation first showed v_max could never un-bind Max's 800s (they
  need 10–10.7 while the 400 corpus demonstrates ≤ 9.4) because the fit's
  own D′ (174 m, pinned by his speed-reserve-less 1500–3000s) says a
  2:04.1 really is peak-level FOR HIM — the IAAF-points "weak at short"
  comparison is population-relative and already encoded in that small D′.
  Max then resolved the perception conflict himself: the 2017-03-30 800
  genuinely was his best lifetime effort to that point (superseded three
  weeks later by three 1600s and an interval workout — a coherent peak);
  "3200s set me apart in HS" was competition density (more proficient HS
  milers than 2-milers), not absolute capability. 800s inform the
  frontier; only sub-120 s sprints stay display-only.

**v_max calibration — an uncertainty interval, two conservative edges
(scripts/calibrate_vmax.py).** Max's two hard invariants — (1) 400 m
races never exceed the frontier, (2) 400/800 predictions never beat the
lifetime PRs — are jointly unsatisfiable by any single curve: they meet
only at equality at the PR's own date, which would additionally have to be
the all-time minimum of the prediction sweep; the 400 PR (2023-08-02) was
run ~3 months after the frontier's 2023 peak (2023-05-06, the 20.5 mi
long-run demo, ~5 s/mi faster than the early-August frontier). A
**time-varying v_max(t)** was considered (Max's suggestion) and rejected
on his own data: the per-race demonstrated v_max series scatters 7.6–9.4
with same-day spreads of 1.3 m/s (2023-08-02: the fresh 57.0 implies
8.91, the fatigued 60.0 implies 7.64) and the 2017 fatigued HS races top
it — execution noise, not a trend; an envelope over it would let single
noisy races define the sprint axis for years. Its robust collapse is the
**lifetime envelope as a constant edge**, which is what shipped:

- **v_evid = 9.5** (evidence edge, high): the 400-corpus sprint-credit
  envelope — the largest per-race implied v_max placing each 400 exactly
  ON the frontier is 9.19 (the 2017 fatigued 57.6, with the 800 demos
  binding nearby) — plus margin. Every 400 ever raced sits at/behind the
  frontier by construction (invariant 1); audit: 0/20 short races past.
- **v_pred = 8.3** (prediction edge, low): under the PR-sweep upper
  bounds (8.55 / 8.52 from the 400/800 anchors) with margin — sweep-min
  predictions 57.56 s and 2:04.50 (+0.56 s / +0.44 s above the PRs),
  binding at the 2023-05-06 peak. Today's dashboard: 400 59.3, 800 2:09.4
  (invariant 2, with room). Note the asymmetry is what makes a BINDING
  800 demo compatible with invariant 2: its conservative evidence read
  goes into the frontier, and the conservative prediction read comes back
  out slower than the race itself — equality is never forced.
- **WORKOUT_VMAX_MPS = 8.7** (workouts.py): the accumulator's deflation
  and the TQ projections keep their own value — a measurement calibration
  anchored to the watch rep corpus (τ = 540 re-validated there, deflation
  reproduces the retired g(d) at rep paces), not a conservatism policy.
  Honest cost: the exact race≡rep invariant becomes approximate across
  the workout/race boundary (a hypothetical standalone max rep would read
  less conservatively than the same race diamond; no such effort exists
  in the corpus).

The two race edges encode that the short↔long conversion is deliberately
LOSSY in both directions — a 400 is weak evidence of 5K fitness, and 5K
fitness is a weak promise of 400 speed — the frontier's one-sided-proof
epistemology extended to the v_max axis, and the model-level encoding of
"simple equivalence tables overrate my short distances". The 400≡mile
pair is now a diagnostic, not an anchor (the 400 reads +8.3 s/mi worse
than the equivalent mile — conservatism working as intended).

**Gate results (as revised by Max's invariants).**
1. *Invariant (race≡rep)*: holds within the race layer and within the
   workout layer; approximate across them (see WORKOUT_VMAX_MPS above).
2. *Population audit*: ZERO short races past the frontier (0/20); 800s
   bind where they were genuine peaks (5 of 7 non-fatigued — by design).
   Band-aid reverted.
3. *CF convergence*: CF median +7.2 vs interval −0.9 (gap 8.1 s/mi, from
   9.6). NOT within the ~2 s/mi target: CF hard 500s genuinely run
   ~0.3 m/s supra-CS, so an honest effort-aware charge remains; the
   residual gap now reads as 2019–20 era effort policy, same epistemic
   status as tempo's +19.
4. *Prediction invariance*: predictions ≥ PRs at EVERY date in the
   historical sweep, +0.4 s margin or better (invariant 2, strict).
5. *Frontier feedback*: 2019 frontier lifted ~+1 s/mi by the CF
   de-pessimization; mile-heavy eras (2017-18) ~+2 s/mi fitter; 800-bound
   bulges remain where the 800s were genuine peaks.
6. *Rep re-validation*: reconciliation untouched (reps.py unchanged);
   RECON_TAU_S = 540 re-checked under the new model and remains the RMS
   optimum (rms 8.15 s/mi at 540 vs 8.51/8.52 at 420/660) — the τ anchor
   holds without refitting.

Rep TQ median moved +5.4 → +1.4 (now between interval and CF — the
scatter-weighted re-entry fully converged).
