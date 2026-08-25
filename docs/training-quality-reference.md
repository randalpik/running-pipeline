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
  *(June 2026: upgraded to the 3-parameter CP projection — implied CS via
  `cp3_implied_cs(D_eff, t_eff, D′₃)`, then `cp3_time(5000, ·)` — the same
  CP3 layer races use; see cs-model-reference.md. The D′₃ bridge preserves
  the CS-implied-5K line exactly, so residual semantics are unchanged; the
  shift is ≤ ~2 s/mi for typical workouts and matters most for short-rep
  days, together with the effort-aware deflation below.)*
- **Cross-distance conversion is the World Athletics hybrid (June 2026).**
  `β_long` is retired everywhere. Each effort homogenises to a 5K-equivalent
  by: efforts whose effective distance >5K (long runs; workout sessions whose
  connected-fatigue `D_eff` >5K) down-convert via the **WA scoring tables**
  (`wa_scoring.py`); efforts ≤5K (most workout `D_eff`s) up-convert via
  **CP3 + v_max** as before. So a short rep session and a 400m race still
  analyze identically, while long aerobic efforts use the empirical WA
  equivalence instead of a fitted fade. See `cs-model-reference.md`.
- **Long-run scope is `duration ≥ 80 min AND miles < 26.2`, global
  across profiles; distance carries no model term (June 2026).** The
  bounds deliberately mix units because they encode different
  mechanisms. The floor is physiological — a run is "long" once
  fueling/hydration become a real concern, which onsets by duration
  regardless of pace — and the residual cliff is razor-sharp at 80 min
  (median raw_resid ~95–105 s/mi below, ~15–42 above; holds within-era,
  so it's not a fitness-mix artifact). The ceiling encodes training
  purpose: a non-race run at marathon distance or beyond is an endurance
  event or informal sub-max time trial, not marathon race prep — true
  regardless of run time, which is why it's distance, not a time cap
  that would bias toward fast runners (20 mi at 9:00/mi is 180 min of
  textbook prep). The ceiling makes the app's marathon-focused-runner
  assumption explicit (already implicit in fatigue categories, the
  dashboard 20 mi card, the CS marathon-pace curve). Strict `< 26.2`
  against logged miles — training marathons are logged as exactly 26.2;
  don't derive the cap from 42195 m (= 26.219 mi). This superseded two
  short-lived June 2026 distance slices ([15.1, 25.3], then [12.0,
  25.3]); on Max's data the time floor and the distance floor at 12
  select near-identical sets (the cliff is the same population) and the
  fits are statistically indistinguishable. The earlier floor analysis:
  sub-cliff labeled-long runs are 2016–17 subjective "long" days, and
  the cliff-to-15.1 band added ~88 runs concentrated in the otherwise
  near-empty 2017–2019 years. The 21mi internal bin's
  coefficient collapsed from ~+20 to +3.5 when route dummies were
  removed (it was mostly route/era mix), and a fresh sweep under the
  physical model found bin@21 *worse*
  than no distance term (ΔAIC +0.8) while every apparently-better
  distance term was era composition in disguise — within-era distance
  slopes flip sign between eras. The lr_lo/lr_hi labels are fully
  retired: the TQ legend renders a single "Long" entry, and the
  dashboard's long-run pace prediction (projected to 20 miles) uses a
  recency-weighted (365d half-life) mean residual over familiar-route
  long runs — the per-bin empirical means differed by 0.3 s/mi (the
  split conditioned on nothing) while recency moves the estimate
  ~+7 s/mi.
- **Full physical route decomposition replaces per-route dummies, then
  the route-constant elevation term (June 2026, watch-stream
  enrichment).** Empirical route betas were almost perfectly
  era-confounded — every named route lives in one contiguous era with its
  own long-run effort policy, so betas encoded "typical effort that year",
  not terrain (physically identical flat routes south lakefront / north
  greenway sat 31 s/mi apart; a moderate 2026 effort out-ranked the
  all-time-best 2023 long run after correction). The first replacement —
  a route-constant `0.17 · elev_per_mile` slope plus a fitted altitude
  term inside `long_run_model.py` — has itself been removed. The route
  conversion now applies the full physical decomposition (grade +
  off-road footing + altitude) as flat / sea-level-equivalent **time**
  corrections, subtracted from each run's time in
  `workouts.project_long_runs` *before* the World Athletics 5K-equivalent
  conversion (so the conversion treats the flat-equivalent as a race).
  Grade prices each run's measured hill segments through the shared
  two-channel `elevation_cost` engine (climb cost + steepness-dependent
  descent benefit);
  footing and altitude are pinned from `recovery_model.physical_route_betas`
  — the single source of truth shared with the recovery and race-CS
  models. `long_run_model.py` no longer fits anything physical
  (`LR_ELEV_SLOPE`/`elev_pm_c` removed); it carries only temp/fatigue. Era
  effort policy still stays visible in the corrected residuals — the
  smoother track reading reverts to "fitness/effort vs CS", no longer
  "within-route trend".
- **Temperature + race-fatigue covariates in the long-run fit; fatigue
  is one fitted scale with the marathon/short contrast pinned from the
  recovery fit (June 2026).** Stage 5b adds `temp_centered` (fit fresh on
  long runs) and a combined race-fatigue regressor
  `fat_race_short + ratio·fat_marathon`, where `ratio` is
  `β_marathon / β_race_short` from the profile's recovery fit
  (`recovery_fatigue_ratio()`, computed per profile at build time;
  fallback 1.0 when the recovery fit is unavailable or the contrast
  unidentified). Rationale: the free long-run marathon beta was
  unidentified — only ~4 long runs carry meaningful marathon-fatigue
  load, and leave-one-out swung it 1.4–8.6 s/mi — while the recovery fit
  pins the contrast on ~2,300 rows (1.82 for Max, June 2026). The fitted
  long-run scale comes out ~2.3× recovery's amplitude (t = 2.12 against
  scale = 1), so transplanting recovery betas wholesale under-corrects
  (tested and rejected in the June 2026 variant experiment); the contrast
  transfers, the amplitude doesn't — a long run holds the fatigued state
  for 12–25 mi nearer threshold. Implied peaks for Max: marathon ≈ +38,
  short ≈ +21 s/mi, strikingly convergent with the old slice's free
  estimate (+36) that had motivated separate terms. Time of day, a strong
  recovery factor, is dead on long runs (t ≈ 0.25) and excluded.
  Covariates are gated at `MIN_COV_N = 30` in-slice runs so sparse
  watch profiles degrade to an intercept-only fit.
- **τ = 210 sec/mi for workout D_eff decay.** Calibrated so 6×1600 @ 3:00/mi
  rest gives D_eff ≈ 5000m (anchor preserved from earlier work).
  *(Superseded June 2026: the connected accumulator's flat decay became
  `RECON_TAU_S=540`s, then the intensity-dependent `recon_tau(v/cs)` =
  662·exp(−3·(v/cs−1)) clamped to Skiba [316,862] — see Stage 2. The flat 540
  remains only for long-run segments; the legacy /210 uniform formula survives
  only as the no-watch fallback.)*
- **Reps excluded from the smoother.** *(Superseded June 2026: reps
  re-entered scatter-weighted once the g(d) anaerobic correction landed —
  see Stage 5a.)*
- **Hill continuous workouts included as a TQ category (April 2026).**
  *(Superseded June 2026: per-loop offsets replaced by the hill model —
  gain/mi + terrain, Stage 3.5 — which also admits the rare loops
  (pwr3, hj, 106th) the n>7 offset rule had to drop.)* Hill repeats
  remain categorically excluded (no per-rep distance is logged, so no
  pace can be recovered).
- **Long runs carry NO model offset — race-equivalent projection only
  (June 2026).** The long-run intercept+covariate model is GONE from TQ:
  its constant (+48 at the end) was subtracted from every long run's
  position and smoother contribution, which amounts to claiming a long
  run predicts a faster 5K than a race at the same distance/pace —
  physically indefensible, and the same class-constant design already
  rejected for workouts. Instead `project_long_runs` projects each long
  run AS IF IT WERE A RACE of its distance, **down-converted to 5K via the
  World Athletics tables** (June 2026 — replaces the old β_long un-bias;
  long runs are all >5K). It carries one extra adjustment a race doesn't: a
  **pause-uncertainty erosion** of the demonstrated distance before the WA score
  (`durability.eroded_deff`). A paused long run is less trustworthy as proof of
  continuous capability, so each pause erodes all *subsequent* confirmed distance
  by `exp(−gate · RATE · pause_sec · lateness)` — driven by pause **LENGTH** (not
  count) and **lateness**, scaled by an **UNCAPPED effort gate** so only
  near/over-race-pace runs are touched and the easy cloud rides on pace alone
  (`LR_EROSION_RATE = 0.001`, `LR_EFFORT_E0 = 0.95`). Watch runs use measured
  stops; pre-watch runs impute the global **P90** stop structure (the uncertainty
  of an unobserved run). **This is an uncertainty model, not a physical recovery
  model** — the earlier durability + W′-balance "pause penalty" (and the crossover
  and longest-segment attempts) over-credited every run at the W′-redline, were
  non-monotone, and were retired. The companion WA conversion is now a monotone
  PCHIP through the real race anchors (5K/10K/HM/marathon), dropping the
  inconsistent 15/20/25/30 k tabs. **See `long-run-pause-uncertainty-reference.md`
  for the full model, the rejected alternatives, and the calibration (incl. the
  2020-11-06 hard case).** The model's PHYSICAL + STATE terms ARE always applied (Max,
  June 2026): resid = raw − (elev + temp + race-fatigue contributions) —
  verified, physically grounded effects, the same family as the hills'
  Minetti term. Only the INTERCEPT (the long-run effort level,
  ~+23 on the race-equivalent scale) is never subtracted — that's the
  class constant that claimed long runs out-predict races. ALTITUDE was
  removed from the model (June 2026): on the race-equivalent scale its
  beta flipped sign (−0.47, "altitude makes you faster") — era/route
  confounding again (Boulder-era runs are Boulder-era effort policy) —
  and a cross-check on workouts found it honestly dead there (β −0.37,
  t = −0.6 after Manhattan Track's 5400 ft was filled in — the one
  at-altitude track tripled the regressor's precision, se 2.1 → 0.64,
  without moving the estimate; the 2σ band [−1.7, +0.9] s/mi per
  1000 ft excludes even textbook altitude costs, consistent with
  residuals being measured against a CS curve whose races already
  absorb altitude), so there is no identified altitude effect anywhere
  to pin from. The same-model-on-workouts spike (June 2026) came back
  NULL across the board: temp +0.05 s/mi/°C (t = 0.7; +0.06 even
  restricted to sustained tempo/fartlek/hill efforts), race fatigue
  −0.5 (t = −0.2, LOO sign-unstable) — fit on 362 quality days with
  category intercepts, temps fully populated. Quality days are environment/state-insensitive in this
  corpus (effort compensates, and scheduling avoids hot/fatigued
  states), unlike long runs (temp +0.37, strongly identified) and
  recovery. So workouts and hills enter TQ with NO covariate
  adjustment — not because it would be a class constant (it wouldn't),
  but because nothing is identified to subtract. Long runs
  enter the smoother at that adjusted residual with a pooled scatter
  weight ((sd_ref/sd)², ~0.53) and ride the shared track-relative prune
  like every other category (the slow 2016 education-hill runs prune
  there naturally; the model's internal MAD prune only robustifies its
  betas). The TQ sidebar table documents the applied terms. The
  dashboard's familiar-route recency mean still uses raw_resid, and its
  20-mi card re-applies β on the way back out (remove the fade going
  in, restore it going out).
- **Long-run watch reconciliation (June 2026).** Long runs were the last
  slice projecting from raw logged values, and a cluster projected
  *faster than any race ever run* (worst corrected residual −48.8). Root
  cause: logged-distance inflation on the two Nashville staples until Max
  re-measured them in April 2022 — log/watch distance ratio 1.09–1.17
  there vs a tight 1.05 baseline everywhere else. Three fixes, owned by
  `src/coros/long_runs.py` (artifacts `long_run_measured.csv`,
  `long_run_calibration.csv`) and consumed in
  `workouts.project_long_runs`:
  - **Distance calibration** (the dominant fix): watch-covered days
    project from `watch_mi·(1+slope) + intercept` — the profile's
    logged-excess curve (`0.262 + 0.0350·mi` for Max; fixed endpoint/
    turnaround slack + proportional GPS underread), MAD-prune-fit on the
    paved recovery+long corpus (the prune ejects the mislogged days
    without a manual list; trailing-strides recovery days are gated out
    explicitly — Max pauses the watch for the strides, so the logged
    miles sit ~+0.32 mi above the recovery-only watch distance, inside
    the prune's reach). Recovery and
    long runs sit on ONE curve — long-run logging is not systematically
    worse than recovery logging. **Application is paved-gated** (June
    2026): trail/mixed/un-typed terrain keeps logged values — GPS
    corner-cutting under tree cover is a route property the paved-fit
    curve can't speak for (10 previously-corrected non-paved days
    reverted).
  - **True times**: watch moving seconds replace pace×miles (logged
    times are minute-rounded down, median −32 s, and exclude pauses).
  - **Pause-aware D_eff**: segments split at watch pauses/standstills
    feed the same `exp(−rest/RECON_TAU_S)` connected accumulator as
    enriched workouts (no anaerobic deflation — long-run segments are
    sub-CS, so the June 2026 effort-aware correction is a no-op for
    them by construction). Median pause is ~15 min/run and D_eff frac ~0.59, but the
    hyperbola is nearly flat at long-run distances so this is only
    ~+1–3 s/mi; it's kept for correctness, not magnitude.
  Pre-watch runs on the two staples (2018-01 → 2022-04-15) get a
  route-era deflation instead (`MISLOGGED_ROUTES`: belle meade & greenway
  both **1.05** flat, pinned 2026-06-17). The real over-logging is
  watch-confirmed at ~5% for both (pre-2022 logged/watch median ÷ the
  post-2022 accurate-logging baseline ≈ 1.057/1.055), essentially flat
  across intensity — the steeper raw trend was a watch GPS-undercount
  artifact that doesn't apply to pre-watch runs. It is **per-route, not
  blanket**: Lake Sammamish's watch ratio is 1.007 (accurately logged), so
  it gets no correction — a flat pre-watch haircut would wrongly deflate
  it. The constant lives in `recovery_model.py` so the recovery fit applies
  the same rules to those routes' recovery rows. Watch enrichment wins over
  the rule. (Distance correction is separate from the pre-watch PAUSE
  penalty above, which imputes the P90 stop structure onto every pre-watch
  run; together they pull the pre-watch Nashville long runs off the
  frontier while leaving accurately-logged routes' distances intact.)
  Distance bracket: true distance is bracketed `watch <= true <= logged`
  — the watch under-reads, the hand log is the ceiling Max never
  under-estimates — so `corr_mi = min(watch+error, logged)`. Time is the
  anchor (it IS the watch time, the log just rounds it to the minute), so
  corr_time stays the watch time and corr_pace is the pure derivative
  corr_time/corr_mi, never clamped. Days the watch beat its average error
  overshoot the log and clamp to it (effectively uncorrected). Replaced
  the old pace-floor clamp (June 2026), which bounded the derivative not
  the distance — letting corr_mi exceed the log and fabricating a phantom
  distance on the days it bound.
  Corrections are projection/display-side only — logged columns are
  never rewritten; the Long Runs plot defaults to logged values with a
  Watch-correction toggle and an opt-in Show-tags ring overlay
  (light blue = snow, yellow = partner-paced — both excluded from
  Training, matching the workout/recovery exclusion methodology;
  gray = watch-enriched, red = rule-corrected without watch data,
  amber = Training's track-relative prune exclusions, vs-track residual
  in the tooltip). A "spurious log" >1σ ring was tried and killed
  same-day: it hit more than half the watch-enriched corpus — an
  arbitrary threshold, and the Watch-correction toggle already shows
  each adjustment directly. Corrected rows stay IN the smoother —
  corrected, never dropped. Combined with the race-equivalent
  projection (previous bullet), no long run projects meaningfully past
  CS anymore (pre-correction the worst claimed a 5K ~2 min faster than
  the PR). Known softness: the 2018–19 rule extension may overcorrect
  by ~10 s/mi (corrected rows sit slightly slower than untouched
  same-era routes); era-specific factors are a possible refinement.

## Pipeline (April 2026)

### Stage 1 — Workout decomposer (`parse_workouts.py`)

Reads `daily.csv` and decomposes quality workout strings into a structured
table: `workout_decomposed_v7.csv` (currently 259 rows). Same schema as the
older `workout_vdot_v6.csv`, with these rule changes:

- **Continuous-fartlek classification**: fartleks 6400–10000m with zero or
  no rest annotation are classified as continuous_fartlek (not interval).
  Boundary at 6400m fixes 2024-03-09 (6800m) and 2024-07-07 (8000m).
- **Continuous-fartlek structure (June 2026)**: every CF day decomposes as
  its known alternation — **500m hard / 300m float**, truncating at the end
  for non-divisible distances (a trailing partial is hard; the one 8200m
  day reads `10×(500+300f) + 200m`). The float:hard pace ratio is
  hand-pinned at **1.25** — dimensionless so it transfers across eras —
  measured 1.253 bin-by-bin on the one watch-covered CF day (2024-07-07:
  hard 5:08/mi even, float 6:26/mi remarkably consistent; the float runs
  ~34 s/mi SLOWER than era recovery-run pace, so a recovery-model anchor
  would have misfired). Hard pace falls out of the blended log pace in
  closed form; the floats act as jog rests in the same effort-aware
  connected accumulator every structured workout uses. Moved the CF
  cluster from +18 to ~+11 raw under g(d). *(June 2026, CP3 unification:
  the distance-only g(d) — whose flat +10.3 s/mi charge on 5K-effort 500s
  was the documented structural compromise — is replaced by the
  effort-aware supra-CS deflation; CF median now +7.2 vs interval −0.9.
  The remaining gap is honest: CF hards genuinely run ~0.3 m/s supra-CS,
  plus 2019–20 era effort policy.)*
- **Implicit decomposition**: workouts written as bare `Nt` / `Ni` / `Nf`
  with no `Nx` reps decompose to standard rep distances (interval ≥ 4800m
  → 1600m reps; 3200–4799 → 800m; rep → 400m; tempo < 7000 → 1000m,
  ≥ 7000 → 1600m). Well-understood staples decompose to their real
  structure (June 2026): **6400t = 4×1600** (the 2017 weekly staple;
  5000t = 5×1000 already falls out of the <7000 rule; 4800f has its
  hardcoded ladder).
- **Default rests**: tempo 60 s/mi, interval 140 s/mi (800m) or 180 s/mi
  (other), rep 420 s/mi, continuous_fartlek 0.
- **Rest estimation** (`build_rest_model` / `effective_rest_per_mile`): rest per
  rep drives the connected-accumulator decay, so it matters. Policy: **2020+**
  interval/rep rest = this profile's OWN watch-measured median by 100 m rep-
  distance bin; **pre-2020** trusts the logged rest when present; **pre-2020 with
  NO logged rest** now falls back to that same **watch per-rep-distance median**
  (June 2026) — keyed by rep distance, so a `10×500` gets far more rest/mile than
  a `4×1600`, instead of the old flat per-type median that under-rested short-rep
  days logged as intervals. The hardcoded defaults above are the last resort.
- **Continuous tempo from the watch** (`reps.extract_tempo_day`, Aug 2026): a
  tempo logged without a `(0:00 rest/mi)` annotation used to take the 60 s/mi
  default *untrusted*, which cost it twice — no connected `d_eff`, and a failed
  `continuous` course-trust rescue in Gate 1, so it landed as `uncertain course`
  and left TQ entirely. (2026-08-25 `47j, 5000t@5:58, 14j` vs 2022-07-29
  `15j, 5000t@5:36 (0:00 rest/mi), 16j`: the annotation was the only difference.)
  The watch settles it. A steady tempo is invisible to block detection — the
  detector's cutoff is CS pace + 20 s/mi (~5:25/mi in Aug 2026) and Max's 5000t
  sits at 5:36–5:58, so every window in it reads as jog and `extract_day`
  returns `no-subset`. But a tempo doesn't need block detection: the log already
  fixes distance and pace, and the only open question is whether the block ran
  unbroken. So we find the single moving segment whose duration matches the
  logged quality time (±5%, with the watch span within 0.85–1.15× the logged
  distance), confirm nothing stands still inside it, and emit ONE rep at the
  LOGGED distance and the MEASURED time — the same division of labour as the
  hill-loop path. The day becomes watch-verified, rest = 0, `d_eff` = the full
  block. Every guard can only reject, and a rejection falls through to the
  ordinary pipeline: 2024-05-04's `10000t@4:56` was really 3×3200 with ~3 min
  rests, no single segment comes near its 1839 s, and it still decomposes the
  old way. Where there is no watch record at all, a bare `Nt@` stays
  `uncertain course` — that is what the gate is for.

Decomposer-level prunes (24 rows go to `workout_pruned_v7.csv`):
2016-07-11 anomaly; tempos with paces over 10:00/mi; `qd < 100`; continuous
tempos with explicit `0:00` rest; sub-4000m fartleks with no/zero rest.

### Stage 2 — Workout projection

Per workout (in the plot pipeline):
1. `decay = exp(−rest_per_mile / 210)`
2. `D_eff = rep_dist · (1 + (rep_count − 1) · decay)`
3. `t_eff = pace_per_mile · D_eff / 1609.344` (seconds at workout pace)
4. `t_5K = cp3_time(5000, cp3_implied_cs(D_eff, t_eff, D′₃_t), D′₃_t)`
   where `D′₃_t` is the CP3 reservoir at the workout date (June 2026 —
   was the plain hyperbola `(5000 − D'_t)·t_eff/(D_eff − D'_t)`)
5. `P5K = t_5K · 1609.344 / 5000` (sec/mi)
6. `raw_resid = P5K − CS_implied_5K_pace_at_t` (sec/mi)

Steps 1–3 are the legacy uniform-rep FALLBACK (no watch data, untrusted
rest; `/210` is a flat per-mile decay). Trusted/enriched days instead
carry `d_eff_m`/`t_eff_s` from the connected accumulator
(`parse_workouts._connected_core`) — the primary path — which uses TWO
distinct time constants (don't conflate them):

- **Reconstitution τ (intensity-dependent, June 2026):** between reps the
  carried-over "connection" decays by `exp(−rest_s / recon_tau(v/cs))`,
  where `recon_tau` SHRINKS as the rep sits further above CS — deep-
  anaerobic reps (W′/PCr) reconstitute fast (small τ → connection resets →
  small D_eff → cool); near-CS efforts reconstitute slow (large τ → D_eff
  preserved). `τ = 662·exp(−3.0·(v/cs−1))`, clamped to Skiba's
  reconstitution band [316, 862] s. CS-relative, so it self-normalizes with
  fitness. See the `recon_tau` block in `workouts.py` for the physiology and
  the non-Max-profile defensibility. (Replaced a flat τ, which decayed by
  absolute rest and cooled a near-CS 1000 m interval as hard as an all-out
  400 m rep off the same rest.)
- **Deflation τ (CP3, unchanged):** each rep's supra-CS speed is scaled by
  `t/(t+τ_defl)`, `τ_defl = D′₃/(v_max−CS)`, CS the workout's own implied CS
  solved as a fixed point (D_eff now inside that fixed point), first rep
  exempt — the effort-aware anaerobic deflation that replaced g(d).

The workout `v_max` is the **race EVIDENCE edge** — `vmax_evidence(cs)`, i.e.
`k_evid · CS(date)` (Max k_evid = 1.97; per-date, not a constant). Workouts read
efforts as evidence exactly like races, so they share that edge — see
[cs-model-reference.md](cs-model-reference.md) for how k_evid is derived (the
smallest CS-multiple keeping short races behind the aerobic frontier). The
separate measured workout cap (the old flat 8.7, with a per-profile median-CS
scaling for others) was retired (June 2026): above ~800 m it proved empirically
inert in the 5K-equiv (the deflation and projection channels cancel), so a
distinct workout value bought nothing.

### Stage 3 — Long run projection

Filter: `run_type == 'long'` AND `duration ≥ 80 min` AND `miles < 26.2`
(`LONG_MIN_MINUTES` / `LONG_CEIL_MILES` in `workouts.py`, global across
profiles; duration = pace × miles, the same quantity the projection
uses). The time floor separates honest long-run effort from shorter
aerobic work at the fueling threshold; the distance ceiling excludes
marathon-distance-plus endurance events that aren't race prep — see
"Decisions locked in" for both rationales. Out-of-scope long runs
aren't displayed in the plot and don't feed the smoother.

Partner-paced long runs are excluded too (June 2026,
`excluded_reason='partners'`): same population logic as the recovery
fit's partner prune — a long run paced by someone else's targets isn't
the runner's own effort policy. Solo and varsity are admitted
(`ADMITTED_PARTNERS`; in 2016-17 the varsity group's recovery pace WAS
Max's pace strategy); 13 of Max's long runs carry the flag. Snow takes
priority when both apply. (Workout partners remain a TRUST signal in
the course-verification gate, not an exclusion — a partnered workout is
still the runner's own quality effort.)

Per in-scope long run:
1. `t_run = recovery_pace_sec_per_mi · miles` (seconds)
2. `D_eff = miles · 1609.344` (continuous, no decay)
3. `t_5K = cp3_time(5000, cp3_implied_cs(D_eff, t_run/β_long, D′₃_t), D′₃_t)`
   (June 2026 — same CP3 swap as workouts; sub-s/mi effect at long-run
   durations, kept for cross-layer consistency)
4. `P5K = t_5K · 1609.344 / 5000`
5. `raw_resid = P5K − CS_implied_5K_pace`
(There is no per-distance bin: the former 21mi lr_lo/lr_hi split was
retired in June 2026 — its original Δ AIC = −52 justification was
computed inside the route-dummy model and didn't survive its removal;
the bin was absorbing route/era mix, not a glycogen regime change. See
"Decisions locked in".)

Long-run residuals are then corrected by an OLS fit (Stage 5b); hills
by the hill model (Stage 3.5); workouts carry no label corrections at
all (Stage 5a).

### Stage 3.5 — Hill continuous projection + hill model

Filter: `run_type == 'hill_cont'` AND the loop has a surveyed
`distance_m`, elevation data, and a terrain class — **covariate-based,
no per-loop session-count gate**, so a brand-new route anywhere
qualifies on its first session. Loop is parsed from the workout string
(`hc-Nx <loop>`); 4 sessions in May–June 2018 missing the inline loop
token are recovered from the `location` column ("rollercoaster" → rc,
"powerline west" → pwr1). Two `hc/rep` hybrid sessions (Sept 2016) are
dropped at parse time.

Per session:
1. `total_dist_m = nreps · loop_distance_m`
2. `actual_pace = (session_min · 60) / (total_dist_m / 1609.344)`
3. CS-hyp projection on `(total_dist_m, actual_pace · total_dist_m / 1609.344)`
   → `t_5K`, `P5K`, `raw_resid` as in Stage 2.
4. Category: `hill_<loop>` — informational grouping only; corrections
   come from the hill model, never from per-loop categories.

**Watch-measured time override (June 2026).** For watch-era days,
`reps.py::extract_hill_day` locates the loop block in the day's GPS
streams (the loop point is found by anchor-crossing search, constrained
by the logged loop count; the regular crossing run isolates the block
even when warmup/cooldown share the recording) and measures its exact
moving time; mid-block watch pauses are subtracted and per-loop splits
recorded (display-only). The override replaces `session_min · 60` —
which is whole-minute quantized (±30 s ≈ ±7 s/mi) and silently includes
standing rest — with measured seconds. **Distance is never taken from
GPS** (reads 2–9% short on these loops); `nreps · loop_distance_m` stays
authoritative, only time moves. Measured 2021–2026 corpus: 24/24 days
extracted with full per-loop splits; measured-vs-logged delta median
+12 s (range −33..+26), i.e. exactly the rounding band. A fitted watch
term came out −2.3 s/mi ≈ noise, so measured and log-timed days share
one model. Pre-watch days are untouched.

**Hill correction (June 2026, `src/shared/hill_model.py`): pinned
Minetti net cost + ONE fitted trail term.** Physical terms replaced
per-loop offsets, the same overhaul the long-run model went through: a
loop's empirical beta encoded "how hard that loop's era was run", and
Love Circle — a moderate paved loop — carrying a +32 s/mi beta was the
tell. The gain correction is PINNED from mechanism (the long-run pinned
elev-slope precedent): Minetti 2002 energy cost of running at gradient
i, applied as a multiplicative NET factor for a loop that climbs and
descends — `(C(i)+C(−i))/2C(0)` with `i = climb/(loop/2)` (symmetric
half-up/half-down approximation; the hills sheet records climb and
distance, not climb fraction). Net cost ≈ +14 s/mi for lc, +26 rc, +48
pwr1 — descents give most of the climb back at moderate grades, which a
free-fitted gain slope ignored (it absorbed effort gap and read a
5:09-pace lc day as a 4:28/mi 5K-equiv; rejected as not grounded —
Minetti reads it 4:55, matching intuition). The only FITTED term is the
binary trail-vs-paved difference of Minetti-corrected residuals,
currently **+26.9 s/mi** (rocky/gravel descents don't give the climb
back; era-confounding caveat: pwr1 is all 2016-18). There is NO
intercept and hills are NOT centered — the hill-class effort gap (~+23
s/mi for the main loops) stays visible like tempo-era effort policy
does. Prune inside the fit is iterative one-sided MAD (drops only
egregious easy days; currently 1). The trail coefficient is persisted
to `hill_model.csv`; the Workouts plot recomputes the Minetti factor
from loop covariates and subtracts the same trail term.

Minetti is deliberately NOT the shared two-channel elevation engine:
replacing it was investigated Aug 2026 and rejected — over-corrects hc
~1.7×, under-prices hill-rep climbs, out of calibration support at
workout grades/verticals, and the descent refund doesn't transfer to
workout effort. See "Two-channel elevation engine for hill workouts"
under Considered and rejected.

### Stage 4 — Corrections

Applied before projection / centering:

- ~~Tempo → Interval reclassification~~ **removed June 2026** — it was a
  pre-watch band-aid (visual-only; tempo and interval share the same
  projection) and the per-category offsets it papered over are gone too.
  Type is logged intent, preserved end-to-end.
- **Snow filter**: drops sessions with `snow` in `workout_raw` or
  `conditions`. Removes 4 workouts and 1 long run; surface conditions
  invalidate pace projection. (Note: 2026-01-19 and 2026-01-22 were on
  ice, not snow — to be reclassified in next data freeze.)
- **XC −6% pace correction**: applied to (a) all sessions in the HS XC
  season window 2016-07-01 through 2016-10-31, and (b) any tempo logged
  with `quality_distance == 5000m` (HS 5K course as 5×segments, three
  summer 2017 entries). **Track locations are categorically exempt**
  (June 2026): both rules target XC-course efforts, and a workout on an
  actual track is neither — 2021-04-26, a 5000m track tempo at Rose
  Park, was the one mis-hit (read −0.3 corrected; reads +18.6 fixed).
  Pace divided by 1.06 before projection. Tooltip marks corrected
  sessions with `[XC-corrected −6%]` in light blue.

### Stage 5a — One shared CS predictor + track-relative prune

**Per-category offsets were REMOVED in June 2026.** Every quality workout
(interval / tempo / rep / cont. fartlek) shares one CS predictor with no
label terms. The old per-category medians (tempo +19.7, cf +17.9,
interval +1.3) were label dummies absorbing **era effort policy**: 41 of
49 tempo days are 2016–17 (classic threshold tempos, medians +28/+17);
the only watch-measured tempo (2024-05-04, long intervals with short
rest) reads +1.7 — interval-grade — yet inherited the −19.7 era discount
and displayed as one of the fastest workouts in the set. CF is entirely
a 2019–20 block at +18 in its era; interval is stable ~0–8 across all
eras (TAU was calibrated on intervals, making them the anchor). The gap
is **intent-as-executed per era** — exactly the "training ahead of /
behind capability" signal this plot exists to show — so it stays in the
residuals, same precedent as the long-run route-dummy removal.

There are **no class constants either** (a brief global-median centering
was removed the same day): every point on the Training graph and the
Workouts graph is the *same number* — the best attempt at predicting 5K
race pace from that session. Workouts enter raw; hills enter
Minetti+trail-corrected raw; long runs enter via their model's corrected
residual (known to have its own issues — one long run currently implies
4:10 pace — to be revisited). Per-category medians are printed as
**diagnostics only** (effort policy, not corrected).

**Track-relative prune.** Outliers are judged against the *surrounding
data*, not against CS or a label baseline: fit the smoother, detrend
every workout/hill point by the track value at its date, drop the slow
side beyond `median + PRUNE_SIGMA·MAD` (σ=3.0) of the detrended
residuals, refit; iterate to a fixed point (converges in ~2 passes). An
era-soft session sits near its era's soft track and survives; a session
that sticks out from its own surroundings goes (currently exactly one:
the 2017-07-17 +69 tempo, +47 above even its local track). Long runs
contribute to the track but are pruned by their own model (Stage 5b);
hills by the hill model's internal prune (Stage 3.5).

Reps and hills are noisier CS signals, so they enter the smoother
scatter-weighted (`(σ_ref/σ)²` clipped to [0.1, 1.0], σ_ref = non-rep
workout residual SD), recomputed inside the prune loop — currently reps
0.79, hills pooled 0.55.

**Two independent gates (June 2026 redesign).** A workout enters TQ and may
bind the frontier only if it clears BOTH. (Replaces the earlier single
"uncertain accuracy" course gate + implausibility ceiling.)

**Gate 1 — uncertain COURSE (where did it happen?).** The projection is only as
trustworthy as the course measurement, and mismeasurement cuts both ways (solo
2020 Powerline intervals read fast on a short gravel course, never replicated
once the watch arrived). Trusted iff **watch-verified** (status exact/watch-only)
∨ **track** (`terrain_type=='track'`) ∨ **varsity partners**. Generic training
partners are NOT enough — a casual group run on an unmarked course is as
mismeasurable as a solo one (this tightened the old "any non-solo" rule). Two
course-known rescues persist: **continuous efforts (0 rest)** — one unbroken
course can't be mismeasured like back-and-forth reps — and the **pre-2018
staples** (`5000t`/`6400t`/`4800f`, run on the HS-era "education hill" that was
really the RHS track; the `≤2017` cap bounds that *location* inference, NOT the
decomposition — see Gate 2). Else → `uncertain course`, dropped from TQ.

**Gate 2 — uncertain STRUCTURE (do we know the rep layout?), binding-
conditioned.** Replaces the implausibility ceiling. Structure is KNOWN only from
watch data, an EXPLICIT `Nx` log (legacy "10x500i" style — vs a bare total like
"2800f" the parser must GUESS into reps), a continuous effort, or a **hardcoded
fixed decomposition**: the `5000t`/`6400t`/`4800f` staple codes in **any year**
(4800f is Max's standard ladder, `parse_workouts.LADDER_4800`) and the
continuous-fartlek 500/300 pattern. Estimated structure is ACCEPTED for the bulk;
it is distrusted ONLY when the workout would **bind** the frontier. Mechanism
(`performance_frontier.gate_estimated_binders`): build the frontier floor from
VERIFIED-structure demos only (kept races + verified-structure workouts); any
estimated-structure, course-OK workout poking above that floor → `uncertain
structure` (excluded, non-binding). Rationale: a guessed layout claiming a
frontier-defining result probably hid shorter reps (e.g. 2017-03-28 varsity read
4:46/mi, years before the real peak; a `2800f` fartlek the parser split into
7×400). Estimated workouts sitting below the floor ride along untouched.

Both flags stay visible on the Workouts plot (`uncertain course` red-orange,
`uncertain structure` magenta); `training_quality_exclusions.csv` annotates every
TQ-excluded session (snow, uncertain course/structure, or outlier cutoff). One-off
mislabels are fixed at the SOURCE (recategorize the log + refreeze) rather than
with code exceptions — e.g. a 2017-04-25 session logged `4800f` but not actually
the ladder was relogged `4800i`, which the structure gate then handles on its own.

### Stage 5b — Long-run model (covariates only)

Long runs use an OLS fit on the in-slice set instead of a per-category
median offset:

```
raw_resid ~ temp_centered
            + fat_race  (= fat_race_short + ratio·fat_marathon,
                         ratio pinned from the recovery fit)
```

- Distance carries no term (June 2026 sweep — see "Decisions locked in");
  the lr_lo/lr_hi labels are fully retired.
- **Physical route terms no longer live here (June 2026, watch-stream
  enrichment).** `long_run_model.py` carries only temp/fatigue;
  `LR_ELEV_SLOPE`/`elev_pm_c` and the fitted altitude term have been
  removed. Grade, off-road footing, and altitude are now applied upstream
  in `workouts.project_long_runs` as flat / sea-level-equivalent time
  corrections — subtracted from the run's time *before* the World Athletics
  5K-equivalent conversion — rather than as residual regressors. Grade is
  each run's per-run measured gain/loss priced through the shared
  `elevation_cost` engine with an effort-aware paved descent refund
  (one grade-resolved refund curve at every effort — the effort schedule
  was retired Aug 2026, see route-normalization-reference.md). Footing
  (trail_frac flat penalty) and altitude
  (≈ +2 s/mi per 1000 ft above a ~3000 ft threshold; per-run measured
  via `per_run_altitude`, not per-city) are pinned from
  `recovery_model.physical_route_betas` — the single source of truth
  shared with the recovery and race-CS models. The effort ranges of
  recovery and long runs coincide (~0.85 frontier-pace fraction), so the
  transfer is constant-effort (no scaling); channels stay separate
  (grade / footing / altitude) with no double-count, under the same
  validity gate and location-terrain bucketing. → see
  route-normalization-reference.md (elevation engine) for the
  `elevation_cost` cost formula, the altitude threshold curve, and the
  `physical_route_betas` internals.
- Covariates reuse the recovery model's encodings —
  `temp_centered = temp_c − 12`, `fat_<cat> = exp(−days_since/τ)` with
  τ = 6d (marathon) / 5d (short race). Temperature's beta is fit on the
  long runs themselves. Race fatigue is a single fitted scale on
  `fat_race_short + ratio·fat_marathon` with the ratio pinned from the
  recovery fit (see "Decisions locked in"); `cov_coefs` exposes the
  ratio-expanded per-category betas so `transferable_contributions`
  and the sidebar see the familiar two-key shape. Missing temp falls to
  the reference (contribution 0) so no row drops from the fit.
- All terms gate on ≥ `MIN_COV_N = 30` in-slice runs; physical terms
  additionally require variation in the data (watch profiles without
  route metadata degrade to an intercept-only fit).
- Iterative MAD-based outlier prune at σ = `PRUNE_SIGMA = 3.0` on the
  corrected residuals (`raw_resid − model.predict()`). Pruned rows
  are dropped from the figure entirely; they don't feed the smoother
  and aren't rendered.
- `route` / `qualifying_routes` (locations with ≥ `MIN_ROUTE_N = 5`
  in-slice runs) are still emitted as descriptive labels — the
  dashboard's familiar-route bin-residual filter and outlier reporting
  use them — but carry no coefficients.

The Long Runs tab's "Normalize" toggle reuses this fit's covariate betas
(via `recovery_model.transferable_contributions`) to shift every plotted
long run — including out-of-slice ones — by its modeled temp + fatigue
contribution.

Caveat to keep in mind: with route dummies gone, era effort variance has
nowhere to hide, and some of it leaks into whatever correlates with era —
the race-fatigue betas roughly tripled (race-dense blocks were
quality-effort blocks) and temperature dropped to ~0. The fatigue terms
are still day-scale (decay τ ≈ 5–6d) so the damage is bounded, but
re-examine the betas once more cross-era data accumulates or after
elev_per_mile gaps are filled.

### Fitted constants / diagnostic medians (June 2026)

These drift as data accumulates — regenerate by running
`plot_training_quality.py` and reading the console output.

Per-category median raw residuals, **diagnostic only** (era effort
policy, deliberately left in the signal; nothing is subtracted):

| Category            | Median resid (sec/mi) | Reading                              |
|---------------------|----------------------:|--------------------------------------|
| interval            |                 −0.91 | The calibration anchor               |
| rep                 |                 +1.36 | effort-aware-deflated, scatter-weighted |
| continuous_fartlek  |                 +7.18 | 500/300-reconstructed (was +11.6 under g(d), +17.9 before that) |
| tempo               |                +19.31 | Mostly 2016–17 era effort policy     |

*(June 2026, post-CP3: reps converged fully — +5.43 → +1.36, now sitting
between interval and CF; CF narrowed +11.6 → +7.2, the remainder being
honest supra-CS effort + era policy.)*

Hill correction (Stage 3.5): Minetti net factor pinned (lc ≈ +14 s/mi,
rc +26, pwr1 +48 at typical paces); fitted trail term **+26.94 s/mi**
(n_kept=126, resid SD 15.6; persisted to `hill_model.csv`). CF float:hard
ratio pinned at **1.25**. Scatter weights: reps 0.97, hills pooled 0.54.

Long-run model (Stage 5b), covariate fit (physical route terms have
moved upstream to `workouts.project_long_runs` — see Stage 5b above —
so `elev_pm_c`/`altitude_kft` are no longer coefficients of this fit):

| Term                        | Coef (sec/mi) |
|-----------------------------|--------------:|
| Intercept                   |       +33.71  |
| temp_centered (per °C)      |        +0.28  |
| fat_marathon (peak)         |       +40.41  |
| fat_race_short (peak)       |       +22.24  |

The pinned physical constants (trail_frac footing, altitude slope
s/mi per 1000 ft above ~3000 ft) come from
`recovery_model.physical_route_betas`, not this fit; regenerate the
covariate coefficients above by running `plot_training_quality.py`. The
R² drop vs the route-dummy model (0.605) is **intentional**: the dummies
were explaining era effort policy, which is signal the graph should
display, not variance the model should remove.

Total kept after all filters and prunes: 235 workouts + 268 long runs
+ 126 hills = 629 (June 2026; 37 days carry the uncertain-accuracy flag).

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
- Hover: shared smart-spikeline scaffold (`_scaffold/cursor_tooltip.js`),
  opted into via `cursor_tooltip=CursorTooltip(...)` on `render_plot()`.
  Plotly's per-trace hover is disabled (`hoverinfo='skip'` everywhere);
  the plot supplies a precomputed daily payload (CS array, smoother
  array with NaN in gaps, sessions list) and a `buildTooltip(day)` body.
  At any cursor x the tooltip shows the date, race fitness (CS),
  training quality (smoother), diff in s/mi, and the nearest session
  with day-difference label and full session detail. Snap mode (cursor
  near a session marker) jumps the spikeline to the marker's x and uses
  the marker's customdata HTML.

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
3. **Era-varying effort prescription.** Continuous fartleks happened only
   2019–2020; HS-era tempos were prescribed differently than modern ones.
   *(June 2026: this is exactly why per-category offsets were removed —
   effort policy now shows in the residuals instead of being averaged
   into a label constant. It still adds noise to per-race predictions.)*

## Considered and rejected

- **Duration term instead of category offsets (June 2026).** Tested
  `raw_resid ~ ln(D_eff/5000)` as a label-free replacement for the
  per-category offsets: it halves the tempo–interval gap but leaves
  tempo +12 above interval, and there is NO within-category duration
  signal (spearman ≈ 0 inside tempo and interval) — the gap is a step
  between labels, not a smooth duration effect.
- **Longest-continuous-piece / structure term (June 2026).** Broken
  3–5-min-piece tempos (n=29) still read +19 vs intervals −1 at the
  same piece length — structure doesn't explain the gap either. What
  does: the label encodes intent-as-executed *per era* (see Stage 5a),
  which is signal, so no term replaces the offsets — they were simply
  removed.
- **Free-fitted hill gain slope (June 2026, lived a few hours).**
  `raw_resid ~ intercept + ft_per_mi + is_trail` fit the observed loops
  but was not grounded: it treated all climbing as pure cost (no descent
  payback), so the slope (+0.36 s/ft) absorbed effort gap and the
  negative intercept (−21.6) compensated in-range — a 5:09-pace lc day
  displayed as a 4:28/mi 5K-equivalent. Gain and terrain are also
  correlated across the six loops (pinning the intercept collapses gain
  to +0.09 and balloons trail to +42), and pwr1 alone pinned the steep
  end (LOO miss +70 s/mi). Replaced by the pinned Minetti net cost —
  which lands precisely on intuition (the 4:28 day reads 4:55).
- **Minetti-style grade adjustment (pre-June-2026 rejection).** Rejected
  when per-loop offsets existed (any uniform per-loop shift collapsed
  into the offset, making it redundant). Adopted June 2026 once the
  offsets were gone — it is now the pinned gain correction.
- **Two-channel elevation engine for hill workouts (Aug 2026).** The
  natural follow-on to the two-channel overhaul — one grade model for
  races, long runs, recovery AND hill workouts — investigated and
  rejected; the engine fails "at least as good as Minetti" in opposite
  directions on the two workout types. **hc:** swapping the engine into
  `project_hill_continuous` (surveyed geometry, floor-excess convention,
  symmetric split — validated against measured hill segments on the 24
  watch days: lc ~5.0–5.5% vs surveyed 5.7%, rc 7.4% vs 7.5%) corrects
  ~1.7× Minetti at every loop (lc 22 vs 13 s/mi at 5:30 pace, pwr1 62 vs
  38) with no dispersion win (resid SD 16.0 vs 15.7, n=127). It breaks
  the cross-loop effort story Minetti produces: trail-corrected gaps
  cluster under Minetti (lc +26 / rc +30 / pwr1 +23 — one hill-effort
  class, and watch-era lc +20.8 ≈ watch-era tempo +21.7) but scatter
  under the engine (lc +17 / rc +24 / pwr1 +4.5 — two same-era trail
  loops 20 s/mi apart, pwr1 at race effort), while half the trail term
  (+25.1 → +13.3) disappears into the steeper grade pricing. Days
  beating the CS line go 6/127 → 16/127, and the pinned intuition anchor
  fails: 2021-12-19 (4mi @ 5:09 on lc) reads 4:47/mi vs Minetti's 4:55
  (line 5:04) — the free-slope failure mode, milder. Root cause is the
  engine's own transfer rule (route-normalization principle 4): the
  descent refund doesn't transfer across effort — b(g) was measured on
  cautious easy-run descents; hc descents are attacked and reclaim what
  Minetti's energy symmetry says they do. **hr:** the climb-only bridge
  (descent is rest, so principle 4 is not violated) under-prices steep
  climbs — linear c(g) extrapolated to 10.3–10.8% sits below Minetti's
  in-domain convex curve (GAP factors 1.54/1.59 vs 1.68/1.72), slowing
  the two measured days' 5K-equivalents 5:15→5:46 and 5:35→6:08, i.e.
  +49/+73 s/mi slow of the line — no effort class lives there. Both
  types also sit at/outside calibration support (loop blocks 150–267
  ft/mi at 5.7–10.1% vs mile-gain p99 176 / climb-grade p99 8.1%; reps
  543–571 ft/mi past the hill-grade p99 ~9.6% and near the 12% clamp)
  and invert the engine's perturbative regime (full-run fractions 2–8%;
  hc 7–23%; hr 54–59% — the correction IS the measurement, amplifying
  parameter error 3–10×). Revisit only if a future calibration gains
  support at ≥10% grades with workout-effort descents, which day-FE
  calibration on recovery/long corpora structurally cannot provide.
  Substrate unification (fused-substrate block geometry auditing the
  surveyed loop constants — e.g. the anomalous 2026-03-27 hj day, +79
  s/mi under Minetti, the fit's one pruned outlier) remains open and
  does not touch the cost curve.
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
  rejected at the time. (Superseded May 2026 by the `[15.1, 25.3]` slice
  with a single 21mi internal bin — see "Decisions locked in".)
- **Long-run upper bound at 26mi.** Removed in April 2026. (Re-introduced
  May 2026 as the `25.3` ceiling of the new slice; the n=7
  marathon-distance fade regime was distorting the model.)
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
- **Any distance term in the long-run model (June 2026, two rounds).**
  Round 1 (inside the route-dummy model): linear/hinge lost to the
  21mi bin by ΔAIC +39 — at the time read as "the bin is a genuine
  regime change". Round 2 (a threshold/knot sweep after route dummies
  were removed): the bin's coefficient collapsed +20 →
  +3.5, bin@21 scored *worse* than no distance term (ΔAIC +0.8), and the
  tempting alternatives (bin@17.5, ΔAIC −62; hinge@18.5, −44) are era
  composition in disguise — the sub-17.5mi band's effect flips sign
  across eras (Nashville −13.5 vs Seattle +13.1 / Chicago +16.4 s/mi),
  within-era distance slopes flip sign era to era, and a real
  physiological boundary wouldn't migrate 3.5mi when route handling
  changes. Distance terms removed entirely, and the lr_lo/lr_hi labels
  with them (single "Long" legend entry; the dashboard's long-run
  prediction is a recency-weighted mean, distance-unconditioned, since
  the per-bin empirical means differed by 0.3 s/mi). Round 1's
  bin-vs-continuous result is thereby
  understood as two route/era proxies competing, the coarser one
  winning.
- **Recovery-sourced covariate betas on long runs (June 2026).** Tested
  as variant V3: subtract the recovery fit's temp/fatigue/TOD
  contributions, then fit bin + route. Underperforms fitting the betas
  fresh — marathon fatigue is ~2.3× stronger on long runs, and TOD is
  recovery-specific. Same conclusion as the route-beta decision: TQ fits
  its own coefficients.
- **Time-of-day covariate in the long-run fit (June 2026).** β ≈ +0.5,
  t ≈ 0.25 — dead, despite being a strong recovery factor (−4.8 s/mi).
  Excluded.
- **Empirical per-route long-run betas (removed June 2026).** Served as
  the Stage 5b route correction until the era confounding proved fatal:
  every named route lives in one contiguous era, so betas measured
  "typical effort that year" (physically identical flat routes 31 s/mi
  apart; east boulder at 5,400 ft carried a *negative* beta; a moderate
  2026 run out-ranked the all-time-best 2023 long run after correction).
  Replaced by physical route terms — see "Decisions locked in".
- **Empirically-fit elevation slope in the long-run model (June 2026).**
  Comes out −0.13 s/mi per ft/mi — wrong-signed, because hilly routes
  are concentrated in quality-effort eras; the same confounding that
  sank route dummies, in continuous form. Briefly pinned to the recovery
  cross-route value (+0.17) instead.
- **Route-constant `0.17 · elev_per_mile` slope + fitted altitude in
  `long_run_model.py` (removed June 2026, watch-stream enrichment).** The
  pinned-slope/fitted-altitude pair (the first replacement for the route
  dummies) was itself superseded by the full physical decomposition
  applied upstream in `workouts.project_long_runs`: grade priced through
  the shared two-channel `elevation_cost` engine off each run's measured
  hill segments, plus footing and
  per-run-measured altitude pinned from `recovery_model.physical_route_betas`,
  all credited as flat / sea-level-equivalent time corrections before the
  World Athletics 5K-equivalent conversion. The route constant priced terrain
  as a static per-route slope; the physical engine prices each run's
  measured grade/footing/altitude and shares one constant per channel with
  the recovery and race-CS models. `LR_ELEV_SLOPE`/`elev_pm_c` removed.

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
  bayes_cs_summary_v11.csv → training_quality.html. Also persists
  `data/training_quality_corpus.csv` (the kept post-prune corpus: date, src,
  category, corrected 5K-equiv pace `p5k_corr_min`, detail) — consumed by
  the Fitness plot's performance frontier (June 2026,
  `src/shared/performance_frontier.py`; see cs-model-reference.md), which is
  why this script runs before bayes_cs_plot in run_plots.sh.

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
