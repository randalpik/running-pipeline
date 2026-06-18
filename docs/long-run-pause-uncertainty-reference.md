# Long-run pause-uncertainty model

**Owner reference. Read this before changing how long-run pauses are handled — it
exists specifically to stop us re-litigating a path we have already walked end to
end.** The model is deliberately an **uncertainty** model, not a physical model of
what a pause does to your body. That distinction is the whole point; the sections
below explain why every physical formulation we tried failed.

Code: `workouts.project_long_runs` (effort, gate, projection), `durability.eroded_deff`
(the erosion law), `durability._pre_watch_profile` / `_impute_segments` (pre-watch
imputation), `wa_scoring.py` (the 5K-equivalent conversion).

## What it's for

Long runs are projected to a **5K-equivalent pace** so they can sit on the
demonstrated-capability frontier next to races and workouts (see
`training-quality-reference.md`, `cs-model-reference.md`). The projection treats a
run's flat-/altitude-corrected time as a performance and down-converts it to 5K
via the World Athletics tables.

The danger: the WA conversion scores a `(distance, time)` pair **as if it were a
maximal race**. A long run is sub-maximal and usually **paused** (stoplights,
water, regroups). Converted naïvely, a comfortable 20-miler can out-convert your
actual 5K/HM race PRs — claiming a fitness you never demonstrated. We cannot know,
from a paused run, what continuous performance it proves. **Pauses are uncertainty,
and this model prices that uncertainty.**

## The thesis (read this first)

A pause does not get *modelled as a physical event*. It **erodes our confidence**
in the distance the run demonstrates. The cleaner (less paused) the run, the more
we trust it; the more (and later, and longer) the stops, the less of the run we
credit. Two consequences we hold to:

1. **The erosion law is universal pause-physics** — identical for watch and
   pre-watch runs. A pause is a pause.
2. **Uncertainty about *unobserved* pauses (pre-watch) lives in the imputation**,
   not in the erosion constant. We impute aggressively (P90) for runs we couldn't
   measure. This keeps "what a pause does" cleanly separate from "we don't know
   how much you paused."

## Why every *physical* pause model failed (do not retry these)

1. **W′-balance "rescue" penalty.** Computed how much a run's stops let it beat a
   continuous effort, via critical-speed + W′-balance (effective CS declines with
   time on feet; running above it drains D′; stops reconstitute it). It priced the
   benefit at the **W′-limited redline** (the *fastest feasible* pace), where every
   run looks W′-limited — so it over-credited the stops on **every** long run
   (~11 s/mi mean), including easy ones that never came near the redline. It also
   violated conservation (a 33 s stop "buying back" >33 s) and was **non-monotone**
   (a late stop could speed up the *whole* run). It placed the wrong runs. Retired.
2. **Crossover gate / first-post-crossover-pause cap.** `d_eff` = distance to the
   first stop after effective CS drops below pace. The crossover landed late, so it
   credited most of the run and still over-rated the fastest runs.
3. **Longest-unpaused-segment cap.** `d_eff` = longest gap-free stretch. Too blunt —
   it obliterated *every* run with any mid-run stop, frontier and cloud alike. "We
   may as well drop long runs entirely."
4. **Route-based imputation as the lever.** Considered route-median / route-P90
   pause profiles. For the canonical hard case the route P90 (Belle Meade, 87 s/mi)
   was within rounding of the global P90 — it didn't move the needle, so we kept the
   global P90 imputation.

Each attempt taught the same lesson: the moment you try to model *what the pause
physically bought you*, you either over-credit (redline) or have to make
indefensible calls about effort you didn't observe. The uncertainty framing
sidesteps all of it.

## The model (final)

Per long run, at its flat-corrected constant pace `v`:

### 1. Effort
`effort = (CS-predicted race pace at the run's distance) ÷ (flat pace)`, where the
CS-predicted race pace is the day's CS-implied 5K up-converted to the run's distance
through the same WA tables.

- `effort < ~0.95` — easy; far below race effort.
- `effort = 1.0` — you ran the long run *at* the pace CS says you could **race** that
  distance.
- `effort > 1.0` — you ran **faster than your predicted race pace**: physically
  near-impossible for a genuine continuous effort, and the single strongest signal
  that pauses (or an over-estimated corrected distance) are flattering the run.

### 2. Effort gate — UNCAPPED
`gate = max(0, (effort − LR_EFFORT_E0) / (1 − LR_EFFORT_E0))`, with
`LR_EFFORT_E0 = 0.95`.

- **0 below 0.95** → the cloud is untouched; easy long runs ride on pace alone and
  fall off the frontier honestly without any pause penalty.
- Rises linearly, and **is intentionally uncapped** so `effort > 1.0` keeps
  escalating — the most-over-race-pace runs are eroded hardest.
- **Do not re-add an upper cap.** We tried `min(…, 1)`: it flattened *every*
  frontier run to the same gate (they're all `effort > 1`), so effort gave zero
  discrimination and the most-suspicious run got no extra penalty. We tried a
  convex/exponential gate: it overshot badly (the canonical run flipped positive).
  Uncapped-linear is the calibrated sweet spot.

### 3. Erosion law — pause LENGTH × LATENESS (`durability.eroded_deff`)
Every second of a pause erodes **all subsequent** moving distance, weighted by how
late the pause falls: a pause of `P` seconds at run-fraction `L` multiplies the
credit of everything after it by `exp(−gate · RATE · P · L)`. So

```
d_eff = Σ_segments  seg · exp( −Σ_{pauses before it}  gate · RATE · Pⱼ · Lⱼ )
```

- **Pause LENGTH drives it, not count.** A 20 s stop contributes almost nothing; a
  5-minute stop a great deal. No lower bound (every second counts), so it can't be
  gamed by resuming and re-pausing.
- **Lateness-weighted, linear in `L`.** A late pause is far more damaging per mile
  than an early one — the later a stop, the less certain we are about what comes
  after it. (Early pauses on a still-fresh runner barely erode.)
- **No hard cap on the result.** Erosion alone places the run; how far below the
  frontier it lands is the model's verdict, not a clamp. (We explicitly tried a
  hard race-anchored cap; it obliterated the runs and the cap is the wrong tool.)

### 4. Erosion constant
`LR_EROSION_RATE = 0.001` (pinned). Calibrated so the canonical hard case
(2020-11-06) lands at residual ≈ −10 (intentionally a hair past).

### 5. Pre-watch imputation
Pre-watch runs have no measured stops, so we **impute** them: the global **P90**
per-mile pause structure (`durability._pre_watch_profile` → `_impute_segments`),
scaled to the run's distance and late-loaded by the watch-corpus median thirds.
P90 is aggressive *on purpose* — it is where the uncertainty of an unobserved run
is injected (see thesis point 2). Watch runs use their real measured stops
(`load_segments`).

### 6. The WA conversion (companion fix — `wa_scoring.py`)
The erosion feeds the WA 5K-equivalent conversion, which was **rebuilt** alongside
this model: a **monotone cubic (PCHIP) through the real race anchors — 5K, 10K, HM,
marathon — in log-distance/log-time**, replacing log-linear interpolation through
*all* the WA tabs. The intermediate road tabs (15/20/25/30 k) are a track/road mix
that don't lie on a smooth curve; they made the conversion **non-monotone at the HM
tab** (a ~3 s discontinuity), which would let erosion *backfire* — shrinking a run's
distance toward ~HM could speed up its 5K-equivalent. The smooth curve is monotone
by construction and exact at the four anchors, so **no race conversion moved** (every
race >5K is at an anchor); only long runs / workouts at in-between distances changed.

## The canonical hard case: 2020-11-06 (documented so we stop re-opening it)

Pre-watch, Belle Meade (Nashville), 16 mi, run ~8 % **faster than its CS-predicted
race pace** — the highest effort in the entire long-run set — on an *estimated*
corrected distance with *imputed* pauses and a *hand-logged* 18 °C. Every input is
an estimate; it is the limit of what the model can resolve. After the model it sits
at residual ≈ **−10.8** (5K-equiv ≈ 4:50) — still the top of the long-run list, but
no longer an isolated spike (the next pre-watch run, 2019-11-23, sits −8.7).

**We accept this placement.** It is:
- supported by a high-residual **measured time trial ~2 weeks later**;
- inside the **CS P95 prediction band**;
- the correct story of an extreme **late-2020 fitness peak cut short by injury** —
  and qualitatively, that peak wasn't matched again until a spring-2022 workout,
  which rings true.

Pushing it lower would require punishing the (trusted) watch-era 2023 frontier in
proportion, or fabricating certainty about inputs we don't have. We chose to stop
here, on the record.

## Knobs

| knob | where | default | effect |
|---|---|---|---|
| `LR_EROSION_RATE` | `workouts.py` | 0.001 | erosion strength (bigger → paused frontier runs erode harder) |
| `LR_EFFORT_E0` | `workouts.py` | 0.95 | gate onset (raise → more of the cloud untouched) |
| `PRE_WATCH_PCTILE` | `durability.py` | 0.90 | aggressiveness of pre-watch pause imputation |

The gate is uncapped **by design** — see §2; do not re-cap it. The erosion constant
is universal — do **not** split it by watch/pre-watch era (uncertainty about
pre-watch belongs in the imputation, §5).
