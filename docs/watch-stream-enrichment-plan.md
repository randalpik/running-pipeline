# Watch-stream enrichment — plan & handoff (June 2026)

Per-second watch data (altitude/speed) turned into physical, era-free route
corrections. Three threads from the June 2026 watch-correction work:
**(1)** elevation enrichment, **(2)** corrected-mileage source-of-truth,
**(3)** terrain-type backlog.

## Status

| Thread | State |
|---|---|
| 1. Elevation enrichment | data layer + **recovery** + **long-run** + **race CS** models SHIPPED |
| 2. Corrected mileage | SHIPPED |
| 3. Terrain backlog | resolved (every route typed; build-join bug fixed) |

---

## SHIPPED components

### Elevation data layer
- **`rich_detail` v2** (`build_current_log.py`): freq points carry
  `[t, dist, heart, gpsLat, gpsLon, altitude, speed]` (raw; **altitude is in
  FEET** in the locations sheet but **meters** in the live stream — the
  stream is what `elevation.py` reads); `elevGain` added to the slim summary.
  Backward-compatible (old consumers index `f[0..4]`); versioned `rich=2`.
- **`src/coros/elevation.py`**: stitches multi-activity days onto one
  monotonic distance axis (`_stitch`), resamples altitude to a 10 m grid with
  a **120 m smoothing window** (matches the device's `summary.elevGain` within
  1–3% on hilly runs), derives gain/loss, a Minetti grade factor, and
  per-**corrected**-mile splits.
- **`scripts/backfill_elevation.py`** (Claude runs it — Coros creds in repo
  `.env`, token cache `data/profiles/coros/coros_token.json`): re-fetches each
  long/recovery/race day, upgrades the cache to rich v2, writes
  `data/elevation_measured.csv` (gain/loss ft, `minetti_factor`) +
  `data/elevation_splits.csv` (per-mile). **1,734 days** (139 long, 38 race,
  1,557 recovery). Gates: watch-validity, sport-type, race surface (below).

### Elevation engine — `src/shared/elevation_cost.py`
`cost = c_up·gain − refund·c_up·loss` (s/mi), all constants tunable:
- `CLIMB_COST` {paved 0.19, mixed/trail 0.26} s/mi per ft/mi climbed.
- `REFUND_RECOVERY` {paved 1.00, mixed/trail 0.34} — descent refund fraction
  at easy effort.
- `REFUND_PAVED_BY_EFFORT` + `paved_refund(effort)` — paved refund falls from
  ~1.0 (recovery) to ~0.85 (race) as effort→1; mixed/trail stay ~0.34 at all
  efforts. **This is the effort-aware knob the long-run/race models need.**

### Recovery route model — `recovery_model.py` (SHIPPED)
Per-route dummies **removed**, replaced by a physical era-free model, fit via
a **backfit** (era smoother ↔ parametric factors iterated). Factors:
- temp / fatigue / TOD (fast-varying, era-immune) — fitted.
- **off-road footing** `is_offroad` ≈ **+4.2 s/mi** (mixed+trail binary, flat
  surface) — fitted.
- **altitude** `alt_kft` ≈ **+0.83 s/mi per 1000 ft** (+4.5 at Boulder) — fitted.
- **pinned grade-aware `elev_cost`** — scales with each route's gain/loss
  (hilly-mixed ≫ flat-mixed).

Three normalization checkboxes expose the decomposition: **Elevation (net)**
= `c_up·(gain−loss)` (zero on loops, applies on point-to-point), **Terrain**
= footing + the mixed descent-braking (refund asymmetry, scales with descent),
**Altitude**. R²_detrended 0.293 / raw 0.627.

### Corrected mileage (SHIPPED)
- **Distance-bracket fix** (`workouts._lr_watch_corrections`,
  `recovery_model.add_watch_corrections`): `corr_mi = min(watch+error, logged)`
  — time is the anchor (= watch time), pace is the derivative. Replaced the old
  pace-floor clamp (which let distance exceed logged and fabricated phantom
  distances). Decrease-only by construction.
- **`shared.effective_mileage.effective_daily_miles`** is the single source of
  truth; the 5 display consumers (annual, dashboard, qualitative-trends,
  geography, world-map) apply it at load (on-disk `daily.csv` untouched).
- Lifetime **−206 mi**; **2023 takes the yearly crown** (3,209.9 vs 2025's
  3,196.7). GHA divergence accepted (see Open items).

### Build fix — `build_dataset._backfill_location_metadata`
Re-attaches terrain/elev/altitude to historically-located routes (education
hill: 524 rows were NaN because `apply_historical` set their location *after*
the metadata join).

---

## Key principles (hard-won — honor these in the next applications)

1. **One watch-validity decision per day.** If the watch failed to record the
   run (`watch_miles/logged ∉ [0.6,1.5]`), NOTHING uses its data — distance,
   time, and elevation all fall back to hand-logged. Validity is universal;
   the *distance* correction layers extra criteria (paved gate, strides,
   `WATCH_FAIL_DEV`) on top; **elevation uses every valid day** (it's the only
   grade source and more reliable than GPS distance).
2. **One source of truth per run for distance/time** — verified-watch or
   hand-logged, never a silent blend of a rejected watch's data.
3. **Pin only what's unidentifiable at the grain; fit everything else.**
   Gain/loss are collinear within loops at run-level, so `elev_cost` is
   *imported* from the per-mile data (where they separate), not "chosen."
   Everything the recovery data can identify (temp/fat/tod/footing/altitude)
   is fitted dynamically.
4. **Backfit for era-correlated factors.** The era trend is a temporal
   smoother; footing and altitude vary on the era timescale (mixed/sea-level
   early, paved/Boulder-altitude late), so a sequential era-detrend *absorbs*
   them (footing collapsed to 0). Backfitting (era ↔ factors) isolates them
   from the overlap years.
5. **Refund is terrain × effort-dependent.** Downhill can't be banked near
   v_max: paved refunds ~fully at easy effort → ~85% at race; mixed/trail
   ~34% at all efforts. So **paved rolling is cheap (~0.3% even racing); the
   textbook 1–2% rolling cost holds only for mixed/trail.** Effort = frontier
   pace at the run's *own distance* / run pace (`performance_frontier`).
6. **Bucket by LOCATION terrain, not race surface** (7 "Road" races run on
   mixed locations — Run for the Pies, etc.).
7. **Altitude is real but modest** (~1% at Boulder), *identifiable* once
   footing is controlled (it was hidden by a feet-as-meters unit bug +
   absence from the model + era-masking). Keep it fitted, don't pin.
8. **Display decomposition** = elevation (net, zero on loops) + terrain
   (footing + refund-asymmetry) + altitude.

---

## NEXT — Thread 1 applications

### A. Long-run 5K conversion — SHIPPED (June 2026)
The route-constant `elev_per_mile × 0.17` was replaced by the full physical
route decomposition (grade + footing + altitude), all **credited as
flat/sea-level-equivalent time corrections** in `workouts.project_long_runs`
*before* the β_long projection (`corr_miles`/D_eff distance correction already
landed; grade layered on top). `long_run_model.py` no longer fits anything
physical (`LR_ELEV_SLOPE`/`elev_pm_c` removed) — it carries only temp/fatigue.

Hard-won findings (course-corrected several wrong turns — honor these):
- **Grade** = per-run measured gain/loss through `elevation_cost`, paved refund
  effort-aware via `paved_refund`. (The premise that long runs are
  *higher*-effort than recovery was WRONG: by the frontier-pace-fraction
  metric both center at effort ~0.85, so paved refund lands ~1.0 — same as
  recovery. The effort ranges fully overlap.)
- **Footing + altitude can't be fit on long runs alone** — era-confounded
  (off-road routes cluster in 2016–22; altitude tracks the Boulder era).
  Naive long-run OLS gives a spurious `is_offroad` +12.7 and a wrong-signed
  altitude −0.5. The within-2023+-era altitude check (+1.18/kft) confirmed the
  cross-era −0.5 was pure era confounding.
- **Effort-dependence of footing/altitude is NOT testable by regression** —
  effort (pace ÷ race-pace-at-distance) is mechanically the residual being fit
  (`corr ≈ −0.99`). The marathon/short-fatigue-ratio trick fails here because
  effort is endogenous, unlike days-since-race.
- **Resolution — `recovery_model.physical_route_betas()`**: the single source
  of truth. Pools recovery + in-slice long runs on a shared `pace − cs_pace`
  scale with an `is_long` level dummy and the recovery era-backfit; the large
  recovery corpus + shared era control discipline the long-run rows (footing
  reads +4.7, not +12.7). Pinned constants: **footing +4.78 s/mi**,
  **altitude +0.86 s/mi/kft** (corpus-stable: recovery-only +4.09/+0.87, both
  shift <1 se when long runs join — footing-up is the real long-run
  fatigue-on-rough-terrain signal). **No effort scaling** (ranges coincide →
  constant-effort transfer, no assumption needed).
  `fit_recovery_model(pin_physical=True)` now consumes the same betas, so one
  constant per channel applies in both models (recovery R² unchanged 0.626).
- **Altitude is per-run MEASURED** (`per_run_altitude`), not a per-location
  estimate: `alt_kft` = midpoint of the watch's smoothed daily min/max
  (`altitude_daily.csv`, the layer feeding the Altitude trend), location
  base-elevation constant only as the pre-watch fallback. Adds within-location
  resolution and fixes hand-set constant errors (watershed 580→~255 ft). Both
  recovery and the long-run conversion use it (grade already uses per-second
  watch gain/loss).
- Channels kept **separate** (grade / footing / altitude) as in recovery — no
  double-count; same validity gate + location-terrain bucketing.

Net effect: off-road long runs credited grade(~10)+footing(4.7)+altitude
(~6.6 at Nederland) ≈ project faster; Boulder paved normalized to sea level
(+4.2 s/mi credit); the dashboard long-run prediction moved +16.4→+14.8 s/mi
vs CS. Frontier (fed by long runs) re-levels accordingly.

### B. Race CS — SHIPPED (June 2026)

**Goal (met).** Correct each watch-covered race's TIME for grade + altitude
(+ footing on off-road) to its flat/sea-level/smooth-equivalent BEFORE it
informs CS, so the demonstrated-capability frontier measures fitness, not the
course. The measured per-run correction **replaces the categorical XC (×1.08)
and Downhill-exclusion** rules where watch data exists; categorical stays as the
pre-watch fallback.

**Single source of truth: `recovery_model.race_physical_correction(races)`**,
applied identically in three places — `cs_projection.project_races_to_5k_pace`
(the displayed diamonds + the frontier they feed; gated by
`apply_physical_correction`, OFF for the actual-pace race plots), the CS fit
(`bayes_cs_fit.main`, subtracted from `race_times` before the β_long un-bias and
the likelihood; `build_eligible` now admits watch-covered Downhill;
`derive_exclusions` corrects on the same times). Footing/altitude pinned from
`physical_route_betas` (same constants recovery/long use); race effort ≈ 1.0 so
paved descents refund at `paved_refund(1.0)≈0.85`. Refit shipped (5 div, 96.2%
95%-coverage, residual bands tight); watch-era CS faster 2–4 s/mi.

**THE decisive course-correction (Max) — barometric race net is noise, use a
DEM.** The watch's per-race barometric *net* is untrustworthy (same Bolder
Boulder course read −19 ft/mi in 2024 vs +5 in 2025), but its *horizontal GPS
track* is reliable and reproducible (start/end agree within metres year-over-
year; Boston resolves Hopkinton→Boston). So races throw away the barometric
vertical and resample elevation from a DEM along the GPS path —
`src/coros/dem_elevation.py` (OpenTopoData NED 10 m / SRTM 30 m fallback, point
cache `data/dem_cache.json`), backfilled into `elevation_measured.csv` race rows
as `dem_gain_ft`/`dem_loss_ft`/`dem_net_ft`/`dem_mean_elev_ft` (top-up pass in
`backfill_elevation.augment_race_dem`; picks the single race activity, excludes
warmup/cooldown). This **subsumes the loop rule** (loops read net≈0 naturally:
Honeywagon +0.2 ft/mi) AND deflates inflated barometric climb totals (Run for
the Pies 37→23 ft/mi), which **organically shrank the off-road grade
over-credit** without any footing band-aid. Recovery/long stay on barometric
(they average out; the betas are fit on them). Validated: Bolder Boulder now a
consistent +10.5 ft/mi (genuinely net *uphill*); Boston DEM −17.2 agrees with
baro (real net downhill, unaffected).

**Footing stays +4.78 (Max's "too generous?" — answered: no).** Mixed/trail
split on the recovery+long corpus (664 mixed / 16 trail rows) gives **mixed-only
+4.92**, *higher* than the pooled +4.78 (pooled is dragged down by the tiny
trail set); and the XC race-effort anchor (≈24 s/mi terrain on a 5K) is 5×
higher — so +4.78 is already conservative for race effort, not generous. The DEM
grade-deflation was the real fix. Kept the pinned pooled constant.

**Known one-pass lag (accepted, as in §A):** `physical_route_betas` depends on
CS; each refit nudges them, so the displayed diamonds use post-fit betas while
the fit used pre-fit ones — a sub-1-s/mi Boulder lag. Verified to converge
geometrically (footing contraction ~0.05, altitude ~0.19; a second refit moved
the CS line <0.32 s/mi/yr, ratio ~0.07-0.12 of the first). Max chose to let the
betas evolve naturally (a new race perturbs them trivially) rather than pin.

**Altitude curve — threshold + linear (June 2026, science-pinned shape, data-fit
slope).** The linear-from-origin altitude term was wrong both ways: it invented a
phantom hypoxia effect at low elevation (a 0.42 s/mi / 11 s "correction" on the
pre-watch Nashville 400 ft marathon, from the per-city base-elevation constant —
which ALSO flipped `has_measured` True and wrongly ADMITTED the pre-watch
Downhill TT to CS) and under-sloped the high end (Magnolia 8-9k ft). Physiology:
VO2max is ~flat below ~914 m (3000 ft) then declines ~linearly (Wehrlin & Hallén
2006, linear to 2800 m — covers Max's 2569 m max; the ~8-11%/1000 m-above-3000 ft
clinical heuristic). Max's altitude data is bimodal (1800 sea-level + ~515
Boulder + only **3** Magnolia runs) so it can't identify a SHAPE — pin the shape
(threshold 3000 ft, linear above) from the literature, fit only the SLOPE.
`recovery_model.altitude_regressor(alt_kft) = max(0, alt_kft − 3.0)`, applied at
every regressor site (the pooled `physical_route_betas` fit → recovery, long-run,
race all consistent). Result: slope **2.28 s/mi per kft above 3 kft** (Boulder
≈5.5, ~unchanged since it anchors the fit; **Magnolia +37%, ≈12.3**; everything
< 3000 ft → exactly 0, killing the phantoms and the Downhill mis-admission).
Recovery R² unchanged (0.298 detrended). Refit shipped (12 div/4000, 96.2%
95%-coverage); the dominant CS change is 2020 +5.5 s/mi (the wrongly-admitted
downhill-assisted mile correctly removed). Tooltip: event / dist in time(pace) /
Course correction: corrected time(pace) at race distance / 5K-equiv. The Fitness
graph also draws faint before→after connectors (corrected-then-converted) for
the 62 races a correction moved >1 s/mi (44 XC categorical + 18 watch-physical).

**Result (frontier).** Off-road 5Ks bind on DEM-grounded credit (The Enforcer
2022 → 4:44/mi, verified: 193 ft real climb, course not short); Boulder-altitude
races credited (Boulderthon, Bolder Boulder consistent, Frank Shorter track =
altitude-only); Boston discounted (+1.4 s/mi) so it informs CS as fitness not as
a fast course. Auto-exclusions stable (12→11; no XC/Downhill watch coverage
exists today, so those paths are armed for the future, no-op now).

#### Original handoff (kept for context)

**Reuse everything from §A — same single source of truth, same conventions.**
- Grade: `elevation_cost(gain_pm, loss_pm, terrain, refund)` with
  **`paved_refund(effort≈1.0) → ~0.85`** (race effort; grade cost bites HARDEST
  here). Per-run measured gain/loss via the same `per_run_elevation` path.
- Footing + altitude: **pinned from `physical_route_betas()`** (footing +4.78,
  altitude +0.86/kft) — the SAME constants recovery and long-runs use. Altitude
  is **per-run measured** (`per_run_altitude`, `altitude_daily.csv`), NOT
  per-city. Do NOT add races to the pool (few races, circular with CS, race
  effort un-identifiable — see §A finding 2).
- Sign/credit convention identical: subtract the cost from the race time. A
  **net-downhill** race (cost<0) gets time ADDED → projects SLOWER → correctly
  **discounts** the assisted time for CS (Boston). A net-uphill/altitude race
  gets credited faster.

**Where it plugs in — TWO points that MUST stay consistent** (a shared
`race_physical_correction(races)` helper feeding both is the clean structure):
1. **The CS fit** — `src/models/bayes_cs_fit.py`. Apply the time correction
   right where the XC pre-correction lives (`main`, ~L362-383: `time_sec *=
   1/(1+xc_correction)`), i.e. on `race_times` before `log_race_times`
   (~L423-425) enters the PyMC likelihood (~L533). This is a **PyMC model-input
   change → needs Max's explicit approval before any run** ([[feedback-no-unapproved-fits]]).
2. **The projection/display** — `cs_projection.project_races_to_5k_pace`
   (~L287-367): apply the SAME correction to `t_race` *before* the β_long
   un-bias (L357) and the CP3 solve. If the fit and the projection disagree, the
   plotted race position won't match what fed CS.

**Race-specific rules.**
- **Distance stays OFFICIAL** (certified > GPS) — do NOT apply the §A
  distance/`D_eff` correction. Grade/altitude/footing are TIME-only corrections;
  `backfill_elevation.py` already uses `distance_m` for race rows (L82-96).
- **Track races: grade gated OFF** (flat; barometric noise; Coros `slope=0` on
  rolling) — but **altitude hypoxia still applies** (a Boulder track 5K is
  altitude-suppressed). Gate grade on surface, not altitude.
- **β_long is orthogonal** to terrain (it's race-execution/glycogen pacing
  fade) — correct the race time for grade FIRST, then β_long un-bias. Don't
  conflate.

---

#### GOTCHAS (the ones that will bite — read before touching code)

1. **PyMC = approval gate.** `bayes_cs_fit.py` is the Bayesian CS fit. Changing
   the race times it observes changes the foundational artifact every plot
   depends on. Present the diff + a dry-run comparison; **never run-and-keep**.
2. **Per-RACE, not per-day, elevation.** `elevation_measured.csv` /
   `altitude_daily.csv` key on `date`; multi-race days exist (`race_seq`: a
   track meet logs 800 m + 3000 m). Attribute the right activity to the right
   race, or the join smears one race's profile onto another. (Moot for track —
   grade off — but altitude and any road double-header are not.)
3. **Don't double-count the categorical.** Where measured grade/footing apply,
   the ×1.08 XC factor and the Downhill exclusion must be TURNED OFF for that
   race — apply measured-where-available, categorical-only-as-fallback, never
   both. Calibrate so a watch-covered XC race lands near its old ×1.08 (the 8%
   was a blend of grade+footing+terrain; the measured channels should
   reconstruct it, not stack on it).
4. **Admitting Downhill races is a behavior change.** Downhill is currently
   HARD-EXCLUDED (`build_eligible`, ~L87). The point of measured grade is to
   ADMIT them with a grade-discounted time (Boston). That changes the eligible
   set fed to CS — flag it explicitly and verify the discount is right before
   admitting.
5. **Circularity is accepted — but pin the curve, don't refit it.** Races feed
   CS and we correct races against CS-derived effort. Per Max: **hardcode the
   refund curve**; race points sit well above CS, which only feeds the top end
   via 5K conversions (secondary). Do NOT fit the refund from the races being
   corrected.
6. **v_max edges are POLICY, not measurement.** `cs_projection`'s race edges
   (`vmax_evidence`, the conservative-high read) are the same boundary as
   `WORKOUT_VMAX_MPS`. A grade correction shifts implied CS, which interacts
   with the v_max interval — apply carefully there, don't let a downhill
   discount push a point through the policy edge unnoticed.
7. **Effort-dependence is still un-regressable** (§A finding 2: effort ≈ the
   residual, corr −0.99). Use the pinned `paved_refund` curve at race effort;
   do NOT try to fit a race-effort footing/altitude term.

(GHA/production parity is deliberately NOT in scope here — local correctness is
the goal; matching CI is a deploy-time step, see Open items.)

**Validation targets (regression checks):** **Boston** net −450 ft → ~3 s/mi
net-downhill discount (slower flat-equivalent, correctly lowering its CS pull);
**North Shore HM** (paved, balanced ~500 ft loop) ~0.4%. Track-at-altitude
(Manhattan Track, Boulder 5400 ft) → altitude credit only, no grade.

**Out of scope:** marathon eccentric-quad-damage / hill *placement* fatigue
(why Boston "feels hard" despite being fast) — the model prices steady-state
grade, not placement; the marathon set is sparse and inconsistent anyway.

---

## Open items
- **GHA/production parity — NOT a concern for this work (Max).** Local builds
  use the watch elevation; CI currently doesn't (no `details` cache), so the
  deployed numbers differ from local. This is BY DESIGN for now — local
  correctness is the goal. When production goes live, GHA will be set up to
  fetch the same watch data and run the pipeline identically; the fix lives
  there (commit the `*_measured.csv` / `altitude_daily.csv` artifacts, or cache
  the details), not in the modeling work. Do not treat it as a blocker or
  re-flag it on every elevation feature.
- **Footing-binary vs grade-engine double-count:** keep the flat-footing term
  and the grade cost as separate channels in the long-run/race models too.
- **Trail terrain unidentified:** 1 location / ~29 mi — trail-specific refund
  uses mixed as a proxy until more trail data exists.
- **~5 watch-failure days** dropped by the validity gate (re-runnable if the
  watch streams are ever re-fetched cleanly).
- `slope`/`adjustedPace` from Coros are too coarse to shortcut any of this
  (`slope=0` on rolling terrain); compute grade from smoothed altitude.
