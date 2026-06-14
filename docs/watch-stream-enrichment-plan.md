# Watch-stream enrichment — plan & handoff (June 2026)

Per-second watch data (altitude/speed) turned into physical, era-free route
corrections. Three threads from the June 2026 watch-correction work:
**(1)** elevation enrichment, **(2)** corrected-mileage source-of-truth,
**(3)** terrain-type backlog.

## Status

| Thread | State |
|---|---|
| 1. Elevation enrichment | data layer + **recovery** model SHIPPED; **long-run & race NEXT** |
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

### A. Long-run 5K conversion
Replace the route-constant `elev_per_mile × LR_ELEV_SLOPE` (0.17) in
`long_run_model.py` with the per-run `elevation_cost` engine.
- **Effort matters here** (unlike recovery, where grade washed out): long runs
  sit at effort ~0.85–0.98, so use `paved_refund(effort)` per run — between
  recovery's full refund and race's 0.85. The grade cost is a **live term**.
- Enter the grade cost as a pace/`t_eff` correction *before* the β_long
  hyperbola projection (the long-run model's existing race-equivalent path);
  the distance correction (`corr_miles`/D_eff) already landed — layer grade on
  top, don't re-correct distance.
- **Reconsider adding altitude.** It was removed from the long-run model for
  era-confounding, but the recovery work shows it's identifiable via backfit +
  footing control, and long-run effort should make it *larger* than recovery's
  ~1%. Worth a backfit-style fit.
- Keep elevation / terrain-footing / altitude **separate** (as in recovery) to
  avoid double-counting the mixed penalty.
- Same validity gate + location-terrain bucketing.

### B. Race CS (the prize)
Watch-covered races → measured grade profile **replaces the XC/Downhill
categorical surface corrections** where data exists.
- **Distance stays OFFICIAL** (certified > GPS). Only grade + altitude apply.
- **Effort ≈ 1.0 → race-effort refund** (paved ~0.85, mixed ~0.34): grade cost
  is *highest* at race effort, so this is where it bites most.
- **Circularity is acceptable, per Max: hardcode the refund curve.** Race
  points sit well above CS; CS only feeds the top end via 5K conversions (a
  secondary effect). So pin the effort-refund curve, don't re-fit it from the
  races we're correcting.
- **Track races: no elevation** (flat; barometric noise) — gated by surface.
- Validated: **Boston** net −450 ft drop captured → ~3 s/mi net-downhill aid
  (correctly *discounts* the assisted time for the CS fit); **North Shore HM**
  (paved, balanced ~500 ft loop) ~0.4%.
- **Out of scope:** marathon eccentric-quad-damage / hill *placement* fatigue
  (why Boston "feels hard" despite being fast) — the model prices steady-state
  grade, not placement; the marathon set is sparse and inconsistent anyway.
- **Conservatism edges:** `cs_projection`'s race edges are policy, not
  measurement (same boundary as `WORKOUT_VMAX_MPS`) — apply grade carefully
  there. Any CS re-fit (PyMC) needs Max's explicit approval.

---

## Open items
- **GHA divergence (accepted):** Max's `details` cache isn't on CI, so the
  deployed build computes only route-rule corrections (no watch elevation).
  Public mileage ≈ logged-minus-rules (~29,030) vs local ~28,860. To make
  them match, commit the small `*_measured.csv` + `elevation_measured.csv`
  artifacts, or cache Max's details. Same caveat will apply to any deployed
  long-run/race elevation correction.
- **Footing-binary vs grade-engine double-count:** keep the flat-footing term
  and the grade cost as separate channels in the long-run/race models too.
- **Trail terrain unidentified:** 1 location / ~29 mi — trail-specific refund
  uses mixed as a proxy until more trail data exists.
- **~5 watch-failure days** dropped by the validity gate (re-runnable if the
  watch streams are ever re-fetched cleanly).
- `slope`/`adjustedPace` from Coros are too coarse to shortcut any of this
  (`slope=0` on rolling terrain); compute grade from smoothed altitude.
