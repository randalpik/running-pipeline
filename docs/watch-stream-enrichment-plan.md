# Watch-stream enrichment — plan (June 2026)

Next steps coming out of the June 2026 watch-correction work (long-run +
recovery distance corrections, paved gating, partner-exclusion
unification). Three threads: per-second elevation/pace enrichment
(supersedes route betas and `elev_per_mile`), the corrected-mileage
"band-aid" decision, and the terrain-type backlog the paved gate
created. Nothing here is implemented yet; the API facts ARE verified.

## 1. Per-second elevation + pace enrichment

### Verified facts (probed live against the Coros API, 2026-06-12)

The raw `/activity/detail/query` response's `frequencyList` carries,
per second:

`timestamp, distance, heart, gpsLat, gpsLon, altitude, speed,
adjustedPace, cadence, cadenceLength, power, slope, groundTime,
verticalStrideRatio, verticalVibration, heartLevel, level`

- **`altitude` is plain meters** — verified on a Boulder run
  (1599–1623 m across the stream, vs Boulder's ~1620 m) and a Chicago
  run (171 m, matching the lakefront). Barometric, so smooth-then-
  differentiate is the right gain/loss extraction.
- `summary.elevGain` (meters) also exists in the raw detail (45.0 on a
  flat Boulder run ≈ 148 ft — plausible), as do `trackClimbInfo`,
  `graphList`, `lapGraphList` (unexplored).
- **Our cache discards all of it.** `sync._project` keeps the rich
  per-second stream only for Track Runs, and `rich_detail` projects
  each point to `[timestamp, distance, heart, gpsLat, gpsLon]` — no
  altitude. Slim records (everything else, incl. nearly all recovery
  and many long runs) keep only summary + weather + one GPS point.
  Current cache: 2,180 details, 311 rich (78 track + 233 backfilled
  workout days).

### Plan

1. **Extend `rich_detail`** to append `altitude` (and optionally
   `speed`) to each freq point. Backward-compatible: `reps._freq_points`
   and `long_runs._freq_points` index `f[0]`/`f[1]` only.
2. **Backfill** raw details for long-run + recovery days (~139 LR +
   ~1,560 recovery, minus already-rich) via the existing
   `scripts/backfill_rich_details.py` precedent. Storage: ~100 KB/rich
   record → ~170 MB if kept whole. Alternative (preferred): compute at
   backfill time and persist a **compact per-day artifact** —
   per-corrected-mile splits with `(pace_s, elev_gain_ft, elev_loss_ft)`
   — and keep the cache slim. ~25 numbers per run instead of 100 KB.
3. **Fit splits to the watch-corrected distance**: rescale the stream's
   distance axis by `corr_miles / watch_miles` so split boundaries land
   on corrected miles; moving-time-only (mask pauses/stalls like
   `long_runs._activity_segments`).
4. **Minetti per-run grade cost** replaces route proxies: integrate
   Minetti cost over the smoothed altitude profile (the
   `hill_model.minetti_net_factor` machinery already exists for hill
   loops) → a per-run multiplicative `t_eff` correction. This
   supersedes `elev_per_mile` (a route-level constant joined from the
   locations sheet) and the pinned recovery-side elevation slope
   (`LR_ELEV_SLOPE = 0.17`) for every enriched run; the route-level
   path stays as the pre-watch fallback.
5. **Races** (the real prize): the same enrichment on watch-covered
   races gives measured elevation profiles + pause-free true times for
   the CS fit. Course distance stays OFFICIAL (certified > GPS); only
   the elevation/grade correction and time verification apply. This
   would replace the XC/Downhill-style categorical surface corrections
   with physical ones where data exists. Needs care at the
   evidence-conservatism edges (cs_projection's race edges are policy,
   not measurement — same boundary as `WORKOUT_VMAX_MPS`).

### Open questions

- Smoothing window for barometric altitude before differentiating
  (drift vs responsiveness; check against surveyed hill loops).
- Whether `slope`/`adjustedPace` from Coros are trustworthy enough to
  shortcut any of this (probed values looked coarse: `slope=0` on
  rolling Chicago terrain).
- Re-fetch etiquette: ~1,700 detail fetches, single-session token —
  run as a slow backfill, reuse `coros_token.json`, expect to be
  logged out of the web app while it runs.
- GHA: Max's details cache is absent on CI (known open item); the
  compact per-day artifact would need to be committed or synced like
  the other `*_measured.csv` artifacts.

## 2. Corrected mileage ("the band-aid") — decision pending

Applying the landed corrections (`corr_miles` where it exists: watch
days through the calibration curve, rule days through route deflation;
stride days and non-paved/unwatched days untouched) to total-mileage
accounting:

- **Lifetime: −170.5 mi** of 29,067 logged (−0.59%). Long runs −124.8
  (watch −46.7 over 129 days; rules −78.2 over 83 days), recovery
  −45.6 (1,179 days).
- **The yearly mileage record reorders.** 2023 takes the crown from
  2025: corrected 2023 = 3,219.6 (logged 3,218.9, +0.7 — its
  east-boulder under-log days clamp to no-ops) vs corrected 2025 =
  3,200.8 (logged 3,232.3, −31.5). No other rank changes; 2019/2020
  each lose ~30, 2018 −21, 2024 −21, 2026-to-date −12.
- Caveat: this is a **lower bound** on the true adjustment — 2016-17
  and most 2018-20 recovery have no watch coverage and no rules, so
  their honest deltas are unknown (and the overread clamp zeroes the
  legitimate under-log corrections).

Decision for Max: whether the Mileage/annual plots and dashboard
lifetime totals switch to corrected, show both, or stay logged. Not
applied anywhere yet — display mileage everywhere still sums logged
`miles`.

## 3. Terrain-type backlog (paved gate fallout)

The paved gate skips watch correction on days whose route has no
`terrain_type` in the locations sheet. Routes needing a type, by row
count (recovery+long, June 2026 freeze): **education hill (415 —
pre-watch, matters for elevation modeling not watch gating), baton
rouge (20), wedgewood (12), williamsburg (12), qingdao (8), chicago
(7), ann arbor (6), 12 south (6), kona (6), paris (6)**, then a long
tail of 1-5-row travel locations (cambridge, tokyo, calais, central
park, pcb, burke-gilman, berlin, boulder, …). 586 of 2,948
recovery+long rows are un-typed in total; the watch-era subset
(~120 rows) is where typing immediately unlocks corrections.
