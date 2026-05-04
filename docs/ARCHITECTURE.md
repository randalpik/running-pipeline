# Architecture & Reference Map

The bird's-eye view of this repo: data sources, the pipeline that processes them, the three independent analysis layers, and which reference doc to open for which concern. Read this before diving into a specific reference doc — it's the routing layer.

## Data flow

```
Google Drive (Running Log xlsx, Lifetime Miles, Routes, Max's Running Data)
        │
        ▼
   drive_snapshot.csv  +  historical_daily.csv (frozen pre-current-year)
        │
        ▼
   build_dataset.py
        │
        ├──▶ daily.csv      (running-only, joined location metadata)
        │
        └──▶ races.csv      (post-adjustment truth)
                │
                ▼
        bayes_cs_fit.py  ──▶ bayes_cs_summary.csv  +  params  +  posterior .nc
                │
                ▼
            plot scripts
                │
                ▼
        output/plots/*.html
```

`daily.csv` is running-only as of April 2026 (zero-mile days pruned). `races.csv` back-propagates `city_state` and `surface` into the corresponding daily race rows for `race_seq=1`.

## Truth sources

The repo's truth sources are checked-in CSVs in `data/` (flat — no raw/processed split). Drive is fetched via `drive_fetch.py` only when those are missing or stale. The `Max's Running Data` workbook (id `1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w`) carries the manual-overlay sheets:

| sheet | purpose |
|---|---|
| `changes` | per-row corrections — `date | race_seq | field | value | note` |
| `additions` | races not in the daily log (mostly pre-2016) — `date | distance_m | time_sec | surface | location | event`. Also produces stub daily rows for race dates not already in daily |
| `locations` | `log_location → city_state / display_name / elev_per_mile / altitude / terrain_type` lookup |
| `hills` | hill-loop metadata used by parser and TQ — `abbrev | location | type | distance_m | elev_gain_up | elev_gain_down | elev_net | elev_per_min` |
| `coordinates` | `city_state → (latitude, longitude)` overrides on top of the Nominatim cache for the world map |
| `historical` | `city_state | min_hist | max_hist | log_location` — surfaces remembered-but-unlogged cities on the world map AND date-range-overrides non-race daily rows. Replaces the legacy 2016-17 `infer_2016_2017_location` date branches |

2018+ is converged: across eight years there's only one location change and four surface changes. New work focuses on pre-2018 rough edges and on 2026+ ongoing data.

## Three analysis layers

The layers are intentionally **not combined**. Each answers a different question, and a combined model was tested and rejected.

### Layer 1 — Critical Speed (CS)

Bayesian estimate of fitness on a single continuous scale, derived from race results only. Replaces earlier VDOT-based approaches. Workouts and long runs do **not** feed back into CS; a combined model was tested in April 2026 and underperformed CS-only on the 2018+ HM+Track5K subset (RMSE 13.4 vs 10.6).

CS-implied 5K is treated as ground truth for fitness. The other layers sit beside it, not inside it.

→ See `cs-model-reference.md`

### Layer 2 — Training Quality (TQ)

A residual layer on top of CS. Translates each workout, long run, and hill effort into a 5K-equivalent pace, then asks how it compares to what CS predicts for that date. Captures fitness building between races, training running ahead of (or behind) realized fitness, and training composition during breakthrough periods.

→ See `training-quality-reference.md`

### Layer 3 — Recovery

Era-windowed regression on recovery-run pace, with a small set of OLS features (temperature, route, recent race exposure, time-of-day). Volume normalizers and bodyweight features were explored and rejected. Captures the slow drift in baseline aerobic fitness independent of race performance.

→ See `recovery-runs-reference.md`

### Cross-cutting — route normalization

Locations propagate location-specific pace costs into both the recovery model and TQ long-run residuals. Lives between the two and informs both. The locations sheet (in `Max's Running Data`) is the single source for `log_location → city_state / display_name / elev_per_mile / altitude / terrain_type`.

→ See `route-normalization-reference.md`

### Diagnostic — qualitative trends

The volume / temperature / weight panel: three stacked subplots with moving-average trendlines on top of vertical-gradient min-max envelopes. Pure visualization, not a model — used for context when interpreting the three analytical layers.

→ See `qualitative-trends-reference.md`

## Reference doc map

| Doc | Layer | Covers |
|-----|-------|--------|
| `cs-model-reference.md` | 1 | PyMC HSGP fit, race exclusion logic, why the combined model was rejected, run flow (`bayes_cs_fit.py --tag vN` then `bayes_cs_plot.py --tag vN`, ~8min). |
| `training-quality-reference.md` | 2 | tau decay (210), distance thresholds, long-run binning, hill-loop offsets, iterative prune threshold (+23.3 s/mi), smoother bandwidth, ESS gates, gap-break logic. |
| `recovery-runs-reference.md` | 3 | Era window (±182d), OLS features, route inclusion threshold, rejected normalizers (volume, bodyweight). |
| `route-normalization-reference.md` | cross | Schema and dataflow of the locations sheet, propagation to recovery and TQ. |
| `qualitative-trends-reference.md` | diagnostic | Window sizes (56/28/56-day MA, 14d min-max with 7d smoothing), gradient-strip rendering, weight gap-align before+after smoothing. |

## Pre-2016 race data

Daily logs go back to 2016. Race history before that came from a one-time enrichment effort:

- `Lifetime Miles` "5K Record Progression" table (sole pre-2016 source in Drive).
- `Routes` "Half Marathon Record Progression" (2016–17 era).
- Athlinks API (76 races, 2008–2025).
- athletic.net HTML (86 entries: track + relay + XC).

Merged into `merged_races.csv` (161 rows, distance-snapped to standard distances) which drove 55 manual additions. The scrapers (`parse_athlinks_api.py`, `parse_athletic_net.py`) live in outputs if persisted; they aren't part of the regular pipeline.

## Drive file IDs

| File | ID |
|------|-----|
| 2026 Running Log | `1zvjx4RUzdZ11lsbyrzibGQJyLXUOR7gRUKwIwIILTuA` |
| Lifetime Miles | `10l629w-jChPdnwpVYQ2Lgj9JiSZH3_mNwTVeP0KusEM` |
| Max's Running Routes | `1tWPI9j8JCJidrOyu8Gw5lJ4aS-8__51gompGjArzUWU` |
| Max's Running Data | `1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w` |
| Drive root folder | `0AEcLRNUY5jL_Uk9PVA` |
