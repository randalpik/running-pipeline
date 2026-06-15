# Watch-stream enrichment — COMPLETE (June 2026)

> **This plan is done and has been collapsed.** All of its durable content was
> folded into the permanent reference docs (below). This file remains only as a
> redirect, because code comments and other docs cite its section labels
> (§A / §B / "thread 1"). For anything substantive, go to the homes listed here.

Per-second watch altitude/GPS turned into physical, era-free route corrections,
across three threads — all shipped.

| Thread / section | What shipped | Permanent home |
|---|---|---|
| **Thread 1 — elevation engine** | grade cost model (`elevation_cost.py`, terrain×effort refund), threshold+linear altitude curve, pinned footing/altitude betas (`physical_route_betas`), per-run features, the race DEM (`dem_elevation.py`) | **`route-normalization-reference.md`** (the elevation-engine hub) + `watch-derived-cache-spec.md` (data layer) |
| **§A — long-run 5K conversion** | grade + footing + altitude credited as flat/sea-level-equivalent time before β_long in `workouts.project_long_runs` (replaced the route-constant `0.17·elev_per_mile`) | `training-quality-reference.md` |
| **§B — race CS** | `recovery_model.race_physical_correction` corrects watch-covered race times to flat/sea-level/smooth-equivalent before CS; replaces categorical XC ×1.08 / Downhill-exclusion where watch data exists | `cs-model-reference.md` |
| Recovery route model | per-route dummies → physical era-free model (footing + altitude + pinned grade), backfit | `recovery-runs-reference.md` |
| **Thread 2 — corrected mileage** | `shared.effective_mileage.effective_daily_miles`; `corr_mi = min(watch+error, logged)`; lifetime −206 mi, 2023 takes the yearly crown | `recovery-runs-reference.md`, `training-quality-reference.md`, `watch-derived-cache-spec.md` |
| **Thread 3 — terrain backlog** | every route typed; `build_dataset._backfill_location_metadata` re-attaches terrain/elev to historically-located routes | `route-normalization-reference.md` |

The science behind the altitude curve (threshold ~3000 ft, then linear):
Wehrlin & Hallén 2006 (J Appl Physiol; linear VO₂max decline in endurance
athletes), Péronnet et al. 1991 (J Appl Physiol; running-performance model).
</content>
