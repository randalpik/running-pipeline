"""Single source of truth for display mileage.

Every consumer that sums/aggregates daily mileage for display (annual plot,
dashboard totals, qualitative-trends volume, geography, world map) should read
``eff_miles`` from this helper instead of the raw logged ``miles`` column:

    daily['eff_miles'] = effective_daily_miles(daily)

``eff_miles`` is the watch/route distance-corrected mileage where a correction
exists (``workouts._lr_watch_corrections`` for long runs,
``recovery_model.add_watch_corrections`` for recovery), else the logged miles.
The correction is the distance bracket ``corr_mi = min(watch+error, logged)``,
so ``eff_miles <= miles`` always — corrections are decrease-only by
construction; no separate accounting clamp is needed.

Builds without the watch-calibration artifacts (e.g. CI without Max's
``details`` cache) simply find no calibration and apply only the route-era
rules; everything else passes logged miles through. So the deployed total can
differ from a local fully-enriched build — an accepted divergence (June 2026).
"""
import pandas as pd

from src.shared import workouts
from src.shared import recovery_model as rm


def effective_daily_miles(daily):
    """Per-day display mileage with distance corrections applied, aligned to
    ``daily.index``. Long/recovery rows pick up their corrected distance where
    one exists; every other row (and every uncorrected day) keeps logged
    miles."""
    eff = daily['miles'].astype(float).copy()

    long_rows = daily[daily['run_type'] == 'long'].dropna(
        subset=['recovery_pace_sec_per_mi', 'miles'])
    if len(long_rows):
        corr = workouts._lr_watch_corrections(long_rows.copy())['corr_miles']
        eff.loc[corr.dropna().index] = corr.dropna().astype(float)

    rec_rows = daily[daily['run_type'] == 'recovery'].dropna(
        subset=['recovery_pace_sec_per_mi'])
    if len(rec_rows):
        corr = rm.add_watch_corrections(rec_rows.copy())['corr_miles']
        eff.loc[corr.dropna().index] = corr.dropna().astype(float)

    return eff
