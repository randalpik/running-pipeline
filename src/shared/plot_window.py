"""Per-profile plot date-window floors.

Plots used to hardcode a 2016-01-01 left bound (Max's logging era). For other
profiles (e.g. a Coros watch import starting 2020) that wastes years of empty
axis. These derive the left bound from the profile's own data instead:

  - daily_floor(): first non-race daily entry  (daily-centric plots)
  - race_floor():  ~1 month before the first race  (race / fitness plots)

No artificial floor — a profile begins at its actual first data. (For Max this
lands daily plots at 2016, since his only pre-2016 rows are race-addition
stubs; his race/fitness plots extend back to his first logged race.)
"""
from __future__ import annotations

import pandas as pd

from src.shared.paths import DATA_DIR


def daily_floor(daily=None) -> pd.Timestamp:
    """Left bound for daily-centric plots: first non-race daily entry."""
    if daily is None:
        daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    non_race = daily[daily['run_type'].astype(str) != 'race']
    src = non_race if len(non_race) else daily
    return pd.Timestamp(src['date'].min())


def pad_range(lo, hi, frac: float = 0.02):
    """Pad an [lo, hi] date range by a small fraction of its span on each side.

    At a fixed plot width, a fraction of the data span maps to a fixed
    fraction of the pixel width — i.e. a near-constant pixel pad regardless of
    the date scale. This keeps edge markers from being clipped without leaving
    a big empty margin at short (e.g. 6-month) scales, which a fixed N-day pad
    does. Returns (padded_lo, padded_hi) as Timestamps.
    """
    lo = pd.Timestamp(lo)
    hi = pd.Timestamp(hi)
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def first_race_date(races=None):
    """Earliest race date (Timestamp), or None if there are no races."""
    if races is None:
        path = DATA_DIR / 'races.csv'
        races = pd.read_csv(path, parse_dates=['date']) if path.exists() else None
    if races is None or not len(races):
        return None
    return pd.Timestamp(races['date'].min())


def race_floor(months_before: int = 1, races=None) -> pd.Timestamp:
    """Left bound for race/fitness plots: ~``months_before`` before first race.

    Falls back to the daily floor if there are no races (those plots won't
    render anyway, but callers get a sane timestamp rather than NaT)."""
    if races is None:
        path = DATA_DIR / 'races.csv'
        races = pd.read_csv(path, parse_dates=['date']) if path.exists() else None
    if races is None or not len(races):
        return daily_floor()
    return pd.Timestamp(races['date'].min()) - pd.DateOffset(months=months_before)
