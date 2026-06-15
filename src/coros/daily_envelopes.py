"""Per-day envelope tables for the Misc. Trends plot, mined from the watch
detail cache (no network):

  data/altitude_daily.csv : date, min_elev_ft, max_elev_ft, n_pts
      Within-day absolute-elevation range across every outdoor run that day
      (Run + Trail Run — trail is where the fourteeners live). Each activity's
      barometric stream is distance-gridded and 120 m-smoothed (the same
      `elevation._gridded_altitude` used for gain/loss) before taking its
      min/max, so single-sample barometric spikes don't define the envelope.
      Activities are NOT stitched (the multi-activity stitch is what produced
      the bogus Minetti factors); we just combine per-activity extremes.

  data/time_daily.csv : date, start_min, end_min, lat, lon, tz_min
      The day's earliest local start and latest local end (minutes-of-day,
      0-1439) across all run activities, plus the representative GPS + tz used
      to compute that day's solar gradient. Indoor-only days fall back to home
      coordinates so the Time panel still renders a band for them.

Both are full rebuilds from the cache each run (local reads only). Like every
watch-derived table (elevation_measured, weather_measured, ...) they are built
wherever the details cache is present — locally for Max, restored-from-cache for
the Maddy profile on GHA — and are gitignored, not committed. The Misc. Trends
plot reads them when present and renders those panels empty otherwise, so a
profile/run without the cache degrades gracefully (no error). This is a no-op
when the cache is absent.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.coros import mappings as M
from src.coros import elevation as E
from src.coros.build_current_log import Activity
from src.shared.hill_model import FT_PER_M
from src.shared.paths import DATA_DIR

DETAILS = DATA_DIR / 'profiles' / 'coros' / 'details'
ACTIVITIES = DATA_DIR / 'watch_activities.csv'
ALT_OUT = DATA_DIR / 'altitude_daily.csv'
TIME_OUT = DATA_DIR / 'time_daily.csv'

# Outdoor GPS sports with a trustworthy barometric stream (see backfill_elevation).
ELEV_SPORTS = {M.SPORT_RUN, M.SPORT_TRAIL_RUN}

# Corrupt-stream backstop: a single activity whose smoothed altitude spans more
# than this is a barometer glitch, not a real climb (a fourteener ascent from a
# trailhead is ~5-6k ft); drop it from the envelope and log.
MAX_ACTIVITY_RANGE_FT = 12000.0

# Home (Boulder, CO) — solar fallback for indoor-only days with no GPS fix.
HOME_LAT, HOME_LON = 40.0150, -105.2705

ALT_COLS = ['date', 'min_elev_ft', 'max_elev_ft', 'n_pts']
TIME_COLS = ['date', 'start_min', 'end_min', 'lat', 'lon', 'tz_min']


def _ids_by_date():
    """{iso_date: [labelId, ...]} from the per-activity index, or None."""
    if not ACTIVITIES.exists():
        return None
    idx = pd.read_csv(ACTIVITIES, dtype={'labelId': str, 'date': str})
    out = {}
    for _, r in idx.iterrows():
        if str(r['labelId']) in M.EXCLUDED_LABEL_IDS:
            continue
        out.setdefault(r['date'], []).append(str(r['labelId']))
    return out


def _load(labelId):
    path = DETAILS / f'{labelId}.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _activity_altitude_range_ft(rec):
    """(min_ft, max_ft) of the smoothed gridded altitude for one rich record,
    or None if there's no usable altitude stream or the range is implausible."""
    if rec.get('rich') != 2:
        return None
    pts = E.alt_points(rec)
    if len(pts) < 3:
        return None
    dist = [p[1] for p in pts]
    alt = [p[2] for p in pts]
    _, galt = E._gridded_altitude(dist, alt)
    if galt is None or len(galt) == 0:
        return None
    # FT_PER_M is 0.3048 (metres per foot, despite the name) — divide to get ft,
    # matching elevation.gain_loss_ft.
    mn = float(np.nanmin(galt)) / FT_PER_M
    mx = float(np.nanmax(galt)) / FT_PER_M
    if not (np.isfinite(mn) and np.isfinite(mx)):
        return None
    if mx - mn > MAX_ACTIVITY_RANGE_FT:
        return None
    return mn, mx


def _local_minutes(dt, tz_min):
    """Minutes-of-day (0-1439, float) of an aware UTC datetime in tz_min, and
    its local date."""
    local = dt.astimezone(timezone(timedelta(minutes=tz_min)))
    return local.hour * 60.0 + local.minute + local.second / 60.0, local.date()


def build():
    ids_by_date = _ids_by_date()
    if ids_by_date is None:
        print('[daily_envelopes] no watch_activities.csv index — '
              'run watch_daily first; skipped')
        return

    alt_rows, time_rows = [], []
    dropped_alt = 0
    for d in sorted(ids_by_date):
        recs = [r for r in (_load(lid) for lid in ids_by_date[d]) if r]
        if not recs:
            continue

        # ---- altitude envelope (outdoor sports, rich stream) ----
        mins_ft, maxs_ft, npts = [], [], 0
        for rec in recs:
            try:
                sport = int((rec.get('summary') or {}).get('sportType'))
            except (TypeError, ValueError):
                continue
            if sport not in ELEV_SPORTS:
                continue
            rng = _activity_altitude_range_ft(rec)
            if rng is None:
                if rec.get('rich') == 2:
                    dropped_alt += 1
                continue
            mins_ft.append(rng[0])
            maxs_ft.append(rng[1])
            npts += len(rec.get('freq') or [])
        if mins_ft:
            alt_rows.append({'date': d,
                             'min_elev_ft': round(min(mins_ft), 1),
                             'max_elev_ft': round(max(maxs_ft), 1),
                             'n_pts': npts})

        # ---- time envelope (all run activities) ----
        acts = []
        for rec in recs:
            try:
                a = Activity(rec)
            except (TypeError, ValueError, KeyError):
                continue
            if a.sport_type in M.RUN_SPORTS:
                acts.append(a)
        if not acts:
            continue
        acts.sort(key=lambda a: a.start_utc)
        starts, ends = [], []
        for a in acts:
            s_min, s_date = _local_minutes(a.start_utc, a.tz_min)
            e_dt = a.start_utc + timedelta(seconds=a.total_s)
            e_min, e_date = _local_minutes(e_dt, a.tz_min)
            if e_date > s_date or e_min <= s_min:
                e_min = 1439.0          # crossed midnight — clamp (rare ultras)
            starts.append(s_min)
            ends.append(e_min)
        # Representative GPS for the solar gradient: first outdoor fix; else home.
        lat = lon = None
        for a in acts:
            if not a.is_indoor and a.lat is not None and a.lon is not None:
                lat, lon = a.lat, a.lon
                break
        if lat is None:
            lat, lon = HOME_LAT, HOME_LON
        time_rows.append({'date': d,
                          'start_min': round(min(starts), 1),
                          'end_min': round(max(ends), 1),
                          'lat': round(lat, 4), 'lon': round(lon, 4),
                          'tz_min': acts[0].tz_min})

    alt = pd.DataFrame(alt_rows, columns=ALT_COLS).sort_values('date')
    time = pd.DataFrame(time_rows, columns=TIME_COLS).sort_values('date')
    alt.to_csv(ALT_OUT, index=False)
    time.to_csv(TIME_OUT, index=False)
    print(f'[daily_envelopes] altitude days={len(alt)} '
          f'(dropped {dropped_alt} glitchy activities), time days={len(time)}')


if __name__ == '__main__':
    build()
