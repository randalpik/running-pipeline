"""SPIKE (June 2026): DEM-along-GPS elevation for every watch-era long run,
compared against the current barometric source. Read-only re: pipeline —
writes only /tmp/long_run_dem_spike.csv and mutates the shared dem_cache.

For each long-run day we gather its rich>=2 run activities, resample each
GPS track to 30 m, look up DEM elevation along the trusted horizontal path
(NED10m / SRTM30m, cached), and compute gain/loss/net/mean with the SAME
gridding+smoothing the barometric and race paths use. Then we line that up
against:
  - barometric gain/loss per mile  (elevation_measured.csv, current grade src)
  - barometric mean elevation      (altitude_daily.csv midpoint, current alt src)
so we can see what would change if long runs moved to DEM like races did.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.coros import mappings as M
from src.coros.dem_elevation import (track_points, _resample, dem_elevations,
                                     _load_cache, _save_cache)
from src.coros.elevation import gain_loss_ft, _gridded_altitude
from src.shared.hill_model import FT_PER_M
from src.shared.paths import DATA_DIR

MILE_M = 1609.344
DETAILS = DATA_DIR / 'profiles' / 'coros' / 'details'
ELEV_SPORTS = {M.SPORT_RUN, M.SPORT_TRAIL_RUN}


def _load(lid):
    p = DETAILS / f'{lid}.json'
    if not p.exists():
        return None
    import json
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def dem_day(recs, cache):
    """DEM gain/loss/net/mean(ft) pooled over a day's rich run activities."""
    g_tot = l_tot = 0.0
    alts, nets, npts = [], [], 0
    for rec in recs:
        if rec.get('rich') != 2:
            continue
        try:
            sport = int((rec.get('summary') or {}).get('sportType'))
        except (TypeError, ValueError):
            continue
        if sport not in ELEV_SPORTS:
            continue
        pts = track_points(rec)
        if len(pts) < 5:
            continue
        d, lat, lon = _resample(pts)
        if len(d) < 3:
            continue
        elevs = dem_elevations(lat, lon, cache)
        valid = np.array([e is not None for e in elevs])
        if valid.sum() < 3:
            continue
        d = d[valid]
        alt = np.array([e for e in elevs if e is not None], float)
        g, l = gain_loss_ft(d, alt)
        _, galt = _gridded_altitude(d, alt)
        g_tot += g
        l_tot += l
        nets.append(float((galt[-1] - galt[0]) / FT_PER_M))
        alts.append(galt / FT_PER_M)
        npts += int(valid.sum())
    if not alts:
        return None
    allalt = np.concatenate(alts)
    return {'dem_gain_ft': round(g_tot, 1), 'dem_loss_ft': round(l_tot, 1),
            'dem_net_ft': round(sum(nets), 1),
            'dem_mean_elev_ft': round(float(allalt.mean()), 1),
            'dem_min_ft': round(float(allalt.min()), 1),
            'dem_max_ft': round(float(allalt.max()), 1),
            'dem_n_pts': npts}


def main():
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    lr = daily[(daily['run_type'] == 'long')
               & (daily['date'] >= pd.Timestamp('2020-12-25'))].copy()
    acts = pd.read_csv(DATA_DIR / 'watch_activities.csv',
                       dtype={'labelId': str, 'date': str})
    ids_by_date = {}
    for _, r in acts.iterrows():
        if str(r['labelId']) in M.EXCLUDED_LABEL_IDS:
            continue
        ids_by_date.setdefault(r['date'], []).append(str(r['labelId']))

    em = pd.read_csv(DATA_DIR / 'elevation_measured.csv', parse_dates=['date'])
    em = em[em['run_type'] == 'long'].set_index('date')
    alt = pd.read_csv(DATA_DIR / 'altitude_daily.csv', parse_dates=['date'])
    alt['baro_mid_ft'] = (alt['min_elev_ft'] + alt['max_elev_ft']) / 2.0
    alt = alt.set_index('date')

    cache = _load_cache()
    rows = []
    for i, (_, drow) in enumerate(lr.sort_values('date').iterrows()):
        dt = drow['date']
        ds = dt.date().isoformat()
        lids = ids_by_date.get(ds, [])
        recs = [r for r in (_load(l) for l in lids) if r]
        dem = dem_day(recs, cache) if recs else None
        _save_cache(cache)  # checkpoint after each day (cheap, resumable)
        row = {'date': ds, 'location': drow['location'],
               'terrain': drow['terrain_type'], 'miles': drow['miles']}
        # current barometric
        if dt in em.index:
            cm = float(em.loc[dt, 'corr_miles'])
            row['baro_gain_pm'] = float(em.loc[dt, 'elev_gain_ft']) / cm
            row['baro_loss_pm'] = float(em.loc[dt, 'elev_loss_ft']) / cm
        row['baro_mean_ft'] = float(alt.loc[dt, 'baro_mid_ft']) if dt in alt.index else np.nan
        row['const_alt_ft'] = drow.get('altitude')
        if dem:
            cm = float(em.loc[dt, 'corr_miles']) if dt in em.index else drow['miles']
            row['dem_gain_pm'] = dem['dem_gain_ft'] / cm
            row['dem_loss_pm'] = dem['dem_loss_ft'] / cm
            row['dem_net_ft'] = dem['dem_net_ft']
            row['dem_mean_ft'] = dem['dem_mean_elev_ft']
            row['dem_n_pts'] = dem['dem_n_pts']
        rows.append(row)
        print(f'[{i+1}/{len(lr)}] {ds} {drow["location"]:<16} '
              f'{"DEM ok" if dem else "no rich track"} '
              f'(cache {len(cache)})', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv('/tmp/long_run_dem_spike.csv', index=False)
    print(f'\nwrote /tmp/long_run_dem_spike.csv ({len(out)} rows, '
          f'{out["dem_mean_ft"].notna().sum() if "dem_mean_ft" in out else 0} with DEM)')


if __name__ == '__main__':
    main()
