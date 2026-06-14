"""Elevation enrichment for long-run, recovery, and race days — gain/loss,
Minetti grade factor, per-corrected-mile splits.

Pipeline producer (incremental, presence-based): candidate days come from the
per-activity index (watch_activities.csv, written by watch_daily — no full
parse), and only days NOT already in elevation_measured.csv are computed
(``--full-regen`` recomputes all). Since the sync now caches outdoor runs rich,
the per-second altitude stream is already present, so this runs WITHOUT any
network call. ``--fetch`` re-enables the one-time historical upgrade (re-fetch
+ rich-v2 cache-upgrade) for any day still cached slim.

Writes:
  data/elevation_measured.csv : date, run_type, watch_miles, corr_miles,
                                elev_gain_ft, elev_loss_ft, minetti_factor,
                                n_alt_pts
  data/elevation_splits.csv   : date, mile, pace_s, gain_ft, loss_ft

corr_miles drives the split axis: long/recovery use the watch/route-corrected
mileage (shared.effective_mileage); races use the OFFICIAL course distance.

Usage:
  python scripts/backfill_elevation.py                 # incremental, no fetch
  python scripts/backfill_elevation.py --full-regen    # recompute every day
  python scripts/backfill_elevation.py --fetch          # upgrade slim days (network)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.shared.env import load_env_file
from src.shared.paths import DATA_DIR
from src.coros import mappings as M
from src.coros.build_current_log import Activity, rich_detail
from src.coros.client import CorosClient
from src.coros import elevation as E
from src.shared.effective_mileage import effective_daily_miles

# Elevation is meaningful only on outdoor GPS runs over real terrain: Run (100)
# and Trail Run (102). Indoor (101) has no altitude; Track (103) is flat by
# definition (its barometric stream is pure drift — the source of the bogus
# 1.5-1.8 "Minetti factors"). Both are excluded.
ELEV_SPORTS = {M.SPORT_RUN, M.SPORT_TRAIL_RUN}

# Watch-validity gate: the watch must have recorded the run (distance within
# this band of the hand-logged distance), else the day is rejected for all
# watch use and gets NO watch elevation.
WATCH_VALID_BAND = (0.6, 1.5)

DETAILS = DATA_DIR / 'profiles' / 'coros' / 'details'
TOKEN = DATA_DIR / 'profiles' / 'coros' / 'coros_token.json'
ACTIVITIES = DATA_DIR / 'watch_activities.csv'
MEAS_OUT = DATA_DIR / 'elevation_measured.csv'
SPLITS_OUT = DATA_DIR / 'elevation_splits.csv'

MEAS_COLS = ['date', 'run_type', 'watch_miles', 'corr_miles', 'elev_gain_ft',
             'elev_loss_ft', 'minetti_factor', 'n_alt_pts']
SPLIT_COLS = ['date', 'mile', 'pace_s', 'gain_ft', 'loss_ft']


def _elev_ids_by_date():
    """{iso_date: [labelId, ...]} of ELEV_SPORTS activities, from the per-
    activity index — no per-second parse. None if the index is absent."""
    if not ACTIVITIES.exists():
        return None
    idx = pd.read_csv(ACTIVITIES, dtype={'labelId': str, 'date': str})
    out = {}
    for _, r in idx.iterrows():
        if int(r['sport_type']) in ELEV_SPORTS:
            out.setdefault(r['date'], []).append(r['labelId'])
    return out


def _targets(daily, races, types):
    """{date: (run_type, corr_miles)} — long/recovery use effective miles,
    races use the official course distance (None -> fall back to watch)."""
    targets = {}
    for rt in ('long', 'recovery'):
        if rt in types:
            for _, r in daily[daily['run_type'] == rt].iterrows():
                targets[r['date'].date().isoformat()] = (rt, float(r['eff_miles']))
    if 'race' in types:
        for _, r in races.iterrows():
            if str(r.get('surface', '')).strip().lower() == 'track':
                continue                      # flat — barometric drift only
            d = r['date'].date().isoformat()
            off_mi = (float(r['distance_m']) / 1609.344
                      if pd.notna(r.get('distance_m')) else None)
            targets.setdefault(d, ('race', off_mi))
    return targets


def main():
    load_env_file()
    p = argparse.ArgumentParser()
    p.add_argument('--run-type', choices=['long', 'recovery', 'race', 'all'],
                   default='all')
    p.add_argument('--full-regen', action='store_true',
                   help='recompute every target day, ignoring the presence cache')
    p.add_argument('--fetch', action='store_true',
                   help='re-fetch + rich-upgrade days still cached slim (network)')
    p.add_argument('--sleep', type=float, default=0.4)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    ids_by_date = _elev_ids_by_date()
    if ids_by_date is None:
        print('[elevation] no watch_activities.csv index — run watch_daily '
              'first; skipped')
        return

    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    logged_miles = {r.date.date().isoformat(): float(r.miles)
                    for r in daily.itertuples() if pd.notna(r.miles)}
    daily['eff_miles'] = effective_daily_miles(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    types = (['long', 'recovery', 'race'] if args.run_type == 'all'
             else [args.run_type])
    targets = _targets(daily, races, types)

    # Existing rows to reuse for days already computed (presence by date).
    done, meas_keep, split_keep = set(), [], []
    if MEAS_OUT.exists() and not args.full_regen:
        em = pd.read_csv(MEAS_OUT, dtype={'date': str})
        done = set(em['date'])
        meas_keep = em.to_dict('records')
        if SPLITS_OUT.exists():
            sp = pd.read_csv(SPLITS_OUT, dtype={'date': str})
            split_keep = sp.to_dict('records')

    todo = [d for d in sorted(targets) if d in ids_by_date and d not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[elevation] targets={len(targets)} pending={len(todo)} "
          f"reused={len(done)}")

    client = None
    meas_rows, split_rows = [], []
    fetched = skipped_slim = skipped_invalid = computed = 0
    for d in todo:
        rt, corr_mi = targets[d]
        recs, watch_m = [], 0.0
        for lid in ids_by_date[d]:
            path = DETAILS / f'{lid}.json'
            if not path.exists():
                continue
            rec = json.loads(path.read_text())
            watch_m += Activity(rec).distance_m / 1609.344
            if rec.get('rich') == 2:
                recs.append(rec)
            elif args.fetch:
                if client is None:
                    client = CorosClient(os.environ['COROS_EMAIL'],
                                         os.environ['COROS_PASSWORD'],
                                         token_cache=TOKEN)
                try:
                    rich = rich_detail(client.activity_detail(
                        lid, (rec.get('summary') or {}).get('sportType')))
                    if rich is not None:
                        path.write_text(json.dumps(rich))
                        recs.append(rich)
                        fetched += 1
                    time.sleep(args.sleep)
                except Exception as e:
                    print(f"  {d} fetch fail {lid}: {e}")
        if not recs:
            skipped_slim += 1                 # cached slim, no --fetch
            continue
        lg = logged_miles.get(d)
        if lg and watch_m and not (WATCH_VALID_BAND[0] <= watch_m / lg
                                   <= WATCH_VALID_BAND[1]):
            skipped_invalid += 1
            continue
        eff_corr = corr_mi if (corr_mi and corr_mi > 0) else watch_m
        res = E.measure_day_elevation(recs, eff_corr, watch_m)
        if res is None:
            continue
        meas_rows.append({'date': d, 'run_type': rt,
                          'watch_miles': round(watch_m, 3),
                          'corr_miles': round(eff_corr, 3),
                          'elev_gain_ft': res['elev_gain_ft'],
                          'elev_loss_ft': res['elev_loss_ft'],
                          'minetti_factor': res['minetti_factor'],
                          'n_alt_pts': res['n_alt_pts']})
        for s in res['splits']:
            split_rows.append({'date': d, **s})
        computed += 1

    meas = pd.DataFrame(meas_keep + meas_rows, columns=MEAS_COLS)
    splits = pd.DataFrame(split_keep + split_rows, columns=SPLIT_COLS)
    if len(meas):
        meas = meas.sort_values('date').reset_index(drop=True)
        meas.to_csv(MEAS_OUT, index=False)
    if len(splits):
        splits = splits.sort_values(['date', 'mile']).reset_index(drop=True)
        splits.to_csv(SPLITS_OUT, index=False)
    print(f"[elevation] computed={computed} reused={len(done)} fetched={fetched} "
          f"slim-skipped={skipped_slim} watch-invalid={skipped_invalid} "
          f"-> {len(meas)} days")


if __name__ == '__main__':
    main()
