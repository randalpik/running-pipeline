"""Backfill per-second altitude/speed (rich>=2) and compute the elevation
enrichment artifacts for long-run, recovery, and race days.

For each target day's run activities: re-fetch the raw detail (the cached slim
record dropped the stream), project to a rich v2 record (altitude+speed),
upgrade the cache file in place, then compute the day's elevation metrics
(gain/loss, Minetti grade factor, per-corrected-mile splits).

Writes two artifacts, both incrementally (resumable — already-done dates are
skipped, and a cache record that's already rich v2 is recomputed without an
API call):
  data/elevation_measured.csv : date, run_type, watch_miles, corr_miles,
                                elev_gain_ft, elev_loss_ft, minetti_factor,
                                n_alt_pts
  data/elevation_splits.csv   : date, mile, pace_s, gain_ft, loss_ft

Distance correction (corr_miles) drives the split axis: long/recovery use the
watch/route-corrected mileage (shared.effective_mileage); races use the
OFFICIAL course distance (certified > GPS) — only elevation/grade is taken
from the watch.

Usage:
  python scripts/backfill_elevation.py --run-type long      # pilot
  python scripts/backfill_elevation.py --run-type all       # long+recovery+race
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from src.shared.env import load_env_file
from src.shared.paths import DATA_DIR
from src.coros import mappings as M
from src.coros.build_current_log import Activity, rich_detail
from src.coros.client import CorosClient
from src.coros import elevation as E
from src.shared.effective_mileage import effective_daily_miles

# Elevation is meaningful only on outdoor GPS runs over real terrain: Run
# (100) and Trail Run (102). Indoor/treadmill (101) has no altitude, and
# Track Run (103) is flat by definition — its barometric stream is pure
# drift (this is what produced the bogus 1.5-1.8 "Minetti factors" on track
# races). Both are excluded from the elevation enrichment.
ELEV_SPORTS = {M.SPORT_RUN, M.SPORT_TRAIL_RUN}

# Universal watch-validity gate: the watch must have actually recorded the run
# (its distance within this band of the hand-logged distance). Days where the
# watch failed — e.g. 2026-05-01, 0.5 of 7 mi recorded — are rejected for ALL
# watch use: the day falls back to hand-logged distance/time and gets NO watch
# elevation. This is the SAME validity verdict the distance correction
# respects; the distance correction layers its extra criteria (paved, strides,
# WATCH_FAIL_DEV) ON TOP. Elevation, being our only source of truth for grade
# and more reliable than GPS distance, uses every VALID day regardless of
# terrain or strides.
WATCH_VALID_BAND = (0.6, 1.5)

DETAILS = DATA_DIR / 'profiles' / 'coros' / 'details'
TOKEN = DATA_DIR / 'profiles' / 'coros' / 'coros_token.json'
MEAS_OUT = DATA_DIR / 'elevation_measured.csv'
SPLITS_OUT = DATA_DIR / 'elevation_splits.csv'


def _cache_by_date():
    """{iso_date: [(path, rec, Activity), ...]} for run activities in cache."""
    by_date = {}
    for path in sorted(DETAILS.glob('*.json')):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if (rec.get('summary') or {}).get('sportType') is None:
            continue
        act = Activity(rec)
        if act.sport_type not in M.RUN_SPORTS:
            continue
        by_date.setdefault(act.local_date.isoformat(), []).append((path, rec, act))
    return by_date


def main():
    load_env_file()
    p = argparse.ArgumentParser()
    p.add_argument('--run-type', choices=['long', 'recovery', 'race', 'all'],
                   default='long')
    p.add_argument('--sleep', type=float, default=0.4, help='s between fetches')
    p.add_argument('--limit', type=int, default=0, help='cap days (0=all)')
    args = p.parse_args()

    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    logged_miles = {r.date.date().isoformat(): float(r.miles)
                    for r in daily.itertuples() if pd.notna(r.miles)}
    daily['eff_miles'] = effective_daily_miles(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])

    # target dates -> (run_type, corr_miles or None for race=official-by-date)
    targets = {}
    types = (['long', 'recovery', 'race'] if args.run_type == 'all'
             else [args.run_type])
    for rt in ('long', 'recovery'):
        if rt in types:
            sub = daily[daily['run_type'] == rt]
            for _, r in sub.iterrows():
                targets[r['date'].date().isoformat()] = (rt, float(r['eff_miles']))
    if 'race' in types:
        for _, r in races.iterrows():
            # Track races are flat by definition — their barometric stream is
            # drift, and on track-meet days the road warmup/cooldown (logged
            # as sport 100) would otherwise leak a bogus profile in. Skip them.
            if str(r.get('surface', '')).strip().lower() == 'track':
                continue
            d = r['date'].date().isoformat()
            off_mi = float(r['distance_m']) / 1609.344 if pd.notna(r.get('distance_m')) else None
            targets.setdefault(d, ('race', off_mi))

    by_date = _cache_by_date()

    done = set()
    if MEAS_OUT.exists():
        done = set(pd.read_csv(MEAS_OUT)['date'].astype(str))
    meas_rows, split_rows = [], []
    client = None
    fetched = skipped = computed = 0

    todo = [d for d in sorted(targets) if d in by_date and d not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"targets={len(targets)} in-cache&pending={len(todo)} already-done={len(done)}")

    for i, d in enumerate(todo):
        rt, corr_mi = targets[d]
        recs, watch_m = [], 0.0
        for path, rec, act in by_date[d]:
            if act.sport_type not in ELEV_SPORTS:
                continue  # skip indoor/track activities (no usable altitude)
            if rec.get('rich') == 2:
                recs.append(rec)
            else:
                if client is None:
                    client = CorosClient(os.environ['COROS_EMAIL'],
                                         os.environ['COROS_PASSWORD'],
                                         token_cache=TOKEN)
                try:
                    raw = client.activity_detail(path.stem, act.sport_type)
                    rich = rich_detail(raw)
                    if rich is not None:
                        path.write_text(json.dumps(rich))
                        recs.append(rich)
                        fetched += 1
                    time.sleep(args.sleep)
                except Exception as e:
                    print(f"  {d} fetch fail {path.stem}: {e}")
                    continue
            watch_m += act.distance_m / 1609.344
        if not recs:
            continue
        # Universal validity gate: reject days where the watch didn't record
        # the run (distance grossly off hand-logged). No watch elevation here.
        lg = logged_miles.get(d)
        if lg and watch_m and not (WATCH_VALID_BAND[0] <= watch_m / lg
                                   <= WATCH_VALID_BAND[1]):
            skipped += 1
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
        if computed % 25 == 0:
            print(f"  [{i+1}/{len(todo)}] computed={computed} fetched={fetched}")
            _flush(meas_rows, split_rows)
            meas_rows, split_rows = [], []

    _flush(meas_rows, split_rows)
    print(f"DONE: computed={computed} fetched={fetched} "
          f"skipped(watch-invalid)={skipped}")


def _flush(meas_rows, split_rows):
    if meas_rows:
        df = pd.DataFrame(meas_rows)
        df.to_csv(MEAS_OUT, mode='a', header=not MEAS_OUT.exists(), index=False)
    if split_rows:
        df = pd.DataFrame(split_rows)
        df.to_csv(SPLITS_OUT, mode='a', header=not SPLITS_OUT.exists(), index=False)


if __name__ == '__main__':
    main()
