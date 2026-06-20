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
from src.coros import dem_elevation as DEM
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
             'elev_loss_ft', 'minetti_factor', 'n_alt_pts',
             'dem_gain_ft', 'dem_loss_ft', 'dem_net_ft', 'dem_mean_elev_ft',
             'dem_n_pts']
SPLIT_COLS = ['date', 'mile', 'pace_s', 'gain_ft', 'loss_ft']

# Flush the DEM point cache to disk every this many days during augmentation.
# The one-time cold seed can run for hours against the public DEM API's 1 req/s
# courtesy limit; a periodic flush means a killed / timed-out run keeps every
# point it already fetched and resumes from there instead of restarting. The
# measurements (elevation_measured.csv) are deliberately NOT flushed mid-loop:
# they recompute for free against a warm cache on the next run, so the cache is
# the only expensive, network-built asset worth persisting incrementally.
FLUSH_EVERY_DAYS = 25


def _flush_cache(cache, prev_size, processed, total, label):
    """Persist the DEM cache mid-loop and report progress. ``+N fetched`` is the
    live network signal (points that came over the wire since the last flush).
    Returns the new baseline point count."""
    DEM._save_cache(cache)
    print(f"[elevation]   ↳ saved {len(cache):,} pts "
          f"(+{len(cache) - prev_size:,} fetched) — {processed}/{total} "
          f"{label} days done", flush=True)
    return len(cache)


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


def augment_race_dem(meas, races, ids_by_date, sleep_s, verbose=False):
    """Fill DEM gain/loss/net/mean for race rows from the GPS track (races-only;
    the watch's barometric net is per-race noise — see dem_elevation.py). Idem-
    potent: only rows missing dem_gain_ft are computed, so a re-run is a cheap
    cache-served top-up. Returns the count newly computed."""
    if 'run_type' not in meas.columns:
        return 0
    off_by_date = {r['date'].date().isoformat(): float(r['distance_m'])
                   for _, r in races.iterrows() if pd.notna(r.get('distance_m'))}
    need = meas[(meas['run_type'] == 'race') & meas['dem_gain_ft'].isna()]
    if need.empty:
        return 0
    cache = DEM._load_cache()
    print(f"[elevation] DEM race: {len(need)} days need lookup "
          f"({len(cache):,} pts cached)", flush=True)
    n, processed, prev_size = 0, 0, len(cache)
    for i, row in need.iterrows():
        processed += 1
        if processed % FLUSH_EVERY_DAYS == 0:
            prev_size = _flush_cache(cache, prev_size, processed, len(need),
                                     'race')
        d = row['date']
        off_m = off_by_date.get(d)
        if off_m is None or d not in ids_by_date:
            continue
        recs = []
        for lid in ids_by_date[d]:
            p = DETAILS / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if not recs:
            continue
        res = DEM.measure_race_elevation(recs, off_m, cache, verbose=verbose)
        if res is None:
            continue
        for k, v in res.items():
            meas.at[i, k] = v
        n += 1
    DEM._save_cache(cache)
    return n


def augment_run_dem(meas, ids_by_date, run_type, verbose=False):
    """Fill DEM gain/loss/net/mean for ``run_type`` (long/recovery) rows from the
    day's pooled GPS track (see dem_elevation.measure_run_elevation). Training
    runs are loops, so the barometric net carries a phantom morning-drift descent
    that DEM removes — same fix as races, applied to the whole-day run rather than
    a single race activity. GPS-corrupt days (false fix / dead-zone) return no DEM
    via the track-quality gate and stay on barometric. Idempotent: only rows
    missing dem_gain_ft are computed, so a re-run is a cheap cache-served top-up.
    Returns the count newly computed."""
    if 'run_type' not in meas.columns:
        return 0
    need = meas[(meas['run_type'] == run_type) & meas['dem_gain_ft'].isna()]
    if need.empty:
        return 0
    cache = DEM._load_cache()
    print(f"[elevation] DEM {run_type}: {len(need)} days need lookup "
          f"({len(cache):,} pts cached)", flush=True)
    n, processed, prev_size = 0, 0, len(cache)
    for i, row in need.iterrows():
        processed += 1
        if processed % FLUSH_EVERY_DAYS == 0:
            prev_size = _flush_cache(cache, prev_size, processed, len(need),
                                     run_type)
        d = row['date']
        if d not in ids_by_date:
            continue
        recs = []
        for lid in ids_by_date[d]:
            p = DETAILS / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if not recs:
            continue
        res = DEM.measure_run_elevation(recs, cache, verbose=verbose)
        if res is None:
            continue
        for k, v in res.items():
            meas.at[i, k] = v
        n += 1
    DEM._save_cache(cache)
    return n


def regate_dem(meas, ids_by_date):
    """Clear dem_* on long/recovery days whose pooled GPS track now fails the
    quality gates (dem_elevation.track_ok) — chiefly rows filled before the gate
    existed. The day falls back to barometric in per_run_elevation. Cheap:
    track_ok is stream arithmetic, no DEM recompute. (Races re-gate through
    augment_race_dem's gated measurement.) Returns the count cleared."""
    if 'dem_gain_ft' not in meas.columns or 'run_type' not in meas.columns:
        return 0
    cols = ['dem_gain_ft', 'dem_loss_ft', 'dem_net_ft', 'dem_mean_elev_ft',
            'dem_n_pts']
    tgt = meas[meas['run_type'].isin(['long', 'recovery'])
               & meas['dem_gain_ft'].notna()]
    n = 0
    for i, row in tgt.iterrows():
        recs = []
        for lid in ids_by_date.get(row['date'], []):
            p = DETAILS / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if recs and any(not DEM.track_ok(rec) for rec in recs):
            for c in cols:
                if c in meas.columns:
                    meas.at[i, c] = float('nan')
            n += 1
    return n


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
    p.add_argument('--dem-verbose', action='store_true',
                   help='per-batch heartbeat from the DEM point lookups (one '
                        'line per ~100 points fetched — chatty)')
    args = p.parse_args()

    # One-time, network-free upgrade of the DEM point cache to the current
    # lookup grid (see dem_elevation.migrate_cache / KEY_DECIMALS). A legacy
    # ~1 m-keyed cache stored ~4x duplicate near-points that never reused across
    # runs; collapsing to the ~10 m grid makes repeated routes cache-hit and
    # shrinks the cache. Idempotent — re-saves only when it actually changed.
    cache = DEM._load_cache()
    migrated = DEM.migrate_cache(cache)
    if len(migrated) < len(cache):
        DEM._save_cache(migrated)
        print(f"[elevation] DEM cache upgraded to {DEM.KEY_DECIMALS}-decimal "
              f"grid: {len(cache):,} -> {len(migrated):,} points", flush=True)

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
    if 'race' in types and len(meas):
        n_dem = augment_race_dem(meas, races, ids_by_date, args.sleep,
                                 verbose=args.dem_verbose)
        print(f"[elevation] DEM race-elevation: {n_dem} newly computed "
              f"(GPS-track lookup; barometric net is per-race noise)")
    for rt in ('long', 'recovery'):
        if rt in types and len(meas):
            n_dem = augment_run_dem(meas, ids_by_date, rt,
                                    verbose=args.dem_verbose)
            print(f"[elevation] DEM {rt}-run elevation: {n_dem} newly computed "
                  f"(GPS-track lookup; barometric net is morning-drift phantom)")
    if len(meas):
        n_clr = regate_dem(meas, ids_by_date)
        if n_clr:
            print(f"[elevation] DEM re-gate: cleared {n_clr} GPS-corrupt "
                  f"long/recovery days (fall back to barometric)")
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
