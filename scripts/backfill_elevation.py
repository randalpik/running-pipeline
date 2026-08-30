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

import numpy as np
import pandas as pd

from src.shared.env import load_env_file
from src.shared.paths import DATA_DIR
from src.coros import mappings as M
from src.coros.build_current_log import Activity, rich_detail, strip_paused
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

# Outdoor run types that get elevation rows (races are handled separately from
# races.csv). long/recovery feed the route models; the rest — trail, hill and
# quality days — carry the steepest descents in the corpus and feed the
# grade-distribution analyses.
RUN_TYPES_ELEV = ('long', 'recovery', 'trail', 'hill_cont', 'hill_rep',
                  'interval', 'fartlek', 'tempo', 'rep')

# Details dir: a watch-import profile (e.g. maddy) carries its own cache at
# DATA_DIR/details; the Max drive profile has none there and pulls from the
# coros (his watch) cache. Resolve to the profile's own cache when present, else
# the coros one — same rule as daily_envelopes.
_OWN_DETAILS = DATA_DIR / 'details'
DETAILS = _OWN_DETAILS if _OWN_DETAILS.exists() else DATA_DIR / 'profiles' / 'coros' / 'details'
TOKEN = DATA_DIR / 'profiles' / 'coros' / 'coros_token.json'
ACTIVITIES = DATA_DIR / 'watch_activities.csv'
MEAS_OUT = DATA_DIR / 'elevation_measured.csv'
SPLITS_OUT = DATA_DIR / 'elevation_splits.csv'
HILLS_OUT = DATA_DIR / 'elevation_hills.csv'
# Day-level failure memo: days that were attempted and legitimately produced
# nothing (no rich details, watch-invalid, measurement/DEM gated) — without it
# they are indistinguishable from "never tried" and re-walk on every run
# (~150 days/run by Aug 2026, growing). Stored INSIDE elevation_measured.csv
# as a ``skip_reason`` column ('stage:reason'): day-stage failures get a stub
# row (all metrics NaN), dem/fuse-stage memos annotate the existing row. In a
# file already on the GHA state-cache path list DELIBERATELY — adding a new
# path invalidates the actions/cache version hash and cold-starts CI (learned
# the hard way, Aug 2026). Entries clear when the day later succeeds, when
# --fetch retries 'slim' days, or wholesale under --full-regen.
LEGACY_SKIPS_OUT = DATA_DIR / 'elevation_skips.csv'   # pre-column sidecar

# A kept day whose corrected distance no longer matches the current
# calibration by more than this is re-derived in full (row + splits + hills).
# With the calibration pinned behind long_runs' adoption deadband this set is
# empty on every ordinary run; it fires once per adoption (≤ ~1/year) and
# heals exactly the corrected days, replacing the old fit-run --full-regen.
CORR_STALE_MI = 0.02

MEAS_COLS = ['date', 'run_type', 'watch_miles', 'corr_miles', 'elev_gain_ft',
             'elev_loss_ft', 'minetti_factor', 'g_gain_pct', 'g_loss_pct',
             'n_alt_pts', 'fused',
             'seg_up_ft', 'seg_dn_ft', 'g_up_pct', 'g_dn_pct',
             'dem_gain_ft', 'dem_loss_ft', 'dem_net_ft', 'dem_mean_elev_ft',
             'dem_n_pts', 'dem_g_gain_pct', 'dem_g_loss_pct', 'skip_reason']
HILL_COLS = ['date', 'act', 'd0', 'd1', 'vert_ft', 'grade_pct', 'kind',
             'lat', 'lon', 'dem_net_ft', 'vetoed']
SPLIT_COLS = ['date', 'mile', 'pace_s', 'gain_ft', 'loss_ft', 'covered',
              'g_up', 'g_down', 'seg_up_ft', 'seg_dn_ft']

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
    Counts only cells carrying a real elevation — the cache also holds
    negative-cached misses (DEM.MISS_NOTE), and counting those would report
    "fetched" for ground NED does not cover.
    Returns the new baseline point count."""
    DEM._save_cache(cache)
    n = DEM.point_count(cache)
    print(f"[elevation]   ↳ saved {n:,} pts "
          f"(+{n - prev_size:,} fetched) — {processed}/{total} "
          f"{label} days done", flush=True)
    return n


def _parse_skips(em):
    """{(stage, date): reason} from the skip_reason column ('stage:reason');
    empty when the column is absent (a pre-memo artifact — the first run with
    this code rebuilds the memo in one walk)."""
    skips = {}
    if 'skip_reason' in em.columns:
        for r in em[em['skip_reason'].notna()].itertuples():
            stage, _, reason = str(r.skip_reason).partition(':')
            skips[(stage, r.date)] = reason
    # Absorb + retire the short-lived sidecar (Aug 2026, never on GHA) —
    # before any column check, so a pre-column artifact still gets the memos
    # without a rebuild walk.
    if LEGACY_SKIPS_OUT.exists():
        df = pd.read_csv(LEGACY_SKIPS_OUT, dtype=str)
        for _, r in df.iterrows():
            skips.setdefault((r['stage'], r['date']), r['reason'])
        LEGACY_SKIPS_OUT.unlink()
        print(f'[elevation] absorbed {len(df)} legacy sidecar memo(s) into '
              f'the skip_reason column')
    return skips


def _materialize_skips(meas, skips, targets):
    """Write the memo dict back into the frame: rebuild skip_reason from
    scratch (stale annotations on kept rows die here), annotate existing rows
    for dem/fuse stages, and add a stub row (metrics all NaN) for each
    day-stage memo with no row."""
    meas['skip_reason'] = pd.Series(pd.NA, index=meas.index, dtype=object)
    by_date = {d: i for i, d in zip(meas.index, meas['date'])}
    stubs = []
    for (stage, d), reason in sorted(skips.items()):
        i = by_date.get(d)
        if i is not None:
            meas.at[i, 'skip_reason'] = f'{stage}:{reason}'
        elif stage == 'day':
            rt = targets.get(d, (None, None))[0]
            stubs.append({'date': d, 'run_type': rt,
                          'skip_reason': f'{stage}:{reason}'})
    if stubs:
        meas = pd.concat([meas, pd.DataFrame(stubs, columns=MEAS_COLS)],
                         ignore_index=True)
    return meas


def _elev_ids_by_date():
    """{iso_date: [labelId, ...]} of ELEV_SPORTS activities. None if no source.

    Prefers the per-activity index (watch_activities.csv) — no per-second parse.
    A watch-import profile has no index (it's written only by the Max-specific
    watch_daily), so fall back to scanning the detail cache directly, reading
    each record's sport to keep the ELEV_SPORTS filter — same pattern as
    daily_envelopes._ids_by_date."""
    if ACTIVITIES.exists():
        idx = pd.read_csv(ACTIVITIES, dtype={'labelId': str, 'date': str})
        out = {}
        for _, r in idx.iterrows():
            if int(r['sport_type']) in ELEV_SPORTS:
                out.setdefault(r['date'], []).append(r['labelId'])
        return out
    if not DETAILS.exists():
        return None
    out = {}
    for p in sorted(DETAILS.glob('*.json')):
        try:
            rec = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if (rec.get('summary') or {}).get('sportType') is None:
            continue
        a = Activity(rec)
        if a.sport_type not in ELEV_SPORTS:
            continue
        out.setdefault(a.local_date.isoformat(), []).append(p.stem)
    return out or None


def _targets(daily, races, types):
    """{date: (run_type, corr_miles)} — daily run types use effective miles,
    races use the official course distance (None -> fall back to watch)."""
    targets = {}
    for rt in RUN_TYPES_ELEV:
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


def _race_rec(recs, off_m):
    """The single rich activity whose GPS-track length is closest to the
    official race distance — the race itself (same endpoint rule as
    dem_elevation.race_activity: MAX cumulative distance, since the trailing
    samples reset the distance field to 0). Race rows measure THIS activity
    alone: the whole-day stitch interleaved warmup/cooldown miles into the
    race's splits and compressed the distance axis by corr/watch (~22% on a
    3-activity HM day)."""
    best, best_err = None, None
    for rec in recs:
        pts = DEM.track_points(rec)
        if len(pts) < 5:
            continue
        err = abs(max(p[0] for p in pts) - off_m)
        if best_err is None or err < best_err:
            best, best_err = rec, err
    return best


def augment_race_dem(meas, races, ids_by_date, sleep_s, cache, skips,
                     verbose=False):
    """Fill DEM gain/loss/net/mean for race rows from the GPS track (races-only;
    the watch's barometric net is per-race noise — see dem_elevation.py).
    Rows missing dem_gain_ft are computed, and rows whose stored point count
    fails the coverage guard (dem_n_pts far below official_dist / SAMPLE_M —
    the measurement covered the wrong segment, e.g. the pre-fix picker
    measuring a warmup) are cleared and re-measured, so a bad row heals on the
    next run instead of freezing. Days that legitimately yield no DEM are
    memoized in ``skips`` (stage 'dem') instead of retrying forever. Returns
    the count newly computed."""
    if 'run_type' not in meas.columns:
        return 0
    off_by_date = {r['date'].date().isoformat(): float(r['distance_m'])
                   for _, r in races.iterrows() if pd.notna(r.get('distance_m'))}
    dem_cols = ['dem_gain_ft', 'dem_loss_ft', 'dem_net_ft', 'dem_mean_elev_ft',
                'dem_n_pts', 'dem_g_gain_pct', 'dem_g_loss_pct']
    race_rows = meas[meas['run_type'] == 'race']
    need_idx = []
    for i, row in race_rows.iterrows():
        if pd.isna(row.get('dem_gain_ft')):
            if (('dem', row['date']) not in skips
                    and ('day', row['date']) not in skips):
                need_idx.append(i)
        elif not DEM.race_dem_covered(row.get('dem_n_pts'),
                                      off_by_date.get(row['date'])):
            for c in dem_cols:          # known-bad: clear so a failed
                meas.at[i, c] = float('nan')   # re-measure doesn't keep it
            need_idx.append(i)
    need = meas.loc[need_idx]
    if need.empty:
        return 0
    print(f"[elevation] DEM race: {len(need)} days need lookup "
          f"({DEM.point_count(cache):,} pts cached)", flush=True)
    n, processed, prev_size = 0, 0, DEM.point_count(cache)
    for i, row in need.iterrows():
        processed += 1
        if processed % FLUSH_EVERY_DAYS == 0:
            prev_size = _flush_cache(cache, prev_size, processed, len(need),
                                     'race')
        d = row['date']
        off_m = off_by_date.get(d)
        if off_m is None or d not in ids_by_date:
            skips[('dem', d)] = 'no-ids'
            continue
        recs = []
        for lid in ids_by_date[d]:
            p = DETAILS / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if not recs:
            skips[('dem', d)] = 'no-rich'
            continue
        res = DEM.measure_race_elevation(recs, off_m, cache, verbose=verbose)
        if res is None:
            skips[('dem', d)] = 'gated-or-none'
            continue
        skips.pop(('dem', d), None)
        for k, v in res.items():
            meas.at[i, k] = v
        n += 1
    return n


def augment_run_dem(meas, ids_by_date, run_type, cache, skips, verbose=False):
    """Fill DEM gain/loss/net/mean for ``run_type`` (long/recovery) rows from the
    day's pooled GPS track (see dem_elevation.measure_run_elevation). Training
    runs are loops, so the barometric net carries a phantom morning-drift descent
    that DEM removes — same fix as races, applied to the whole-day run rather than
    a single race activity. GPS-corrupt days (false fix / dead-zone) return no DEM
    via the track-quality gate and stay on barometric — and are memoized in
    ``skips`` (stage 'dem'), not retried forever. Idempotent: only rows missing
    dem_gain_ft with no failure memo are computed, so a re-run is a no-op.
    Returns the count newly computed."""
    if 'run_type' not in meas.columns:
        return 0
    need = meas[(meas['run_type'] == run_type) & meas['dem_gain_ft'].isna()
                & ~meas['date'].map(lambda d: ('dem', d) in skips
                                    or ('day', d) in skips)]
    if need.empty:
        return 0
    print(f"[elevation] DEM {run_type}: {len(need)} days need lookup "
          f"({DEM.point_count(cache):,} pts cached)", flush=True)
    n, processed, prev_size = 0, 0, DEM.point_count(cache)
    for i, row in need.iterrows():
        processed += 1
        if processed % FLUSH_EVERY_DAYS == 0:
            prev_size = _flush_cache(cache, prev_size, processed, len(need),
                                     run_type)
        d = row['date']
        if d not in ids_by_date:
            skips[('dem', d)] = 'no-ids'
            continue
        recs = []
        for lid in ids_by_date[d]:
            p = DETAILS / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if not recs:
            skips[('dem', d)] = 'no-rich'
            continue
        res = DEM.measure_run_elevation(recs, cache, verbose=verbose)
        if res is None:
            skips[('dem', d)] = 'gated-or-none'
            continue
        skips.pop(('dem', d), None)
        for k, v in res.items():
            meas.at[i, k] = v
        n += 1
    return n


def regate_dem(meas, ids_by_date):
    """Clear dem_* on non-race days whose pooled GPS track now fails the
    quality gates (dem_elevation.track_ok) — rows filled before the current
    gate logic existed. The day falls back to barometric in per_run_elevation.
    (Races re-gate through augment_race_dem's gated measurement.) Runs only
    under --full-regen (Aug 2026): a day's track never changes, so re-checking
    every dem-filled day on every run was a full-corpus JSON walk that could
    only matter after a gate-logic change — which is exactly when a full regen
    is warranted anyway. Returns the count cleared."""
    if 'dem_gain_ft' not in meas.columns or 'run_type' not in meas.columns:
        return 0
    cols = ['dem_gain_ft', 'dem_loss_ft', 'dem_net_ft', 'dem_mean_elev_ft',
            'dem_n_pts', 'dem_g_gain_pct', 'dem_g_loss_pct']
    tgt = meas[(meas['run_type'] != 'race') & meas['dem_gain_ft'].notna()]
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


def _geo_hills(d, res, recs, profiles, k):
    """Hill rows for one day: geography and the aligned DEM net that the veto
    post-pass adjudicates. Hills arrive on the corrected stitched axis; each is
    mapped back to its activity (raw axis) for lat/lon and the DEM check."""
    bounds = res['act_bounds_raw']
    tracks = [None] * len(recs)
    rows = []
    for (d0, d1, vert, grade, kind, _pit, _fl) in res['hills']:
        mid_raw = (d0 + d1) / 2.0 / k
        ai = next((i for i, b in enumerate(bounds)
                   if b and b[0] <= mid_raw < b[1]), None)
        lat = lon = np.nan
        dem_net = np.nan
        if ai is not None:
            if tracks[ai] is None:
                tp = DEM.track_points(recs[ai])
                if len(tp) >= 3:
                    td = np.array([q[0] for q in tp], float)
                    keep = np.concatenate(([True], np.diff(td) > 0))
                    tracks[ai] = (td[keep],
                                  np.array([q[1] for q in tp], float)[keep],
                                  np.array([q[2] for q in tp], float)[keep])
                else:
                    tracks[ai] = ()
            if tracks[ai]:
                td, tla, tlo = tracks[ai]
                loc = mid_raw - bounds[ai][0]
                lat = float(np.interp(loc, td, tla))
                lon = float(np.interp(loc, td, tlo))
            prof = profiles[ai]
            if prof is not None:
                a0 = d0 / k - bounds[ai][0]
                a1 = d1 / k - bounds[ai][0]
                dem_net = DEM.aligned_net(prof[0], prof[1], a0, a1, kind)
        rows.append({'date': d, 'act': ai if ai is not None else -1,
                     'd0': round(d0, 1), 'd1': round(d1, 1),
                     'vert_ft': round(vert, 1), 'grade_pct': round(grade, 2),
                     'kind': int(kind), 'lat': round(lat, 6),
                     'lon': round(lon, 6),
                     'dem_net_ft': (round(dem_net, 1)
                                    if np.isfinite(dem_net) else np.nan),
                     'vetoed': 0})
    return rows


# --- Persistence-aware hill veto ---------------------------------------------
# DEM refutes, never confirms: a hill is vetoed only when (a) DEM shows nearly
# flat ground under it after +/-80 m alignment (net < 25% of the claim AND
# under the 12 ft hill floor), and (b) the disagreement is a one-off — the same
# disagreement recurring at the same coordinates on other dates is structure
# bare-earth lidar cannot see (arch bridges, boardwalks, post-lidar
# construction), where baro is right. Demanding DEM *confirmation* instead was
# tested and rejected: it deleted real terrain wholesale (fit R2 fell
# monotonically with vertical removed and the coefficients went unphysical).
VETO_FRAC = 0.25
VETO_ABS_FT = 12.0
VETO_CELL_DEG = 0.0005        # ~55 m; clusters merge 8-neighbourhoods so GPS
                              # drift cannot fragment one structure into
                              # several "locations" (it did: one arch bridge
                              # read as seven)
VETO_MIN_DATES = 2            # recurrence on >=2 dates = structure


def apply_hill_veto(meas, splits, hills):
    """Flag one-off DEM-refuted hills, then rebuild the per-run and per-mile
    segment quantities from the surviving hills. Runs over the FULL hills
    table every build (it is small), so a structure's first visit is
    retroactively un-vetoed the day its second visit lands."""
    if hills.empty:
        return meas, splits, hills, 0
    h = hills.copy()
    cand = (h.dem_net_ft.notna() & (h.dem_net_ft < VETO_FRAC * h.vert_ft)
            & (h.dem_net_ft < VETO_ABS_FT) & h.lat.notna())
    hc = h[cand]
    cy = np.round(hc.lat / VETO_CELL_DEG).astype(int)
    cx = np.round(hc.lon / VETO_CELL_DEG).astype(int)
    cells = set(zip(cy, cx))
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for (y, x) in cells:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (y + dy, x + dx)
                if n in cells:
                    ra, rb = find((y, x)), find(n)
                    if ra != rb:
                        parent[ra] = rb
    cl = pd.Series([str(find(c)) for c in zip(cy, cx)], index=hc.index)
    n_dates = hc.groupby(cl)['date'].transform('nunique')
    h['vetoed'] = 0
    h.loc[n_dates.index[(n_dates < VETO_MIN_DATES).to_numpy()], 'vetoed'] = 1

    kept = h[h.vetoed == 0]
    mi_m = 1609.344
    # per-run segment sums
    run = {}
    for d, g in kept.groupby('date'):
        uv = (g.vert_ft.abs() * (g.kind > 0)).sum()
        ug = (g.vert_ft.abs() * g.grade_pct * (g.kind > 0)).sum()
        dv = (g.vert_ft.abs() * (g.kind < 0)).sum()
        dg = (g.vert_ft.abs() * g.grade_pct * (g.kind < 0)).sum()
        run[d] = (uv, ug / uv if uv else 0.0, dv, dg / dv if dv else 0.0)
    meas = meas.copy()
    vals = meas['date'].map(run)
    meas['seg_up_ft'] = [round(v[0], 1) if isinstance(v, tuple) else 0.0
                         for v in vals]
    meas['g_up_pct'] = [round(v[1], 2) if isinstance(v, tuple) else 0.0
                        for v in vals]
    meas['seg_dn_ft'] = [round(v[2], 1) if isinstance(v, tuple) else 0.0
                         for v in vals]
    meas['g_dn_pct'] = [round(v[3], 2) if isinstance(v, tuple) else 0.0
                        for v in vals]
    # per-mile segment sums (prorated overlap; overwrite what mile_splits
    # wrote pre-veto so fit and application see one statistic)
    by_day = {d: g for d, g in kept.groupby('date')}
    su, gu, sd, gd = [], [], [], []
    for d, mi in zip(splits['date'], splits['mile']):
        g = by_day.get(d)
        uv = ug = dv = dg = 0.0
        if g is not None:
            lo, hi = mi * mi_m, (mi + 1) * mi_m
            ov = (np.minimum(g.d1, hi) - np.maximum(g.d0, lo)).clip(lower=0)
            f = np.where(g.d1 > g.d0, ov / (g.d1 - g.d0), 0.0)
            v = g.vert_ft.abs() * f
            uv = float((v * (g.kind > 0)).sum())
            ug = float((v * g.grade_pct * (g.kind > 0)).sum())
            dv = float((v * (g.kind < 0)).sum())
            dg = float((v * g.grade_pct * (g.kind < 0)).sum())
        su.append(round(uv, 1))
        gu.append(round(ug / uv, 2) if uv else 0.0)
        sd.append(round(dv, 1))
        gd.append(round(dg / dv, 2) if dv else 0.0)
    splits = splits.copy()
    splits['seg_up_ft'], splits['g_up'] = su, gu
    splits['seg_dn_ft'], splits['g_down'] = sd, gd
    return meas, splits, h, int(h.vetoed.sum())



def main():
    load_env_file()
    p = argparse.ArgumentParser()
    p.add_argument('--run-type',
                   choices=list(RUN_TYPES_ELEV) + ['race', 'all'],
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
    # ONE cache for the whole process. It used to be loaded and re-saved inside
    # each augment_* call — five load/save cycles of a 6.4 MB JSON per build.
    cache = DEM._load_cache()
    migrated = DEM.migrate_cache(cache)
    if len(migrated) < len(cache):
        print(f"[elevation] DEM cache upgraded to {DEM.KEY_DECIMALS}-decimal "
              f"grid: {len(cache):,} -> {len(migrated):,} points", flush=True)
        cache = migrated
        DEM._save_cache(cache)

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
    types = (list(RUN_TYPES_ELEV) + ['race'] if args.run_type == 'all'
             else [args.run_type])
    targets = _targets(daily, races, types)

    # Day-level failure memo — persisted in the skip_reason column of
    # elevation_measured.csv (see the note at LEGACY_SKIPS_OUT). --full-regen
    # wipes it (retry everything); --fetch retries 'slim' days below.
    skips = {}

    # Existing rows to reuse for days already computed (presence by date).
    # SCHEMA GUARD: artifacts written by the pre-Aug-2026 engine lack the
    # fused/hill columns (and the hills artifact entirely); reusing them would
    # mark every day "done" and leave the new model's inputs empty corpus-wide
    # (the exact failure the first CI run after this lands would hit, since
    # the state cache restores the old files). Stale schema -> recompute all.
    done, meas_keep, split_keep, hill_keep = set(), [], [], []
    refuse = set()
    if MEAS_OUT.exists() and not args.full_regen:
        em = pd.read_csv(MEAS_OUT, dtype={'date': str})
        _need = {'fused', 'seg_up_ft', 'g_up_pct'}
        if not (_need <= set(em.columns)) or not HILLS_OUT.exists():
            print('[elevation] stale artifact schema (pre two-channel) — '
                  'recomputing all days')
        else:
            skips = _parse_skips(em)
            # --fetch retries 'slim' days (fetching can upgrade them): drop
            # their memos AND their stub rows so they reprocess.
            fetch_retry = set()
            if args.fetch:
                fetch_retry = {d for (s, d), r in skips.items()
                               if s == 'day' and r == 'slim'}
                skips = {k: v for k, v in skips.items()
                         if k[1] not in fetch_retry}
            # Corr-staleness heal: a kept day whose stored corrected distance
            # no longer matches the current (pinned) calibration is re-derived
            # in full. Empty on ordinary runs; fires once per calibration
            # adoption (long_runs' deadband) and replaces the old fit-run
            # --full-regen for elevation.
            stale = set()
            for r in em.itertuples():
                tgt = targets.get(r.date)
                if (tgt is not None and tgt[1] and pd.notna(r.corr_miles)
                        and abs(float(r.corr_miles) - float(tgt[1]))
                        > CORR_STALE_MI):
                    stale.add(r.date)
            if stale:
                print(f'[elevation] corrected-distance change on '
                      f'{len(stale)} day(s) (calibration adoption?) — '
                      f're-deriving them')
                skips = {k: v for k, v in skips.items() if k[1] not in stale}
            # Fusion heal: a pure-baro row (fused=0) on a day whose track HAS
            # a DEM measurement (dem_gain_ft present) predates its own cache
            # cells — the degraded outcome of a first build whose fetch-first
            # failed (DEM unreachable), or a pre-fetch-first row. Re-derive:
            # the profile is now cache-served. A day that STILL can't fuse
            # (genuinely thin/gappy coverage) is memoized ('fuse') rather
            # than retried forever; --full-regen clears the memo.
            refuse = {r.date for r in em.itertuples()
                      if r.fused == 0 and pd.notna(r.dem_gain_ft)
                      and ('fuse', r.date) not in skips} - stale
            if refuse:
                print(f'[elevation] fusion heal: {len(refuse)} pure-baro '
                      f'day(s) now have DEM point coverage — re-deriving '
                      f'their fused substrate')
            stale |= refuse | fetch_retry
            done = set(em['date']) - stale
            meas_keep = [r for r in em.to_dict('records')
                         if r['date'] not in stale]
            if SPLITS_OUT.exists():
                sp = pd.read_csv(SPLITS_OUT, dtype={'date': str})
                split_keep = [r for r in sp.to_dict('records')
                              if r['date'] not in stale]
            hp = pd.read_csv(HILLS_OUT, dtype={'date': str})
            hill_keep = [r for r in hp.to_dict('records')
                         if r['date'] not in stale]

    todo = [d for d in sorted(targets)
            if d in ids_by_date and d not in done and ('day', d) not in skips]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[elevation] targets={len(targets)} pending={len(todo)} "
          f"reused={len(done)} memoized-skips={len(skips)}")

    client = None
    meas_rows, split_rows, hill_rows = [], [], []
    dem_cache = cache
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
                recs.append(strip_paused(rec))
            elif args.fetch:
                if client is None:
                    client = CorosClient(os.environ['COROS_EMAIL'],
                                         os.environ['COROS_PASSWORD'],
                                         token_cache=TOKEN)
                try:
                    rich = rich_detail(client.activity_detail(
                        lid, (rec.get('summary') or {}).get('sportType')))
                    if rich is not None:
                        path.write_text(json.dumps(rich))   # cache stays raw
                        recs.append(strip_paused(rich))
                        fetched += 1
                    time.sleep(args.sleep)
                except Exception as e:
                    print(f"  {d} fetch fail {lid}: {e}")
        if not recs:
            skipped_slim += 1                 # cached slim, no --fetch
            skips[('day', d)] = 'slim'
            continue
        lg = logged_miles.get(d)
        if lg and watch_m and not (WATCH_VALID_BAND[0] <= watch_m / lg
                                   <= WATCH_VALID_BAND[1]):
            skipped_invalid += 1
            skips[('day', d)] = 'watch-invalid'
            continue
        # Race days measure the race activity alone (warmup/cooldown vertical
        # and miles stay out of the race's totals and splits); the validity
        # gate above still judges the whole day against the hand log.
        meas_recs, meas_watch_m = recs, watch_m
        if rt == 'race' and corr_mi:
            rrec = _race_rec(recs, corr_mi * 1609.344)
            if rrec is not None:
                meas_recs = [rrec]
                meas_watch_m = Activity(rrec).distance_m / 1609.344
        eff_corr = corr_mi if (corr_mi and corr_mi > 0) else meas_watch_m
        # Fetch-first (Aug 2026): seed the DEM point cache for this day's
        # tracks BEFORE building the fusion profiles, so a brand-new day
        # fuses on its own first build instead of landing pure-baro until
        # the augment stage fetches its cells (activity_dem_profile is
        # deliberately cache-only). Same total network, reordered — the
        # augment's later lookups become cache-served. Soft-fails like the
        # augment (unreachable DEM leaves the day barometric; the fusion
        # heal above converges it on a later build).
        for rec in meas_recs:
            if not DEM.track_ok(rec):
                continue
            pts = DEM.track_points(rec)
            if len(pts) >= 5:
                _, la, lo = DEM._resample(pts)
                DEM.dem_elevations(la, lo, cache)
        profiles = [DEM.activity_dem_profile(r, dem_cache)
                    for r in meas_recs]
        res = E.measure_day_elevation(meas_recs, eff_corr, meas_watch_m,
                                      dem_profiles=profiles)
        if res is None:
            skips[('day', d)] = 'no-measure'
            continue
        skips.pop(('day', d), None)
        meas_rows.append({'date': d, 'run_type': rt,
                          'watch_miles': round(meas_watch_m, 3),
                          'corr_miles': round(eff_corr, 3),
                          'elev_gain_ft': res['elev_gain_ft'],
                          'elev_loss_ft': res['elev_loss_ft'],
                          'minetti_factor': res['minetti_factor'],
                          'g_gain_pct': res['g_gain_pct'],
                          'g_loss_pct': res['g_loss_pct'],
                          'fused': int(any(pr is not None for pr in profiles)),
                          'n_alt_pts': res['n_alt_pts']})
        for s in res['splits']:
            split_rows.append({'date': d, **s})
        kk = (eff_corr / meas_watch_m) if meas_watch_m else 1.0
        hill_rows.extend(_geo_hills(d, res, meas_recs, profiles, kk))
        computed += 1
        # Same resume story as the augment: a killed cold-seed run keeps the
        # points its fetch-first calls already pulled.
        if computed % FLUSH_EVERY_DAYS == 0:
            DEM._save_cache(cache)

    meas = pd.DataFrame(meas_keep + meas_rows, columns=MEAS_COLS)
    splits = pd.DataFrame(split_keep + split_rows, columns=SPLIT_COLS)
    hills = pd.DataFrame(hill_keep + hill_rows, columns=HILL_COLS)
    meas, splits, hills, n_veto = apply_hill_veto(meas, splits, hills)
    print(f"[elevation] hill veto: {n_veto} one-off DEM-refuted hills vetoed "
          f"of {len(hills)} ({int((meas['fused'] == 1).sum())} fused days)")
    if 'race' in types and len(meas):
        n_dem = augment_race_dem(meas, races, ids_by_date, args.sleep, cache,
                                 skips, verbose=args.dem_verbose)
        print(f"[elevation] DEM race-elevation: {n_dem} newly computed "
              f"(GPS-track lookup; barometric net is per-race noise)")
    for rt in RUN_TYPES_ELEV:
        if rt in types and len(meas):
            n_dem = augment_run_dem(meas, ids_by_date, rt, cache, skips,
                                    verbose=args.dem_verbose)
            print(f"[elevation] DEM {rt}-run elevation: {n_dem} newly computed "
                  f"(GPS-track lookup; barometric net is morning-drift phantom)")
    DEM._save_cache(cache)
    # Fusion-heal outcome: a re-derived day that still couldn't fuse has
    # genuinely thin cache coverage — memoize so it isn't re-walked forever
    # (--full-regen clears; a later successful fuse pops it).
    if refuse and len(meas):
        fused_by_date = meas.drop_duplicates('date').set_index('date')['fused']
        for d in refuse:
            v = fused_by_date.get(d)
            if v is not None and pd.notna(v) and int(v) == 0:
                skips[('fuse', d)] = 'profile-thin'
            else:
                skips.pop(('fuse', d), None)
    if args.full_regen and len(meas):
        n_clr = regate_dem(meas, ids_by_date)
        if n_clr:
            print(f"[elevation] DEM re-gate: cleared {n_clr} GPS-corrupt "
                  f"long/recovery days (fall back to barometric)")
    if len(meas):
        meas = _materialize_skips(meas, skips, targets)
        meas = meas.sort_values('date').reset_index(drop=True)
        meas.to_csv(MEAS_OUT, index=False)
    if len(splits):
        splits = splits.sort_values(['date', 'mile']).reset_index(drop=True)
        splits.to_csv(SPLITS_OUT, index=False)
    if len(hills):
        hills = hills.sort_values(['date', 'd0']).reset_index(drop=True)
        hills.to_csv(HILLS_OUT, index=False)
    print(f"[elevation] computed={computed} reused={len(done)} fetched={fetched} "
          f"slim-skipped={skipped_slim} watch-invalid={skipped_invalid} "
          f"-> {len(meas)} days")


if __name__ == '__main__':
    main()
