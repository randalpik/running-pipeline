"""DEM-based race elevation from the GPS track (June 2026, watch-stream §B).

The watch's *barometric* altitude is per-race noise: the same Bolder Boulder
course read −19 ft/mi net in 2024 and +5 ft/mi in 2025 (Max). But the watch's
*horizontal* GPS track is reliable and reproducible (start/end agree within
metres year-over-year, Boston resolves Hopkinton→Boston). So for races — where
each course must look right qualitatively, not just average out over thousands
of points — we throw away the barometric vertical and resample elevation from a
DEM along the trusted GPS path.

Source: OpenTopoData public API — USGS NED 10 m for CONUS, SRTM 30 m fallback
elsewhere (Berlin). Free, no key, deterministic. Point elevations are cached in
``data/dem_cache.json`` keyed by rounded lat/lon so repeated courses (Boston
×6, Bolder Boulder, Run for the Pies …) and re-runs cost no network.

Gain/loss/net are computed with the SAME gridding+smoothing as the barometric
path (``elevation._gridded_altitude``, ``gain_loss_ft``) so the two sources are
directly comparable; only the altitude source differs.

LONG RUNS use DEM too (June 2026, ``measure_run_elevation``): they are loops, so
true net ≈ 0, but the barometer drifts through a morning run and registers a
systematic phantom descent (Max's watch-era long runs: barometric median net
−17 ft, mean |net| 31 ft, 97/140 net-negative; DEM cuts those to −3 ft / 5 ft —
verified June 2026). That tilt is exactly what DEM-along-GPS removed for races,
so long runs prefer DEM gain/loss in ``per_run_elevation``. Recovery runs stay
on barometric for now (smaller per-run net, large corpus the betas lean on).
"""
import json
import time
import urllib.parse
import urllib.request

import numpy as np

from src.coros.elevation import gain_loss_ft, _gridded_altitude
from src.shared.hill_model import FT_PER_M
from src.shared.paths import DATA_DIR

SAMPLE_M = 30.0          # resample the GPS track to this spacing before lookup
BATCH = 100              # OpenTopoData locations per request
SLEEP_S = 1.0            # public-API courtesy rate limit (1 req/s)
CACHE_PATH = DATA_DIR / 'dem_cache.json'
_NED = 'https://api.opentopodata.org/v1/ned10m'
_SRTM = 'https://api.opentopodata.org/v1/srtm30m'


def _sc(v):
    """Coros lat/lon are integer micro-degrees (×1e7); pass through if already
    decimal."""
    return v / 1e7 if abs(v) > 1000 else float(v)


def track_points(rec):
    """Monotonic (cumulative_dist_m, lat, lon) from one rich>=2 activity, using
    the GPS-derived horizontal distance (f[1]) and lat/lon (f[3]/f[4])."""
    out = []
    for f in rec.get('freq') or []:
        if len(f) < 6 or f[1] is None or not f[3] or not f[4]:
            continue
        out.append((f[1] / 100.0, _sc(f[3]), _sc(f[4])))
    return out


def race_activity(recs, off_m):
    """The single activity whose distance is closest to the official race
    distance — the race itself, excluding any warmup/cooldown activities (their
    vertical would inflate the race's gain/loss). ``recs`` are rich records."""
    best, best_err = None, None
    for rec in recs:
        pts = track_points(rec)
        if len(pts) < 5:
            continue
        err = abs(pts[-1][0] - off_m)
        if best_err is None or err < best_err:
            best, best_err = pts, err
    return best


def _resample(pts):
    """Resample (dist, lat, lon) to an even SAMPLE_M spacing along distance."""
    d = np.array([p[0] for p in pts], float)
    keep = np.concatenate(([True], np.diff(d) > 0))
    d = d[keep]
    lat = np.array([p[1] for p in pts], float)[keep]
    lon = np.array([p[2] for p in pts], float)[keep]
    if len(d) < 3 or d[-1] - d[0] < SAMPLE_M:
        return d, lat, lon
    grid = np.arange(d[0], d[-1], SAMPLE_M)
    return grid, np.interp(grid, d, lat), np.interp(grid, d, lon)


def _load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache))


def _query(latlons, dataset):
    """Batched DEM lookup; returns list of elevations (m), None for misses."""
    locs = '|'.join(f'{la:.6f},{lo:.6f}' for la, lo in latlons)
    url = f'{dataset}?{urllib.parse.urlencode({"locations": locs})}'
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return [r.get('elevation') for r in data.get('results', [])]


def dem_elevations(lat, lon, cache, sleep_s=SLEEP_S, verbose=False):
    """Elevation (m) at each (lat, lon) via NED10m (CONUS) with SRTM30m
    fallback for misses. Cached per rounded point. Mutates ``cache``."""
    n = len(lat)
    out = [None] * n
    todo = []
    for i in range(n):
        key = f'{lat[i]:.5f},{lon[i]:.5f}'
        if key in cache:
            out[i] = cache[key]
        else:
            todo.append(i)
    for ds in (_NED, _SRTM):
        pend = [i for i in todo if out[i] is None]
        if not pend:
            break
        for b in range(0, len(pend), BATCH):
            idx = pend[b:b + BATCH]
            try:
                elevs = _query([(lat[i], lon[i]) for i in idx], ds)
            except Exception as e:  # network/HTTP — leave as misses
                if verbose:
                    print(f'  DEM query failed ({ds.split("/")[-1]}): {e}')
                elevs = [None] * len(idx)
            for i, ev in zip(idx, elevs):
                if ev is not None:
                    out[i] = ev
                    cache[f'{lat[i]:.5f},{lon[i]:.5f}'] = ev
            time.sleep(sleep_s)
    return out


def measure_run_elevation(recs, cache, verbose=False):
    """DEM gain/loss/net/mean(ft) for a long/recovery run, pooled over the day's
    rich run activities (NO race-distance activity-picking — a training run has no
    warmup/cooldown to exclude). Same gridding+smoothing as the race path, so the
    two are directly comparable; only the activity selection differs. Net is the
    sum of per-activity smoothed end−start drops — on a loop the DEM net is ~0
    (the closed-loop ground truth the drifting barometer gets wrong). Returns a
    dict or None if no activity has a usable GPS track / the DEM is unreachable.

    ``recs`` should already be filtered to outdoor-run rich>=2 records by the
    caller (backfill_elevation's ELEV_SPORTS gate)."""
    g_tot = l_tot = net_tot = 0.0
    alts, npts = [], 0
    for rec in recs:
        pts = track_points(rec)
        if len(pts) < 5:
            continue
        d, lat, lon = _resample(pts)
        if len(d) < 3:
            continue
        elevs = dem_elevations(lat, lon, cache, verbose=verbose)
        valid = np.array([e is not None for e in elevs])
        if valid.sum() < 3:
            continue
        d = d[valid]
        alt = np.array([e for e in elevs if e is not None], float)
        g, l = gain_loss_ft(d, alt)
        _, galt = _gridded_altitude(d, alt)
        g_tot += g
        l_tot += l
        net_tot += float((galt[-1] - galt[0]) / FT_PER_M)
        alts.append(galt / FT_PER_M)
        npts += int(valid.sum())
    if not alts:
        return None
    allalt = np.concatenate(alts)
    return {'dem_gain_ft': round(g_tot, 1), 'dem_loss_ft': round(l_tot, 1),
            'dem_net_ft': round(net_tot, 1),
            'dem_mean_elev_ft': round(float(allalt.mean()), 1),
            'dem_n_pts': int(npts)}


def measure_race_elevation(recs, off_m, cache, verbose=False):
    """DEM gain/loss/net/mean for one race from its GPS track. ``off_m`` is the
    official race distance (m); ``cache`` is the shared point-elevation cache
    (mutated). Returns a dict or None if the track is unusable / DEM unreachable.

    gain/loss use the same smoothed gridding as the barometric path; net is the
    smoothed end−start drop (the de-drifted quantity the watch gets wrong);
    mean_elev_ft anchors the altitude (hypoxia) term."""
    pts = race_activity(recs, off_m)
    if pts is None:
        return None
    d, lat, lon = _resample(pts)
    if len(d) < 3:
        return None
    elevs = dem_elevations(lat, lon, cache, verbose=verbose)
    valid = np.array([e is not None for e in elevs])
    if valid.sum() < 3:
        return None
    d, alt = d[valid], np.array([e for e in elevs if e is not None], float)
    g, l = gain_loss_ft(d, alt)
    grid, galt = _gridded_altitude(d, alt)
    net_ft = float((galt[-1] - galt[0]) / FT_PER_M)
    mean_ft = float(galt.mean() / FT_PER_M)
    return {'dem_gain_ft': round(g, 1), 'dem_loss_ft': round(l, 1),
            'dem_net_ft': round(net_ft, 1), 'dem_mean_elev_ft': round(mean_ft, 1),
            'dem_n_pts': int(valid.sum())}
