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
so long AND recovery runs prefer DEM gain/loss in ``per_run_elevation``. Recovery
adds genuine point-to-point net the barometer's drift masked (e.g. a real −315 ft
descent logged as a loop). A two-gate track-quality filter (``track_ok``: pace +
coverage) drops the ~2% of runs with a false-fix or GPS dead-zone back to
barometric, so a corrupt horizontal track never feeds the elevation.
"""
import json
import math
import time
import urllib.parse
import urllib.request

import numpy as np

from src.coros.elevation import gain_loss_ft, _gridded_altitude
from src.shared.hill_model import FT_PER_M
from src.shared.paths import REPO_ROOT

SAMPLE_M = 30.0          # resample the GPS track to this spacing before lookup
BATCH = 100              # OpenTopoData locations per request
SLEEP_S = 1.0            # public-API courtesy rate limit (1 req/s)
# Profile-INDEPENDENT shared cache: a DEM elevation is a pure function of (lat,
# lon) — identical no matter which profile ran there — so every profile shares
# one cache at the repo-root data dir instead of re-seeding the same routes per
# profile. Anchored to REPO_ROOT, NOT the RP_DATA_DIR-routed DATA_DIR. (Matches
# reverse_coords.csv, also shared at root; already in the CI cache as this path.)
CACHE_PATH = REPO_ROOT / 'data' / 'dem_cache.json'
_NED = 'https://api.opentopodata.org/v1/ned10m'
_SRTM = 'https://api.opentopodata.org/v1/srtm30m'

# Cache/query grid. 4 decimals ≈ 10 m, matching the DEM's own 10 m (NED) / 30 m
# (SRTM) resolution, so snapping the lookup to the grid is lossless (it's below
# what the data resolves, and gain/loss is computed from a profile smoothed over
# SMOOTH_M=120 m anyway). It also makes REPEATED routes reuse cached points: a
# finer ~1 m key (5 decimals) never collided across runs — GPS jitter (~3-10 m)
# plus the shifting 30 m resample grid put every pass in a fresh cell, so the
# cache stored ~4x near-duplicates and re-fetched whole tracks every time. The
# query point is the cell itself (parsed back from the key), so the cached value
# and the queried point are always consistent. See migrate_cache for the one-
# time, network-free collapse of a legacy finer cache onto this grid.
KEY_DECIMALS = 4


def _key(la, lo):
    """On-grid cache key for a coordinate."""
    return f'{la:.{KEY_DECIMALS}f},{lo:.{KEY_DECIMALS}f}'


def _sc(v):
    """Coros lat/lon are integer micro-degrees (×1e7); pass through if already
    decimal."""
    return v / 1e7 if abs(v) > 1000 else float(v)


# GPS track-quality gates (June 2026, validated on Max's recovery corpus). DEM
# resamples elevation along the horizontal GPS track, so a corrupt track puts
# the lookup in the wrong place — the barometer (immune to horizontal error) is
# the better source there, so a flagged run returns no DEM and falls back to it.
# Two physical failure modes:
#   * False fix: the watch holds a wrong position then jumps to truth, so the
#     GPS-derived speed briefly exceeds anything humanly runnable. Flag if the
#     fastest ~60 s window of GPS-haversine speed tops PACE_CEIL_MPS (14 m/s ≈
#     1:55/mi for a full minute — deep in teleport territory). Set here, not at
#     a tighter "max human pace": a brief 8-12 m/s blip (a stride + jitter, or a
#     mid-run glitch that doesn't move the endpoints) leaves the net intact, so
#     it isn't worth dropping a run for. Across Max's 1556 recovery runs the
#     fastest-60s tops out at 10.2 m/s at the 99.9th pct, then a clean gap to the
#     ONE genuinely net-corrupted run at 19.6 m/s (net −308 ft on a loop); 14
#     sits in that gap, catching it while keeping every benign blip.
#   * Dead-zone: GPS never locks for a stretch and the watch dead-reckons
#     distance with no track, so the GPS-haversine length falls well short of
#     the watch's reported distance. Flag if that ratio drops below
#     COVERAGE_FLOOR. (Clean runs sit ~1.0-1.05 — jitter nudges haversine
#     slightly OVER the watch; severe dead-zones land 0.2-0.6.)
PACE_CEIL_MPS = 14.0
COVERAGE_FLOOR = 0.80


def _hav(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def track_quality(rec):
    """(fastest_60s_mps, coverage) for one activity, or None when there's too
    little GPS to judge. ``fastest_60s_mps`` is the max GPS-haversine speed over
    any ~60 s window (false-fix detector); ``coverage`` is the GPS-haversine
    total over the watch's reported distance (dead-zone detector)."""
    pts = [(f[0] / 100.0, _sc(f[3]), _sc(f[4])) for f in rec.get('freq') or []
           if len(f) >= 6 and f[0] is not None and f[3] and f[4]]
    if len(pts) < 10:
        return None
    t = [p[0] for p in pts]
    cum, c = [0.0], 0.0
    for i in range(len(pts) - 1):
        c += _hav(pts[i][1], pts[i][2], pts[i + 1][1], pts[i + 1][2])
        cum.append(c)
    best, j = 0.0, 0
    for i in range(len(t)):
        if j < i:
            j = i
        while j < len(t) - 1 and t[j] - t[i] < 60:
            j += 1
        if t[j] - t[i] >= 50:
            best = max(best, (cum[j] - cum[i]) / (t[j] - t[i]))
    watch_m = (rec.get('summary') or {}).get('distance')
    coverage = cum[-1] / (watch_m / 100.0) if watch_m else float('nan')
    return best, coverage


def track_ok(rec):
    """Whether one activity's GPS track passes both quality gates. A track with
    too little GPS to judge passes (downstream produces no DEM anyway)."""
    q = track_quality(rec)
    if q is None:
        return True
    best, cov = q
    if best > PACE_CEIL_MPS:
        return False
    if cov == cov and cov < COVERAGE_FLOOR:   # cov==cov: not NaN
        return False
    return True


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


def migrate_cache(cache):
    """Collapse a point cache onto the current ``_key`` grid, averaging any
    points that fall in the same cell. Network-free — it only re-keys points
    already fetched, so upgrading a finer legacy cache (5-decimal, ~1 m) keeps
    every fetched point while ~4x-deduplicating near-identical neighbours.
    Idempotent: a cache already on-grid round-trips to an equivalent dict."""
    agg = {}
    for k, v in cache.items():
        la, lo = k.split(',')
        nk = _key(float(la), float(lo))
        s, c = agg.get(nk, (0.0, 0))
        agg[nk] = (s + v, c + 1)
    return {k: s / c for k, (s, c) in agg.items()}


def _save_cache(cache):
    # Atomic write: a mid-loop flush can be killed by a CI step timeout while
    # the file is being written. Serialize to a temp file then os-rename so the
    # reader (and the always-on CI cache save) never sees a truncated JSON that
    # _load_cache would discard as corrupt — the seed would then restart cold.
    tmp = CACHE_PATH.with_name(CACHE_PATH.name + '.tmp')
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_PATH)


def _query(latlons, dataset):
    """Batched DEM lookup; returns list of elevations (m), None for misses."""
    locs = '|'.join(f'{la:.6f},{lo:.6f}' for la, lo in latlons)
    url = f'{dataset}?{urllib.parse.urlencode({"locations": locs})}'
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return [r.get('elevation') for r in data.get('results', [])]


def dem_elevations(lat, lon, cache, sleep_s=SLEEP_S, verbose=False):
    """Elevation (m) at each (lat, lon) via NED10m (CONUS) with SRTM30m fallback
    for misses. Cached per on-grid cell (``_key``); the query point is the cell
    itself, so the cached value and the queried point are consistent and
    repeated routes reuse cached cells. Mutates ``cache``. Returns one elevation
    (or None) per input point."""
    keys = [_key(lat[i], lon[i]) for i in range(len(lat))]
    # Unique not-yet-cached cells: a 30 m-resampled track can put several points
    # in one ~10 m cell, and repeated routes share cells — query each only once.
    pend0 = list(dict.fromkeys(k for k in keys if k not in cache))
    for ds in (_NED, _SRTM):
        pend = [k for k in pend0 if k not in cache]
        if not pend:
            break
        for b in range(0, len(pend), BATCH):
            chunk = pend[b:b + BATCH]
            latlons = [tuple(map(float, k.split(','))) for k in chunk]
            try:
                elevs = _query(latlons, ds)
            except Exception as e:  # network/HTTP — leave as misses
                if verbose:
                    print(f'  DEM query failed ({ds.split("/")[-1]}): {e}')
                elevs = [None] * len(chunk)
            for k, ev in zip(chunk, elevs):
                if ev is not None:
                    cache[k] = ev
            if verbose:
                done = min(b + BATCH, len(pend))
                print(f"      [dem] {ds.split('/')[-1]} {done}/{len(pend)} "
                      f"cells ({len(cache):,} cached)", flush=True)
            time.sleep(sleep_s)
    return [cache.get(k) for k in keys]


def measure_run_elevation(recs, cache, verbose=False):
    """DEM gain/loss/net/mean(ft) for a long/recovery run, pooled over the day's
    rich run activities (NO race-distance activity-picking — a training run has no
    warmup/cooldown to exclude). Same gridding+smoothing as the race path, so the
    two are directly comparable; only the activity selection differs. Net is the
    sum of per-activity smoothed end−start drops — on a loop the DEM net is ~0
    (the closed-loop ground truth the drifting barometer gets wrong). Returns a
    dict or None if no activity has a usable GPS track / the DEM is unreachable.

    ``recs`` should already be filtered to outdoor-run rich>=2 records by the
    caller (backfill_elevation's ELEV_SPORTS gate). A day with ANY GPS-corrupt
    activity (failing track_ok) yields no DEM — one bad track poisons the day's
    pooled net, so the whole day falls back to barometric."""
    if any(not track_ok(rec) for rec in recs):
        return None
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
    mean_elev_ft anchors the altitude (hypoxia) term. GPS-corrupt activities are
    dropped before the race-activity pick, so a false-fix / dead-zone track never
    feeds the race elevation (it falls back to barometric)."""
    recs = [rec for rec in recs if track_ok(rec)]
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
