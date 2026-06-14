"""Per-run elevation / grade enrichment from the rich detail stream (rich>=2).

Produces, per watch day:
  - smoothed total elevation gain / loss (ft),
  - a Minetti per-run grade-cost factor (multiplicative energy-vs-flat ratio,
    used downstream as a t_eff correction), and
  - per-corrected-mile splits (pace_s, gain_ft, loss_ft) for display and a
    future pace-vs-grade physiology model.

Design notes
------------
* Barometric altitude is noisy, so it's resampled onto an even DISTANCE grid
  (robust to pace) and smoothed before differencing. ``GRID_M`` / ``SMOOTH_M``
  are the knobs; pick them by matching the device's own ``summary.elevGain``
  and surveyed hill-loop climb (see scripts/backfill_elevation.py --validate).
* Unlike the distance calibration (paved-gated — GPS corner-cutting under
  tree cover is a route property), barometric altitude is trustworthy on
  trail too, so elevation enrichment is NOT paved-gated.
* The distance axis is rescaled by ``corr_miles / watch_miles`` so split
  boundaries land on watch-corrected miles. Split pace uses moving time
  (stream-gap / pause seconds removed), never the raw ``speed`` field.
* Minetti grade is clipped to the model's fitted domain (|i| <= 0.45) so the
  quintic never extrapolates on a noisy spike.
"""
import numpy as np
import pandas as pd

from src.shared.hill_model import minetti_cost, FT_PER_M

GRID_M = 10.0      # even-distance resample step (m)
SMOOTH_M = 120.0   # altitude smoothing window (m of distance). Validated
                   # 2026-06-13: at 120 m, computed gain matches the device's
                   # summary.elevGain within ~1-3% on hilly Boulder runs
                   # (east boulder, boulder turnpike); flat runs sit a little
                   # high (barometric noise dominates when real gain is small,
                   # where the Minetti effect is negligible anyway). Smaller
                   # windows over-count gain badly.
GRADE_CLIP = 0.45  # Minetti domain bound on |grade|
STREAM_GAP_S = 10.0  # gap beyond this between samples = a pause (drop its time)


def alt_points(rec):
    """(t_s, dist_m, alt_m) from a rich>=2 record. Keeps the leading 0-distance
    sample; drops points missing altitude or distance."""
    out = []
    for f in rec.get('freq') or []:
        if len(f) < 6 or f[1] is None or f[5] is None:
            continue
        out.append((f[0] / 100.0, f[1] / 100.0, float(f[5])))
    return out


def _gridded_altitude(dist, alt):
    """Resample altitude onto an even GRID_M distance grid and smooth it.
    Returns (grid_dist, grid_alt_smoothed) or (None, None) if too short."""
    dist = np.asarray(dist, float)
    alt = np.asarray(alt, float)
    # Enforce strictly increasing distance (GPS/clock jitter can stall it).
    keep = np.concatenate(([True], np.diff(dist) > 0))
    dist, alt = dist[keep], alt[keep]
    if len(dist) < 3 or dist[-1] - dist[0] < GRID_M:
        return None, None
    grid = np.arange(dist[0], dist[-1], GRID_M)
    galt = np.interp(grid, dist, alt)
    win = max(1, int(round(SMOOTH_M / GRID_M)))
    if win > 1 and len(galt) > 1:
        # Centered rolling mean; min_periods shrinks the window at the ends so
        # no index gymnastics and no edge crash on short/sparse segments.
        galt = (pd.Series(galt).rolling(win, center=True, min_periods=1)
                .mean().to_numpy())
    return grid, galt


def gain_loss_ft(dist, alt):
    """(gain_ft, loss_ft) from the smoothed gridded profile."""
    _, galt = _gridded_altitude(dist, alt)
    if galt is None:
        return 0.0, 0.0
    d = np.diff(galt)
    return float(d[d > 0].sum() / FT_PER_M), float(-d[d < 0].sum() / FT_PER_M)


def minetti_factor(dist, alt):
    """Distance-weighted Minetti energy cost over the smoothed profile,
    divided by flat cost — a multiplicative grade-cost factor (>=1 ~ flat).
    Each GRID_M step is equal-distance, so this is just mean(cost)/cost(0)."""
    grid, galt = _gridded_altitude(dist, alt)
    if galt is None or len(galt) < 2:
        return 1.0
    grade = np.clip(np.diff(galt) / GRID_M, -GRADE_CLIP, GRADE_CLIP)
    return float(minetti_cost(grade).mean() / minetti_cost(0.0))


def _moving_time(t):
    """Elapsed seconds minus stream gaps (pauses) over a (sorted) time array."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    return float(dt[dt <= STREAM_GAP_S].sum())


def mile_splits(pts, k):
    """Per-corrected-mile splits from (t_s, dist_m, alt_m) points, with the
    distance axis scaled by ``k = corr_miles / watch_miles``. Returns a list of
    dicts {mile, pace_s, gain_ft, loss_ft} (pace = moving s/mi)."""
    if len(pts) < 3:
        return []
    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts]) * k / 1609.344  # corrected miles
    a = np.array([p[2] for p in pts])
    out = []
    last = int(np.floor(d[-1]))
    for mi in range(last + 1):
        m = (d >= mi) & (d < mi + 1)
        if m.sum() < 2:
            continue
        seg_d = d[m]
        covered = seg_d[-1] - seg_d[0]
        if covered <= 0:
            continue
        mov = _moving_time(t[m])
        g, l = gain_loss_ft(seg_d * 1609.344, a[m])
        out.append({'mile': mi,
                    'pace_s': round(mov / covered, 1),
                    'gain_ft': round(g, 1),
                    'loss_ft': round(l, 1)})
    return out


def _stitch(recs):
    """Concatenate multiple activities into ONE monotonic (t, d, a) profile.

    Each activity's distance stream restarts at 0, so a naive concatenation
    of a warmup+main+cooldown day yields non-monotonic distance and a
    scrambled altitude profile (np.interp needs sorted x). Offset each
    activity's distance by the cumulative prior distance, and rebase its time
    after a >STREAM_GAP_S marker so split moving-time skips the rest between
    activities."""
    out = []
    d_off = t_off = 0.0
    for rec in recs:
        pts = alt_points(rec)
        if not pts:
            continue
        t0 = pts[0][0]
        for (t, d, a) in pts:
            out.append((t - t0 + t_off, d + d_off, a))
        d_off += pts[-1][1]
        t_off = out[-1][0] + STREAM_GAP_S + 1.0
    return out


def measure_day_elevation(recs, corr_miles, watch_miles):
    """Aggregate elevation metrics for one day's run activities (``recs`` is a
    list of rich records, time-ordered). ``corr_miles`` is the watch-corrected
    distance (for the split-axis rescale and per-mile pace); ``watch_miles`` is
    the raw measured distance. Returns a dict or None if no altitude stream."""
    all_pts = _stitch(recs)
    if len(all_pts) < 3:
        return None
    # The stream's distance axis is the watch's (it under-reads), so scale it
    # to the corrected distance before computing grade — grade = dAlt/dDist
    # depends on the horizontal axis. Vertical gain/loss is axis-invariant but
    # the rescale keeps everything on one (corrected) axis.
    k = (corr_miles / watch_miles) if (watch_miles and watch_miles > 0) else 1.0
    pts = [(t, d * k, a) for (t, d, a) in all_pts]
    dist = np.array([p[1] for p in pts], float)
    alt = np.array([p[2] for p in pts], float)
    g, l = gain_loss_ft(dist, alt)
    mf = minetti_factor(dist, alt)
    return {
        'elev_gain_ft': round(g, 1),
        'elev_loss_ft': round(l, 1),
        'minetti_factor': round(mf, 5),
        'splits': mile_splits(pts, 1.0),   # pts already on the corrected axis
        'n_alt_pts': len(all_pts),
    }
