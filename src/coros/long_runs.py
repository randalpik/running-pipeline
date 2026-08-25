"""Watch-side run measurement + log/watch distance calibration.

Plays reps.py's role for long-run AND recovery days: reconciles each
hand-logged run against the day's watch activities and writes three
artifacts — the long-run pair consumed by
``shared.workouts.project_long_runs``, plus the recovery one consumed by
``shared.recovery_model.add_watch_corrections``:

``long_run_measured.csv`` / ``recovery_measured.csv`` — one row per
long-run (resp. recovery) day with watch activities:
    date, n_acts, status (rich|slim), complete (time gate, below),
    gps_ok (space gate — see GPS_LAG_CEIL_S), gps_lag_s, gps_coverage,
    watch_miles, watch_moving_s, watch_total_s, pause_s, stall_s,
    n_segs, d_eff_frac, longest_seg_mi

The two gates are independent and both must pass before a day's watch
distance is used: ``complete`` asks whether the watch recorded the whole run
in TIME, ``gps_ok`` whether it recorded it in SPACE.

``long_run_calibration.csv`` — the profile's log-vs-watch distance curve:
    intercept_mi, slope, sigma_mi, n_fit, n_pruned

The calibration models the LOGGED excess over watch distance as
``excess_mi = intercept + slope * watch_mi`` — a fixed per-run slack
(GPS settle at endpoints, one turnaround per run) plus a proportional
underread (corner-cutting). Fit June 2026 on Max's paved outdoor corpus:
``0.27 + 0.035·mi`` (ratio 1.052 at 16 mi), recovery and long runs on one
curve. The fit pools paved recovery + long days and relies on an iterative
3σ MAD prune to eject mislogged days (the pre-Apr-2022 Nashville staples
deviate 4–8σ and converge out without a manual list — verified June 2026:
the prune recovers the same coefficients as a hand-cleaned fit). Trail /
mixed terrain is excluded: GPS corner-cutting under tree cover is a route
property, not a logging-behaviour signal. ``sigma_mi`` is the kept LONG-RUN
residual sd (long runs scatter wider than recovery; falls back to the
overall kept sd when the profile has no long runs) — diagnostic only, no
pipeline consumer.

Pause/rest handling mirrors the workout enrichment scheme: watch pauses
(button presses) come from the rich record's pause list, standing time
while recording is detected from the per-second stream, and the day's
connected-fatigue effective distance uses the same ``exp(-rest/RECON_TAU_S)``
dissipation as ``parse_workouts._connected_core`` — but WITHOUT the
anaerobic short-rep correction (long-run segments are aerobic; the
correction targets rep-pace efforts). Slim records (no per-second stream)
degrade to one segment per activity with the activity's pause total as the
rest after it.

Usage (Max's watch cache + hand log):
  python src/coros/long_runs.py --details-dir data/profiles/coros/details
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.coros import dem_elevation as DEM
from src.shared.paths import DATA_DIR
from src.shared.recovery_model import STRIDE_SUFFIX_RX
from src.shared.workouts import RECON_TAU_S
from src.shared.units import METERS_PER_MILE as MILE_M

STALL_SPEED_MPS = 0.5   # below this = standing (walking is ~1.2 m/s)
STALL_MIN_S = 15.0      # minimum standstill to count (GPS jitter floor)
STREAM_GAP_S = 10.0     # stream gap beyond this = watch pause, not a sample

# Logged-vs-watch time completeness gate: a day whose watch moving time
# disagrees with the logged duration by more than this fraction has an
# unrecorded segment or a dead watch — the watch can't vouch for the run.
TIME_COMPLETE_FRAC = 0.10

# GPS track-quality gate for DISTANCE use (Aug 2026). The time gate above and
# the recovery model's route-deviation guard both judge a day the watch
# RECORDED; neither can see a day the watch recorded incompletely in SPACE.
# Two failure modes, both taken from the elevation layer rather than re-derived:
#   * late first fix (dem_elevation.LAG_CEIL_S) — the watch dead-reckons the
#     un-fixed opening stretch, so its distance is short by pace x lag while
#     moving time stays whole. This is the dominant one: 108 of Max's 1300
#     watch-corrected recovery days, incl. 2026-08-24 (518 s lag, 1782 m open
#     loop, 1.3 mi of a 10 mi run untracked).
#   * mid-run dead-zone (dem_elevation.COVERAGE_FLOOR) — orthogonal; catches a
#     track that dies in the middle rather than at the start.
# A day failing either keeps its LOGGED distance and pace, and drops out of the
# calibration corpus. See docs/recovery-runs-reference.md.
GPS_LAG_CEIL_S = DEM.LAG_CEIL_S
GPS_COVERAGE_FLOOR = DEM.COVERAGE_FLOOR

# Calibration fit knobs (match the house MAD-prune style).
CAL_PRUNE_SIGMA = 3.0
CAL_EXCESS_CAP_MI = 2.0   # |excess| beyond this is data corruption, not signal
CAL_MIN_N = 50            # below this, don't write a calibration at all


def _freq_points(rec):
    """Scaled (t_s, dist_m) from a rich record, glitch zeros dropped (same
    convention as reps._freq_points)."""
    return [(f[0] / 100.0, f[1] / 100.0) for f in rec.get('freq') or [] if f[1]]


def _pauses(rec):
    """Sorted [(start_s, dur_s)] from the rich record's pause list."""
    return sorted((ps / 100.0, dur / 100.0)
                  for ps, _e, dur in rec.get('pauses') or [] if ps and dur)


def _detect_stalls(pts):
    """Maximal standstills [(t_start, dur_s)] in a (t, d) stream: windows
    where distance advances slower than STALL_SPEED_MPS for at least
    STALL_MIN_S. Stream gaps (watch pauses) are skipped — those are counted
    from the pause list."""
    if len(pts) < 3:
        return []
    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    out = []
    i, n = 0, len(pts)
    while i < n - 1:
        dt = t[i + 1] - t[i]
        if dt > STREAM_GAP_S:
            i += 1
            continue
        if (d[i + 1] - d[i]) / max(dt, 1e-9) < STALL_SPEED_MPS:
            j = i
            while j < n - 1:
                dtj = t[j + 1] - t[j]
                if dtj > STREAM_GAP_S:
                    break
                if (d[j + 1] - d[j]) / max(dtj, 1e-9) >= STALL_SPEED_MPS:
                    break
                j += 1
            dur = t[j] - t[i]
            if dur >= STALL_MIN_S:
                out.append((float(t[i]), float(dur)))
            i = j + 1
        else:
            i += 1
    return out


def _activity_segments(rec, act):
    """([(dist_m, rest_after_s), ...], stall_s) for one activity, split at
    watch pauses and detected standstills. The last segment's rest is 0 —
    the caller adds inter-activity gaps. Slim records (no stream) degrade
    to one segment with the activity's pause total as its rest."""
    pts = _freq_points(rec)
    pause_s = max(act.total_s - act.moving_s, 0.0)
    if not pts:
        return [[act.distance_m, pause_s]], 0.0
    breaks = [(ts, dur, 'pause') for ts, dur in _pauses(rec)]
    stalls = _detect_stalls(pts)
    breaks += [(ts, dur, 'stall') for ts, dur in stalls]
    breaks.sort()
    if not breaks:
        return [[act.distance_m, 0.0]], 0.0

    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    segs, prev_d = [], 0.0
    for bt, dur, _kind in breaks:
        d_at = float(np.interp(bt, t, d))
        segs.append([d_at - prev_d, dur])
        prev_d = d_at
    segs.append([float(d[-1]) - prev_d, 0.0])
    # Fold degenerate slivers (<30 m — GPS jitter around a pause point) into
    # the previous segment's rest.
    clean = []
    for seg_d, rest in segs:
        if seg_d < 30 and clean:
            clean[-1][1] += rest
        elif seg_d >= 30:
            clean.append([seg_d, rest])
    # Rescale to the activity's official distance (stream truncation).
    tot = sum(s[0] for s in clean)
    if tot > 0 and act.distance_m > 0:
        k = act.distance_m / tot
        for s in clean:
            s[0] *= k
    return clean, sum(dur for _t, dur in stalls)


def _connected_d_eff(segs):
    """Connected-fatigue effective distance over (dist_m, rest_after_s)
    segments — parse_workouts._connected_core's accumulator, no anaerobic
    term (see module docstring)."""
    conn = d_eff = 0.0
    for dist, rest in segs:
        conn += dist
        d_eff = max(d_eff, conn)
        conn *= math.exp(-rest / RECON_TAU_S)
    return d_eff


def measure_day(acts):
    """Pure-watch per-day measurement for a day's time-ordered run activities
    ``[(rec, Activity), ...]``. Independent of the hand log — the ``run_type``
    filter and the ``complete`` time-gate are applied by the consumer, so this
    is exactly what watch_daily caches. Recovery days are mostly slim records
    (no per-second stream); their segment/stall fields degrade to one segment
    per activity, but the distance/moving-time fields are exact either way."""
    day_segs, stall_s = [], 0.0
    lags, covs = [], []
    for i, (rec, act) in enumerate(acts):
        # GPS track quality (see GPS_LAG_CEIL_S). Activities with no GPS at all
        # — indoor runs — are skipped, not treated as an infinite lag; the
        # `any_indoor` flag already speaks for those.
        lag = DEM.first_fix_lag_s(rec)
        if lag is not None:
            lags.append(lag)
            q = DEM.track_quality(rec)
            if q is not None:
                covs.append(q[1])
        segs, st = _activity_segments(rec, act)
        stall_s += st
        if i + 1 < len(acts):
            nxt = acts[i + 1][1]
            gap = (nxt.start_utc - act.start_utc).total_seconds() - act.total_s
            segs[-1][1] += max(gap, 0.0)
        day_segs.extend(segs)
    dist_m = sum(a.distance_m for _r, a in acts)
    moving_s = sum(a.moving_s for _r, a in acts)
    total_s = sum(a.total_s for _r, a in acts)
    return {
        'n_acts': len(acts),
        'status': 'rich' if all('freq' in r for r, _a in acts) else 'slim',
        'any_indoor': any(a.is_indoor for _r, a in acts),
        'watch_miles': dist_m / MILE_M,
        'watch_moving_s': round(moving_s, 1),
        'watch_total_s': round(total_s, 1),
        'pause_s': round(total_s - moving_s, 1),
        'stall_s': round(stall_s, 1),
        'n_segs': len(day_segs),
        'd_eff_frac': _connected_d_eff(day_segs) / dist_m if dist_m else np.nan,
        'longest_seg_mi': max((s[0] for s in day_segs), default=0.0) / MILE_M,
        # Worst activity of the day on each axis: one incompletely-tracked leg
        # is enough to make the day's summed distance untrustworthy.
        'gps_lag_s': round(max(lags), 1) if lags else np.nan,
        'gps_coverage': round(min(covs), 4) if covs else np.nan,
    }


def _f(v):
    """float(v) with a NaN for anything missing — watch_daily is read back as
    strings, and older rows predate the GPS columns entirely."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def gps_ok(m):
    """Whether a watch_daily row's GPS track is complete enough to price the
    day's DISTANCE (see GPS_LAG_CEIL_S). Missing values pass: pre-schema-5 rows
    and streams with no GPS to judge shouldn't retroactively lose their
    correction — the failure this catches is positive evidence, not absence."""
    lag = _f(m.get('gps_lag_s'))
    cov = _f(m.get('gps_coverage'))
    if lag == lag and lag > GPS_LAG_CEIL_S:
        return False
    if cov == cov and cov < GPS_COVERAGE_FLOOR:
        return False
    return True


def measure_runs(daily, wd, run_type='long'):
    """``run_type`` measured rows: the cached watch_daily measurement joined by
    date, plus the ``complete`` time-gate (watch moving time vs logged minutes).
    No cache parse — reads the precomputed scalars from ``wd``."""
    lr = daily[daily['run_type'] == run_type]
    rows = []
    for _, drow in lr.iterrows():
        dt = drow['date'].date().isoformat()
        if dt not in wd.index:
            continue
        m = wd.loc[dt]
        moving_s = float(m['watch_moving_s'])
        log_s = float(drow['minutes']) * 60.0 if pd.notna(drow.get('minutes')) else 0.0
        complete = (log_s > 0 and moving_s > 0
                    and abs(moving_s - log_s) / log_s <= TIME_COMPLETE_FRAC)
        rows.append({
            'date': dt,
            'n_acts': int(m['n_acts']),
            'status': m['status'],
            'complete': complete,
            'gps_ok': gps_ok(m),
            'gps_lag_s': _f(m.get('gps_lag_s')),
            'gps_coverage': _f(m.get('gps_coverage')),
            'watch_miles': float(m['watch_miles']),
            'watch_moving_s': moving_s,
            'watch_total_s': float(m['watch_total_s']),
            'pause_s': float(m['pause_s']),
            'stall_s': float(m['stall_s']),
            'n_segs': int(m['n_segs']),
            'd_eff_frac': float(m['d_eff_frac']),
            'longest_seg_mi': float(m['longest_seg_mi']),
        })
    return pd.DataFrame(rows)


def fit_calibration(daily, wd):
    """MAD-pruned OLS of logged excess miles on watch miles over the paved
    outdoor recovery + long corpus (see module docstring). Reads watch_daily
    scalars (no cache parse). Returns a one-row DataFrame, or None when the
    corpus is too small."""
    d = daily[daily['run_type'].isin(['recovery', 'long'])]
    xs, ys, is_lr = [], [], []
    for _, drow in d.iterrows():
        dt = drow['date'].date().isoformat()
        if dt not in wd.index:
            continue
        m = wd.loc[dt]
        if bool(m['any_indoor']):
            continue
        if str(drow.get('terrain_type')).lower() != 'paved':
            continue
        # Trailing-strides days: Max pauses the watch for the strides, so
        # the watch records the recovery portion only while the logged
        # miles include the strides — logged sits +0.32 mi (median) ABOVE
        # the calibrated watch distance vs −0.01 on normal days. Leaving
        # them in biases the excess curve up; gate explicitly rather than
        # rely on the MAD prune (the bias is inside its reach).
        if STRIDE_SUFFIX_RX.search(str(drow.get('workout_raw') or '')):
            continue
        moving_s = float(m['watch_moving_s'])
        log_s = float(drow['minutes']) * 60.0 if pd.notna(drow.get('minutes')) else 0.0
        if not (log_s > 0 and moving_s > 0
                and abs(moving_s - log_s) / log_s <= TIME_COMPLETE_FRAC):
            continue
        # A late-lock / dead-zone day is short in SPACE, so its logged "excess"
        # is inflated — leaving it in biases the curve toward over-correcting
        # every other day. Same gate the corrections themselves use.
        if not gps_ok(m):
            continue
        watch_mi = float(m['watch_miles'])
        excess = float(drow['miles']) - watch_mi
        if watch_mi <= 0 or abs(excess) > CAL_EXCESS_CAP_MI:
            continue
        xs.append(watch_mi)
        ys.append(excess)
        is_lr.append(drow['run_type'] == 'long')
    if len(xs) < CAL_MIN_N:
        return None
    x = np.asarray(xs)
    y = np.asarray(ys)
    is_lr = np.asarray(is_lr)
    X = np.vstack([np.ones(len(x)), x]).T
    kept = np.ones(len(x), bool)
    for _ in range(10):
        (c, m), *_ = np.linalg.lstsq(X[kept], y[kept], rcond=None)
        r = y - (c + m * x)
        med = float(np.median(r[kept]))
        sd_rob = 1.4826 * float(np.median(np.abs(r[kept] - med)))
        new = np.abs(r - med) <= CAL_PRUNE_SIGMA * sd_rob
        if (new == kept).all():
            break
        kept = new
    resid = y - (c + m * x)
    lr_kept = kept & is_lr
    sigma_pool = resid[lr_kept] if lr_kept.sum() >= 10 else resid[kept]
    return pd.DataFrame([{
        'intercept_mi': round(float(c), 4),
        'slope': round(float(m), 5),
        'sigma_mi': round(float(np.std(sigma_pool)), 4),
        'n_fit': int(kept.sum()),
        'n_pruned': int(len(x) - kept.sum()),
    }])


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    # --details-dir accepted for back-compat but unused: measurement now comes
    # from the precomputed watch_daily.csv (built incrementally by watch_daily),
    # so this step does no per-second parse.
    p.add_argument('--details-dir', type=Path,
                   default=DATA_DIR / 'profiles' / 'coros' / 'details')
    p.add_argument('--daily', type=Path, default=DATA_DIR / 'daily.csv')
    p.add_argument('--watch-daily', type=Path,
                   default=DATA_DIR / 'watch_daily.csv')
    p.add_argument('--out-measured', type=Path,
                   default=DATA_DIR / 'long_run_measured.csv')
    p.add_argument('--out-recovery-measured', type=Path,
                   default=DATA_DIR / 'recovery_measured.csv')
    p.add_argument('--out-calibration', type=Path,
                   default=DATA_DIR / 'long_run_calibration.csv')
    args = p.parse_args()

    if not args.watch_daily.exists():
        print(f'long_runs: no {args.watch_daily.name} — skipped '
              '(run watch_daily first)')
        return
    daily = pd.read_csv(args.daily, parse_dates=['date'])
    wd = pd.read_csv(args.watch_daily).set_index('date')

    for run_type, out_path in (('long', args.out_measured),
                               ('recovery', args.out_recovery_measured)):
        measured = measure_runs(daily, wd, run_type)
        measured.to_csv(out_path, index=False)
        n_rich = int((measured['status'] == 'rich').sum()) if len(measured) else 0
        n_inc = int((~measured['complete']).sum()) if len(measured) else 0
        print(f'{run_type}_measured: {len(measured)} days '
              f'({n_rich} rich, {n_inc} incomplete) -> {out_path}')

    cal = fit_calibration(daily, wd)
    if cal is None:
        print(f'calibration: corpus too small (<{CAL_MIN_N} paved gated days), '
              f'not written')
    else:
        cal.to_csv(args.out_calibration, index=False)
        r = cal.iloc[0]
        print(f'calibration: excess = {r.intercept_mi:+.3f} + {r.slope:.4f}*mi '
              f'(n={int(r.n_fit)}, pruned {int(r.n_pruned)}, '
              f'sigma={r.sigma_mi:.3f}) -> {args.out_calibration}')


if __name__ == '__main__':
    main()
