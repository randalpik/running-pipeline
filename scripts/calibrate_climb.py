"""Live calibration of the two-channel grade model (Aug 2026, continuous).

Fits the four engine constants with a CONTINUOUS sliding-window regression —
one observation per 30 m of every eligible run, window length WINDOW_MI — so no
window placement exists anywhere in the model (the failure mode of fixed mile
bins: a half-mile boundary shift moved coefficients 20-35%). The window SCALE
is a physical constant, not a tuning knob, bounded on both sides by the Aug
2026 scale sweep (scratch/scale_sweep.csv, artifact ba121160):

  * >= ~1.25 mi: a window must contain a hill PLUS its smeared pace response,
    or the steepness slopes attenuate and dump into the base rates (priced
    totals stay right — the split doesn't);
  * <= ~2 mi: identification thins beyond that (few windows per run; short
    runs leave the corpus). Gain/loss collinearity is NOT the limiter
    (measured flat at -0.51 across scales).
  1.5 mi is the middle of that band. Priced rates at corpus-typical grades are
  scale-invariant within ~2% from 0.25-2.0 mi, so this choice steers only how
  cost is split between base and steepness (i.e. extrapolation behaviour).

Model (fraction of pace; regressors in ft per mile of window):

    y ~ c0·gain + c1·SUM_climb_hills(vert·grade) − b0·loss − b1·SUM_desc(...)

MAGNITUDE is gross fused gain/loss — ALL vertical (a 1% grade climbed for 10
miles is 500 real feet). STEEPNESS enters through the veto-cleaned hill
segments' vert·grade mass (data/elevation_hills.csv): the hills machinery
measures how steep the real climbs/descents are (reproducible to ~0.1% across
course reruns) and contributes nothing else.

Also measures FLOOR_G/FLOOR_L — the flat-ground pedestal (median per-mile
gain/loss on miles with no hills at all). The day demeaning absorbs it in the
fit, so the slopes were never estimated on it; application must therefore
price vertical IN EXCESS of the floor (elevation_cost.py) or every run pays a
phantom ~2 s/mi (Berlin read +52 s before this).

Controls: day demeaning (FE), quintile drift dummies on window-centre position
(within-run pace drift is non-monotonic), 4-sigma MAD prune, interior windows
(>= 1 mile from the start, clear of the finish), recovery + long across
paved/mixed/trail (terrain is day-constant -> absorbed by the FE).

Writes data/elevation_calibration.csv: one row —
  c0, c1, b0, b1, floor_g, floor_l, n_windows, n_days, k_natexp, n_natexp,
  source(fit|default)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.coros import elevation as E
from src.coros.build_current_log import strip_paused

WINDOW_MI = 1.5
STEP = 10.0
MI = 1609.344
STRIDE = 3                        # window starts every 30 m

# Defaults = the Aug 2026 continuous fit on Max's fused corpus.
DEFAULT_C0 = 0.49e-3
DEFAULT_C1 = 0.052e-3             # per % hill grade
DEFAULT_B0 = 0.46e-3
DEFAULT_B1 = -0.036e-3            # per % hill grade
DEFAULT_FLOOR_G = 14.8            # ft per mile — flat-ground pedestal
DEFAULT_FLOOR_L = 14.5

MIN_DAYS = 120
NATEXP_BAND = (0.6e-3, 1.5e-3)
EASY_TIERS = {'recovery', 'long'}

_OWN_DETAILS = DATA_DIR / 'details'
DETAILS = (_OWN_DETAILS if _OWN_DETAILS.exists()
           else DATA_DIR / 'profiles' / 'coros' / 'details')
OUT = DATA_DIR / 'elevation_calibration.csv'


def _streams():
    """{date: (t, d, a)} stitched, k-corrected, running-max monotonic — the
    easy-tier calibration corpus."""
    idx = pd.read_csv(DATA_DIR / 'watch_activities.csv',
                      dtype={'labelId': str, 'date': str})
    meas = pd.read_csv(DATA_DIR / 'elevation_measured.csv',
                       dtype={'date': str})
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily['dk'] = daily['date'].dt.date.astype(str)
    meta = daily.drop_duplicates('dk').set_index('dk')
    kk = dict(zip(meas.date, np.where(meas.watch_miles > 0,
                                      meas.corr_miles / meas.watch_miles,
                                      1.0)))
    ids = {}
    for _, r in idx.iterrows():
        ids.setdefault(r['date'], []).append(r['labelId'])
    out = {}
    for dte in meas[meas.run_type.isin(EASY_TIERS)].date:
        if dte not in meta.index:
            continue
        terr = str(meta.loc[dte, 'terrain_type']).strip().lower()
        if terr not in ('paved', 'mixed', 'trail'):
            continue
        pts, od, ot = [], 0.0, 0.0
        for lid in ids.get(dte, []):
            p = DETAILS / f'{lid}.json'
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            if rec.get('rich') != 2:
                continue
            ap = E.alt_points(strip_paused(rec))
            if len(ap) < 30:
                continue
            t0 = ap[0][0]
            for (t, d, a) in ap:
                pts.append((t - t0 + ot, d + od, a))
            od = pts[-1][1]
            ot = pts[-1][0] + E.STREAM_GAP_S + 1
        if len(pts) < 60:
            continue
        k = kk.get(dte, 1.0)
        t = np.array([q[0] for q in pts])
        d = np.array([q[1] for q in pts]) * k
        a = np.array([q[2] for q in pts])
        rm = np.maximum.accumulate(d)
        keep = np.concatenate(([True], d[1:] > rm[:-1]))
        out[dte] = (t[keep], d[keep], a[keep])
    return out


def _cumulants(t, d, a, hills):
    grid = np.arange(d[0], d[-1], STEP)
    if len(grid) < 60:
        return None
    dt = np.diff(t)
    dt[dt > E.STREAM_GAP_S] = 0.0
    mov = np.concatenate(([0.0], np.cumsum(dt)))
    T = np.interp(grid, d, mov)
    _, galt = E._gridded_altitude(d, a)
    galt = (galt[:len(grid)] if len(galt) >= len(grid)
            else np.pad(galt, (0, len(grid) - len(galt)), 'edge'))
    cap = E.SPIKE_GRADE_CAP * E.GRID_M
    steps = np.clip(np.diff(galt), -cap, cap) / 0.3048
    P = np.concatenate(([0.0], np.cumsum(np.maximum(steps, 0.0))))
    N = np.concatenate(([0.0], np.cumsum(np.maximum(-steps, 0.0))))
    U = np.zeros(len(grid))
    D = np.zeros(len(grid))
    for (h0, h1, vert, grade, kind) in hills:
        i0, i1 = np.searchsorted(grid, h0), np.searchsorted(grid, h1)
        if i1 <= i0:
            continue
        rate = abs(vert) * grade / (i1 - i0)
        (U if kind > 0 else D)[i0:i1] += rate
    U = np.concatenate(([0.0], np.cumsum(U)))
    D = np.concatenate(([0.0], np.cumsum(D)))
    return grid, T, P[:len(grid)], N[:len(grid)], U[:len(grid)], D[:len(grid)]


def _fit(streams, hills_by_date):
    ys, Xs = [], []
    n_days = 0
    W = int(round(WINDOW_MI * MI / STEP))
    for dte, (t, d, a) in streams.items():
        cum = _cumulants(t, d, a, hills_by_date.get(dte, []))
        if cum is None:
            continue
        grid, T, P, N, U, D = cum
        lo = int(round(MI / STEP))
        hi = len(grid) - W - int(round(320 / STEP))
        if hi <= lo:
            continue
        idx = np.arange(lo, hi, STRIDE)
        j = idx + W
        pace = (T[j] - T[idx]) / WINDOW_MI
        ok = (pace > 240) & (pace < 780)
        if ok.sum() < 20:
            continue
        idx, j, pace = idx[ok], j[ok], pace[ok]
        X = np.column_stack([P[j] - P[idx], U[j] - U[idx],
                             N[j] - N[idx], D[j] - D[idx]]) / WINDOW_MI
        pos = (grid[idx] + grid[j]) / 2 / grid[-1]
        y = pace / pace.mean() - 1.0
        qb = np.clip((pos * 5).astype(int), 0, 4)
        Q = np.zeros((len(pos), 4))
        for q in range(1, 5):
            Q[:, q - 1] = (qb == q)
        Xd = np.column_stack([X, Q])
        Xd = Xd - Xd.mean(0)
        ys.append(y.astype(np.float32))
        Xs.append(Xd.astype(np.float32))
        n_days += 1
    if not ys:
        return None
    y = np.concatenate(ys)
    X = np.vstack(Xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    keep = np.abs(r - np.median(r)) <= 4.0 * mad
    b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return (float(b[0]), float(b[1]), float(-b[2]), float(-b[3]),
            int(keep.sum()), n_days)


def _floor_and_natexp():
    """Flat-ground pedestal + the position-matched sustained-climb
    cross-check, both from the mile splits."""
    fg, fl = DEFAULT_FLOOR_G, DEFAULT_FLOOR_L
    k_ne, n_ne = np.nan, 0
    sp_path = DATA_DIR / 'elevation_splits.csv'
    if not sp_path.exists():
        return fg, fl, k_ne, n_ne
    sp = pd.read_csv(sp_path, dtype={'date': str})
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily['dk'] = daily['date'].dt.date.astype(str)
    meta = daily.drop_duplicates('dk').set_index('dk')
    sp['run_type'] = sp['date'].map(meta['run_type'])
    sp = sp[sp.run_type.isin(EASY_TIERS) & (sp.get('covered', 1.0) >= 0.99)]
    last = sp.groupby('date')['mile'].transform('max')
    sp = sp[(sp.mile > 0) & (sp.mile < last)]
    flat = sp[(sp.g_up == 0) & (sp.g_down == 0)]
    if len(flat) >= 200:
        fg, fl = float(flat.gain_ft.median()), float(flat.loss_ft.median())
    ks = []
    climbs = sp[(sp.gain_ft >= 150) & (sp.loss_ft < 40)]
    for d, g in climbs.groupby('date'):
        day = sp[sp['date'] == d]
        base_pool = day[(day.gain_ft < 40) & (day.loss_ft < 40)]
        for _, r in g.iterrows():
            near = base_pool[(base_pool.mile >= r.mile - 2)
                             & (base_pool.mile <= r.mile + 2)]
            if not len(near):
                continue
            base = near.pace_s.median()
            ks.append(((r.pace_s - base) / base) / (r.gain_ft - r.loss_ft))
    if ks:
        k_ne, n_ne = float(np.median(ks)), len(ks)
    return fg, fl, k_ne, n_ne


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    row = {'c0': DEFAULT_C0, 'c1': DEFAULT_C1, 'b0': DEFAULT_B0,
           'b1': DEFAULT_B1, 'floor_g': DEFAULT_FLOOR_G,
           'floor_l': DEFAULT_FLOOR_L, 'n_windows': 0, 'n_days': 0,
           'k_natexp': np.nan, 'n_natexp': 0, 'source': 'default'}
    hills_path = DATA_DIR / 'elevation_hills.csv'
    fit_out = None
    if hills_path.exists():
        hills = pd.read_csv(hills_path, dtype={'date': str})
        hills = hills[hills.vetoed == 0]
        hb = {d: [(r.d0, r.d1, r.vert_ft, r.grade_pct, r.kind)
                  for r in g.itertuples()]
              for d, g in hills.groupby('date')}
        streams = _streams()
        if len(streams) >= MIN_DAYS:
            fit_out = _fit(streams, hb)
    if fit_out is not None:
        c0, c1, b0, b1, n_win, n_days = fit_out
        fg, fl, k_ne, n_ne = _floor_and_natexp()
        warn = []
        if not (c0 > 0 and b0 > 0):
            warn.append(f'sign check failed (c0 {c0:.2e}, b0 {b0:.2e})')
        if b1 >= 0:
            warn.append(f'b1 {b1:.2e} not negative — no steepness decline')
        if np.isfinite(k_ne) and n_ne >= 10 and not (
                NATEXP_BAND[0] <= k_ne <= NATEXP_BAND[1]):
            warn.append(f'natexp {k_ne:.2e} outside {NATEXP_BAND}')
        for w in warn:
            print(f'[calibrate_climb] WARN: {w}')
        if c0 > 0 and b0 > 0:
            row.update(c0=c0, c1=c1, b0=b0, b1=b1, floor_g=fg, floor_l=fl,
                       n_windows=n_win, n_days=n_days, k_natexp=k_ne,
                       n_natexp=n_ne, source='fit')
        else:
            print('[calibrate_climb] unphysical fit — keeping defaults')
    else:
        print(f'[calibrate_climb] corpus too thin — writing defaults')

    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(f"[calibrate_climb] c0={row['c0']*1e3:.4f}e-3 "
          f"c1={row['c1']*1e3:.4f}e-3 b0={row['b0']*1e3:.4f}e-3 "
          f"b1={row['b1']*1e3:.4f}e-3 floor={row['floor_g']:.1f}/"
          f"{row['floor_l']:.1f} ft/mi ({row['source']}, "
          f"{row['n_days']} days, {row['n_windows']:,} windows; "
          f"natexp {row['k_natexp']} n={row['n_natexp']}) -> {OUT}")


if __name__ == '__main__':
    main()
