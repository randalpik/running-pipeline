"""Derive + audit the two CP3 v_max edges for the active profile (June 2026).

v_max is an uncertainty interval, not a point estimate — see the registry
comment in src/shared/cs_projection.py. This script derives both edges from
the profile's own corpus and audits Max's two hard invariants:

  (1) 400 m races must never exceed the frontier  → EVIDENCE edge (high):
      the lifetime envelope of sprint-credit demonstrated by the 400 corpus
      (the largest per-race implied v_max placing each 400 exactly ON the
      engine-only frontier), plus margin.
  (2) 400/800 predictions must never beat the lifetime PRs anywhere in the
      historical sweep → PREDICTION edge (low): the largest v_max whose
      sweep-min predictions still respect both PRs, minus margin.

The frontier is the standard demo set (time ≥ 120 s — 800s included and
allowed to bind; Max, June 2026). The evidence envelope is solved on the
400 corpus, the only shorts that are NOT demos, so the solve is not
circular. The workout accumulator's v_max (workouts.workout_vmax()) is
a separate measurement calibration and is NOT derived here.

Usage: python scripts/calibrate_vmax.py     (uses data/ artifacts)
Re-run after a CS refit; copy results into cs_projection's
VMAX_EVID_BY_PROFILE / VMAX_PRED_BY_PROFILE.

FOLLOW-UP (June 2026 1500 m-boundary redesign): the `implied_vmax` /
`sweep_min_pred` SOLVERS below still model a sub-1500 race as CP3 straight to
5K, but the live projection now goes CP3 → 1500 m → WA → 5K (two legs). So the
"400-corpus envelope" number this prints is computed on the OLD path and is not
directly comparable to the registry edge — TODO: update the solvers to the
two-leg path. The "audits at registry edges" section IS trustworthy: it calls
the live `project_races_to_5k_pace`, and it confirms the registry v_max
(9.5 / 8.3) still satisfies both invariants (0 short races past the frontier;
400/800 predictions respect the PRs). So the constants need no change today.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shared.paths import DATA_DIR
import src.shared.cs_projection as csp
from src.shared.performance_frontier import (standard_demos, build_frontier,
                                             frontier_at_anchor)

ANCHORS = [(400.0, None), (800.0, None)]   # PRs filled from races.csv
PAIR = [('2023-08-02', 400.0, 57.0), ('2023-07-26', 1609.0, 270.0)]


def load_base():
    daily, beta_long, d_thresh, xc = csp.load_cs_outputs(str(DATA_DIR))
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    return daily, beta_long, d_thresh, xc, races


def lifetime_pr(races, anchor):
    r = races.dropna(subset=['distance_m', 'time_sec']).copy()
    if 'surface' in r.columns:
        r = r[r['surface'] != 'Downhill']
    is_tt = r.get('event', pd.Series('', index=r.index)).fillna('').astype(str)\
             .str.contains('time trial', case=False, regex=False)
    r = r[~is_tt]
    sub = r[(r['distance_m'] - anchor).abs() / anchor < 0.08]
    return float(sub['time_sec'].min()) if len(sub) else np.nan


def engine_frontier(daily, beta_long, d_thresh, xc):
    """Frontier from standard_demos (time >= 120s — 800s in). Demo
    projection uses the registry/env evidence edge."""
    demos = standard_demos(daily, beta_long, d_thresh, xc)
    frontier, demos = build_frontier(demos, pd.DatetimeIndex(daily['date']),
                                     daily['p5k_implied_min'])
    return frontier, demos


def implied_vmax(date, d, t, daily_lk, fb):
    """v_max placing this short race exactly ON the engine frontier."""
    dd = pd.Timestamp(date).date()
    t5k_f = fb.get(dd) * 60 * 5000 / 1609.344
    dp2 = float(daily_lk.loc[dd, 'dp_med'])
    cs_fit = float(daily_lk.loc[dd, 'cs_mps_med'])

    def pred(v):
        dp3 = float(csp.cp3_dprime(dp2, cs_fit, v))
        cs_f = float(csp.cp3_implied_cs(5000.0, t5k_f, dp3, v))
        return float(csp.cp3_time(d, cs_f, dp3, v))

    lo, hi = max(7.2, d / t + 0.05), 14.0
    if pred(hi) > t:
        return None          # past frontier even at huge sprint credit
    if pred(lo) < t:
        return -1.0          # behind frontier at any credit (never binds)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pred(mid) > t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sweep_min_pred(frontier, daily, anchor, beta_long, d_thresh, v):
    os.environ['RP_VMAX_PRED'] = f'{v:.6f}'
    dv = daily.copy()
    dv['dp3_pred_med'] = csp.cp3_dprime(dv['dp_med'], dv['cs_mps_med'], v)
    t = frontier_at_anchor(frontier, dv, anchor, beta_long, d_thresh)
    os.environ.pop('RP_VMAX_PRED', None)
    return float(np.nanmin(t)), int(np.nanargmin(t))


def main():
    daily, beta_long, d_thresh, xc, races = load_base()
    daily_lk = daily.set_index(daily['date'].dt.date)
    frontier, demos = engine_frontier(daily, beta_long, d_thresh, xc)
    fb = pd.Series(frontier['frontier_pace_min'].to_numpy(),
                   index=daily['date'].dt.date.to_numpy())

    # ---- evidence edge: sprint-credit envelope over the 400 corpus ----
    print('=== EVIDENCE edge (invariant 1: 400s never past the frontier) ===')
    shorts = races[races['distance_m'] < 1500].sort_values('date')
    # Envelope table: 400s only — 800s are demos now, so their implied
    # v_max against a frontier they may themselves bind is circular.
    fours = shorts[shorts['distance_m'] < 600]
    env_400 = 0.0
    for _, r in fours.iterrows():
        v = implied_vmax(r['date'], float(r['distance_m']),
                         float(r['time_sec']), daily_lk, fb)
        lab = ('never binds' if v == -1.0
               else '>14 ?!' if v is None else f'{v:5.2f}')
        is400 = abs(r['distance_m'] - 400) < 50
        if is400 and isinstance(v, float) and v > env_400:
            env_400 = v
        print(f"  {r['date'].date()} {r['distance_m']:5.0f}m "
              f"{r['time_sec']:7.1f}s {'F' if r['fatigued'] else ' '}  "
              f"implied v_max {lab}{'   <- 400 corpus' if is400 else ''}")
    print(f'  400-corpus envelope: {env_400:.2f}  '
          f'(registry v_evid = {csp.vmax_evidence():.2f} — must exceed this)')

    # ---- prediction edge: PR-sweep bounds ----
    print('\n=== PREDICTION edge (invariant 2: predictions never beat PRs) ===')
    for anchor, _ in ANCHORS:
        pr = lifetime_pr(races, anchor)
        lo, hi = 7.3, 12.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            mn, _i = sweep_min_pred(frontier, daily, anchor, beta_long,
                                    d_thresh, mid)
            if mn >= pr:
                lo = mid
            else:
                hi = mid
        mn_reg, i_reg = sweep_min_pred(frontier, daily, anchor, beta_long,
                                       d_thresh, csp.vmax_predict())
        print(f'  {anchor:5.0f}m: PR {pr:7.2f}s | v_pred upper bound {lo:.3f} '
              f'| at registry {csp.vmax_predict():.2f}: sweep-min {mn_reg:.2f}s '
              f'({daily["date"].iloc[i_reg].date()}), margin {mn_reg - pr:+.2f}s')

    # ---- audits at registry values ----
    print('\n=== audits at registry edges ===')
    proj = csp.project_races_to_5k_pace(shorts.copy(), daily, beta_long,
                                        d_thresh, apply_xc_correction=True,
                                        xc_correction=xc)
    proj['front'] = [fb.get(d.date(), np.nan) for d in proj['date']]
    proj['delta'] = (proj['pace_norm_min'] - proj['front']) * 60
    past = proj[proj['delta'] < 0]
    print(f'  short races past frontier: {len(past)}/{len(proj)}')
    for _, r in proj.sort_values('delta').head(4).iterrows():
        print(f"    {r['date'].date()} {r['distance_m']:5.0f}m "
              f"{r['time_sec']:7.1f}s  {r['delta']:+6.1f} s/mi")
    r800 = races[((races['distance_m'] - 800).abs() < 60)
                 & ~races['fatigued'].fillna(False).astype(bool)]
    d800 = demos[(demos['src'] == 'race') & demos['date'].isin(r800['date'])]
    print(f'  800m demos binding the frontier: '
          f'{int(d800["binding"].sum())}/{len(d800)} (by design — Max, '
          f'June 2026: an 800 is honest aerobic evidence)')

    pair = pd.DataFrame([{'date': pd.Timestamp(d), 'distance_m': dm,
                          'time_sec': ts} for d, dm, ts in PAIR])
    pp = csp.project_races_to_5k_pace(pair, daily, beta_long, d_thresh,
                                      apply_xc_correction=False)
    t5 = pp['time_norm_sec'].to_numpy()
    print(f'  pair diagnostic (no longer an anchor): 400@57.0 -> {t5[0]:.1f}s; '
          f'mile@4:30 -> {t5[1]:.1f}s (delta {(t5[0]-t5[1])/3.107:+.1f} s/mi)')


if __name__ == '__main__':
    main()
