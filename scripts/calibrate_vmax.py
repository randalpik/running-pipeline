"""Derive the two CP3 v_max CS-multiples for the active profile (Max, June 2026).

v_max is under-identified from the data, so we don't estimate it — we BRACKET it
with two conservative CS-multiples, each the binding extreme of a constraint
against demonstrated performance. No manual tuning: this script derives both;
re-run after a CS refit and copy the values into cs_projection's
VMAX_EVID_CS_RATIO_BY_PROFILE / VMAX_PRED_CS_RATIO_BY_PROFILE.

  k_evid (evidence / DOWN-projection, HIGH): the smallest multiple that keeps
    every short (<700 m) race behind the AEROBIC (>=1500 m) race frontier — the
    races reliably WA-convertible to a 5K. The >=1500 frontier is pure World
    Athletics (v_max-INDEPENDENT), so this is a monotonic root-find with NO
    fixed-point. 800s are deliberately neither the reference (a model-projected
    short race isn't a yardstick) nor constrained (they bind the displayed
    frontier as real demos). "A 400 never defines the frontier" then holds by
    construction.

  k_pred (prediction / UP-projection, LOW): the largest multiple whose CP3-up
    400/800 predictions never beat the lifetime PRs (binding at peak fitness),
    measured on the displayed frontier built at k_evid.

Defaults (no qualifying short race, e.g. a new watch profile): k_evid=2.00,
k_pred=1.50 — a wide conservative bracket the first real short race tightens.

Usage: python scripts/calibrate_vmax.py     (uses the active profile's data/)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE
import src.shared.cs_projection as csp
from src.shared.performance_frontier import build_frontier

SHORT_MAX_M = 700.0          # races below this are CONSTRAINED (400s, 200s)
AERO_MIN_M = csp.CP3_IAAF_BOUNDARY_M   # 1500 — aerobic reference frontier
K_EVID_DEFAULT = csp.VMAX_EVID_CS_RATIO_DEFAULT
K_PRED_DEFAULT = csp.VMAX_PRED_CS_RATIO_DEFAULT
PRED_ANCHORS = (400.0, 800.0)


def _mss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def lifetime_pr(races, anchor):
    r = races.dropna(subset=['distance_m', 'time_sec']).copy()
    if 'surface' in r.columns:
        r = r[r['surface'] != 'Downhill']
    ev = r.get('event', pd.Series('', index=r.index)).fillna('').astype(str)
    r = r[~ev.str.contains('time trial', case=False, regex=False)]
    sub = r[(r['distance_m'] - anchor).abs() / anchor < 0.08]
    return float(sub['time_sec'].min()) if len(sub) else np.nan


def main():
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    ap.add_argument('--write', action='store_true',
                    help='write DATA_DIR/vmax_ratios.csv (the pipeline source); '
                         'omit for a dry-run audit')
    args = ap.parse_args()

    daily, beta_long, d_thresh, xc = csp.load_cs_outputs(str(DATA_DIR))
    grid = pd.DatetimeIndex(daily['date'])
    gd = grid.to_numpy('datetime64[D]').astype(float)
    floor = daily['p5k_implied_min'].to_numpy(float)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    profile = os.environ.get('RP_PROFILE', 'max')
    print(f"profile: {profile}")

    # ---- aerobic reference frontier: >=1500 m races only (WA, k-independent) ----
    aero = races[(~races['fatigued'].fillna(False).astype(bool))
                 & (races.get('surface', 'X') != 'Downhill')
                 & (races['distance_m'] >= AERO_MIN_M)].copy()
    short = races[races['distance_m'] < SHORT_MAX_M].copy()

    def frontier_pace_at(dates, fp):
        return np.array([float(np.interp(np.datetime64(d, 'D').astype(float), gd, fp))
                         for d in dates])

    # ===== k_evid: smallest multiple keeping every <700 m race behind aero frontier
    print("\n=== k_evid (short races behind the >=1500 m aerobic frontier) ===")
    if aero.empty or short.empty:
        k_evid = K_EVID_DEFAULT
        print(f"  no {'aerobic races' if aero.empty else 'short races'} "
              f"-> DEFAULT k_evid = {k_evid:.2f}")
    else:
        aproj = csp.project_races_to_5k_pace(aero, daily, beta_long, d_thresh,
                                             apply_xc_correction=True, xc_correction=xc)
        aero_demos = pd.DataFrame({'date': aproj['date'], 'pace_min': aproj['pace_norm_min'],
                                   'src': 'race', 'category': 'race', 'detail': ''})
        aero_front, _ = build_frontier(aero_demos, grid, floor)
        afp = aero_front['frontier_pace_min'].to_numpy(float)

        def worst_past(k):     # max over short races of (aero_frontier - proj), + = PAST
            os.environ['RP_VMAX_EVID_RATIO'] = f'{k:.5f}'
            pr = csp.project_races_to_5k_pace(short.copy(), daily, beta_long, d_thresh,
                                              apply_xc_correction=True, xc_correction=xc)
            os.environ.pop('RP_VMAX_EVID_RATIO', None)
            fpa = frontier_pace_at(pr['date'], afp)
            d = (fpa - pr['pace_norm_min'].to_numpy(float)) * 60.0
            return d, pr
        k_evid = None
        for k in np.arange(1.40, 2.61, 0.01):
            d, _ = worst_past(k)
            if np.nanmax(d) <= 0:
                k_evid = round(float(k), 2)
                break
        d, pr = worst_past(k_evid if k_evid else 2.60)
        i = int(np.nanargmax(d))
        b = pr.iloc[i]
        if k_evid is None:        # no multiple in range cleared it -> conservative default
            k_evid = K_EVID_DEFAULT
            print(f"  no multiple <=2.60 keeps all short races behind -> DEFAULT {k_evid}")
        else:
            print(f"  k_evid = {k_evid}   binding: {b['date'].date()} "
                  f"{int(b['distance_m'])}m {b['time_sec']:.1f}s "
                  f"(margin {np.nanmax(d):+.2f} s/mi behind frontier)")

    # ===== k_pred: largest multiple whose 400/800 predictions don't beat PRs =====
    print("\n=== k_pred (short predictions never beat lifetime PRs) ===")
    prs = {a: lifetime_pr(races, a) for a in PRED_ANCHORS}
    have = {a: t for a, t in prs.items() if np.isfinite(t)}
    if not have:
        k_pred = K_PRED_DEFAULT
        print(f"  no short PRs -> DEFAULT k_pred = {k_pred:.2f}")
    else:
        # Race-only frontier (no workout corpus) so k_pred depends solely on
        # CS + races — no plot-phase artifact, no build-order loop. Built at the
        # derived k_evid (sub-1500 race demos use the evidence edge).
        os.environ['RP_VMAX_EVID_RATIO'] = f'{k_evid:.5f}'
        elig = races[(~races['fatigued'].fillna(False).astype(bool))
                     & (races.get('surface', 'X') != 'Downhill')
                     & (races['time_sec'] >= 120)].copy()
        ep = csp.project_races_to_5k_pace(elig, daily, beta_long, d_thresh,
                                          apply_xc_correction=True, xc_correction=xc)
        rdemos = pd.DataFrame({'date': ep['date'], 'pace_min': ep['pace_norm_min'],
                               'src': 'race', 'category': 'race', 'detail': ''})
        front, _ = build_frontier(rdemos, grid, floor)
        os.environ.pop('RP_VMAX_EVID_RATIO', None)
        cs_g = daily['cs_mps_med'].to_numpy(float)
        dp2_g = daily['dp_med'].to_numpy(float)
        t5k_front = front['frontier_pace_min'].to_numpy(float) * 60.0 * 5000.0 / METERS_PER_MILE

        def min_pred(a, k):    # fastest predicted time at anchor a over all dates
            v = k * cs_g
            dp3 = csp.cp3_dprime(dp2_g, cs_g, v)
            ci = csp.cp3_implied_cs(5000.0, t5k_front, dp3, v)
            return float(np.nanmin(np.asarray(csp.cp3_time(a, ci, dp3, v), float)))

        k_pred = None
        for k in np.arange(2.00, 1.19, -0.01):   # descend; first k respecting all PRs
            if all(min_pred(a, k) >= have[a] for a in have):
                k_pred = round(float(k), 2)
                break
        k_pred = k_pred if k_pred else K_PRED_DEFAULT
        for a in have:
            mp = min_pred(a, k_pred)
            print(f"  {int(a)}m PR {have[a]:.1f}s | pred {mp:.1f}s "
                  f"(margin {have[a]-mp:+.1f}s)")
        print(f"  k_pred = {k_pred}")

    # ---- result + write the pipeline artifact ----
    reg_e = csp.VMAX_EVID_CS_RATIO_BY_PROFILE.get(profile, K_EVID_DEFAULT)
    reg_p = csp.VMAX_PRED_CS_RATIO_BY_PROFILE.get(profile, K_PRED_DEFAULT)
    print(f"\n=== result ===\n  derived: k_evid={k_evid}  k_pred={k_pred}")
    print(f"  registry fallback: k_evid={reg_e}  k_pred={reg_p}"
          + ("  (match)" if (k_evid == reg_e and k_pred == reg_p) else ""))
    if args.write:
        out = DATA_DIR / 'vmax_ratios.csv'
        pd.DataFrame([{'profile': profile, 'k_evid': k_evid, 'k_pred': k_pred}]
                     ).to_csv(out, index=False)
        print(f"  wrote {out}")
    else:
        print("  (dry run; pass --write to update data/vmax_ratios.csv)")


if __name__ == '__main__':
    main()
