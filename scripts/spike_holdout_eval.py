"""Spike: evaluate held-out 2019-06..2020-12 race predictions.

Arms (races_holdout.csv = races.csv minus the window):
  A: race-only fit on the reduced set        (spike_loo_raceonly)
  B: A + near-race obs RE-SELECTED against A (spike_loo_nearrace)
Reference: production race-only fit that saw the races (spike_raceonly).

Eval set: window races that the production race-only fit KEPT (its
residuals CSV), i.e. real eligible races, no bonks/fatigued/downhill.
Prediction: t_hat = (d - dp_med(t))/cs_mps(t) * (1 + beta*log(d/10k))[d>10k],
with XC actual times divided by 1.08 first (fit convention).

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DEBUG = ROOT / 'output' / 'debug'

W_LO, W_HI = '2019-06-01', '2021-01-01'

kept = pd.read_csv(DEBUG / 'bayes_cs_residuals_spike_raceonly.csv',
                   parse_dates=['date'])
ev = kept[(kept['date'] >= W_LO) & (kept['date'] < W_HI)].copy()
# actual_sec in the residuals file is already XC-pre-corrected
print(f'Eval set: {len(ev)} held-out kept races')


def predict(tag, dates, dists):
    s = pd.read_csv(DATA / f'bayes_cs_summary_{tag}.csv', parse_dates=['date'])
    p = pd.read_csv(DATA / f'bayes_cs_params_{tag}.csv')
    beta = float(p['beta_long_med'].iloc[0])
    day = (s['date'] - s['date'].min()).dt.days.astype(float)
    d_obs = (dates - s['date'].min()).days.astype(float)
    cs = np.interp(d_obs, day, s['cs_mps_med'])
    dp = np.interp(d_obs, day, s['dp_med'])
    bias = np.where(dists > 10000, 1 + beta * np.log(np.maximum(dists, 10000) / 10000), 1.0)
    return (dists - dp) / cs * bias


dates = pd.DatetimeIndex(ev['date'])
dists = ev['distance_m'].to_numpy(float)
actual = ev['actual_sec'].to_numpy(float)

enr_tag = sys.argv[1] if len(sys.argv) > 1 else 'spike_loo_nearrace'
res = {}
for tag in ('spike_loo_raceonly', enr_tag, 'spike_raceonly'):
    try:
        pred = predict(tag, dates, dists)
    except FileNotFoundError:
        print(f'  ({tag} not available yet)')
        continue
    err = (actual / pred - 1) * 100
    res[tag] = err
    print(f'\n--- {tag} ---')
    print(f'  mean|err| {np.mean(np.abs(err)):.2f}%   RMSE {np.sqrt(np.mean(err**2)):.2f}%   '
          f'mean err {np.mean(err):+.2f}% (+ = actual slower than predicted)')

if 'spike_loo_raceonly' in res and enr_tag in res:
    print(f'\n--- per-race (errors %, + = ran slower than predicted) ---')
    print(f'{"date":<12}{"dist":>7}  {"A race-only":>11}  {"B enriched":>11}  '
          f'{"full-fit":>9}  event')
    for i, (_, r) in enumerate(ev.iterrows()):
        full = res.get('spike_raceonly')
        print(f"{str(r['date'].date()):<12}{int(r['distance_m']):>7}  "
              f"{res['spike_loo_raceonly'][i]:>+10.2f}%  "
              f"{res[enr_tag][i]:>+10.2f}%  "
              f"{(full[i] if full is not None else float('nan')):>+8.2f}%  "
              f"{r['event'][:32]}")
