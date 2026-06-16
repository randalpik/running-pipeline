"""Spike: evaluate held-out race predictions for the censored delta model.

Usage:
    spike_holdout_eval2.py <window_lo> <window_hi> <race_arm_tag> <delta_tag>

Scores three predictors on the kept races inside the window:
  A: race-only LOO curve (bayes_cs_summary_<race_arm_tag>.csv)
  B: training-informed = A + delta (cs_training_summary_<delta_tag>.csv)
  R: production full fit (saw the races; reference)

All summary/params files are read from output/debug/spike_cs_enrichment/;
the kept-race list from output/debug/bayes_cs_residuals_spike_raceonly.csv
(actual_sec already XC-pre-corrected, matching fit conventions).

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
DEBUG = ROOT / 'output' / 'debug'

w_lo, w_hi, race_tag, delta_tag = sys.argv[1:5]

kept = pd.read_csv(DEBUG / 'bayes_cs_residuals_spike_raceonly.csv',
                   parse_dates=['date'])
ev = kept[(kept['date'] >= w_lo) & (kept['date'] < w_hi)].copy()
print(f'Eval set: {len(ev)} held-out kept races in [{w_lo}, {w_hi})')

dates = pd.DatetimeIndex(ev['date'])
dists = ev['distance_m'].to_numpy(float)
actual = ev['actual_sec'].to_numpy(float)


def predict(summary_path, params_path, cs_col):
    s = pd.read_csv(summary_path, parse_dates=['date'])
    p = pd.read_csv(params_path)
    beta = float(p['beta_long_med'].iloc[0])
    day = (s['date'] - s['date'].min()).dt.days.astype(float)
    d_obs = (dates - s['date'].min()).days.astype(float)
    cs = np.interp(d_obs, day, s[cs_col])
    dp = np.interp(d_obs, day, s['dp_med'])
    bias = np.where(dists > 10000,
                    1 + beta * np.log(np.maximum(dists, 10000) / 10000), 1.0)
    return (dists - dp) / cs * bias


arms = {
    'A race-only LOO': (SPIKE / f'bayes_cs_summary_{race_tag}.csv',
                        SPIKE / f'bayes_cs_params_{race_tag}.csv', 'cs_mps_med'),
    'B train-informed': (SPIKE / f'cs_training_summary_{delta_tag}.csv',
                         SPIKE / f'bayes_cs_params_{race_tag}.csv',
                         'cs_train_mps_med'),
    'R full fit (ref)': (SPIKE / 'bayes_cs_summary_spike_raceonly.csv',
                         SPIKE / 'bayes_cs_params_spike_raceonly.csv',
                         'cs_mps_med'),
}
errs = {}
for name, (sp, pp, col) in arms.items():
    pred = predict(sp, pp, col)
    err = (actual / pred - 1) * 100
    errs[name] = err
    print(f'{name}: mean|err| {np.mean(np.abs(err)):.2f}%   '
          f'RMSE {np.sqrt(np.mean(err**2)):.2f}%   mean err {np.mean(err):+.2f}%')

print(f'\n{"date":<12}{"dist":>7}  {"A":>8}  {"B":>8}  {"R":>8}  event')
for i, (_, r) in enumerate(ev.iterrows()):
    print(f"{str(r['date'].date()):<12}{int(r['distance_m']):>7}  "
          f"{errs['A race-only LOO'][i]:>+7.2f}%  "
          f"{errs['B train-informed'][i]:>+7.2f}%  "
          f"{errs['R full fit (ref)'][i]:>+7.2f}%  {r['event'][:32]}")
