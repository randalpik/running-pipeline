"""Spike: production acceptance analysis for the training-informed delta fit.

Reads cs_training_summary_<tag>.csv (delta over the production race anchor)
and reports: delta by year, race-contradiction check over all kept races,
fall-2020 peak / spring-2021 trough, current-day reading, delta CI width,
and the influence list of load-bearing fast points.

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
DEBUG = ROOT / 'output' / 'debug'

tag = sys.argv[1] if len(sys.argv) > 1 else 'prod_hc'
s = pd.read_csv(SPIKE / f'cs_training_summary_{tag}.csv', parse_dates=['date'])

pace_s = s['cs_pace_med'] * 60.0
d_smi = (s['cs_train_pace_med'] - s['cs_pace_med']) * 60.0  # + = train slower
s['year'] = s['date'].dt.year

print('=== delta by year (s/mi at asymptotic pace; - = capability faster than race CS) ===')
for y, g in d_smi.groupby(s['year']):
    if y < 2016:
        continue
    print(f'  {y}: mean {g.mean():+6.2f}   min {g.min():+6.2f}   max {g.max():+6.2f}')

# Race-contradiction check on all kept races
rr = pd.read_csv(DEBUG / 'bayes_cs_residuals_spike_raceonly.csv', parse_dates=['date'])
p = pd.read_csv(SPIKE / 'bayes_cs_params_spike_raceonly.csv')
beta = float(p['beta_long_med'].iloc[0])
day = (s['date'] - s['date'].min()).dt.days.astype(float)
d_obs = (pd.DatetimeIndex(rr['date']) - s['date'].min()).days.astype(float)
dists = rr['distance_m'].to_numpy(float)
actual = rr['actual_sec'].to_numpy(float)
bias = np.where(dists > 10000, 1 + beta * np.log(np.maximum(dists, 10000) / 10000), 1.0)
preds = {}
for col, lbl in [('cs_mps_med', 'race-only'), ('cs_train_mps_med', 'train-informed')]:
    cs_at = np.interp(d_obs, day, s[col])
    dp_at = np.interp(d_obs, day, s['dp_med'])
    pred = (dists - dp_at) / cs_at * bias
    err = (actual / pred - 1) * 100
    preds[lbl] = err
    print(f'\n{lbl}: race mean|err| {np.mean(np.abs(err)):.2f}%   mean err {np.mean(err):+.2f}%')
worse = np.abs(preds['train-informed']) - np.abs(preds['race-only'])
print(f'races degraded >0.5%: {(worse > 0.5).sum()} / {len(rr)}; improved >0.5%: {(worse < -0.5).sum()}')
idx = np.argsort(worse)[-6:][::-1]
for i in idx:
    r = rr.iloc[i]
    print(f'  {r["date"].date()}  {int(r["distance_m"]):>6}m  '
          f'{preds["race-only"][i]:+.2f}% -> {preds["train-informed"][i]:+.2f}%  {r["event"][:38]}')

def at(datestr, col):
    i = (s['date'] - pd.Timestamp(datestr)).abs().idxmin()
    return float(s.loc[i, col])

def mss(m):
    sec = m * 60
    return f'{int(sec//60)}:{sec%60:04.1f}'

print('\n=== fall-2020 peak / spring-2021 trough (asymptotic CS pace) ===')
for dt_ in ('2020-10-15', '2020-11-21', '2021-04-15', '2021-07-01'):
    ro, ti = at(dt_, 'cs_pace_med'), at(dt_, 'cs_train_pace_med')
    print(f'  {dt_}: race-only {mss(ro)}/mi   train-informed {mss(ti)}/mi   ({(ti-ro)*60:+.1f} s/mi)')
w20 = s[(s['date'] >= '2020-08-01') & (s['date'] <= '2020-12-15')]
w21 = s[(s['date'] >= '2021-03-01') & (s['date'] <= '2021-08-01')]
for lbl, col in [('race-only', 'cs_pace_med'), ('train-informed', 'cs_train_pace_med')]:
    pk, tr = w20[col].min(), w21[col].max()
    print(f'  {lbl}: peak {mss(pk)}/mi  trough {mss(tr)}/mi  drop {(tr-pk)*60:+.1f} s/mi')

print('\n=== current reading ===')
today = s[s['date'] <= '2026-06-12'].iloc[-1]
print(f"  {today['date'].date()}: race-only {mss(today['cs_pace_med'])}/mi   "
      f"train-informed {mss(today['cs_train_pace_med'])}/mi   "
      f"delta {(today['cs_train_pace_med']-today['cs_pace_med'])*60:+.1f} s/mi")
print(f"  grid end {s['date'].iloc[-1].date()}: "
      f"race-only {mss(s['cs_pace_med'].iloc[-1])}/mi   "
      f"train-informed {mss(s['cs_train_pace_med'].iloc[-1])}/mi")

print('\n=== delta 95% CI width (s/mi) by era ===')
ciw = (s['cs_train_pace_hi95'] - s['cs_train_pace_lo95']) * 60
race_ciw = (s['cs_pace_hi95'] - s['cs_pace_lo95']) * 60
for y0, y1 in [(2016, 2019), (2020, 2022), (2023, 2026)]:
    m = (s['year'] >= y0) & (s['year'] <= y1)
    print(f'  {y0}-{y1}: delta-only CI {ciw[m].median():.1f}   '
          f'(race-only CS CI {race_ciw[m].median():.1f})')

# Influence: fastest obs vs anchor, with delta at their dates
obs = pd.read_csv(SPIKE / 'obs_full_prod.csv', parse_dates=['date'])
cs_at_o = np.interp((obs['date'] - s['date'].min()).dt.days.astype(float), day, s['cs_mps_med'])
log_anchor = np.log((5000 - obs['dp_fixed_m']) / cs_at_o)
obs['r_log'] = np.log(obs['t5k_sec']) - log_anchor
obs['r_smi'] = obs['resid']
d_at_o = np.interp((obs['date'] - s['date'].min()).dt.days.astype(float), day, d_smi)
obs['delta_here'] = d_at_o
corpus = pd.read_csv(SPIKE / 'tq_corpus.csv', parse_dates=['date'])
obs = obs.merge(corpus[['date', 'src', 'category', 'watch_verified', 'is_track', 'detail']],
                on=['date', 'src', 'category'], how='left', suffixes=('', '_c'))
print('\n=== influence list: 12 fastest points vs race CS (the load-bearing evidence) ===')
top = obs.nsmallest(12, 'r_smi')
for _, r in top.iterrows():
    ver = 'watch' if r.get('watch_verified') else ('track' if r.get('is_track') else 'other')
    print(f'  {r["date"].date()}  {r["src"]:<8} {str(r["category"]):<10} '
          f'resid {r["r_smi"]:+6.1f} s/mi  delta@date {r["delta_here"]:+5.1f}  '
          f'[{ver}]  {str(r["detail_c"])[:42]}')
