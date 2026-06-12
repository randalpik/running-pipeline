"""Spike: compare race-only vs near-race-enriched CS fits.

Reads bayes_cs_summary_{tagA,tagB}.csv from data/ and the corresponding
diagnostics/residuals from output/debug/. Reports:
  - CS curve shift (median/max |delta|, overall and by year)
  - shift at race dates vs race-sparse dates (>60d from nearest kept race)
  - 95% CI width change (between-race sharpening — the intended value)
  - D' shift (should be ~0; Gate 2 check)
  - beta_long shift
  - race residuals under each fit (does enrichment degrade race fit?)

One-off spike tooling (June 2026).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DEBUG = ROOT / 'output' / 'debug'

p = argparse.ArgumentParser()
p.add_argument('--base', default='spike_raceonly')
p.add_argument('--enr', default='spike_nearrace')
args = p.parse_args()

a = pd.read_csv(DATA / f'bayes_cs_summary_{args.base}.csv', parse_dates=['date'])
b = pd.read_csv(DATA / f'bayes_cs_summary_{args.enr}.csv', parse_dates=['date'])
assert len(a) == len(b) and (a['date'] == b['date']).all(), 'grid mismatch'

# CS pace shift in s/mi (positive = enriched is SLOWER)
d_pace = (b['cs_pace_med'] - a['cs_pace_med']) * 60.0
print(f'=== CS curve shift ({args.enr} - {args.base}), s/mi at asymptotic CS pace ===')
print(f'  median |shift| = {d_pace.abs().median():.2f}   '
      f'mean = {d_pace.mean():+.2f}   '
      f'max |shift| = {d_pace.abs().max():.2f} '
      f'on {a.loc[d_pace.abs().idxmax(), "date"].date()}')

a['year'] = a['date'].dt.year
by_year = pd.DataFrame({'year': a['year'], 'd': d_pace})
print('\n  by year (mean shift, max |shift|):')
for y, g in by_year.groupby('year'):
    print(f'    {y}: {g["d"].mean():+6.2f}  max|{g["d"].abs().max():5.2f}|')

# Race-anchored vs race-sparse
rr = pd.read_csv(DEBUG / f'bayes_cs_residuals_{args.base}.csv', parse_dates=['date'])
race_days = rr['date'].values.astype('datetime64[D]').astype(float)
grid_days = a['date'].values.astype('datetime64[D]').astype(float)
dist_to_race = np.array([np.min(np.abs(race_days - g)) for g in grid_days])
near = dist_to_race <= 30
sparse = dist_to_race > 60
print(f'\n=== Race-anchored vs race-sparse ===')
print(f'  within 30d of a race  (n={near.sum()}): mean {d_pace[near].mean():+.2f}, '
      f'median |{d_pace[near].abs().median():.2f}|, max |{d_pace[near].abs().max():.2f}|')
print(f'  >60d from any race    (n={sparse.sum()}): mean {d_pace[sparse].mean():+.2f}, '
      f'median |{d_pace[sparse].abs().median():.2f}|, max |{d_pace[sparse].abs().max():.2f}|')

# CI width change
wa = (a['cs_pace_hi95'] - a['cs_pace_lo95']) * 60
wb = (b['cs_pace_hi95'] - b['cs_pace_lo95']) * 60
print(f'\n=== 95% CI width (s/mi) ===')
print(f'  race-only: median {wa.median():.1f}   enriched: median {wb.median():.1f}   '
      f'median change {(wb - wa).median():+.1f}')
print(f'  at race-anchored points: {(wb - wa)[near].median():+.1f}   '
      f'at race-sparse points: {(wb - wa)[sparse].median():+.1f}')

# D'
dd = b['dp_med'] - a['dp_med']
print(f'\n=== D\' shift (m) ===')
print(f'  median |shift| = {dd.abs().median():.1f}   max |shift| = {dd.abs().max():.1f}   '
      f'mean = {dd.mean():+.1f}')
print(f'  race-only D\' median over grid: {a["dp_med"].median():.0f}m, '
      f'enriched: {b["dp_med"].median():.0f}m')

# beta_long
pa = pd.read_csv(DATA / f'bayes_cs_params_{args.base}.csv')
pb = pd.read_csv(DATA / f'bayes_cs_params_{args.enr}.csv')
print(f'\n=== beta_long ===')
print(f'  race-only {pa["beta_long_med"].iloc[0]:.4f}  ->  '
      f'enriched {pb["beta_long_med"].iloc[0]:.4f}')

# Race residuals under each fit
rb = pd.read_csv(DEBUG / f'bayes_cs_residuals_{args.enr}.csv', parse_dates=['date'])
m = rr.merge(rb, on=['date', 'distance_m'], suffixes=('_a', '_b'))
print(f'\n=== Race residuals (pct, abs) ===')
print(f'  race-only:  mean|resid| {m["pct_resid_a"].abs().mean():.2f}%   '
      f'sd {m["pct_resid_a"].std():.2f}%')
print(f'  enriched:   mean|resid| {m["pct_resid_b"].abs().mean():.2f}%   '
      f'sd {m["pct_resid_b"].std():.2f}%')
worse = (m['pct_resid_b'].abs() - m['pct_resid_a'].abs())
print(f'  races whose |resid| grew >0.5%: {(worse > 0.5).sum()} / {len(m)}; '
      f'shrank >0.5%: {(worse < -0.5).sum()}')
top = m.assign(w=worse).nlargest(5, 'w')
for _, r in top.iterrows():
    print(f'    {r["date"].date()}  {int(r["distance_m"]):>6}m  '
          f'{r["pct_resid_a"]:+.2f}% -> {r["pct_resid_b"]:+.2f}%  {r["event_a"][:35]}')

# Largest curve moves: where and which way
print(f'\n=== 10 largest |CS shifts| ===')
idx = d_pace.abs().nlargest(10).index
for i in sorted(idx):
    print(f'  {a.loc[i, "date"].date()}  {d_pace[i]:+6.2f} s/mi  '
          f'(dist to nearest race {dist_to_race[i]:.0f}d)')
