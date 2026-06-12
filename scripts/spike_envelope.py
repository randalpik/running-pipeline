"""Spike: performance-frontier envelope ("red line") constructor.

Demonstrated-capability frontier: every kept TQ point and every eligible
race 5K-equivalent is PROOF of 5K capability at its date. Each demonstration
projects a cone — capability at least c_i, receding backward in time at the
empirical max GAIN rate and forward at the empirical max DECAY rate (q90
rates from the race structure function, spike_cone_slopes.py). The frontier
is the upper envelope (fastest bound) of all cones. Cones do not propagate
across >90-day demonstration gaps (GAP_BREAK_DAYS convention — the 2020-21
labrum cliff stays a cliff).

Semantics: "the fastest 5K I could provably have run that day" — a race
PREDICTION line, not a CS estimate. No posterior, no CI: it is an envelope
over evidence, and its accuracy contract lives in TQ/race point selection.

Outputs performance_frontier.csv (date, frontier 5K pace, CS-implied 5K pace,
binding point id) + binding-point audit table to stdout.

One-off spike tooling (June 2026); productionize on approval.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
DEBUG = ROOT / 'output' / 'debug'

GAIN_PER_DAY = 0.000797   # log-time per day, backward reach (q90 gain)
DECAY_PER_DAY = 0.000475  # log-time per day, forward reach (q90 decay)
GAP_BREAK_DAYS = 90

# ---- demonstrations ----
rr = pd.read_csv(DEBUG / 'bayes_cs_residuals_spike_raceonly.csv', parse_dates=['date'])
p = pd.read_csv(SPIKE / 'bayes_cs_params_spike_raceonly.csv')
beta = float(p['beta_long_med'].iloc[0])
d = rr['distance_m'].to_numpy(float)
t = rr['actual_sec'].to_numpy(float)
dp = rr['dp_med_at_race'].to_numpy(float)
t_unb = np.where(d > 10000, t / (1 + beta * np.log(np.maximum(d, 10000) / 10000)), t)
races = pd.DataFrame({
    'date': rr['date'],
    'log_t5k': np.log((5000.0 - dp) * t_unb / (d - dp)),
    'src': 'race',
    'detail': rr['event'].astype(str) + ' (' + rr['distance_m'].astype(int).astype(str) + 'm)',
})

corpus = pd.read_csv(SPIKE / 'tq_corpus.csv', parse_dates=['date'])
tq = pd.DataFrame({
    'date': corpus['date'],
    'log_t5k': np.log((corpus['p5k_cs_min'] * 60 + corpus['resid']) * 5000 / 1609.344),
    'src': corpus['src'],
    'detail': corpus['detail'],
})
demos = pd.concat([races, tq], ignore_index=True).sort_values('date').reset_index(drop=True)
print(f'{len(demos)} demonstrations ({len(races)} races + {len(tq)} TQ points), '
      f'{demos["date"].min().date()} .. {demos["date"].max().date()}')

# ---- gap segments ----
dd = demos['date'].to_numpy('datetime64[D]').astype(float)
gaps = np.nonzero(np.diff(dd) > GAP_BREAK_DAYS)[0]
seg_bounds = [demos['date'].iloc[0]]
for g in gaps:
    print(f'  gap break: {demos["date"].iloc[g].date()} -> {demos["date"].iloc[g+1].date()} '
          f'({int(dd[g+1]-dd[g])} d)')
seg_starts = np.concatenate([[0], gaps + 1])
seg_ends = np.concatenate([gaps, [len(demos) - 1]])
demos['seg'] = 0
for s_i, (a, b) in enumerate(zip(seg_starts, seg_ends)):
    demos.loc[a:b, 'seg'] = s_i

# ---- envelope on the CS grid ----
cs = pd.read_csv(SPIKE / 'bayes_cs_summary_spike_raceonly.csv', parse_dates=['date'])
grid = cs[['date', 'cs_pace_med', 'cs_mps_med', 'dp_med']].copy()
gd = grid['date'].to_numpy('datetime64[D]').astype(float)
grid['cs_t5k'] = np.log((5000.0 - grid['dp_med']) / grid['cs_mps_med'])

env = np.full(len(grid), np.inf)
binder = np.full(len(grid), -1)
for i, row in demos.iterrows():
    ti = np.datetime64(row['date'], 'D').astype(float)
    back = row['log_t5k'] + GAIN_PER_DAY * (ti - gd)   # t < ti
    fwd = row['log_t5k'] + DECAY_PER_DAY * (gd - ti)   # t > ti
    cone = np.where(gd < ti, back, fwd)
    # restrict to this demo's segment (no crossing gap breaks)
    seg = row['seg']
    a = np.datetime64(demos[demos['seg'] == seg]['date'].iloc[0], 'D').astype(float)
    b = np.datetime64(demos[demos['seg'] == seg]['date'].iloc[-1], 'D').astype(float)
    cone = np.where((gd >= a - 1) & (gd <= b + GAP_BREAK_DAYS), cone, np.inf)
    take = cone < env
    env[take] = cone[take]
    binder[take] = i

valid = np.isfinite(env)
out = grid.copy()
out['frontier_log_t5k'] = np.where(valid, env, np.nan)
out['frontier_pace_min'] = np.exp(out['frontier_log_t5k']) * 1609.344 / 5000 / 60
out['cs_p5k_pace_min'] = np.exp(grid['cs_t5k']) * 1609.344 / 5000 / 60
out['binder'] = binder
out[['date', 'frontier_pace_min', 'cs_p5k_pace_min', 'binder']].to_csv(
    SPIKE / 'performance_frontier.csv', index=False)
print(f'Wrote {SPIKE / "performance_frontier.csv"} '
      f'({int(valid.sum())}/{len(out)} grid points covered)')

# ---- readings ----
def mss(m):
    if pd.isna(m):
        return '   — '
    s = m * 60
    return f'{int(s//60)}:{s%60:04.1f}'

print('\n=== key readings (5K-equivalent pace: frontier vs CS-implied) ===')
for dt_ in ('2017-05-15', '2019-07-15', '2020-10-15', '2020-11-21', '2021-04-15',
            '2022-04-15', '2022-11-12', '2023-05-06', '2024-06-15', '2025-10-12',
            '2026-06-06'):
    i = (out['date'] - pd.Timestamp(dt_)).abs().idxmin()
    f_, c_ = out.loc[i, 'frontier_pace_min'], out.loc[i, 'cs_p5k_pace_min']
    b = int(out.loc[i, 'binder'])
    btxt = (f"{demos.loc[b, 'date'].date()} {demos.loc[b, 'src']}: "
            f"{str(demos.loc[b, 'detail'])[:38]}") if b >= 0 else '—'
    print(f'  {dt_}: frontier {mss(f_)}/mi  CS-5K {mss(c_)}/mi  '
          f'({(f_-c_)*60:+6.1f} s/mi)  <- {btxt}')

print('\n=== frontier vs CS-implied 5K, share of grid faster ===')
m = valid & out['date'].ge('2016-01-01')
diff = (out.loc[m, 'frontier_pace_min'] - out.loc[m, 'cs_p5k_pace_min']) * 60
print(f'  2016+: frontier faster than CS-5K on {(diff < 0).mean()*100:.0f}% of days; '
      f'median {diff.median():+.1f} s/mi; p5 {diff.quantile(0.05):+.1f}; '
      f'p95 {diff.quantile(0.95):+.1f}')

# ---- binding audit ----
print('\n=== binding points (define the frontier somewhere), 2016+ ===')
bind_ids = [b for b in pd.unique(out.loc[valid, 'binder']) if b >= 0]
rows = []
for b in bind_ids:
    days = out[out['binder'] == b]
    if days['date'].max() < pd.Timestamp('2016-01-01'):
        continue
    rows.append((demos.loc[b, 'date'], demos.loc[b, 'src'],
                 str(demos.loc[b, 'detail'])[:44], len(days),
                 days['date'].min().date(), days['date'].max().date()))
rows.sort(key=lambda r: r[0])
print(f'{"demo date":<12}{"src":<9}{"detail":<46}{"grid-days bound":>16}  span')
for dt_, src, det, n, a, b_ in rows:
    print(f'{str(dt_.date()):<12}{src:<9}{det:<46}{n:>16}  {a}..{b_}')
print(f'\n{len(rows)} binding demonstrations (2016+) of {len(demos)} total')
