"""Spike: derive performance-frontier cone slopes from the demonstrations.

Structure-function analysis: for pairs of max-effort demonstrations at time
lag dt, E[(c_j - c_i)^2] = 2*sigma_noise^2 + (real capability change over dt)^2.
The short-lag plateau (nugget) is repeatability noise — independently known
(~0.029 for races, June 2026 pair analysis) — and the growth beyond it is the
true rate at which 5K capability moves. Races are the primary input (max
effort by assumption); kept TQ points run alongside for comparison (their
nugget includes effort slack, so only the SHAPE is comparable).

Signed split: pairs where the later demo is faster measure the GAIN
direction; later-slower measures DECAY (for races, both ends are capability;
for workouts, decay pairs are slack-confounded — read races only).

Output: per-lag-bin table + suggested cone slopes (s/mi per week at the
corpus median 5K pace).

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
DEBUG = ROOT / 'output' / 'debug'

# ---- demonstrations: races' 5K-equivalents (fit conventions) ----
rr = pd.read_csv(DEBUG / 'bayes_cs_residuals_spike_raceonly.csv', parse_dates=['date'])
p = pd.read_csv(SPIKE / 'bayes_cs_params_spike_raceonly.csv')
beta = float(p['beta_long_med'].iloc[0])
d = rr['distance_m'].to_numpy(float)
t = rr['actual_sec'].to_numpy(float)  # already XC-corrected
dp = rr['dp_med_at_race'].to_numpy(float)
t_unb = np.where(d > 10000, t / (1 + beta * np.log(np.maximum(d, 10000) / 10000)), t)
t5k = (5000.0 - dp) * t_unb / (d - dp)
races = pd.DataFrame({'date': rr['date'], 'c': np.log(t5k)}).sort_values('date')

# TQ corpus comparison set
corpus = pd.read_csv(SPIKE / 'tq_corpus.csv', parse_dates=['date'])
corpus['c'] = np.log((corpus['p5k_cs_min'] * 60 + corpus['resid']) * 5000 / 1609.344)
tq = corpus[['date', 'c']].sort_values('date')

MED_PACE = float((corpus['p5k_cs_min'] * 60).median())  # s/mi scale factor

BINS = [(0, 10), (10, 21), (21, 35), (35, 56), (56, 91), (91, 140), (140, 210), (210, 365)]


def pairs(df, max_lag=365):
    dts, dcs = [], []
    dates = df['date'].to_numpy('datetime64[D]').astype(float)
    cs = df['c'].to_numpy(float)
    for i in range(len(df)):
        j = i + 1
        while j < len(df) and dates[j] - dates[i] <= max_lag:
            if dates[j] > dates[i]:
                dts.append(dates[j] - dates[i])
                dcs.append(cs[j] - cs[i])
            j += 1
    return np.array(dts), np.array(dcs)


def table(name, dts, dcs):
    print(f'\n=== {name}: structure function ===')
    print(f'{"lag bin (d)":<14}{"n":>6}{"rms dc":>10}{"implied |drift|":>18}'
          f'{"gain q90":>12}{"decay q90":>12}')
    rows = []
    for lo, hi in BINS:
        m = (dts >= lo) & (dts < hi)
        if m.sum() < 8:
            continue
        rms = float(np.sqrt(np.mean(dcs[m] ** 2)))
        # drift implied after removing the lag->0 nugget (estimated from
        # the first populated bin)
        rows.append((lo, hi, int(m.sum()), rms,
                     float(np.quantile(-dcs[m][dcs[m] < 0], 0.9)) if (dcs[m] < 0).any() else np.nan,
                     float(np.quantile(dcs[m][dcs[m] > 0], 0.9)) if (dcs[m] > 0).any() else np.nan))
    nugget = rows[0][3] if rows else np.nan
    for lo, hi, n, rms, gq, dq in rows:
        drift2 = max(rms ** 2 - nugget ** 2, 0.0)
        drift = np.sqrt(drift2)
        mid_wk = (lo + hi) / 2 / 7
        per_wk = drift / mid_wk * MED_PACE if mid_wk > 0 else np.nan
        print(f'{lo:>4}-{hi:<9}{n:>6}{rms:>10.4f}{drift:>10.4f} '
              f'({per_wk:>4.1f} s/mi/wk){gq:>12.4f}{dq:>12.4f}')
    print(f'  nugget (first bin rms) = {nugget:.4f} '
          f'(~{nugget/np.sqrt(2)*1:.4f} per-obs sd; expect ~0.029 for races)')
    return rows, nugget


dts_r, dcs_r = pairs(races)
rows_r, nug_r = table('RACES (max-effort demonstrations)', dts_r, dcs_r)

dts_t, dcs_t = pairs(tq)
table('TQ points (slack-confounded; shape comparison only)', dts_t, dcs_t)

# Suggested slopes: fit sqrt(rms^2 - nugget^2) ~ rate * lag on the race
# bins beyond the nugget zone (>= 21d), separately for signed directions
# via signed q90 growth.
print('\n=== suggested cone slopes (races, drift fit over 21-210d bins) ===')
mids, drifts = [], []
for lo, hi, n, rms, gq, dq in rows_r:
    if lo < 21 or lo >= 210:
        continue
    mids.append((lo + hi) / 2)
    drifts.append(np.sqrt(max(rms ** 2 - nug_r ** 2, 0)))
mids, drifts = np.array(mids), np.array(drifts)
rate = float(np.sum(mids * drifts) / np.sum(mids ** 2))  # through-origin LS
print(f'  rms drift rate: {rate:.6f} /day = {rate * 7 * MED_PACE:.2f} s/mi per week '
      f'(at median pace {MED_PACE:.0f} s/mi)')

# Signed: growth of the q90 envelope per direction
for lbl, col in [('gain (later faster)', 4), ('decay (later slower)', 5)]:
    m_, d_ = [], []
    for r_ in rows_r:
        if r_[0] < 21 or r_[0] >= 210 or np.isnan(r_[col]):
            continue
        m_.append((r_[0] + r_[1]) / 2)
        d_.append(np.sqrt(max(r_[col] ** 2 - nug_r ** 2, 0)))
    m_, d_ = np.array(m_), np.array(d_)
    if len(m_) >= 2:
        rt = float(np.sum(m_ * d_) / np.sum(m_ ** 2))
        print(f'  {lbl:<22}: q90 {rt:.6f} /day = {rt * 7 * MED_PACE:.2f} s/mi per week')
