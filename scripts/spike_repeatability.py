"""Spike: selection-free noise benchmark via close-in-time pairs.

The in-band sd is biased low by the band truncation (selection on the very
quantity being measured). A selection-free estimate: for points of the same
source within K days of each other, the SD of their residual DIFFERENCE / sqrt(2)
estimates the short-horizon repeatability noise — true fitness barely moves
inside the window, and the CS curve cancels out of the difference. Apply the
same estimator to races for an apples-to-apples sigma_race.

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
K_DAYS = 21


def pair_sigma(dates, fracs, k_days=K_DAYS):
    """sd(diff)/sqrt(2) over all pairs within k_days (each point can appear
    in multiple pairs; fine for an estimate). Returns (sigma, n_pairs)."""
    d = np.asarray(dates, dtype='datetime64[D]').astype(float)
    f = np.asarray(fracs, dtype=float)
    order = np.argsort(d)
    d, f = d[order], f[order]
    diffs = []
    for i in range(len(d)):
        j = i + 1
        while j < len(d) and d[j] - d[i] <= k_days:
            if d[j] > d[i]:  # skip same-day duplicates (doubles)
                diffs.append(f[j] - f[i])
            j += 1
    if len(diffs) < 5:
        return float('nan'), len(diffs)
    return float(np.std(diffs) / np.sqrt(2)), len(diffs)


corpus = pd.read_csv(SPIKE / 'tq_corpus.csv', parse_dates=['date'])
corpus['frac'] = corpus['resid'] / (corpus['p5k_cs_min'] * 60.0)
B = float(-corpus['resid'].min())
band = corpus[corpus['resid'].abs() <= B]

print(f'Pair-based sigma (window {K_DAYS}d), fractional log-time scale:')
for label, g in [('corpus ALL', corpus), ('in-band ALL', band)]:
    s, n = pair_sigma(g['date'].values, g['frac'].values)
    print(f'  {label:<22} sigma={s:.4f}  ({n} pairs)')
for src in ['workout', 'long_run', 'hill']:
    g = corpus[corpus['src'] == src]
    s, n = pair_sigma(g['date'].values, g['frac'].values)
    gb = band[band['src'] == src]
    sb, nb = pair_sigma(gb['date'].values, gb['frac'].values)
    print(f'  {src:<22} full sigma={s:.4f} ({n} pairs) | in-band sigma={sb:.4f} ({nb} pairs)')

# Watch-verified workouts only (highest-quality slice)
g = corpus[(corpus['src'] == 'workout') & corpus['watch_verified']]
s, n = pair_sigma(g['date'].values, g['frac'].values)
print(f'  {"workout watch-only":<22} full sigma={s:.4f} ({n} pairs)')

# Races, same estimator, same scale (log residual ~ frac deviation)
rr = pd.read_csv(ROOT / 'output' / 'debug' / 'bayes_cs_residuals.csv',
                 parse_dates=['date'])
rr['frac'] = np.log(rr['actual_sec'] / rr['predicted_sec'])
s, n = pair_sigma(rr['date'].values, rr['frac'].values)
print(f'  {"races ALL":<22} sigma={s:.4f} ({n} pairs)')
sub = rr[rr['distance_m'] < 8000]
s, n = pair_sigma(sub['date'].values, sub['frac'].values)
print(f'  {"races <8K":<22} sigma={s:.4f} ({n} pairs)')
