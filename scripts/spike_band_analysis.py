"""Spike: near-race band selection analysis + Gate 1 sigma benchmark.

Band: |corrected resid| <= |min corrected resid| over the kept TQ corpus
(the user's rule: the fastest point defines how far "near race-derived CS"
extends in both directions).

Gate 1: compare the band-selected workout scatter against per-race residual
scatter from the production race-only Bayes fit, both expressed as
fractional (log-time) deviations so the units match.

One-off spike tooling (June 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'
corpus = pd.read_csv(SPIKE / 'tq_corpus.csv', parse_dates=['date'])

B = float(-corpus['resid'].min())
print(f'Band half-width B = {B:.2f} s/mi (fastest corrected residual)')

band = corpus[corpus['resid'].abs() <= B].copy()
out = corpus[corpus['resid'].abs() > B].copy()
print(f'\nIn-band: {len(band)} / {len(corpus)} points '
      f'({len(band)/len(corpus)*100:.0f}%)')

print('\n--- In-band composition by source ---')
print(band.groupby('src').agg(n=('resid', 'size'),
                              med=('resid', 'median'),
                              sd=('resid', 'std')).round(2))

print('\n--- In-band composition by category ---')
print(band.groupby('category').agg(n=('resid', 'size'),
                                   med=('resid', 'median'),
                                   sd=('resid', 'std')).round(2))

band['year'] = band['date'].dt.year
corpus['year'] = corpus['date'].dt.year
print('\n--- In-band by year: n, share of that year\'s corpus, type mix ---')
for y, g in band.groupby('year'):
    total_y = (corpus['year'] == y).sum()
    mix = g.groupby('src').size().to_dict()
    mix_s = ', '.join(f'{k}={v}' for k, v in sorted(mix.items()))
    print(f'  {y}: {len(g):3d} / {total_y:3d} in band ({len(g)/total_y*100:3.0f}%)  [{mix_s}]')

print('\n--- In-band long runs by year (the era-shift check) ---')
lr = corpus[corpus['src'] == 'long_run'].copy()
lr['year'] = lr['date'].dt.year
for y, g in lr.groupby('year'):
    inb = (g['resid'].abs() <= B).sum()
    neg = (g['resid'] < 0).sum()
    print(f'  {y}: {inb:2d}/{len(g):2d} in band, {neg} faster than CS, '
          f'median resid {g["resid"].median():+6.1f}')

print('\n--- Watch verification of in-band points ---')
print(band.groupby(['src', 'watch_verified']).size())

# ---------- Gate 1 ----------
# sigma_workout: in-band corrected residuals, converted to fractional
# deviation of the 5K-equivalent time: resid_s_per_mi / cs_pace_s_per_mi.
band['frac'] = band['resid'] / (band['p5k_cs_min'] * 60.0)
corpus['frac'] = corpus['resid'] / (corpus['p5k_cs_min'] * 60.0)

print('\n=== Gate 1: sigma benchmark (fractional log-time scale) ===')
print(f"in-band ALL:      sd={band['frac'].std():.4f}  n={len(band)}  "
      f"mean={band['frac'].mean():+.4f}")
for s, g in band.groupby('src'):
    print(f"in-band {s:<9}: sd={g['frac'].std():.4f}  n={len(g)}  "
          f"mean={g['frac'].mean():+.4f}")
print(f"full corpus:      sd={corpus['frac'].std():.4f}  n={len(corpus)}  "
      f"mean={corpus['frac'].mean():+.4f}")

# Truncation correction: if the underlying population were Normal(mu, sigma)
# and we keep |x| <= B, the observed sd understates sigma. Report the
# truncated-normal MLE sigma for the in-band set as the honest noise figure.
from scipy.stats import truncnorm
from scipy.optimize import minimize

def trunc_mle(x, B_frac):
    def nll(params):
        mu, log_sd = params
        sd = np.exp(log_sd)
        a, b = (-B_frac - mu) / sd, (B_frac - mu) / sd
        return -np.sum(truncnorm.logpdf(x, a, b, loc=mu, scale=sd))
    x = np.asarray(x)
    res = minimize(nll, [x.mean(), np.log(x.std())], method='Nelder-Mead')
    return res.x[0], float(np.exp(res.x[1]))

# B in fractional units varies per point (pace varies); use median pace.
med_pace = float((corpus['p5k_cs_min'] * 60).median())
B_frac = B / med_pace
mu_t, sd_t = trunc_mle(band['frac'].values, B_frac)
print(f"\nTruncated-normal MLE on in-band set (B_frac={B_frac:.4f}): "
      f"mu={mu_t:+.4f}, sigma={sd_t:.4f}")

# sigma_race: per-race residuals from the production race-only fit.
resid_path = ROOT / 'output' / 'debug' / 'bayes_cs_residuals.csv'
rr = pd.read_csv(resid_path, parse_dates=['date'])
rr['log_resid'] = np.log(rr['actual_sec'] / rr['predicted_sec'])
print(f"\nsigma_race (per-race log residuals, May-1 posterior, n={len(rr)}): "
      f"sd={rr['log_resid'].std():.4f}  mean={rr['log_resid'].mean():+.4f}")
for lo, hi, lbl in [(0, 8000, '<8K'), (8000, 15000, '8-15K'),
                    (15000, 100000, 'HM+')]:
    g = rr[(rr['distance_m'] >= lo) & (rr['distance_m'] < hi)]
    print(f"  {lbl:<6}: sd={g['log_resid'].std():.4f}  n={len(g)}")

band.to_csv(SPIKE / 'near_race_band.csv', index=False)
print(f"\nWrote {SPIKE / 'near_race_band.csv'}")
