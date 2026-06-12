"""Spike: build the near-race workout observation CSV for bayes_cs_fit
--workout-obs.

Each in-band TQ point becomes a 5K-equivalent observation:
    t5k_sec    = (p5k_cs_pace_s + corrected_resid) * 5000/1609.344
i.e. the corrected residual the TQ smoother consumes, re-expressed as an
absolute 5K time at that date. dp_fixed_m is the production race-fit D'
median interpolated to the date (the same D' the projection itself used).
sigma_obs is the per-source pair-based repeatability sigma measured by
spike_repeatability.py (selection-free, June 2026):
    workout 0.0272, long_run 0.0325, hill 0.0336
optionally scaled by --sigma-scale for sensitivity variants.

One-off spike tooling (June 2026).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.shared.workouts import load_cs

SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'

SIGMA_BY_SRC = {'workout': 0.0272, 'long_run': 0.0325, 'hill': 0.0336}

p = argparse.ArgumentParser()
p.add_argument('--sigma-scale', type=float, default=1.0)
p.add_argument('--corpus', default=str(SPIKE / 'tq_corpus.csv'))
p.add_argument('--cs-summary', default='',
               help='Alternate bayes_cs_summary CSV for dp_fixed_m '
                    '(default: production)')
p.add_argument('--band', type=float, default=0.0,
               help='Fixed band half-width s/mi (default 0 = derive from '
                    'the corpus fastest residual)')
p.add_argument('--out', default=str(SPIKE / 'workout_obs.csv'))
args = p.parse_args()

corpus = pd.read_csv(args.corpus, parse_dates=['date'])
B = args.band if args.band > 0 else float(-corpus['resid'].min())
band = corpus[corpus['resid'].abs() <= B].copy()

if args.cs_summary:
    import src.shared.workouts as workouts_mod
    workouts_mod.CS_PATH = Path(args.cs_summary)
cs, epoch = load_cs()
band['day'] = (band['date'] - epoch).dt.days.astype(float)
band['dp_fixed_m'] = np.interp(band['day'], cs['day'].values, cs['dp_med'].values)

pace_s = band['p5k_cs_min'] * 60.0 + band['resid']
band['t5k_sec'] = pace_s * 5000.0 / 1609.344
band['sigma_obs'] = band['src'].map(SIGMA_BY_SRC) * args.sigma_scale

out_cols = ['date', 't5k_sec', 'dp_fixed_m', 'sigma_obs', 'src', 'category', 'resid', 'detail']
band[out_cols].to_csv(args.out, index=False)
print(f'Wrote {args.out}: {len(band)} obs (band ±{B:.2f} s/mi, '
      f'sigma scale {args.sigma_scale})')
print(band.groupby('src')['sigma_obs'].agg(['size', 'first']))
