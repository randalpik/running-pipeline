"""Dump physical betas + recovery-fit residual stats for a before/after diff."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.shared.recovery_model import physical_route_betas, fit_recovery_model

tag = sys.argv[1]
pb = physical_route_betas()
daily = pd.read_csv('data/daily.csv', parse_dates=['date'])
races = pd.read_csv('data/races.csv', parse_dates=['date'])
cs = pd.read_csv('data/bayes_cs_summary.csv', parse_dates=['date'])
if 'cs_pace_sec' not in cs.columns:
    cs['cs_pace_sec'] = cs['cs_pace_med'] * 60.0
fr = fit_recovery_model(daily, races, cs, verbose=False)
rec = fr.rec[~fr.rec['is_pruned']]
print(f'[{tag}] betas is_offroad={pb["is_offroad"]:.3f} alt_kft={pb["alt_kft"]:.3f} '
      f'| recovery n={len(rec)} resid_sd={rec["residual_raw"].std():.3f}')
rec[['date', 'residual_raw']].to_csv(f'/tmp/recfit_{tag}.csv', index=False)
