"""Report what changes if watch-era long runs move from barometric to DEM
elevation. Reads the spike output and the live elevation engine to translate
elevation deltas into pace-cost / projection deltas."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.shared.recovery_model import (physical_route_betas, altitude_regressor)
from src.shared.elevation_cost import elevation_cost, paved_refund, REFUND_RECOVERY

df = pd.read_csv('/tmp/long_run_dem_spike.csv')
d = df[df['dem_mean_ft'].notna()].copy()
print(f'{len(df)} long runs, {len(d)} with a DEM track\n')

# ---- altitude (hypoxia) channel ----
d['baro_alt_kft'] = d['baro_mean_ft'].fillna(d['const_alt_ft']) / 1000.0
d['dem_alt_kft'] = d['dem_mean_ft'] / 1000.0
pb = physical_route_betas()
alt_beta = pb['alt_kft']
d['alt_cost_baro'] = alt_beta * altitude_regressor(d['baro_alt_kft'])
d['alt_cost_dem'] = alt_beta * altitude_regressor(d['dem_alt_kft'])
d['alt_cost_delta'] = d['alt_cost_dem'] - d['alt_cost_baro']

# ---- grade channel ----
terr = d['terrain'].astype(str).str.strip().str.lower()
terr = terr.where(terr.isin(['paved', 'mixed', 'trail']), 'paved')
refund = terr.map(REFUND_RECOVERY).astype(float).to_numpy(copy=True)
refund[(terr == 'paved').to_numpy()] = paved_refund(1.0)  # ~race effort for long
for src in ('baro', 'dem'):
    g = d[f'{src}_gain_pm'].fillna(0).to_numpy()
    l = d[f'{src}_loss_pm'].fillna(0).to_numpy()
    d[f'grade_cost_{src}'] = elevation_cost(g, l, terr.to_numpy(), refund=refund)
d['grade_cost_delta'] = d['grade_cost_baro'] - d['grade_cost_dem']  # baro - dem

pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: f'{x:7.1f}')

print('=== MEAN ELEVATION (hypoxia term) ===')
print('mean |baro-dem| ft:', round((d['baro_mean_ft'] - d['dem_mean_ft']).abs().mean(), 1))
big = d.reindex((d['baro_mean_ft'] - d['dem_mean_ft']).abs().sort_values(ascending=False).index)
print(big[['date', 'location', 'baro_mean_ft', 'dem_mean_ft', 'const_alt_ft',
           'alt_cost_delta']].head(12).to_string(index=False))

print('\n=== GRADE (gain/loss per mile) ===')
d['gain_dpm'] = d['baro_gain_pm'] - d['dem_gain_pm']
print('mean baro gain/mi:', round(d['baro_gain_pm'].mean(), 1),
      '| mean dem gain/mi:', round(d['dem_gain_pm'].mean(), 1))
print('mean baro loss/mi:', round(d['baro_loss_pm'].mean(), 1),
      '| mean dem loss/mi:', round(d['dem_loss_pm'].mean(), 1))
big2 = d.reindex(d['grade_cost_delta'].abs().sort_values(ascending=False).index)
print(big2[['date', 'location', 'terrain', 'baro_gain_pm', 'dem_gain_pm',
            'baro_loss_pm', 'dem_loss_pm', 'grade_cost_delta']].head(12).to_string(index=False))

print('\n=== NET PROJECTION-PACE IMPACT (s/mi removed from run time) ===')
# total physical credit delta = how much faster/slower the flat-equivalent pace
# becomes. positive = DEM makes the run project FASTER than baro did.
d['total_delta'] = d['grade_cost_delta'] + d['alt_cost_delta']
print('per-run |total delta| s/mi: mean',
      round(d['total_delta'].abs().mean(), 2),
      '| max', round(d['total_delta'].abs().max(), 2))
print(d.reindex(d['total_delta'].abs().sort_values(ascending=False).index)
      [['date', 'location', 'terrain', 'alt_cost_delta', 'grade_cost_delta',
        'total_delta']].head(15).to_string(index=False))
