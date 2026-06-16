"""Dump project_long_runs key columns + the pooled physical betas to a CSV tag,
so we can diff barometric vs DEM long-run elevation. Usage: capture <tag>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.shared.workouts import load_cs, project_long_runs
from src.shared.recovery_model import physical_route_betas

tag = sys.argv[1]
cs, epoch = load_cs()
lr = project_long_runs(cs, epoch)
keep = ['date', 'location', 'terrain_type', 'miles', 'excluded_reason',
        'elev_gain_pm', 'elev_loss_pm', 'grade_cost_s_per_mi', 'alt_cost_s_per_mi',
        'p5k_min', 'p5k_cs_min', 'raw_resid']
lr[keep].to_csv(f'/tmp/lrproj_{tag}.csv', index=False)
pb = physical_route_betas()
print(f'[{tag}] rows={len(lr)} in-slice={lr["excluded_reason"].isna().sum()} '
      f'betas: is_offroad={pb["is_offroad"]:.3f} alt_kft={pb["alt_kft"]:.3f}')
