"""Full decomposition for the 2023-05-06 long run on the Training graph."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.shared.workouts import load_cs, project_long_runs
from src.shared.long_run_model import fit_long_run_model

pd.set_option('display.width', 200)
TARGET = pd.Timestamp('2023-05-06').date()

cs, epoch = load_cs()
lr = project_long_runs(cs, epoch)
lr = lr[lr['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
lr2, fit, _ = fit_long_run_model(lr)

print(f'fit: intercept={fit.intercept:+.2f} (NOT applied in TQ)  temp_ref={fit.temp_ref:.2f}')
print(f'cov_coefs: {", ".join(f"{k}={v:+.3f}" for k,v in fit.cov_coefs.items())}')
print(f'phys_coefs: {fit.phys_coefs}')

row = lr2[pd.to_datetime(lr2['date']).dt.date == TARGET]
if row.empty:
    print('NOT in fit set; checking full projection...')
    row = lr[pd.to_datetime(lr['date']).dt.date == TARGET]
r = row.iloc[0]

def g(k, fmt='{:+.2f}'):
    v = r.get(k)
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)

print(f'\n=== 2023-05-06 ===')
for k in ['miles', 'location', 'temp_c', 'recovery_pace_sec_per_mi',
          'lr_watch', 'lr_rule', 'pause_s']:
    print(f'  {k:28s} {g(k)}')
print('  --- per-run physical costs (s/mi, priced upstream in project_long_runs) ---')
for k in ['grade_cost_s_per_mi', 'footing_cost_s_per_mi', 'alt_cost_s_per_mi',
          'elev_gain_pm', 'elev_loss_pm', 'terrain_type', 'altitude']:
    print(f'  {k:28s} {g(k)}')
print('  --- model decomposition ---')
for k in ['raw_resid', 'phys_contrib', 'cov_contrib', 'temp_centered',
          'fat_marathon', 'fat_race_short', 'model_offset', 'corrected',
          'p5k_min', 'p5k_cs_min', 'is_outlier']:
    print(f'  {k:28s} {g(k)}')

# temp contribution breakdown
tc = fit.cov_coefs.get('temp_centered', 0.0)
print(f'\n  temp contribution = {tc:+.3f} * {g("temp_centered","{:.2f}")} = '
      f'{tc*float(r["temp_centered"]):+.2f} s/mi (subtracted from raw_resid)')
print(f'  (air temp {g("temp_c","{:.0f}")}C -> hinge {max(0,float(r["temp_c"])-6):.1f}'
      f' -> centered {float(r["temp_centered"]):.2f})')

# --- projection internals ---
from src.shared.workouts import _load_beta_long, METERS_PER_MILE
beta_long, d_thresh = _load_beta_long()
import math
print(f'\n=== PROJECTION INTERNALS (2023-05-06) ===')
for k in ['corr_miles', 'corr_time_s', 'd_m', 'd_eff_m', 't_run', 'effort', 'dp3_t']:
    print(f'  {k:18s} {g(k, "{:.1f}")}')
dm = float(r['d_m']); tr = float(r['t_run'])
movpace = tr / (dm / METERS_PER_MILE)
ps = float(r.get('pause_s') or 0)
elapsed = tr + ps
print(f'  moving pace        {movpace:.0f} s/mi = {int(movpace//60)}:{movpace%60:04.1f}/mi')
print(f'  pause              {ps:.0f}s ({ps/elapsed*100:.0f}% of {elapsed/60:.0f} min elapsed)')
print(f'  elapsed pace       {elapsed/(dm/METERS_PER_MILE):.0f} s/mi (incl. pauses)')
deff = float(r['d_eff_m']) if pd.notna(r.get('d_eff_m')) else dm
print(f'  d_eff/d_m          {deff/dm:.3f}  (pause-aware fatigue distance shrinks {(1-deff/dm)*100:.0f}%)')
beta = 1.0 + beta_long*math.log(max(dm, d_thresh)/d_thresh) if dm > d_thresh else 1.0
print(f'  beta_long un-bias  beta={beta:.3f}  (divides time -> speeds equiv by ~{(1-1/beta)*100:.0f}%)')
print(f'  d_thresh           {d_thresh:.0f}m   beta_long={beta_long:.4f}')

def mmss(pace_min):
    s = pace_min*60.0; return f'{int(s//60)}:{s%60:04.1f}/mi'
def t5k(pace_min):
    s = pace_min*60.0*5000.0/METERS_PER_MILE; return f'{int(s//60)}:{s%60:04.1f}'
print(f'\n  CS-implied 5K pace (this date): {mmss(r["p5k_cs_min"])}  -> 5K {t5k(r["p5k_cs_min"])}')
print(f'  PROJECTED from long run:        {mmss(r["p5k_min"])}  -> 5K {t5k(r["p5k_min"])}')

# actual 5K races for comparison
races = pd.read_csv('data/races.csv', parse_dates=['date'])
fivek = races[(races['distance_m'].between(4900, 5100))].copy()
fivek['pace_s'] = fivek['time_sec']/ (fivek['distance_m']/METERS_PER_MILE) if 'time_sec' in fivek else np.nan
print(f'\n=== actual 5K races (fastest 5) ===')
tcol = 'time_sec' if 'time_sec' in fivek else ('seconds' if 'seconds' in fivek else None)
if tcol:
    fivek = fivek.sort_values(tcol)
    for _, x in fivek.head(5).iterrows():
        tt = x[tcol]; print(f'  {x["date"].date()}  {int(tt//60)}:{tt%60:04.1f}  {x.get("city_state","")}')
else:
    print('  (time column not found; cols:', list(races.columns)[:12], ')')

# --- DISPLAYED value (what the graph plots): CS-implied + (raw_resid - model_adj) ---
from src.shared.workouts import METERS_PER_MILE as _MPM
def t5k_from_pacemin(pace_min):
    s = pace_min*60.0*5000.0/_MPM
    return f'{int(s//60)}:{s%60:04.1f}'
print('\n=== RAW p5k_min vs DISPLAYED (raw_resid - model_adj) ===')
for dstr in ('2023-05-06', '2023-02-17', '2023-03-30'):
    x = lr2[pd.to_datetime(lr2['date']).dt.date == pd.Timestamp(dstr).date()]
    if x.empty:
        print(f'  {dstr}: not in fit set'); continue
    x = x.iloc[0]
    model_adj = float(x['phys_contrib']) + float(x['cov_contrib'])
    resid = float(x['raw_resid']) - model_adj          # s/mi, what the graph uses
    disp_pace = float(x['p5k_cs_min']) + resid/60.0      # min/mi
    print(f'  {dstr}: raw p5k_min={t5k_from_pacemin(x["p5k_min"])}  '
          f'model_adj={model_adj:+.1f} (phys {x["phys_contrib"]:+.1f} + cov {x["cov_contrib"]:+.1f})  '
          f'-> DISPLAYED 5K {t5k_from_pacemin(disp_pace)}')

# rank among all long runs by p5k_min
lr2_sorted = lr2.sort_values('p5k_min')
print(f'\n=== fastest 6 long runs by p5k_min ===')
for _, x in lr2_sorted.head(6).iterrows():
    m, s = divmod(x['p5k_min']*60, 60)
    print(f'  {pd.to_datetime(x["date"]).date()}  p5k={int(m)}:{s:04.1f}  '
          f'{x["miles"]:.0f}mi  {x["location"]}  raw_resid={x["raw_resid"]:+.1f} '
          f'corrected={x["corrected"]:+.1f} cov={x["cov_contrib"]:+.1f} phys={x["phys_contrib"]:+.1f}')
