"""make_recovery_plots.py - Recovery runs analyzed against CS pace.

Two side-by-side panels in one HTML:
  - Left: absolute recovery pace + CS gold reference curve
  - Right: residual (pace - CS) + flat zero gold reference

A right-sidebar checkbox UI applies normalization factors. Most factors
adjust BOTH panels simultaneously (subtracting β × x from each point's
y-value), but the "Era trend" toggle is special and applies ONLY to the
residual panel — the absolute pace panel always shows the actual recorded
pace minus only the cross-sectional factor adjustments. This lets you
ask "how did my absolute pace compare to CS over time, controlling for
these factors?" without the chart's baseline shifting underneath you.

Era trend is a centered ±182-day rolling mean of residual. Toggling it
on subtracts (era_trend − global_mean_residual) from each residual,
which centers the points around the global mean rather than collapsing
them toward zero. The gold zero line on the residual panel remains the
CS reference.

Per-day factors are fit via OLS on the era-detrended residual:

  residual_detrended ~ β_temp · temp_centered
                     + Σ β_r · route_dummy_r       (n ≥ MIN_ROUTE_N)
                     + β_marathon · fatigue_marathon
                     + β_race    · fatigue_race_short
                     + β_long    · fatigue_long

Sleep cycles and recovery distance were tested in earlier model versions
and produced near-zero coefficients with no useful explanatory power. They
are excluded.

Hill workouts (continuous + reps) and tempo/interval/rep/fartlek workouts
were also tested as fatigue features and showed near-zero or noise-level
βs. Only races and long runs leave detectable next-day pace effects.

Training-quality residual (the smoothed training-derived deviation from CS
that plot_training_quality.py produces) was tested as a feature and found
to be redundant with era trend — the two are highly collinear over their
shared range, since both are smoothed long-term tracks of fitness vs CS.
Era trend wins because it's data-driven from recovery itself.

Three classes of rows are pruned from BOTH the era-trend window contents
AND the OLS fit, but remain on the chart with hover notes. **A single row
can fall into multiple classes** — the three flags are independent.

  1. Bad conditions — conditions ∈ {snow, icy}, OR the workout string
     contains "snow" (catches "[2" snow]" annotations the conditions
     field missed). Inside/treadmill/indoor-track runs are kept (still
     valid pace data on a stable surface).
  2. Partner runs — any partners entry that isn't blank/solo/none.
     Concentrated in 2016-2017 HS team easy runs; different population.
  3. Outliers — |residual from leave-one-out 28-day local mean| > 45
     sec/mi, where the local mean is computed against the *clean*
     neighbor pool (rows that are neither bad-cond nor partner-run).
     Removes "clearly something happened" days (travel, illness,
     extreme post-marathon fatigue). Computed for every recovery row,
     so a partner-run or bad-cond day can also be flagged outlier if
     its pace is anomalous against the clean local mean.

The visibility section has three independent toggles ("Hide bad
conditions", "Hide non-solo", "Hide outliers"), each with its own count
and an All/None group. Because flags can overlap, the counts may sum
to more than the unique pruned total.

Hidden points have y=null on the trace (not just opacity=0) so hover
events don't fire on them.

Hover annotates each point with display_name and city_state read directly
from columns on daily.csv (populated by build_dataset.py from the locations
sheet of the adjustments doc). These are purely informational — they don't
feed the model. Daily rows whose location isn't in the locations sheet
fall back to the raw log_location string.

Workflow:
  python make_recovery_plots.py --tag v11

Default paths assume the script lives next to bayes_cs_summary_{tag}.csv,
bayes_cs_params_{tag}.csv, daily.csv, races.csv. Override with --in-dir,
--daily, --races, --out-dir.

Dependencies: pandas, numpy, plotly.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_projection import load_cs_outputs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN_DIR = SCRIPT_DIR
DEFAULT_DAILY  = os.path.join(SCRIPT_DIR, 'daily.csv')
DEFAULT_RACES  = os.path.join(SCRIPT_DIR, 'races.csv')
DEFAULT_OUT    = SCRIPT_DIR


# ---------- conditions / pruning excluded from fit ----------
# Time-of-day binary indicator. Recovery pace runs ~4.5 sec/mi slower in
# the early/morning window than the afternoon/late window (p<1e-13 in
# fully-normalized residuals), driven by overnight fueling state and
# joint stiffness. Encoded as 1 = PM (afternoon|late), 0 = AM (early|
# morning). `early` (n≈30) and `late` (n≈90) are too sparse to keep as
# their own levels; their means align with morning and afternoon
# respectively to within their CIs.
TOD_PM_VALUES = {'afternoon', 'late'}

EXCLUDED_CONDITIONS = {'snow', 'icy'}

# Workout-string snow detection. Catches "[2\" snow]" and similar annotations
# in workout_raw that the conditions field missed.
SNOW_IN_WORKOUT_RE = re.compile(r'\bsnow\b', re.IGNORECASE)

# Partner-run detection: any partners entry that isn't blank/solo/none counts
# as a non-solo run. Partner runs are a different population (HS team easy
# days, the occasional Maddy run) — their pace targets and route choices
# differ from solo recovery, and including them inflates within-period
# variance. They're pruned from the fit alongside snow/ice/inside.
NULL_PARTNERS = {'', 'nan', 'solo', 'none'}

# Outlier detection: residual from leave-one-out 28-day local mean exceeds
# OUTLIER_THRESHOLD_SEC. Catches "clearly something happened" days
# (travel/jet-lag, illness, extreme post-marathon fatigue) that no feature
# in the model is set up to capture. ±45 prunes ~16 of ~2,200 eligible
# rows (Apr 2026); about 3σ on the LOO residual distribution.
OUTLIER_THRESHOLD_SEC      = 45.0
OUTLIER_WINDOW_HALF_DAYS   = 14
OUTLIER_MIN_NEIGHBORS      = 5


# ---------- model parameters ----------
TEMP_REFERENCE_C       = 12.0
# Recent-race fatigue uses exponential decay: contribution = exp(−t/τ) at
# day t post-race, fitted via OLS for amplitude. Per-category τ in days,
# from empirical curve fits on days-since-last-race vs marathon-revealed
# residual (see April 2026 analysis).
FATIGUE_TAU_DAYS = {'marathon': 6.0, 'race_short': 5.0}
# Hover threshold — only show the "Recent race" line if within this many
# days of the most recent race. Cosmetic only; doesn't affect the model.
FATIGUE_HOVER_DAYS = 14
ERA_WINDOW_HALF_DAYS   = 182
ERA_WINDOW_MIN_POINTS  = 30
MIN_ROUTE_N            = 13
MARATHON_DISTANCE_M    = 42000


CS_LINE_COLOR = 'rgb(255,180,80)'
CS_LINE_WIDTH = 2.5
TREND_COLOR   = 'rgb(220,220,220)'
TREND_WIDTH   = 2.0
TREND_SMOOTH_SIGMA_DAYS = 28  # Gaussian σ; FWHM ≈ 66d, ~95% mass within ±56d


# ---------- helpers ----------

def days_since(daily, source_dates_sorted):
    if not source_dates_sorted:
        return np.full(len(daily), np.nan)
    sd = np.array([d.value for d in source_dates_sorted])
    out = []
    for d in daily['date']:
        idx = np.searchsorted(sd, d.value, side='left')
        if idx == 0:
            out.append(np.nan)
        else:
            prev = pd.Timestamp(sd[idx - 1])
            out.append((d - prev).days)
    return np.array(out, dtype=float)


def centered_rolling_mean(target_dates, source_dates, source_values,
                           half_days, min_points):
    target_ms = np.array([d.value // 10**6 for d in target_dates])
    source_ms = np.array([d.value // 10**6 for d in source_dates])
    source_v = np.asarray(source_values, dtype=float)
    half_ms = half_days * 86_400_000

    out = np.full(len(target_ms), np.nan)
    lo = hi = 0
    n_src = len(source_ms)
    for i, t in enumerate(target_ms):
        while lo < n_src and source_ms[lo] < t - half_ms:
            lo += 1
        while hi < n_src and source_ms[hi] <= t + half_ms:
            hi += 1
        if hi - lo >= min_points:
            out[i] = source_v[lo:hi].mean()
    return out


def sanitize_route(name):
    s = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
    return f'rt_{s}'

def sec_to_mss(s):
    if s is None or pd.isna(s):
        return ''
    sign = '-' if s < 0 else ''
    s = abs(int(round(float(s))))
    return f'{sign}{s // 60}:{s % 60:02d}'


def signed_sec(s):
    if s is None or pd.isna(s):
        return ''
    s = int(round(float(s)))
    return f'{"+" if s >= 0 else "−"}{abs(s)}'


def thin_yearly_ticks(x_lo, x_hi, max_labels=11):
    y_lo = int(pd.Timestamp(x_lo).year)
    y_hi = int(pd.Timestamp(x_hi).year)
    years = list(range(y_lo, y_hi + 1))
    if not years:
        return [], []
    tickvals = [pd.Timestamp(f'{y}-01-01') for y in years]
    if len(years) <= max_labels:
        return tickvals, [str(y) for y in years]
    step = -(-len(years) // max_labels)
    txt = [str(y) if ((len(years) - 1 - i) % step == 0) else ''
           for i, y in enumerate(years)]
    return tickvals, txt


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v11')
    ap.add_argument('--in-dir', default=DEFAULT_IN_DIR)
    ap.add_argument('--daily', default=DEFAULT_DAILY)
    ap.add_argument('--races', default=DEFAULT_RACES)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f'Loading daily.csv from {args.daily}...')
    daily = pd.read_csv(args.daily, parse_dates=['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    print(f'Loading races.csv from {args.races}...')
    races = pd.read_csv(args.races, parse_dates=['date'])

    print(f'Loading CS outputs (tag={args.tag}) from {args.in_dir}...')
    cs_summary, beta_long, d_thresh, xc = load_cs_outputs(args.in_dir, args.tag)
    cs_summary['cs_pace_sec'] = cs_summary['cs_pace_med'] * 60.0

    # Quality-day categorization
    marathon_dates = set(races.loc[races['distance_m'] >= MARATHON_DISTANCE_M,
                                    'date'].tolist())

    def categorize_quality(row):
        if row['run_type'] == 'race':
            return 'marathon' if row['date'] in marathon_dates else 'race_short'
        if row['run_type'] == 'long':
            return 'long'
        return None
    daily['quality_category'] = daily.apply(categorize_quality, axis=1)

    QUALITY_CATS = ['marathon', 'race_short']
    for cat in QUALITY_CATS:
        cat_dates = sorted(daily.loc[daily['quality_category']==cat,
                                       'date'].tolist())
        daily[f'dsq_{cat}'] = days_since(daily, cat_dates)

    # Recovery subset
    rec = daily[daily['run_type'] == 'recovery'].copy()
    rec = rec.dropna(subset=['recovery_pace_sec_per_mi'])
    rec = rec.merge(cs_summary[['date', 'cs_pace_sec']], on='date', how='left')
    rec = rec.dropna(subset=['cs_pace_sec'])
    rec = rec.sort_values('date').reset_index(drop=True)
    print(f'  {len(rec)} recovery days with valid pace and CS')

    # Features (sleep_centered and miles_centered are intentionally NOT computed)
    rec['temp_centered'] = rec['temp_c'] - TEMP_REFERENCE_C
    rec['residual_raw'] = rec['recovery_pace_sec_per_mi'] - rec['cs_pace_sec']

    # Time-of-day binary indicator: 1 for PM (afternoon/late), 0 for AM
    # (early/morning). Recovery rows have 100% TOD coverage; missing or
    # unknown values fall to 0 (AM baseline).
    tod_clean = rec['time_of_day'].astype(str).str.strip().str.lower()
    rec['tod_is_pm'] = tod_clean.isin(TOD_PM_VALUES).astype(float)

    rec['conditions_clean'] = rec['conditions'].astype(str).str.strip().str.lower()
    cond_excluded = rec['conditions_clean'].isin(EXCLUDED_CONDITIONS)
    workout_snow = rec['workout_raw'].fillna('').astype(str).apply(
        lambda w: bool(SNOW_IN_WORKOUT_RE.search(w)))
    rec['is_bad_cond'] = cond_excluded | workout_snow
    rec['workout_snow_only'] = workout_snow & ~cond_excluded

    # Partner-run detection
    rec['partners_clean'] = rec['partners'].astype(str).str.strip().str.lower()
    rec['is_partner_run'] = ~rec['partners_clean'].isin(NULL_PARTNERS)

    # Outlier detection — leave-one-out 28-day rolling mean of recovery
    # pace. The neighbor pool (the source of "normal" pace) is restricted
    # to rows that are neither bad-cond nor partner-run, so the local
    # baseline reflects typical solo recovery state. The LOO residual is
    # computed for EVERY recovery row, so a bad-cond or partner-run day
    # can also receive the outlier flag if its pace is anomalous against
    # the clean local mean. This means the three flags overlap, and a
    # single point can be in multiple classes.
    neighbor_mask = ~rec['is_bad_cond'] & ~rec['is_partner_run']
    rec['loo_resid'] = np.nan
    if neighbor_mask.sum() > 0:
        nbr_dates_ms = np.array(
            [d.value // 10**6 for d in rec.loc[neighbor_mask, 'date']])
        nbr_pace = rec.loc[neighbor_mask, 'recovery_pace_sec_per_mi'].values
        all_dates_ms = np.array([d.value // 10**6 for d in rec['date']])
        all_pace = rec['recovery_pace_sec_per_mi'].values
        in_pool = neighbor_mask.values
        half_ms = OUTLIER_WINDOW_HALF_DAYS * 86_400_000
        loo_resid = np.full(len(rec), np.nan)
        for i, d_ms in enumerate(all_dates_ms):
            lo = np.searchsorted(nbr_dates_ms, d_ms - half_ms, side='left')
            hi = np.searchsorted(nbr_dates_ms, d_ms + half_ms, side='right')
            n_in = hi - lo
            if in_pool[i]:
                # Leave one out: this row IS a neighbor of itself.
                if n_in < OUTLIER_MIN_NEIGHBORS + 1:
                    continue
                local_mean = (nbr_pace[lo:hi].sum() - all_pace[i]) / (n_in - 1)
            else:
                # Bad-cond / partner row — not in pool, no leave-out needed.
                if n_in < OUTLIER_MIN_NEIGHBORS:
                    continue
                local_mean = nbr_pace[lo:hi].mean()
            loo_resid[i] = all_pace[i] - local_mean
        rec['loo_resid'] = loo_resid
    rec['is_outlier_loo'] = (
        rec['loo_resid'].abs() > OUTLIER_THRESHOLD_SEC).fillna(False)

    # Combined "pruned from fit" flag — drives both OLS exclusion and the
    # three independent visibility toggles on the chart. Flags are NOT
    # mutually exclusive; a single row can have more than one set.
    rec['is_pruned'] = (rec['is_bad_cond']
                       | rec['is_partner_run']
                       | rec['is_outlier_loo'])

    # Outlier breakdown by which other flags also apply
    n_outlier_clean = int((rec['is_outlier_loo']
                           & ~rec['is_bad_cond']
                           & ~rec['is_partner_run']).sum())
    n_outlier_bad = int((rec['is_outlier_loo'] & rec['is_bad_cond']).sum())
    n_outlier_partner = int((rec['is_outlier_loo'] & rec['is_partner_run']).sum())
    print(f'  pruned from fit: {rec["is_pruned"].sum()} unique '
          f'({rec["is_bad_cond"].sum()} bad-cond '
          f'[{cond_excluded.sum()} cond + {workout_snow.sum()} workout-snow], '
          f'{rec["is_partner_run"].sum()} partner-runs, '
          f'{rec["is_outlier_loo"].sum()} outliers '
          f'[{n_outlier_clean} clean + {n_outlier_bad} also bad-cond + '
          f'{n_outlier_partner} also partner])')

    for cat in QUALITY_CATS:
        tau = FATIGUE_TAU_DAYS[cat]
        rec[f'fat_{cat}'] = np.exp(-rec[f'dsq_{cat}'] / tau).fillna(0)

    # Qualifying routes
    route_counts = (rec.loc[~rec['is_pruned'], 'location']
                     .dropna().value_counts())
    qualifying_routes = (route_counts[route_counts >= MIN_ROUTE_N]
                         .index.tolist())
    qualifying_routes = sorted(qualifying_routes,
                                key=lambda x: -route_counts[x])
    print(f'\nQualifying routes (n >= {MIN_ROUTE_N}): {len(qualifying_routes)}')

    route_col_map = {r: sanitize_route(r) for r in qualifying_routes}
    for r in qualifying_routes:
        rec[route_col_map[r]] = (rec['location'] == r).astype(float)

    # Era trend
    pool = rec[~rec['is_pruned']].copy()
    rec['era_trend'] = centered_rolling_mean(
        rec['date'].tolist(),
        pool['date'].tolist(),
        pool['residual_raw'].values,
        ERA_WINDOW_HALF_DAYS,
        ERA_WINDOW_MIN_POINTS,
    )
    rec['residual_detrended'] = rec['residual_raw'] - rec['era_trend']

    # Global mean residual — used to center era contributions so toggling
    # era_trend doesn't collapse points to zero.
    global_mean_residual = float(pool['residual_raw'].mean())
    print(f'  global mean residual: {global_mean_residual:+.2f} sec/mi '
          f'(reference for era-detrended view; computed on non-pruned pool)')

    # OLS fit
    base_features = (['temp_centered']
                     + [f'fat_{c}' for c in QUALITY_CATS]
                     + ['tod_is_pm'])
    route_features = [route_col_map[r] for r in qualifying_routes]
    feature_cols = base_features + route_features

    rec_fit = rec[~rec['is_pruned']].dropna(
        subset=feature_cols + ['residual_detrended']).copy()

    X = rec_fit[feature_cols].values.astype(float)
    y = rec_fit['residual_detrended'].values.astype(float)
    X_int = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    intercept = float(coef[0])
    betas = {f: float(b) for f, b in zip(feature_cols, coef[1:])}

    yhat = X_int @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_detrended = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    raw_y = rec_fit['residual_raw'].values
    raw_yhat = rec_fit['era_trend'].values + yhat
    ss_res_raw = float(np.sum((raw_y - raw_yhat) ** 2))
    ss_tot_raw = float(np.sum((raw_y - raw_y.mean()) ** 2))
    r2_raw = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw > 0 else 0.0

    print(f'\nOLS results (n={len(rec_fit)}):')
    print(f'  R² on detrended:    {r2_detrended:.3f}')
    print(f'  R² on raw residual: {r2_raw:.3f}')
    print(f'  intercept: {intercept:+.2f}')
    print(f'\n  Per-day factors:')
    for f in base_features:
        print(f'    {f:18s}  β = {betas[f]:+.3f}')
    print(f'\n  Route offsets (vs unspecified-location baseline):')
    for r in qualifying_routes:
        b = betas[route_col_map[r]]
        print(f'    {r:25s}  β = {b:+.2f}')
    # ---------- per-point contributions (4 channels) ----------
    rec['contrib_temp'] = betas['temp_centered'] * rec['temp_centered'].fillna(0)
    rec['contrib_quality'] = sum(
        betas[f'fat_{c}'] * rec[f'fat_{c}'].fillna(0) for c in QUALITY_CATS)
    rec['contrib_tod'] = betas['tod_is_pm'] * rec['tod_is_pm'].fillna(0)

    def route_contrib(loc):
        if loc in qualifying_routes:
            return betas[route_col_map[loc]]
        return 0.0
    rec['contrib_route'] = rec['location'].apply(route_contrib)

    # Era contribution centered on global mean
    rec['contrib_era'] = rec['era_trend'].fillna(global_mean_residual) - global_mean_residual

    # ---------- build figure ----------
    cs_for_plot = cs_summary[(cs_summary['date'] >= rec['date'].min()) &
                              (cs_summary['date'] <= rec['date'].max())].copy()

    def build_hover(row):
        date_label = f"{row['date'].date()} ({row['date'].strftime('%a')})"
        parts = [f"<b>{date_label}</b>",
                 f"Pace: {sec_to_mss(row['recovery_pace_sec_per_mi'])}/mi  "
                 f"({row['miles']:.1f} mi)",
                 f"CS pace: {sec_to_mss(row['cs_pace_sec'])}/mi",
                 f"Residual: {signed_sec(row['residual_raw'])} sec/mi"]
        if pd.notna(row.get('temp_c')):
            parts.append(f"Temp: {row['temp_c']:.0f}°C")
        loc = row.get('location')
        if pd.notna(loc) and str(loc) != 'nan':
            disp_raw = row.get('display_name')
            cs_raw = row.get('city_state')
            disp = (str(disp_raw).strip()
                    if pd.notna(disp_raw) and str(disp_raw).strip() else None)
            cs = (str(cs_raw).strip()
                  if pd.notna(cs_raw) and str(cs_raw).strip() else None)
            if disp:
                label = f"{disp}, {cs}" if cs else disp
            elif cs:
                label = cs
            else:
                label = str(loc)
            parts.append(f"<i>{label}</i>")
        if row.get('is_bad_cond'):
            if row.get('workout_snow_only'):
                parts.append(f"<i>Snow noted in workout (excluded from fit)</i>")
            else:
                parts.append(f"<i>Conditions: {row['conditions_clean']} "
                             f"(excluded from fit)</i>")
        elif (pd.notna(row.get('conditions_clean'))
              and str(row['conditions_clean']) not in ('nan', '')):
            parts.append(f"Conditions: {row['conditions_clean']}")
        if row.get('is_partner_run'):
            parts.append(f"<i>Partners: {row['partners']} "
                         f"(excluded from fit)</i>")
        elif pd.notna(row.get('partners')) and str(row['partners']) != 'nan':
            parts.append(f"With: {row['partners']}")
        if row.get('is_outlier_loo'):
            parts.append(f"<i>Outlier (excluded from fit)</i>")
        # Most recent race within fatigue decay window only
        most_recent = None  # (cat, days_ago)
        for cat in QUALITY_CATS:
            d = row.get(f'dsq_{cat}')
            if pd.notna(d) and d <= FATIGUE_HOVER_DAYS:
                if most_recent is None or d < most_recent[1]:
                    most_recent = (cat, int(d))
        if most_recent is not None:
            cat, d = most_recent
            label = "Recent marathon" if cat == 'marathon' else "Recent race"
            parts.append(f"{label}: {d}d ago")
        tod_raw = row.get('time_of_day')
        if pd.notna(tod_raw) and str(tod_raw).strip():
            parts.append(f"Time of day: {str(tod_raw).strip().lower()}")
        return '<br>'.join(parts)

    rec['_hover'] = rec.apply(build_hover, axis=1)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=('Absolute pace', 'Residual vs CS'),
                         horizontal_spacing=0.07)

    fig.add_trace(go.Scatter(
        x=cs_for_plot['date'], y=cs_for_plot['cs_pace_sec'],
        mode='lines', line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH),
        name='CS pace', hoverinfo='skip', showlegend=True,
        meta={'role': 'reference'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[rec['date'].min(), rec['date'].max()], y=[0, 0],
        mode='lines', line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH),
        name='Zero', hoverinfo='skip', showlegend=False,
        meta={'role': 'reference'},
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(color=TREND_COLOR, width=TREND_WIDTH),
        name='Trend', hoverinfo='skip', showlegend=True,
        meta={'role': 'trend_pace'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(color=TREND_COLOR, width=TREND_WIDTH),
        name='Trend', hoverinfo='skip', showlegend=False,
        meta={'role': 'trend_resid'},
    ), row=1, col=2)

    # Customdata channels — ORDER MUST MATCH FACTOR_ORDER in JS:
    # 0=temp, 1=route, 2=recent_effort, 3=tod, 4=era
    contrib_arr = np.stack([
        rec['contrib_temp'].values,
        rec['contrib_route'].values,
        rec['contrib_quality'].values,
        rec['contrib_tod'].values,
        rec['contrib_era'].values,
    ], axis=1).tolist()

    fig.add_trace(go.Scatter(
        x=rec['date'],
        y=rec['recovery_pace_sec_per_mi'],
        mode='markers',
        marker=dict(
            size=4.5,
            color=rec['temp_c'],
            coloraxis='coloraxis',
            opacity=0.6,
            line=dict(width=0),
        ),
        customdata=contrib_arr,
        text=rec['_hover'].tolist(),
        hovertemplate='%{text}<extra></extra>',
        name='Recovery (pace)',
        showlegend=False,
        meta={'role': 'pace',
              'raw_y': rec['recovery_pace_sec_per_mi'].tolist()},
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rec['date'],
        y=rec['residual_raw'],
        mode='markers',
        marker=dict(
            size=4.5,
            color=rec['temp_c'],
            coloraxis='coloraxis',
            opacity=0.6,
            line=dict(width=0),
        ),
        customdata=contrib_arr,
        text=rec['_hover'].tolist(),
        hovertemplate='%{text}<extra></extra>',
        name='Recovery (residual)',
        showlegend=False,
        meta={'role': 'residual',
              'raw_y': rec['residual_raw'].tolist(),
              'date_ms': [int(d.value // 10**6) for d in rec['date']],
              'is_bad_cond': rec['is_bad_cond'].astype(bool).tolist(),
              'is_partner_run': rec['is_partner_run'].astype(bool).tolist(),
              'is_outlier': rec['is_outlier_loo'].astype(bool).tolist()},
    ), row=1, col=2)

    # ---------- axes ----------
    x_lo = rec['date'].min() - pd.Timedelta(days=30)
    x_hi = rec['date'].max() + pd.Timedelta(days=30)
    tickvals_x, ticktext_x = thin_yearly_ticks(x_lo, x_hi)

    left_y_lo = max(180, np.floor(rec['recovery_pace_sec_per_mi'].quantile(0.001) / 30) * 30 - 30)
    left_y_hi = min(720, np.ceil(rec['recovery_pace_sec_per_mi'].quantile(0.999) / 30) * 30 + 30)
    left_ticks = list(range(int(left_y_lo), int(left_y_hi) + 1, 30))

    res_lo = float(rec['residual_raw'].quantile(0.005))
    res_hi = float(rec['residual_raw'].quantile(0.995))
    right_y_lo = np.floor(res_lo / 30) * 30 - 30
    right_y_hi = np.ceil(res_hi / 30) * 30 + 30
    right_ticks = list(range(int(right_y_lo), int(right_y_hi) + 1, 30))

    for col in (1, 2):
        fig.update_xaxes(tickvals=tickvals_x, ticktext=ticktext_x,
                          range=[x_lo, x_hi],
                          gridcolor='rgba(255,255,255,0.08)',
                          row=1, col=col)

    fig.update_yaxes(tickvals=left_ticks,
                      ticktext=[sec_to_mss(t) for t in left_ticks],
                      range=[left_y_hi, left_y_lo],
                      title_text='Recovery pace (per mile)',
                      gridcolor='rgba(255,255,255,0.08)',
                      row=1, col=1)
    fig.update_yaxes(tickvals=right_ticks,
                      ticktext=[signed_sec(t) if t != 0 else '0' for t in right_ticks],
                      range=[right_y_hi, right_y_lo],
                      title_text='Residual (sec/mi above CS)',
                      gridcolor='rgba(255,255,255,0.08)',
                      zeroline=False,
                      row=1, col=2)

    title_text = ('Recovery runs vs. race fitness'
                  '<br><sub style="font-size:13px;color:#bbb">'
                  'Absolute pace and relative fitness signal, '
                  'controlling for combinations of factors'
                  '</sub>')
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', y=0.965,
                    yanchor='top'),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#eee', size=12),
        margin=dict(l=70, r=300, t=110, b=70),
        hoverlabel=dict(bgcolor='#222', bordercolor='#555',
                        font=dict(color='#eee', size=12)),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.005, xanchor='left',
            y=0.10, yanchor='top',  # just below the colorbar (which ends at y≈0.15)
            bgcolor='rgba(26,26,26,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color='#eee', size=11),
            itemsizing='constant',
        ),
        coloraxis=dict(
            # Match the Running Log conditional formatting rule (column G temp):
            # -10°C #00B0F0 (light blue) → 22°C #92D050 (green) → 40°C #FF0000 (red)
            colorscale=[
                [0.00, 'rgb(0,176,240)'],     # -10°C
                [0.64, 'rgb(146,208,80)'],    # 22°C  ((22−(−10))/(40−(−10)) = 0.64)
                [1.00, 'rgb(255,0,0)'],       # 40°C
            ],
            cmin=-10, cmax=40,
            colorbar=dict(
                title=dict(text='°C', side='right',
                            font=dict(color='#eee', size=11)),
                orientation='v', x=1.005, xanchor='left',
                y=0.45, yanchor='top',  # below the toolbar's typical footprint
                len=0.30, thickness=10,
                tickvals=[-10, 0, 10, 22, 30, 40],
                tickfont=dict(color='#eee', size=10),
                outlinewidth=0,
            ),
        ),
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#eee', size=14)

    out_path = os.path.join(args.out_dir, 'recovery_pace.html')
    write_dark_html(fig, out_path,
                     extra_html=build_normalization_ui(
                         betas, intercept, r2_detrended, r2_raw, len(rec_fit),
                         int(rec['is_bad_cond'].sum()),
                         int(rec['is_partner_run'].sum()),
                         int(rec['is_outlier_loo'].sum()),
                         int(rec['is_pruned'].sum()),
                         qualifying_routes, route_col_map, route_counts,
                         global_mean_residual))
    print(f'\nWrote {out_path}')

    # Export route betas for downstream consumption (e.g. long-run TQ
    # corrections). The route betas are recovery-only, so they're effort-
    # uncontaminated and represent the pace cost of each route at easy
    # effort. The elev/terrain model parameters from the cross-route
    # exploration are NOT exported here since they're a separate analysis.
    betas_csv = os.path.join(args.out_dir, f'route_betas_{args.tag}.csv')
    betas_df = pd.DataFrame([
        {'route': r, 'n': int(route_counts[r]), 'beta_sec_per_mi': betas[route_col_map[r]]}
        for r in qualifying_routes
    ]).sort_values('beta_sec_per_mi')
    betas_df.to_csv(betas_csv, index=False)
    print(f'Wrote {betas_csv} ({len(betas_df)} routes)')


# ---------- HTML output ----------
def write_dark_html(fig, path, extra_html=''):
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                    config={'responsive': True})
    css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'background:#1a1a1a;color:#eee;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}'
        '.plotly-graph-div,.js-plotly-plot{width:100%!important;height:100vh!important;}'
        '@media (max-width:760px){'
        '#norm-filter{position:static!important;margin:8px;width:auto!important;}'
        '}'
        '</style>')
    with open(path, 'r') as f:
        html = f.read()
    html = html.replace('<head>', '<head>' + css, 1)
    if extra_html:
        html = html.replace('</body>', extra_html + '</body>')
    with open(path, 'w') as f:
        f.write(html)


def build_normalization_ui(betas, intercept, r2_detrended, r2_raw, n_fit,
                            n_bad_cond, n_partner_runs, n_outliers, n_pruned_unique,
                            qualifying_routes, route_col_map,
                            route_counts, global_mean_residual):
    """Right-sidebar UI. 4 normalization toggles + 3 visibility toggles.

    Normalization order MUST match customdata channel order:
    0=temp, 1=route, 2=recent_effort, 3=time_of_day, 4=era

    Era applies ONLY to residual panel (not absolute pace). Other
    normalization factors apply to both panels. The visibility section
    has three independent toggles (Hide bad conditions, Hide non-solo,
    Hide outliers) each with its own count and a shared All/None group.
    Hidden points have y set to null on the trace so hover events don't
    fire on them; the rolling-mean trend line ignores hidden points.
    """
    q_betas = [betas['fat_marathon'], betas['fat_race_short']]

    factors = [
        ('era',           'Era trend'),
        ('temp',          'Temperature'),
        ('route',         'Route'),
        ('recent_effort', 'Recent race'),
        ('time_of_day',   'Time of day'),
    ]
    cb_html = '\n'.join(
        f'  <label class="nf-row"><input type="checkbox" data-factor="{key}" data-mode="norm"> {label}</label>'
        for key, label in factors)

    filter_items = [
        ('hide_bad_cond', 'Hide bad conditions', n_bad_cond),
        ('hide_partner',  'Hide non-solo',       n_partner_runs),
        ('hide_outlier',  'Hide outliers',       n_outliers),
    ]
    filter_html = '\n'.join(
        f'  <label class="nf-row"><input type="checkbox" data-factor="{key}" data-mode="filter"> '
        f'{label} <span style="color:#888">({n})</span></label>'
        for key, label, n in filter_items)

    routes_by_beta = sorted(qualifying_routes,
                              key=lambda r: betas[route_col_map[r]])
    rt_rows = '\n'.join(
        f'<tr><td>{r}</td><td>{int(route_counts[r])}</td>'
        f'<td style="text-align:right">{betas[route_col_map[r]]:+.2f}</td></tr>'
        for r in routes_by_beta)

    return f"""
<style>
#norm-filter {{
  position: fixed; right: 12px; top: 48px;
  background: rgba(26,26,26,0.92);
  border: 1px solid #444;
  padding: 12px 14px;
  border-radius: 4px;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 12px;
  z-index: 100;
  width: 240px;
  user-select: none;
  max-height: calc(100vh - 100px);
  overflow-y: auto;
}}
#norm-filter .nf-title {{ font-weight: 500; margin-bottom: 4px; color: #eee; font-size: 13px; }}
#norm-filter .nf-sub {{ font-size: 11px; color: #999; margin-bottom: 8px; line-height: 1.4; }}
#norm-filter .nf-buttons {{ margin-bottom: 8px; display: flex; gap: 4px; }}
#norm-filter .nf-buttons button {{
  background: transparent; border: 1px solid #555; color: #eee;
  padding: 2px 8px; cursor: pointer; font-size: 12px; border-radius: 3px;
  flex: 1;
}}
#norm-filter .nf-buttons button:hover {{ background: #2a2a2a; }}
#norm-filter .nf-row {{ display: block; margin: 5px 0; cursor: pointer; line-height: 1.35; }}
#norm-filter input[type=checkbox] {{ margin-right: 6px; vertical-align: top; cursor: pointer; accent-color: #4aa3ff; margin-top: 3px; }}
#norm-filter .nf-stats {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #333; font-size: 11px; color: #888; line-height: 1.4; }}
#norm-filter details {{ margin-top: 10px; }}
#norm-filter summary {{ cursor: pointer; font-size: 11px; color: #aaa; }}
#norm-filter table {{ width: 100%; font-size: 10.5px; margin-top: 6px; border-collapse: collapse; }}
#norm-filter td, #norm-filter th {{ padding: 2px 4px; }}
#norm-filter tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
#norm-filter .nf-noteworthy {{ font-size: 10.5px; color: #777; margin-top: 8px; line-height: 1.4; font-style: italic; }}
#norm-filter .nf-detail-row {{ font-size: 11px; color: #bbb; margin-top: 6px; line-height: 1.4; }}
#norm-filter .nf-detail-row b {{ color: #ddd; }}
#norm-filter .nf-divider {{ border-top: 1px solid #333; margin: 8px 0 4px; }}
/* Suppress Plotly's built-in hover label — we render our own */
.hovertext {{ display: none !important; }}
#rec-tooltip {{
  position: fixed; top: 0; left: 0;
  background: rgba(26,26,26,0.96);
  color: #eee;
  border: 1px solid #555;
  padding: 8px 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 12px;
  line-height: 1.45;
  border-radius: 4px;
  pointer-events: none;
  z-index: 9999;
  max-width: 360px;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}
/* Arrow pointing to the data point. Tooltip carries data-side="right"
   when it's drawn to the right of the point (arrow on its left edge),
   or "left" when drawn to the left (arrow on its right edge). */
#rec-tooltip::before, #rec-tooltip::after {{
  content: '';
  position: absolute;
  width: 0; height: 0;
  top: 50%;
  border-top: 8px solid transparent;
  border-bottom: 8px solid transparent;
}}
#rec-tooltip[data-side='right']::before {{
  left: -9px; margin-top: -8px;
  border-right: 9px solid #555;
}}
#rec-tooltip[data-side='right']::after {{
  left: -8px; margin-top: -8px;
  border-right: 9px solid rgba(26,26,26,0.96);
}}
#rec-tooltip[data-side='left']::before {{
  right: -9px; margin-top: -8px;
  border-left: 9px solid #555;
}}
#rec-tooltip[data-side='left']::after {{
  right: -8px; margin-top: -8px;
  border-left: 9px solid rgba(26,26,26,0.96);
}}
</style>
<div id="rec-tooltip"></div>
<div id="norm-filter">
  <div class="nf-title">Normalize</div>
  <div class="nf-sub">Subtract each factor's modeled contribution.</div>
  <div class="nf-buttons">
    <button id="nf-norm-all">All</button>
    <button id="nf-norm-none">None</button>
  </div>
{cb_html}
  <div class="nf-divider"></div>
  <div class="nf-title" style="margin-top:6px">Hide from chart</div>
  <div class="nf-buttons">
    <button id="nf-hide-all">All</button>
    <button id="nf-hide-none">None</button>
  </div>
{filter_html}
  <details>
    <summary>Coefficient details</summary>
    <div class="nf-detail-row"><b>Era trend:</b> residual panel only; centers around {global_mean_residual:+.0f} s/mi</div>
    <div class="nf-detail-row"><b>Temperature:</b> β = {betas["temp_centered"]:+.2f} sec/mi per °C from {int(TEMP_REFERENCE_C)}°C</div>
    <div class="nf-detail-row"><b>Recent race</b> (exponential decay):
      marathon {q_betas[0]:+.1f} (τ={FATIGUE_TAU_DAYS['marathon']:.0f}d),
      short race {q_betas[1]:+.1f} (τ={FATIGUE_TAU_DAYS['race_short']:.0f}d)</div>
    <div class="nf-detail-row"><b>Time of day:</b> β = {betas["tod_is_pm"]:+.2f} sec/mi for afternoon/late (vs early/morning)</div>
    <div class="nf-detail-row"><b>Route offsets</b> (n ≥ {MIN_ROUTE_N}):</div>
    <table>
      <thead><tr><th style="text-align:left">Route</th><th>n</th><th style="text-align:right">β</th></tr></thead>
      <tbody>{rt_rows}</tbody>
    </table>
    <div class="nf-noteworthy">
      Sleep cycles, run distance, shoes, rain and wind were tested and
      excluded as non-factors. Non-race quality efforts were also found
      to have no detectable next-day pace effect.
    </div>
  </details>
  <div class="nf-stats">
    n={n_fit:,} in fit; {n_pruned_unique} excluded (classes overlap)<br>
    R² = {r2_detrended:.3f} (factors only on detrended)<br>
    R² = {r2_raw:.3f} (era + factors on raw)<br>
    Trend: gaussian kernel, σ={TREND_SMOOTH_SIGMA_DAYS}d
  </div>
</div>
<script>
(function() {{
  // ORDER MUST MATCH customdata channel order in Python
  var FACTOR_ORDER = ['temp', 'route', 'recent_effort', 'time_of_day', 'era'];
  var ERA_INDEX = FACTOR_ORDER.indexOf('era');

  var TREND_SIGMA_MS = {TREND_SMOOTH_SIGMA_DAYS} * 86400000;
  var TREND_TRUNC_MS = 4 * TREND_SIGMA_MS;  // truncate kernel at 4σ
  var TREND_STEP_MS = 1 * 86400000;          // daily step
  var TREND_TWO_SIGSQ = 2 * TREND_SIGMA_MS * TREND_SIGMA_MS;
  var BASE_OPACITY = 0.6;

  function getPlot() {{ return document.querySelector('.plotly-graph-div'); }}

  function findTraces(plot) {{
    var idx = {{pace:-1, residual:-1, trendPace:-1, trendResid:-1}};
    plot.data.forEach(function(t, i) {{
      if (!t.meta) return;
      if (t.meta.role === 'pace') idx.pace = i;
      else if (t.meta.role === 'residual') idx.residual = i;
      else if (t.meta.role === 'trend_pace') idx.trendPace = i;
      else if (t.meta.role === 'trend_resid') idx.trendResid = i;
    }});
    return idx;
  }}

  function rollingTrend(dateMs, ys, mask) {{
    // Gaussian-kernel smoother, σ = TREND_SIGMA_MS, truncated at ±4σ.
    // mask: optional boolean array. True = include this point in the trend.
    if (dateMs.length === 0) return {{x:[], y:[]}};
    var t0 = dateMs[0], t1 = dateMs[dateMs.length - 1];
    var trendX = [], trendY = [];
    var lo = 0, hi = 0;
    for (var t = t0; t <= t1; t += TREND_STEP_MS) {{
      var lo_target = t - TREND_TRUNC_MS, hi_target = t + TREND_TRUNC_MS;
      while (lo < dateMs.length && dateMs[lo] < lo_target) lo++;
      while (hi < dateMs.length && dateMs[hi] <= hi_target) hi++;
      var sumWY = 0, sumW = 0, count = 0;
      for (var k = lo; k < hi; k++) {{
        if (mask && !mask[k]) continue;
        if (ys[k] == null || isNaN(ys[k])) continue;
        var dt = dateMs[k] - t;
        var w = Math.exp(-(dt*dt) / TREND_TWO_SIGSQ);
        sumWY += ys[k] * w;
        sumW  += w;
        count++;
      }}
      if (count >= 5) {{
        trendX.push(new Date(t));
        trendY.push(sumWY / sumW);
      }}
    }}
    return {{x: trendX, y: trendY}};
  }}

  function update() {{
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) {{ setTimeout(update, 100); return; }}
    var idx = findTraces(plot);
    if (idx.pace < 0 || idx.residual < 0) return;

    var checked = {{}};
    document.querySelectorAll('#norm-filter input[type=checkbox]').forEach(function(cb) {{
      checked[cb.dataset.factor] = cb.checked;
    }});

    var paceTrace = plot.data[idx.pace];
    var residTrace = plot.data[idx.residual];
    var rawPace = paceTrace.meta.raw_y;
    var rawResid = residTrace.meta.raw_y;
    var dateMs = residTrace.meta.date_ms;
    var isBadCond = residTrace.meta.is_bad_cond;
    var isPartner = residTrace.meta.is_partner_run;
    var isOutlier = residTrace.meta.is_outlier;
    var custom = paceTrace.customdata;

    var hideBadCond = !!checked['hide_bad_cond'];
    var hidePartner = !!checked['hide_partner'];
    var hideOutlier = !!checked['hide_outlier'];

    var n = rawPace.length;
    var newPace = new Array(n);
    var newResid = new Array(n);
    var newOpacity = new Array(n);
    var visibleMask = new Array(n);
    for (var i = 0; i < n; i++) {{
      var hidden = (hideBadCond && isBadCond[i]) ||
                   (hidePartner && isPartner[i]) ||
                   (hideOutlier && isOutlier[i]);
      if (hidden) {{
        // null y suppresses both rendering AND hover hit-testing
        newPace[i] = null;
        newResid[i] = null;
        newOpacity[i] = 0;
        visibleMask[i] = false;
        continue;
      }}
      var adjPace = 0, adjResid = 0;
      var c = custom[i];
      for (var j = 0; j < FACTOR_ORDER.length; j++) {{
        if (!checked[FACTOR_ORDER[j]]) continue;
        if (j === ERA_INDEX) {{
          adjResid += c[j];
        }} else {{
          adjPace += c[j];
          adjResid += c[j];
        }}
      }}
      newPace[i] = rawPace[i] - adjPace;
      newResid[i] = rawResid[i] - adjResid;
      newOpacity[i] = BASE_OPACITY;
      visibleMask[i] = true;
    }}

    Plotly.restyle(plot,
                   {{y: [newPace, newResid],
                     'marker.opacity': [newOpacity, newOpacity]}},
                   [idx.pace, idx.residual]);

    if (idx.trendPace >= 0 && idx.trendResid >= 0) {{
      var tp = rollingTrend(dateMs, newPace, visibleMask);
      var tr = rollingTrend(dateMs, newResid, visibleMask);
      Plotly.restyle(plot, {{x: [tp.x, tr.x], y: [tp.y, tr.y]}},
                     [idx.trendPace, idx.trendResid]);
    }}
  }}

  function setGroup(mode, on) {{
    var sel = '#norm-filter input[data-mode="' + mode + '"]';
    document.querySelectorAll(sel).forEach(function(cb) {{ cb.checked = on; }});
    update();
  }}

  function initialRender() {{
    var plot = getPlot();
    if (!plot || !plot.data || !window.Plotly) {{ setTimeout(initialRender, 100); return; }}
    update();
  }}

  document.querySelectorAll('#norm-filter input[type=checkbox]').forEach(function(cb) {{
    cb.addEventListener('change', update);
  }});
  document.getElementById('nf-norm-all').addEventListener('click', function() {{ setGroup('norm', true); }});
  document.getElementById('nf-norm-none').addEventListener('click', function() {{ setGroup('norm', false); }});
  document.getElementById('nf-hide-all').addEventListener('click', function() {{ setGroup('filter', true); }});
  document.getElementById('nf-hide-none').addEventListener('click', function() {{ setGroup('filter', false); }});

  // Custom tooltip — snaps to the actual data point with an arrow,
  // flips to the left side of the point when the point is in the
  // rightmost FLIP_LEFT_PX of the viewport (a fixed threshold so the
  // tooltip doesn't toggle back and forth as the cursor moves).
  function setupCustomTooltip() {{
    var plot = getPlot();
    if (!plot || !plot.on) {{ setTimeout(setupCustomTooltip, 100); return; }}
    var tt = document.getElementById('rec-tooltip');
    if (!tt) return;

    var FLIP_LEFT_PX = 360;  // flip when point is within this many px of right edge
    var POINT_GAP = 9;       // gap between point and tooltip edge (matches arrow)

    function pointPixelPos(p) {{
      var pdiv = getPlot();
      if (!pdiv || !pdiv._fullLayout) return null;
      var rect = pdiv.getBoundingClientRect();
      var xa = p.xaxis, ya = p.yaxis;
      if (!xa || !ya || !xa.c2p) return null;
      try {{
        var xVal = p.x;
        if (xVal instanceof Date) xVal = xVal.getTime();
        else if (typeof xVal === 'string') xVal = new Date(xVal).getTime();
        var pxX = xa.c2p(xVal);
        var pxY = ya.c2p(p.y);
        if (pxX == null || pxY == null) return null;
        return {{
          x: rect.left + xa._offset + pxX,
          y: rect.top + ya._offset + pxY
        }};
      }} catch (e) {{ return null; }}
    }}

    function position(pointScreenX, pointScreenY) {{
      var ttW = tt.offsetWidth, ttH = tt.offsetHeight;
      var flipLeft = pointScreenX > window.innerWidth - FLIP_LEFT_PX;
      var x, side;
      if (flipLeft) {{
        x = pointScreenX - POINT_GAP - ttW;
        side = 'left';
      }} else {{
        x = pointScreenX + POINT_GAP;
        side = 'right';
      }}
      var y = pointScreenY - ttH / 2;
      if (y < 4) y = 4;
      if (y + ttH > window.innerHeight - 4) y = window.innerHeight - ttH - 4;
      tt.setAttribute('data-side', side);
      tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    }}

    plot.on('plotly_hover', function(ev) {{
      if (!ev || !ev.points || !ev.points.length) return;
      var p = ev.points[0];
      var html = (p.text != null) ? p.text : '';
      if (!html) return;
      tt.innerHTML = html;
      tt.style.display = 'block';
      var pos = pointPixelPos(p);
      if (pos) position(pos.x, pos.y);
    }});

    plot.on('plotly_unhover', function() {{
      tt.style.display = 'none';
    }});

    plot.addEventListener('mouseleave', function() {{
      tt.style.display = 'none';
    }});
  }}
  setupCustomTooltip();

  initialRender();
}})();
</script>
"""


if __name__ == '__main__':
    main()
