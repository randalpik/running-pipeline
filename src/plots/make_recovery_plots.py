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
  python src/plots/make_recovery_plots.py

Defaults read CS fit outputs, daily.csv, and races.csv from data/ and write
the HTML into output/. Override with --in-dir, --daily, --races, --out-dir.

Dependencies: pandas, numpy, plotly.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            sec_to_mss, FG,
                            CS_LINE, CS_LINE_WIDTH, TREND_LINE, TREND_WIDTH,
                            GRID, gaussian_rolling_trend)
from src.shared.cs_projection import load_cs_outputs


DEFAULT_IN_DIR = str(DATA_DIR)
DEFAULT_DAILY  = str(DATA_DIR / 'daily.csv')
DEFAULT_RACES  = str(DATA_DIR / 'races.csv')
DEFAULT_OUT    = str(OUTPUT_DIR)


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

def signed_sec(s):
    """Plot-local signed-seconds formatter — emits ``+30`` / ``-30`` (no
    M:SS conversion). Differs from formatters.signed_sec, which emits
    ``+0:30`` / ``-0:30``. Recovery's residual ticks and hover already
    carry the ``sec/mi`` label so a count is clearer than a duration."""
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
    ap.add_argument('--tag', default='')
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
        # Per-run HTML for the smart-spikeline scaffold's snap mode and the
        # smooth-mode "Nearest run" section. The cursor scaffold renders the
        # date and the CS-pace/Trend-pace/CS-residual/Trend-residual rows
        # itself, so this content focuses on what's run-specific.
        parts = [f"Pace: {sec_to_mss(row['recovery_pace_sec_per_mi'])}/mi  "
                 f"({row['miles']:.1f} mi)"]
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
        mode='lines', line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
        name='CS pace', hoverinfo='skip', showlegend=True,
        meta={'role': 'reference'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[rec['date'].min(), rec['date'].max()], y=[0, 0],
        mode='lines', line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
        name='Zero', hoverinfo='skip', showlegend=False,
        meta={'role': 'reference'},
    ), row=1, col=2)

    # Pre-compute the initial (no-normalization-applied, no-hides-applied)
    # trend so the line renders on first paint. The JS rollingTrend reruns
    # with the same algorithm whenever the user toggles a factor or filter.
    rec_dates_ms = np.array([d.value // 10**6 for d in rec['date']],
                            dtype=np.int64)
    init_pace_x, init_pace_y = gaussian_rolling_trend(
        rec_dates_ms, rec['recovery_pace_sec_per_mi'].values,
        sigma_days=TREND_SMOOTH_SIGMA_DAYS)
    init_resid_x, init_resid_y = gaussian_rolling_trend(
        rec_dates_ms, rec['residual_raw'].values,
        sigma_days=TREND_SMOOTH_SIGMA_DAYS)
    init_pace_dates  = pd.to_datetime(init_pace_x,  unit='ms')
    init_resid_dates = pd.to_datetime(init_resid_x, unit='ms')

    fig.add_trace(go.Scatter(
        x=init_pace_dates, y=init_pace_y, mode='lines',
        line=dict(color=TREND_LINE, width=TREND_WIDTH),
        name='Trend', hoverinfo='skip', showlegend=True,
        meta={'role': 'trend_pace'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=init_resid_dates, y=init_resid_y, mode='lines',
        line=dict(color=TREND_LINE, width=TREND_WIDTH),
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

    # Snap HTML lives on the residual trace's text field — same content as
    # the pace trace's, since both panels represent the same run. (The
    # per-trace hover html lookup falls back to text when customdata is a
    # structured array, which is the case here — customdata holds the
    # factor-contribution vector used by update().)
    hover_html = rec['_hover'].tolist()
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
        text=hover_html,
        hoverinfo='skip',
        name='Recovery (pace)',
        showlegend=False,
        meta={'role': 'pace',
              'snap_eligible': True,
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
        text=hover_html,
        hoverinfo='skip',
        name='Recovery (residual)',
        showlegend=False,
        meta={'role': 'residual',
              'snap_eligible': True,
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
                          gridcolor=GRID,
                          row=1, col=col)

    fig.update_yaxes(tickvals=left_ticks,
                      ticktext=[sec_to_mss(t) for t in left_ticks],
                      range=[left_y_hi, left_y_lo],
                      title_text='Recovery pace (per mile)',
                      gridcolor=GRID,
                      row=1, col=1)
    fig.update_yaxes(tickvals=right_ticks,
                      ticktext=[signed_sec(t) if t != 0 else '0' for t in right_ticks],
                      range=[right_y_hi, right_y_lo],
                      title_text='Residual (sec/mi above CS)',
                      gridcolor=GRID,
                      zeroline=False,
                      row=1, col=2)

    apply_default_layout(
        fig,
        font=dict(color=FG, size=12),
        margin=dict(l=70, r=300, t=40, b=70),
        hoverlabel=dict(bgcolor='#222', bordercolor='#555',
                        font=dict(color=FG, size=12)),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.005, xanchor='left',
            y=0.10, yanchor='top',  # just below the colorbar (which ends at y≈0.15)
            bgcolor='rgba(26,26,26,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color=FG, size=11),
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
                            font=dict(color=FG, size=11)),
                orientation='v', x=1.005, xanchor='left',
                y=0.45, yanchor='top',  # below the toolbar's typical footprint
                len=0.30, thickness=10,
                tickvals=[-10, 0, 10, 22, 30, 40],
                tickfont=dict(color=FG, size=10),
                outlinewidth=0,
            ),
        ),
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color=FG, size=14)

    # ---------- spikeline tooltip payload ----------
    # Per-day arrays the JS uses for the trend section, plus a sorted list
    # of sessions for nearest-run lookup. Both smooth and snap modes read
    # from the same payload — the only difference is the snap path uses
    # the snapped point's day directly (so trend rows reflect that exact
    # day) and skips the date header + "Nearest run" caption.
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start = pd.Timestamp(rec['date'].min().normalize() - pd.Timedelta(days=30))
    plot_end   = pd.Timestamp(rec['date'].max().normalize() + pd.Timedelta(days=30))
    grid_dates = pd.date_range(plot_start, plot_end, freq='D')
    grid_day_idx = ((grid_dates - js_epoch).days).astype(int)

    def _interp_to_grid(xs_ms, ys):
        if len(xs_ms) == 0:
            return [None] * len(grid_day_idx)
        xs_day = xs_ms.astype(np.float64) / 86_400_000.0
        out = np.interp(grid_day_idx.astype(float), xs_day, ys,
                        left=np.nan, right=np.nan)
        return [None if np.isnan(v) else round(float(v), 2) for v in out]

    cs_pace_per_day    = _interp_to_grid(
        np.array([d.value // 10**6 for d in cs_for_plot['date']], dtype=np.int64),
        cs_for_plot['cs_pace_sec'].values)
    trend_pace_per_day = _interp_to_grid(init_pace_x,  init_pace_y)
    trend_resid_per_day = _interp_to_grid(init_resid_x, init_resid_y)

    sessions = []
    for _, r in rec.iterrows():
        sessions.append({
            'day':   int((r['date'] - js_epoch).days),
            'resid': float(r['residual_raw']) if pd.notna(r['residual_raw']) else None,
            'html':  r['_hover'],
        })
    sessions.sort(key=lambda s: s['day'])

    payload = {
        'first_day':   int(grid_day_idx[0]),
        'cs_pace':     cs_pace_per_day,
        'trend_pace':  trend_pace_per_day,
        'trend_resid': trend_resid_per_day,
        'sessions':    sessions,
        'nearest_window_days': 60,
    }
    smooth_build_js = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.cs_pace.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function paceMSS(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    var mn = Math.floor(x / 60), sc = x % 60;
    return mn + ':' + (sc < 10 ? '0' : '') + sc;
  }
  function signedSec(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    if (x === 0) return '0';
    return (x > 0 ? '+' : '−') + Math.abs(x);
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  // Date header in both modes — in snap mode it identifies the snapped
  // point's day, in smooth mode it identifies the cursor's date.
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: trend info for the absolute-pace panel.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>CS pace</span><b>' + paceMSS(P.cs_pace[idx]) + '/mi</b></div>';
  html += '<div class="tt-row"><span>Trend pace</span><b>' + paceMSS(P.trend_pace[idx]) + '/mi</b></div>';
  html += '</div>';

  // Section 2: residual + run details. In smooth mode, "Nearest run [+/-d]"
  // header points at a run within ±nearest_window_days; in snap mode the
  // section is unlabeled and references the snapped point directly.
  var run = null;
  var s = P.sessions;
  if (isSnap) {
    // Find session at the snapped day (recovery has at most one per day).
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    if (s[lo] && s[lo].day === day) run = s[lo];
  } else if (s.length) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    var cands = [s[lo]];
    if (lo > 0) cands.push(s[lo - 1]);
    var best = null, bestAbs = 9999;
    for (var k = 0; k < cands.length; k++) {
      var ad = Math.abs(cands[k].day - day);
      if (ad < bestAbs) { bestAbs = ad; best = cands[k]; }
    }
    if (best && bestAbs <= P.nearest_window_days) run = best;
  }
  if (run) {
    var residIdx = run.day - P.first_day;
    var trResid = (residIdx >= 0 && residIdx < P.trend_resid.length)
                  ? P.trend_resid[residIdx] : null;
    html += '<div class="tt-section">';
    if (!isSnap) {
      var dd2 = run.day - day;
      var lbl = dd2 === 0 ? 'same day'
              : (dd2 > 0 ? '+' + dd2 + ' day' + (dd2 === 1 ? '' : 's')
                         :  dd2 + ' day' + (dd2 === -1 ? '' : 's'));
      html += '<div class="tt-section-title">Nearest run [' + lbl + ']</div>';
    }
    html += '<div class="tt-row"><span>CS residual</span><b>' + signedSec(run.resid) + ' sec/mi</b></div>';
    html += '<div class="tt-row"><span>Trend residual</span><b>' + signedSec(trResid) + ' sec/mi</b></div>';
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    first_day = int(grid_day_idx[0])
    last_day  = int(grid_day_idx[-1])

    out_path = os.path.join(args.out_dir, 'recovery_pace.html')
    render_plot(
        fig, out_path,
        title_slug='recovery_pace',
        page_title='Recovery',
        title='Recovery runs vs. race fitness',
        subtitle='Absolute pace and relative fitness signal, '
                 'controlling for combinations of factors',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=smooth_build_js,
            first_day=first_day,
            last_day=last_day,
        ),
        overlay_html=build_normalization_ui(
            betas, intercept, r2_detrended, r2_raw, len(rec_fit),
            int(rec['is_bad_cond'].sum()),
            int(rec['is_partner_run'].sum()),
            int(rec['is_outlier_loo'].sum()),
            int(rec['is_pruned'].sum()),
            qualifying_routes, route_col_map, route_counts,
            global_mean_residual),
        extra_head_css=(
            '@media (max-width:760px){'
            '#norm-filter{position:static!important;'
            'margin:8px;width:auto!important;}}'
        ),
    )
    print(f'\nWrote {out_path}')

    # Export route betas for downstream consumption (e.g. long-run TQ
    # corrections). The route betas are recovery-only, so they're effort-
    # uncontaminated and represent the pace cost of each route at easy
    # effort. The elev/terrain model parameters from the cross-route
    # exploration are NOT exported here since they're a separate analysis.
    suffix = f'_{args.tag}' if args.tag else ''
    betas_csv = os.path.join(args.out_dir, f'route_betas{suffix}.csv')
    betas_df = pd.DataFrame([
        {'route': r, 'n': int(route_counts[r]), 'beta_sec_per_mi': betas[route_col_map[r]]}
        for r in qualifying_routes
    ]).sort_values('beta_sec_per_mi')
    betas_df.to_csv(betas_csv, index=False)
    print(f'Wrote {betas_csv} ({len(betas_df)} routes)')


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
/* Suppress Plotly's built-in hover label — the smart spikeline scaffold
   renders the tooltip via .rp-tooltip / .rp-spike. */
.hovertext {{ display: none !important; }}
</style>
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

  // Initial paint shows the no-normalization-applied scatter and trend
  // exactly as Python wrote them — no JS recompute needed. update() only
  // fires from now on when the user toggles a checkbox or button.

  document.querySelectorAll('#norm-filter input[type=checkbox]').forEach(function(cb) {{
    cb.addEventListener('change', update);
  }});
  document.getElementById('nf-norm-all').addEventListener('click', function() {{ setGroup('norm', true); }});
  document.getElementById('nf-norm-none').addEventListener('click', function() {{ setGroup('norm', false); }});
  document.getElementById('nf-hide-all').addEventListener('click', function() {{ setGroup('filter', true); }});
  document.getElementById('nf-hide-none').addEventListener('click', function() {{ setGroup('filter', false); }});

  // Tooltip rendering is handled by the smart spikeline scaffold (see
  // src/plotting/_scaffold/cursor_tooltip.js); this overlay only owns the
  // normalization sidebar and the plotly_restyle recompute loop.
}})();
</script>
"""


if __name__ == '__main__':
    main()
