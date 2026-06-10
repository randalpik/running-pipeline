"""Recovery-pace factor model: OLS on era-detrended residual vs CS.

Lifted out of ``src/plots/make_recovery_plots.py`` so both the Recovery plot
and the Long Runs plot can import the same fit. ``run_plots.sh`` builds Long
Runs *before* Recovery, so a file artifact written by the recovery plot would
be stale or missing — an import is build-order-independent. Same precedent as
``long_run_model.py`` (lifted out of ``plot_training_quality.py``).

Model (fit on the era-detrended residual = pace − CS − era_trend):

  residual_detrended ~ β_temp · temp_centered
                     + Σ β_r · route_dummy_r       (n ≥ MIN_ROUTE_N)
                     + β_marathon · fatigue_marathon
                     + β_race    · fatigue_race_short
                     + β_tod     · tod_is_pm

Three classes of rows are pruned from BOTH the era-trend window contents AND
the OLS fit (flags are independent — a row can be in multiple classes):
bad conditions (snow/icy or "snow" in the workout string), partner runs, and
LOO outliers (|residual from leave-one-out 28-day local mean| > 45 sec/mi
against the clean neighbor pool). See the recovery plot docstring for the
full rationale and ``docs/recovery-runs-reference.md`` for the analysis
history (sleep, distance, shoes, non-race quality efforts tested and
excluded as non-factors).

``TRANSFERABLE_FEATURES`` names the per-day factors that are meaningful for
runs other than recovery runs (temperature, recent-race fatigue, time of
day). Route and era are recovery-specific: long-run route betas are fit
inside the TQ model instead (altitude/era confounding — see
``docs/route-normalization-reference.md``), and era trend is a smoothed
track of recovery residuals themselves.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
ERA_WINDOW_HALF_DAYS   = 182
ERA_WINDOW_MIN_POINTS  = 30
MIN_ROUTE_N            = 13
MARATHON_DISTANCE_M    = 42000

QUALITY_CATS = ['marathon', 'race_short']

# Per-day factors whose betas are meaningful beyond recovery runs (applied
# by the Long Runs plot's normalization toggle and the TQ long-run model
# experiments). Route and era are intentionally absent — see module docstring.
TRANSFERABLE_FEATURES = ['temp_centered', 'fat_marathon', 'fat_race_short',
                         'tod_is_pm']


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
            prev = pd.Timestamp(int(sd[idx - 1]))
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


def quality_category_dates(daily, races):
    """Sorted race-day dates per fatigue category, derived the same way the
    recovery fit derives them: daily race rows, categorized as 'marathon'
    when the date appears in races.csv at marathon distance or beyond."""
    marathon_dates = set(races.loc[races['distance_m'] >= MARATHON_DISTANCE_M,
                                    'date'].tolist())
    is_race = daily['run_type'] == 'race'
    out = {}
    out['marathon'] = sorted(
        daily.loc[is_race & daily['date'].isin(marathon_dates), 'date'].tolist())
    out['race_short'] = sorted(
        daily.loc[is_race & ~daily['date'].isin(marathon_dates), 'date'].tolist())
    return out


def add_quality_features(df, quality_dates):
    """Add ``dsq_<cat>`` (days since most recent <cat> race) and ``fat_<cat>``
    (exponential fatigue decay, exp(−days/τ)) columns. Returns a copy."""
    df = df.copy()
    for cat in QUALITY_CATS:
        df[f'dsq_{cat}'] = days_since(df, quality_dates.get(cat, []))
        df[f'fat_{cat}'] = np.exp(-df[f'dsq_{cat}'].fillna(np.inf)
                                  / FATIGUE_TAU_DAYS[cat])
    return df


def tod_is_pm(df):
    """1.0 for afternoon/late, 0.0 for early/morning or missing (AM baseline)."""
    tod_clean = df['time_of_day'].astype(str).str.strip().str.lower()
    return tod_clean.isin(TOD_PM_VALUES).astype(float)


def transferable_contributions(df, betas, quality_dates):
    """Per-row modeled pace contribution (sec/mi) of the transferable
    factors — temperature, recent-race fatigue, time of day — for any
    daily-frame subset (e.g. long runs). Missing temp/TOD contribute 0.
    Subtracting the result from observed pace normalizes those factors out.
    """
    df = add_quality_features(df, quality_dates)
    contrib = betas.get('temp_centered', 0.0) * (
        (df['temp_c'] - TEMP_REFERENCE_C).fillna(0.0))
    for cat in QUALITY_CATS:
        contrib = contrib + betas.get(f'fat_{cat}', 0.0) * df[f'fat_{cat}']
    contrib = contrib + betas.get('tod_is_pm', 0.0) * tod_is_pm(df)
    return contrib.to_numpy(dtype=float)


# ---------- fit ----------

def fit_recovery_model(daily, races, cs_summary, verbose=True):
    """Fit the recovery factor model on loaded frames.

    ``daily`` is the (already era-filtered) daily log; ``races`` is
    races.csv; ``cs_summary`` needs ``date`` plus ``cs_pace_sec`` (computed
    from ``cs_pace_med`` if absent). Returns a SimpleNamespace with
    everything the recovery plot consumes — ``rec`` (recovery rows with
    feature/flag/contrib columns), ``betas``, ``intercept``, ``r2_detrended``,
    ``r2_raw``, ``n_fit``, ``qualifying_routes``, ``route_col_map``,
    ``route_counts``, ``global_mean_residual``, ``quality_dates`` — or
    ``None`` when there isn't enough data to fit (sparse profiles).
    """
    if 'cs_pace_sec' not in cs_summary.columns:
        cs_summary = cs_summary.copy()
        cs_summary['cs_pace_sec'] = cs_summary['cs_pace_med'] * 60.0

    quality_dates = quality_category_dates(daily, races)

    # Recovery subset
    rec = daily[daily['run_type'] == 'recovery'].copy()
    rec = rec.dropna(subset=['recovery_pace_sec_per_mi'])
    rec = rec.sort_values('date').reset_index(drop=True)
    if rec.empty:
        if verbose:
            print('  0 recovery days with valid pace — skipping fit')
        return None
    # CS belief at each run's date via np.interp, which HOLDS the endpoint value
    # beyond the CS summary's range — the same extrapolation add_cs() uses for
    # every other plot. A run is therefore never dropped for lack of an exact
    # CS-date match, so recovery stays current even if the Bayesian fit is
    # stale. With the fit now spanning the full run history (bayes_cs_fit grid),
    # the lookup is in-range in the common case; the hold-last is the safety net
    # that previously caused recent runs to vanish (when this was a merge+dropna).
    _epoch = pd.Timestamp('1970-01-01')
    rec['cs_pace_sec'] = np.interp(
        (rec['date'] - _epoch).dt.days.to_numpy(),
        (cs_summary['date'] - _epoch).dt.days.to_numpy(),
        cs_summary['cs_pace_sec'].values,
    )
    if verbose:
        print(f'  {len(rec)} recovery days with valid pace')

    # Features (sleep_centered and miles_centered are intentionally NOT computed)
    rec['temp_centered'] = rec['temp_c'] - TEMP_REFERENCE_C
    rec['residual_raw'] = rec['recovery_pace_sec_per_mi'] - rec['cs_pace_sec']

    # Time-of-day binary indicator: 1 for PM (afternoon/late), 0 for AM
    # (early/morning). Recovery rows have 100% TOD coverage; missing or
    # unknown values fall to 0 (AM baseline).
    rec['tod_is_pm'] = tod_is_pm(rec)

    rec['conditions_clean'] = rec['conditions'].astype(str).str.strip().str.lower()
    cond_excluded = rec['conditions_clean'].isin(EXCLUDED_CONDITIONS)
    workout_snow = rec['workout_raw'].fillna('').astype(str).apply(
        lambda w: bool(SNOW_IN_WORKOUT_RE.search(w)))
    rec['is_bad_cond'] = cond_excluded | workout_snow
    rec['workout_snow_only'] = workout_snow & ~cond_excluded

    # Partner-run detection
    # fillna('') before astype(str): pandas 3.0's astype(str) leaves NaN as NaN
    # (not the string 'nan'), which would fall outside NULL_PARTNERS and flag
    # every blank-partner run as a partner run — catastrophic for a profile
    # with no partners data at all (the whole fit gets pruned away).
    rec['partners_clean'] = (rec['partners'].fillna('')
                             .astype(str).str.strip().str.lower())
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
        nbr_pace = rec.loc[neighbor_mask, 'recovery_pace_sec_per_mi'].to_numpy()
        all_dates_ms = np.array([d.value // 10**6 for d in rec['date']])
        all_pace = rec['recovery_pace_sec_per_mi'].to_numpy()
        in_pool = neighbor_mask.to_numpy()
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
    if verbose:
        print(f'  pruned from fit: {rec["is_pruned"].sum()} unique '
              f'({rec["is_bad_cond"].sum()} bad-cond '
              f'[{cond_excluded.sum()} cond + {workout_snow.sum()} workout-snow], '
              f'{rec["is_partner_run"].sum()} partner-runs, '
              f'{rec["is_outlier_loo"].sum()} outliers '
              f'[{n_outlier_clean} clean + {n_outlier_bad} also bad-cond + '
              f'{n_outlier_partner} also partner])')

    rec = add_quality_features(rec, quality_dates)

    # Qualifying routes
    route_counts = (rec.loc[~rec['is_pruned'], 'location']
                     .dropna().value_counts())
    qualifying_routes = (route_counts[route_counts >= MIN_ROUTE_N]
                         .index.tolist())
    qualifying_routes = sorted(qualifying_routes,
                                key=lambda x: -route_counts[x])
    if verbose:
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
    if verbose:
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
    if len(rec_fit) <= len(feature_cols) + 1:
        if verbose:
            print(f'  only {len(rec_fit)} fit-eligible rows for '
                  f'{len(feature_cols)} features — skipping fit')
        return None

    X = rec_fit[feature_cols].to_numpy().astype(float)
    y = rec_fit['residual_detrended'].to_numpy().astype(float)
    X_int = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    intercept = float(coef[0])
    betas = {f: float(b) for f, b in zip(feature_cols, coef[1:])}

    yhat = X_int @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_detrended = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    raw_y = rec_fit['residual_raw'].to_numpy()
    raw_yhat = rec_fit['era_trend'].to_numpy() + yhat
    ss_res_raw = float(np.sum((raw_y - raw_yhat) ** 2))
    ss_tot_raw = float(np.sum((raw_y - raw_y.mean()) ** 2))
    r2_raw = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw > 0 else 0.0

    if verbose:
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

    return SimpleNamespace(
        rec=rec,
        betas=betas,
        intercept=intercept,
        r2_detrended=r2_detrended,
        r2_raw=r2_raw,
        n_fit=len(rec_fit),
        qualifying_routes=qualifying_routes,
        route_col_map=route_col_map,
        route_counts=route_counts,
        global_mean_residual=global_mean_residual,
        quality_dates=quality_dates,
    )
