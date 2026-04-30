"""Shared 5K-equivalent projection logic for race data points.

Both bayes_cs_plot.py (CS-timeline diamonds) and make_race_plots.py
(event-focused race plots) need to project race times to a common
5K-equivalent pace anchored to the posterior CS model. This module owns
that logic so the projection stays consistent across plots.

Projection
----------
At the race's date, look up D' from the model. Treat the race as a single
(d, t) sample of the hyperbolic distance-time relationship d = CS·t + D',
giving an implied CS_race = (d_race - D') / t_race. Project to the anchor
distance with the same D':

    t_anchor = (D_anchor - D') · t_race / (d_race - D')

For races above d_thresh_long (=10K), un-bias by dividing t_race by
(1 + β_long · log(d / d_thresh)) BEFORE projecting — the model's β_long
captures the systematic pacing fade above 10K. Without un-biasing, the
implied CS collapses toward the actual race pace and the 5K-equivalent
becomes uninformative for HM/marathon.

XC pre-correction (dividing time by 1+xc_correction) is OPTIONAL via the
`apply_xc_correction` flag. The CS plot applies it (matches the fit). The
event-focused all-races plot does NOT — we want to see the actual XC pace.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


def load_cs_outputs(in_dir, tag=''):
    """Load CS fit outputs and interpolate the summary to a daily grid.

    Looks for ``bayes_cs_summary{_tag}.csv`` (required) and
    ``bayes_cs_params{_tag}.csv`` (optional — falls back to β_long=0,
    d_thresh=10000m, xc_correction=0 with a stderr warning).

    Returns
    -------
    daily_summary : DataFrame
        Daily-resolution copy of the summary, columns include the originals
        (cs_pace_med, cs_pace_lo50/hi50/lo95/hi95, cs_mps_med, dp_med, ...)
        plus a derived ``p5k_implied_min`` column: the model's posterior-median
        5K-equivalent pace in min/mi (no β_long applied — this represents the
        CS belief, not race performance).
    beta_long, d_thresh_long, xc_correction : float
    """
    suffix = f'_{tag}' if tag else ''
    summary_path = os.path.join(in_dir, f'bayes_cs_summary{suffix}.csv')
    params_path  = os.path.join(in_dir, f'bayes_cs_params{suffix}.csv')

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'Summary not found: {summary_path}')

    summary = (pd.read_csv(summary_path, parse_dates=['date'])
                 .sort_values('date').reset_index(drop=True))

    if os.path.exists(params_path):
        params = pd.read_csv(params_path)
        beta_long = float(params['beta_long_med'].iloc[0])
        d_thresh  = float(params['d_thresh_long'].iloc[0])
        if 'xc_correction' in params.columns:
            xc = float(params['xc_correction'].iloc[0])
        elif 'beta_xc_med' in params.columns:
            # Backward-compat with older fit outputs
            xc = float(params['beta_xc_med'].iloc[0])
        else:
            xc = 0.0
    else:
        print(f'WARNING: params file not found: {params_path}\n'
              f'  Falling back to β_long=0, d_thresh=10000m, xc_correction=0.',
              file=sys.stderr)
        beta_long, d_thresh, xc = 0.0, 10000.0, 0.0

    daily = _interpolate_daily(summary)
    # CS-implied 5K pace in min/mi (no β_long applied — this IS the model's CS belief)
    daily['p5k_implied_min'] = (1609.344 * (5000.0 - daily['dp_med'])
                                / daily['cs_mps_med'] / 5000.0 / 60.0)
    return daily, beta_long, d_thresh, xc


def _interpolate_daily(summary):
    """Cubic-interpolate every numeric column of `summary` to daily resolution."""
    grid_t = (summary['date'] - summary['date'].iloc[0]).dt.days.values.astype(float)
    daily_dates = pd.date_range(summary['date'].iloc[0],
                                 summary['date'].iloc[-1], freq='D')
    daily_t = (daily_dates - summary['date'].iloc[0]).days.values.astype(float)
    out = {'date': daily_dates}
    for c in summary.columns:
        if c == 'date':
            continue
        out[c] = CubicSpline(grid_t, summary[c].values)(daily_t)
    return pd.DataFrame(out)


def _beta_factor(d, beta_long, d_thresh_long, beta_short, d_thresh_short):
    """The β factor at distance d. >1 for d>d_thresh_long (long-distance fade)
    and for d<d_thresh_short (short-distance calibration), 1 in between."""
    if d > d_thresh_long and beta_long > 0:
        return 1.0 + beta_long * np.log(d / d_thresh_long)
    if d < d_thresh_short and beta_short > 0:
        return 1.0 + beta_short * np.log(d_thresh_short / d)
    return 1.0


def project_races_to_5k_pace(races, daily_summary, beta_long, d_thresh,
                              *, apply_xc_correction=True, xc_correction=0.0,
                              norm_dist_m=5000.0,
                              beta_short=0.0, d_thresh_short=800.0):
    """Project each race row to a target ANCHOR distance via the hyperbolic CS model.

    The function name preserves backward compatibility — the by-distance plot
    calls this with norm_dist_m set per panel, not just 5000m. β factors are
    applied symmetrically (race distance and anchor distance both get one),
    so the formula is

        t_anchor = t_race × (norm_dist_m − D') / (d_race − D') × β_anchor / β_race

    where β_x = 1 + β_long·log(x/d_thresh_long) if x > d_thresh_long, else
    1 + β_short·log(d_thresh_short/x) if x < d_thresh_short, else 1.

    For the historical 5K-anchor call site, β_anchor = 1 (5K is in the
    unbiased zone), so this generalization is a no-op for that case.

    Parameters
    ----------
    races : DataFrame with at least date, distance_m, time_sec; surface optional.
    daily_summary : output of ``load_cs_outputs`` (must cover the race dates).
    beta_long, d_thresh : long-distance bias parameters from the CS fit.
    apply_xc_correction : when True, XC race times are divided by
        (1 + xc_correction) before projection — matches the fit. Set False
        for event-focused plots that should display actual XC race times.
    xc_correction : the correction factor (only used if apply_xc_correction).
    norm_dist_m : anchor distance in meters (default 5000m).
    beta_short, d_thresh_short : OPTIONAL plot-specific calibration for very
        short races. The CS+D' model breaks down below ~800m because peak
        speed limits and anaerobic capacity dominate, not sustained capacity.
        For d < d_thresh_short, t_eff is divided by
        (1 + β_short · log(d_thresh_short / d)) before projection.
        Default β_short=0 disables this (CS plot uses default — these are
        a hand-calibrated visualization adjustment, not a model fact).

    Returns
    -------
    DataFrame: copy of `races` with these columns added/preserved
        time_sec_original         : the original race time (always preserved)
        pace_sec_per_mi_original  : original pace if input had pace_sec_per_mi
        time_sec                  : the time fed to the projection (post-XC if applied)
        pace_sec_per_mi           : likewise
        pace_norm_min             : anchor-equivalent pace in min/mi (NaN if undefined)
        pace_norm_sec             : same in sec/mi
        time_norm_sec             : anchor-equivalent total time in seconds
    Rows where the projection is undefined (date outside summary, distance ≤ D',
    or non-positive time) keep NaN in pace_norm_min — caller decides whether
    to drop them.
    """
    df = races.copy()
    df['time_sec_original'] = df['time_sec'].astype(float)
    if 'pace_sec_per_mi' in df.columns:
        df['pace_sec_per_mi_original'] = df['pace_sec_per_mi'].astype(float)

    if apply_xc_correction and xc_correction > 0 and 'surface' in df.columns:
        is_xc = df['surface'].fillna('').astype(str).str.upper() == 'XC'
        if is_xc.any():
            factor = 1.0 / (1.0 + xc_correction)
            df.loc[is_xc, 'time_sec'] = df.loc[is_xc, 'time_sec'].astype(float) * factor
            if 'pace_sec_per_mi' in df.columns:
                df.loc[is_xc, 'pace_sec_per_mi'] = (
                    df.loc[is_xc, 'pace_sec_per_mi'].astype(float) * factor)

    lookup = daily_summary.set_index(daily_summary['date'].dt.date)
    beta_anchor = _beta_factor(norm_dist_m, beta_long, d_thresh, beta_short, d_thresh_short)

    def _proj(r):
        d = r['date']
        if hasattr(d, 'date'):
            d = d.date()
        if d not in lookup.index:
            return np.nan
        dp = float(lookup.loc[d, 'dp_med'])
        d_race = float(r['distance_m'])
        t_race = float(r['time_sec'])
        if d_race <= dp or t_race <= 0 or norm_dist_m <= dp:
            return np.nan
        beta_race = _beta_factor(d_race, beta_long, d_thresh, beta_short, d_thresh_short)
        t_eff = t_race / beta_race
        t_anchor = (norm_dist_m - dp) * t_eff / (d_race - dp) * beta_anchor
        return t_anchor / (norm_dist_m / 1609.344) / 60.0

    df['pace_norm_min'] = df.apply(_proj, axis=1)
    df['pace_norm_sec'] = df['pace_norm_min'] * 60.0
    df['time_norm_sec'] = df['pace_norm_sec'] * (norm_dist_m / 1609.344)
    return df


def cs_line_at_anchor(daily_summary, anchor_dist_m, beta_long, d_thresh_long,
                       *, beta_short=0.0, d_thresh_short=800.0):
    """CS-predicted total time (seconds) at anchor_dist_m for each daily date.

    t(date) = (anchor − D'(date)) / CS_mps(date) × β_anchor

    β_anchor handles long-distance bias (anchor > d_thresh_long) and
    short-distance calibration (anchor < d_thresh_short). Returns NaN for
    days where anchor ≤ D' (would be physically nonsensical).
    """
    dp = daily_summary['dp_med'].values
    cs_mps = daily_summary['cs_mps_med'].values
    beta_anchor = _beta_factor(anchor_dist_m, beta_long, d_thresh_long,
                                beta_short, d_thresh_short)
    t = (anchor_dist_m - dp) / cs_mps * beta_anchor
    t = np.where(anchor_dist_m > dp, t, np.nan)
    return t


def cubic_at_anchor(daily_summary, cubic_coefs, t0, handdrawn_start, handdrawn_end,
                     anchor_dist_m, beta_long, d_thresh_long,
                     *, beta_short=0.0, d_thresh_short=800.0):
    """Convert a hand-drawn 5K-equiv pace cubic into time at anchor_dist_m.

    The cubic gives pace in min/mi at 5K-equivalent (the same projection that
    placed race diamonds on the all-races plot). Per date in the cubic
    window, derive cs_mps from the cubic's pace_5k value and D'(date), then
    project cs_mps to a time at the new anchor with that anchor's β factor.

    Returns (dotted_dates, dotted_times_sec). Dates outside the daily
    summary range are dropped (we can't look up D' for them).
    """
    # Clip handdrawn window to summary range so D' lookup succeeds
    sm_min = daily_summary['date'].min()
    sm_max = daily_summary['date'].max()
    hd_start = max(pd.Timestamp(handdrawn_start), pd.Timestamp(sm_min))
    hd_end   = min(pd.Timestamp(handdrawn_end),   pd.Timestamp(sm_max))
    if hd_end <= hd_start:
        return pd.DatetimeIndex([]), np.array([])

    dotted_dates = pd.date_range(hd_start, hd_end, freq='D')
    dotted_ts = (dotted_dates - t0).days.values.astype(float)
    pace_5k_min = np.polyval(cubic_coefs, dotted_ts)
    # 5K time in seconds: pace (min/mi) × 5/1.609344 (miles) × 60 (sec/min)
    t_5k_sec = pace_5k_min * 5.0 / 1.609344 * 60.0

    dp_lookup = daily_summary.set_index(daily_summary['date'].dt.date)['dp_med']
    dp_values = np.array([float(dp_lookup.loc[d.date()]) for d in dotted_dates])

    # cs_mps from 5K time. The cubic was calibrated assuming 5K is in the
    # unbiased zone for both β_long (5K < 10K) and β_short (5K > 800m).
    cs_mps = (5000.0 - dp_values) / t_5k_sec

    beta_anchor = _beta_factor(anchor_dist_m, beta_long, d_thresh_long,
                                beta_short, d_thresh_short)
    t_anchor_sec = (anchor_dist_m - dp_values) / cs_mps * beta_anchor
    return dotted_dates, t_anchor_sec
