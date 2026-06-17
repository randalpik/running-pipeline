"""Shared 5K-equivalent projection logic for race data points.

Both bayes_cs_plot.py (CS-timeline diamonds) and make_race_plots.py
(event-focused race plots) need to project race times to a common
5K-equivalent pace anchored to the posterior CS model. This module owns
that logic so the projection stays consistent across plots.

Projection — the 3-parameter critical-power model (CP3, June 2026)
------------------------------------------------------------------
Every effort is treated as a sample of the Morton 3-parameter CP curve

    v(t) = CS + D′ / (t + τ),   τ = D′ / (v_max − CS)

i.e. the classic hyperbola d = CS·t + D′ bent at short durations so the
model's instantaneous speed is capped at a finite v_max instead of
diverging as t → 0. This replaced the former β_short display knob (a
time-inflation factor below 875 m, calibrated to two PR outcomes): the
2-parameter hyperbola is structurally wrong below ~800 m — at the
prediction level it over-credits (a 39 s 400 m "prediction"), at the
projection level it under-credits (a 57 s 400 m read as a 20-min-5K
fitness) — and the CP3 bend fixes both directions with a shape instead of
a knob. v_max itself is an uncertainty interval whose two edges serve the
two directions (see the v_max section below).
See docs/cs-model-reference.md ("Projection method: CP3").

At the race's date, look up D′ from the model and solve the race's own
implied CS from (d, t) — a closed-form quadratic, since τ depends on the
CS being solved for. The solve is SELF-CONSISTENT: the effort defines its
own curve, and the fit's current-fitness CS never enters (short races
stay independent evidence, not circular echoes of the CS line). Then
forward-solve the time at the anchor distance on that implied curve.

For races above d_thresh_long (=10K), un-bias by dividing t_race by
(1 + β_long · log(d / d_thresh)) BEFORE projecting — the model's β_long
captures the systematic pacing fade above 10K. Without un-biasing, the
implied CS collapses toward the actual race pace and the 5K-equivalent
becomes uninformative for HM/marathon. (β_long is race-execution
physiology above 10K and is orthogonal to the short-duration bend.)

XC pre-correction (dividing time by 1+xc_correction) is OPTIONAL via the
`apply_xc_correction` flag. The CS plot applies it (matches the fit). The
event-focused all-races plot does NOT — we want to see the actual XC pace.

D′ bridging (CP2 fit → CP3 projection layer)
--------------------------------------------
The Bayesian fit stays 2-parameter (races < 1500 m are excluded from it,
where CP2 is fine). Its D′ posterior is therefore the EFFECTIVE anaerobic
distance delivered over efforts in the fitted range, anchored at 5K — under
CP3 the same fit data implies a slightly larger full reservoir. Per date:

    D′₃ = D′₂ / (1 − D′₂ / ((v_max − CS) · t5K)),  t5K = (5000 − D′₂)/CS

chosen so the CP3 curve through (CS, D′₃) reproduces the fit's 5K
prediction EXACTLY — the CS-implied-5K line, the frontier floor, and the
TQ residual frame are all unchanged by the CP3 switch, and the two models
agree within ~1 s/mi across the fitted 1500 m–10K range. Effects are
confined to where the fit has no opinion: sub-1500 m efforts.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

from src.shared.units import METERS_PER_MILE


# ---------------------------------------------------------------------------
# v_max — the CP3 model's instantaneous maximum speed (m/s), per profile.
#
# v_max is an UNCERTAINTY INTERVAL, not a point estimate (Max, June 2026).
# The sprint corpus can't identify it (the per-race demonstrated values
# scatter 7.6–9.4 with same-day spreads of 1.3 m/s — execution noise, not
# a trend; a time-varying v_max(t) was considered and rejected on those
# numbers). Max's two hard invariants — (1) 400 m races must never exceed
# the frontier, (2) 400/800 predictions must never beat the lifetime PRs —
# are jointly unsatisfiable by any single curve: the 400 PR was run ~3
# months after the frontier's 2023 peak, so a curve that never promises
# sub-PR at the peak must read the August PR as beating the August
# frontier. Each direction therefore takes its conservative edge:
#
#   EVIDENCE (reading short races as 5K proof — diamonds): v_max_evidence,
#   the HIGH edge — "the sprint could have carried much of that 400;
#   credit the engine modestly." Calibrated as the lifetime envelope of
#   sprint-credit demonstrated by the 400 corpus: the largest per-race
#   implied v_max (9.19, a 2017 fatigued 57.4 — with the 800 demos
#   binding the frontier nearby) + margin, so every 400 ever raced sits
#   at/behind the frontier by construction. 800s and longer are frontier
#   DEMOS (≥ 120 s) and may bind it — Max verified the binding 800s are
#   coherent genuine peaks; a ≥ 1500 m demo cutoff was tried and reverted.
#
#   PREDICTION (forward-solving short times from aerobic fitness — the
#   dashboard cards, per-panel frontier lines): v_max_predict, the LOW
#   edge — "assume little sprint contribution." Calibrated under the
#   PR-sweep bounds (8.52/8.55 from the 800/400 anchors) with ~0.5 s
#   margin: no date in history gets a 400/800 prediction faster than the
#   lifetime PRs. The asymmetry is also what lets a BINDING 800 demo
#   coexist with this invariant: conservative read in, conservative
#   prediction out — equality is never forced.
#
# This is the frontier's own epistemology (demonstrations are one-sided
# proofs, not symmetric estimates) extended to the v_max axis, and it
# encodes Max's self-knowledge that equivalence tables overrate his short
# distances: conversions are deliberately LOSSY both ways — a 400 is weak
# evidence of 5K fitness, and 5K fitness is a weak promise of 400 speed.
# Setting both edges equal recovers a symmetric model. Derivations:
# scripts/calibrate_vmax.py; re-run after a CS refit.
#
# The workout accumulator deliberately does NOT use these edges — its
# deflation v_max (workouts.workout_vmax()) is a measurement calibration
# anchored to Max's watch rep corpus (CS-scaled for other profiles), not a
# conservatism policy.
#
# Other profiles: no sprint corpus → symmetric literature-plausible
# defaults for a trained distance runner.
# ---------------------------------------------------------------------------
VMAX_EVID_DEFAULT = 8.5
VMAX_PRED_DEFAULT = 8.5
VMAX_EVID_BY_PROFILE = {
    'max': 9.5,    # 400-corpus sprint-credit envelope (9.19) + margin
}
VMAX_PRED_BY_PROFILE = {
    'max': 8.3,    # below the PR-sweep bounds (8.52 / 8.55), ~0.5 s margin
}


def vmax_evidence():
    """The active profile's evidence-side (high-edge) v_max (m/s).
    RP_VMAX_EVID overrides for calibration sweeps."""
    env = os.environ.get('RP_VMAX_EVID')
    if env:
        return float(env)
    profile = os.environ.get('RP_PROFILE', 'max')
    return VMAX_EVID_BY_PROFILE.get(profile, VMAX_EVID_DEFAULT)


def vmax_predict():
    """The active profile's prediction-side (low-edge) v_max (m/s).
    RP_VMAX_PRED overrides for calibration sweeps."""
    env = os.environ.get('RP_VMAX_PRED')
    if env:
        return float(env)
    profile = os.environ.get('RP_PROFILE', 'max')
    return VMAX_PRED_BY_PROFILE.get(profile, VMAX_PRED_DEFAULT)


def cp3_dprime(dp2, cs_mps, vmax):
    """CP3 full anaerobic reservoir D′₃ from the CP2 fit's (D′₂, CS) — the
    5K-preserving bridge (see module docstring). vmax must be the SAME edge
    the resulting D′₃ will be used with, so each direction's curve is
    self-consistent. Vectorized."""
    dp2 = np.asarray(dp2, float)
    cs = np.asarray(cs_mps, float)
    t5k = (5000.0 - dp2) / cs
    return dp2 / (1.0 - dp2 / ((vmax - cs) * t5k))


def cp3_implied_cs(d_m, t_s, dp3, vmax):
    """Implied CS (m/s) of an effort covering d_m metres in t_s seconds on
    its own CP3 curve with reservoir dp3 and cap vmax.

    Solving v = CS + D′/(t + D′/(v_max − CS)) for CS with v = d/t gives a
    quadratic in u = v_max − CS:

        u = [ (v_max − v) + sqrt((v_max − v)² + 4·D′·(v_max − v)/t) ] / 2

    (positive root; CS = v_max − u). CP2 limit checks out: v_max → ∞ gives
    CS → v − D′/t. NaN where v ≥ v_max (the effort is off the model — can't
    happen for honest data once v_max is calibrated above the fastest race
    speed) or where the implied CS is non-positive. Vectorized.
    """
    d = np.asarray(d_m, float)
    t = np.asarray(t_s, float)
    dp3 = np.asarray(dp3, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = d / t
        gap = vmax - v
        u = 0.5 * (gap + np.sqrt(gap * gap + 4.0 * dp3 * gap / t))
        cs = vmax - u
        cs = np.where((gap > 0) & (t > 0) & (cs > 0), cs, np.nan)
    return cs


def cp3_time(d_m, cs_mps, dp3, vmax):
    """Time (s) to cover d_m metres on the CP3 curve (cs, dp3, vmax) — the
    forward direction (predictions, lines-at-anchor).

    d = CS·t + D′·t/(t + τ) is a quadratic in t:

        CS·t² + (CS·τ + D′ − d)·t − d·τ = 0,   τ = D′/(v_max − CS)

    Positive root. Unlike CP2 there is no d ≤ D′ blow-up — the curve is
    well-defined down to d → 0 (t → d/v_max). NaN where cs ≥ vmax or
    cs ≤ 0. Vectorized.
    """
    d = np.asarray(d_m, float)
    cs = np.asarray(cs_mps, float)
    dp3 = np.asarray(dp3, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        tau = dp3 / (vmax - cs)
        b = cs * tau + dp3 - d
        t = (-b + np.sqrt(b * b + 4.0 * cs * d * tau)) / (2.0 * cs)
        t = np.where((cs > 0) & (cs < vmax) & (d > 0), t, np.nan)
    return t


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
    # CS-implied 5K pace in min/mi (no β_long applied — this IS the model's CS
    # belief). Identical under CP2 and CP3 — the D′₃ bridge preserves the 5K
    # prediction by construction, on BOTH edges.
    daily['p5k_implied_min'] = (METERS_PER_MILE * (5000.0 - daily['dp_med'])
                                / daily['cs_mps_med'] / 5000.0 / 60.0)
    # CP3 full anaerobic reservoir per date, one per v_max edge (see the
    # module docstring): evidence (reading efforts) and prediction (forward
    # solves) each get a self-consistent curve.
    daily['dp3_evid_med'] = cp3_dprime(daily['dp_med'], daily['cs_mps_med'],
                                       vmax_evidence())
    daily['dp3_pred_med'] = cp3_dprime(daily['dp_med'], daily['cs_mps_med'],
                                       vmax_predict())
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


def _beta_long_factor(d, beta_long, d_thresh_long):
    """The β_long factor at distance d: >1 for d > d_thresh_long (the fit's
    long-distance race-execution fade), 1 otherwise. Short distances carry
    no factor — the CP3 bend handles them structurally."""
    if d > d_thresh_long and beta_long > 0:
        return 1.0 + beta_long * np.log(d / d_thresh_long)
    return 1.0


def project_races_to_5k_pace(races, daily_summary, beta_long, d_thresh,
                              *, apply_xc_correction=True, xc_correction=0.0,
                              apply_physical_correction=True, daily=None,
                              norm_dist_m=5000.0):
    """Project each race row to a target ANCHOR distance via the CP3 model.

    The function name preserves backward compatibility — the by-distance plot
    calls this with norm_dist_m set per panel, not just 5000m. Per race:

        t_eff     = t_race / β_long(d_race)          (un-bias the fade >10K)
        cs_imp    = cp3_implied_cs(d_race, t_eff)    (the race's own curve)
        t_anchor  = cp3_time(norm_dist_m, cs_imp) × β_long(norm_dist_m)

    For the historical 5K-anchor call site both β_long factors are 1.
    The former β_short display knob is gone — sub-875m races now ride the
    CP3 bend like every other distance (see module docstring).

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

    # Physical route correction (grade + off-road footing + altitude) — convert
    # each watch-covered race to its flat / sea-level / smooth-equivalent time
    # so the projection (and the frontier it feeds) matches what informs the CS
    # fit (docs/cs-model-reference.md "Race physical correction"). MUST stay consistent with
    # the same helper applied in bayes_cs_fit. Where a race has a measured
    # correction the categorical XC factor is turned OFF for it (measured wins;
    # categorical is the pre-watch fallback). dt_sec is 0 on unmeasured races,
    # so this is a no-op there.
    has_measured = pd.Series(False, index=df.index)
    if apply_physical_correction:
        from src.shared.recovery_model import race_physical_correction
        corr = race_physical_correction(df, daily)
        has_measured = corr['has_measured']
        df['phys_dt_sec'] = corr['dt_sec'].to_numpy()
        df['time_sec'] = df['time_sec'].astype(float) - df['phys_dt_sec']
        if 'pace_sec_per_mi' in df.columns:
            df['pace_sec_per_mi'] = (df['pace_sec_per_mi'].astype(float)
                                     - corr['total_s_per_mi'].to_numpy())

    if apply_xc_correction and xc_correction > 0 and 'surface' in df.columns:
        is_xc = (df['surface'].fillna('').astype(str).str.upper() == 'XC') & ~has_measured
        if is_xc.any():
            factor = 1.0 / (1.0 + xc_correction)
            df.loc[is_xc, 'time_sec'] = df.loc[is_xc, 'time_sec'].astype(float) * factor
            if 'pace_sec_per_mi' in df.columns:
                df.loc[is_xc, 'pace_sec_per_mi'] = (
                    df.loc[is_xc, 'pace_sec_per_mi'].astype(float) * factor)

    lookup = daily_summary.set_index(daily_summary['date'].dt.date)
    beta_anchor = _beta_long_factor(norm_dist_m, beta_long, d_thresh)
    vmax = vmax_evidence()   # races are read as EVIDENCE — conservative-high edge

    def _proj(r):
        d = r['date']
        if hasattr(d, 'date'):
            d = d.date()
        if d not in lookup.index:
            return np.nan
        dp3 = float(lookup.loc[d, 'dp3_evid_med'])
        d_race = float(r['distance_m'])
        t_race = float(r['time_sec'])
        if t_race <= 0 or d_race <= 0:
            return np.nan
        t_eff = t_race / _beta_long_factor(d_race, beta_long, d_thresh)
        cs_imp = float(cp3_implied_cs(d_race, t_eff, dp3, vmax))
        t_anchor = float(cp3_time(norm_dist_m, cs_imp, dp3, vmax)) * beta_anchor
        if not np.isfinite(t_anchor):
            return np.nan
        return t_anchor / (norm_dist_m / METERS_PER_MILE) / 60.0

    df['pace_norm_min'] = df.apply(_proj, axis=1)
    df['pace_norm_sec'] = df['pace_norm_min'] * 60.0
    df['time_norm_sec'] = df['pace_norm_sec'] * (norm_dist_m / METERS_PER_MILE)
    return df


def cs_line_at_anchor(daily_summary, anchor_dist_m, beta_long, d_thresh_long):
    """CS-predicted total time (seconds) at anchor_dist_m for each daily date:
    the CP3 forward solve on (CS(date), D′₃(date)), times the β_long factor
    for anchors above d_thresh_long. Forward direction → prediction edge."""
    beta_anchor = _beta_long_factor(anchor_dist_m, beta_long, d_thresh_long)
    return cp3_time(anchor_dist_m, daily_summary['cs_mps_med'].values,
                    daily_summary['dp3_pred_med'].values,
                    vmax_predict()) * beta_anchor


def pace5k_series_to_anchor(p5k_min, daily_summary, anchor_dist_m,
                            beta_long, d_thresh_long):
    """Project an arbitrary 5K-equivalent pace series (min/mi, aligned to
    daily_summary) to total time (s) at anchor_dist_m via the CP3
    prediction-edge curve: back out the implied CS at each date from the 5K
    pace against D′₃(date), then forward-solve the anchor time with the
    β_long factor.

    Generalizes cs_line_at_anchor to any 5K-pace floor (e.g. the blended
    hiatus-floor + CS-implied line the race plots draw). For a pace series
    that equals the model's own p5k_implied_min, the result matches
    cs_line_at_anchor exactly (the D′₃ bridge preserves the 5K solve).
    Forward direction → prediction edge."""
    vmax = vmax_predict()
    dp3 = daily_summary['dp3_pred_med'].to_numpy(float)
    t5k = np.asarray(p5k_min, float) * 60.0 * 5000.0 / METERS_PER_MILE
    cs_mps = cp3_implied_cs(5000.0, t5k, dp3, vmax)
    beta_anchor = _beta_long_factor(anchor_dist_m, beta_long, d_thresh_long)
    return cp3_time(anchor_dist_m, cs_mps, dp3, vmax) * beta_anchor


def cubic_at_anchor(daily_summary, cubic_coefs, t0, handdrawn_start, handdrawn_end,
                     anchor_dist_m, beta_long, d_thresh_long):
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

    dp3_lookup = daily_summary.set_index(daily_summary['date'].dt.date)['dp3_pred_med']
    dp3_values = np.array([float(dp3_lookup.loc[d.date()]) for d in dotted_dates])

    # cs_mps from the 5K time via the CP3 inversion (the cubic was calibrated
    # at the 5K anchor, which carries no β_long factor). This helper feeds
    # lines-at-anchor — the forward/prediction direction.
    vmax = vmax_predict()
    cs_mps = cp3_implied_cs(5000.0, t_5k_sec, dp3_values, vmax)

    beta_anchor = _beta_long_factor(anchor_dist_m, beta_long, d_thresh_long)
    t_anchor_sec = cp3_time(anchor_dist_m, cs_mps, dp3_values, vmax) * beta_anchor
    return dotted_dates, t_anchor_sec
