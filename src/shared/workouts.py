"""Shared workout-decomposition + 5K-equivalent projection helpers.

Used by both the Training plot (which filters and smooths) and the Workouts /
Long Runs plots (which display every session). Each projection function
returns ALL rows it can decompose; rows that the Training plot would normally
prune are flagged via the ``excluded_reason`` column (None when in-scope) so
both consumers can share the same upstream pipeline.

Excluded reasons used:
    - 'snow'              : snow in workout_raw or conditions (quality workouts)
    - 'partners'          : long run with a partners entry outside the
                            recovery model's ADMITTED_PARTNERS (solo and
                            varsity admitted — same population logic as the
                            recovery fit's partner prune)
    - 'long_out_of_slice' : long run under LONG_MIN_MINUTES or at/over
                            LONG_CEIL_MILES
    - 'hc_rep_hybrid'     : 2016-09 hybrid 'hc/rep' sessions
    - 'hc_loop_other'     : hill_cont on a loop outside HC_LOOPS (n<7)
(Workouts have no category-based exclusion beyond snow; sub-threshold quality
days are removed by the Training plot's residual-cutoff outlier prune instead.
Workout partners are a TRUST signal there — see the course-verification gate —
not an exclusion: a partnered workout is still Max's own quality effort.)

Training plot consumes ``project_*(...)`` and immediately filters to
``df[df['excluded_reason'].isna()]`` (with the long-run slice + outlier prune
on top). Workouts plot keeps all rows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.hill_model import minetti_net_factor
from src.shared.cs_projection import cp3_dprime, cp3_implied_cs, cp3_time
from src.shared.recovery_model import ADMITTED_PARTNERS, MISLOGGED_ROUTES
from src.parsers.snapshot import find_snapshot, read_snapshot


WORKOUTS_PATH = DATA_DIR / 'workout_decomposed.csv'
DAILY_PATH    = DATA_DIR / 'daily.csv'
CS_PATH       = DATA_DIR / 'bayes_cs_summary.csv'

# ---------- pipeline parameters ----------
# TAU / DECAY_FLOOR drive the LEGACY uniform-rep decay used for hand-log-only
# quality days (no watch data): decay = max(exp(-rest_per_mile/TAU),
# DECAY_FLOOR), D_eff = rep_dist*(1+(rep_count-1)*decay). The floor stops a
# fully-recovered set from collapsing D_eff toward D' and exploding the 5K-
# equivalent hyperbola (a 12x400 read as one 440m effort); it sets a LEVEL
# the TQ category offsets absorb, not the workout-to-workout signal. Watch-
# ENRICHED days bypass this entirely — they carry a CS-free connected-fatigue
# D_eff (parse_workouts._measured_d_eff, RECON_TAU_S below), so the floor's
# arbitrariness no longer touches the enriched corpus.
TAU = 210.0
DECAY_FLOOR = 0.17
# Connected-fatigue reconstitution time constant (REAL seconds), used by the
# enriched per-rep D_eff. Fit June 2026 by minimizing SSE of workout-implied
# CS vs race-fit CS over Max's 58 enriched days (best of a 120-1200s grid;
# residual mean +0.8, sd 8.4 s/mi). Lands inside the W' reconstitution
# literature range (Skiba ~316-862s) — independent corroboration, not a free
# knob. Unlike the legacy decay this needs no floor: the accumulator is
# bounded in [longest rep, total] by construction. Provisional pending the
# race-residual weighting step (workout effort runs sub-max vs races — a
# longest-rep-correlated bias, +0.25 Spearman, still in the residual).
# Re-checked June 2026 under the CP3 + effort-aware-deflation model: 540
# remains the RMS optimum on the enriched corpus (rms 8.15 s/mi, vs
# 8.51/8.52 at 420/660) — the anchor held without refitting.
RECON_TAU_S = 540.0
# v_max for the WORKOUT side of CP3 — the accumulator's effort-aware
# deflation and the TQ projections (workouts / long runs / hills). This is
# deliberately NOT one of cs_projection's conservative race edges: it's a
# MEASUREMENT calibration, anchored to the watch rep corpus (the τ=540
# re-validation above ran at this value; the deflation reproduces the
# retired empirically-fitted g(d) at rep paces here). The race edges
# encode evidence/prediction conservatism policy; this encodes "what a
# rep day actually demonstrates", and the TQ corpus's accuracy is owned
# by its own gates (course verification, implausibility ceiling).
WORKOUT_VMAX_MPS = 8.7
# Short-effort anaerobic handling (June 2026, CP3 unification): the former
# distance-only pace add g(d) = K·(1/d − 1/d0)+ is GONE, replaced by the
# effort-aware per-rep deflation inside parse_workouts._connected_core —
# each rep's supra-CS speed is scaled by t/(t+τ) (the CP3 model's anaerobic
# availability ratio at the rep's duration, τ = D′₃/(v_max − CS)), with CS
# the workout's OWN implied CS solved self-consistently (no CS-fit input —
# the accumulator stays an independent fitness signal) and the FIRST rep
# exempt (its anaerobic deployment is priced by the projection's D′ term;
# this is also what makes a single max rep analyze identically to a race).
# At rep paces the deflation reproduces the retired g(d) almost exactly
# (≈+12 s/mi at 400m rep pace vs g's +14.3; +4.6 vs +4.3 at 800); at race
# paces it scales ~4× — the race/rep invariant the distance-only form
# violated. CF hard 500s at 5K effort now draw ≈+2 s/mi instead of g's
# +10.3 (the documented structural compromise, resolved). See
# docs/short-effort-unification-plan.md and [[project-workout-enrichment]].
# Watch-vs-log disagreement gate (Max, June 2026): a large RAW watch-vs-log
# pace gap means the watch data is wrong in some way (GPS error large enough
# that the rep decomposition can't be trusted). The consequence is DEMOTION,
# never exclusion: the day falls back to the string-parser estimate — the
# same quality as the pre-watch era or any other watch failure case — and is
# admitted to Training through the standard criteria like any non-enriched
# day. (It also stops counting as watch-VERIFIED for the course gate and the
# implausibility ceiling: the watch data we just rejected can't vouch for
# anything.) The gate is PER MEASURED REP, not whole-day: GPS error accrues
# at watch stops, so a 13-rep 12x400 day legitimately carries more total gap
# than a 4x1600 day (2021-06-07, +11.5 s/mi over 13 reps, is clean). The
# line sits in the distribution's natural break: the corpus runs ...1.42,
# 1.62 | 1.98, 3.61, 4.12, 4.40 s/mi-per-rep — the four egregious days are
# separated by a 1.6 gap above and 0.36 below.
# Deliberately NO manual bad-decomp list (Max): days with known decomp flaws
# (individual reps off by ~100m, e.g. 2023-10-22 / 2023-12-28) keep their
# enrichment as long as the per-rep pace impact is below the gate — what's
# admitted to CS is the pace signal, and a structurally imperfect decomp
# with small pace error is still good signal.
WATCH_LOG_MISMATCH_PER_REP_S = 1.75


# Global long-run slice (June 2026, replacing the per-profile distance
# slice). The two bounds deliberately mix units because they encode
# different mechanisms:
#
# Floor — TIME. A run becomes "long" once fueling/hydration become a real
# concern, which onsets by duration regardless of pace. Empirically the
# residual cliff is razor-sharp at 80 min (median raw_resid ~95-105 s/mi
# below, ~15-42 above) and holds within-era, so it isn't a fitness-mix
# artifact. Duration = recovery_pace_sec_per_mi × miles / 60, the same
# quantity the 5K-equivalent projection uses.
#
# Ceiling — DISTANCE, strict less-than. Filters endurance events that
# aren't marathon race prep: a non-race run at marathon distance or
# beyond is at most an informal sub-max time trial, regardless of how
# long it takes. This hardcodes the app's marathon-focused-runner
# assumption (already implicit in MARATHON_DISTANCE_M fatigue categories,
# the dashboard's 20mi card, the CS marathon-pace curve). A time ceiling
# would instead bias toward fast runners — a 20 mi prep run at 9:00/mi is
# 180 min of textbook marathon prep. NOTE: compare against the LOGGED
# miles value with the log's rounding convention (training marathons are
# logged as exactly 26.2); deriving the cap from 42195 m (= 26.219 mi)
# would wrongly admit them.
LONG_MIN_MINUTES = 80.0
LONG_CEIL_MILES  = 26.2

# ---------- long-run watch enrichment ----------
LR_MEASURED_PATH = DATA_DIR / 'long_run_measured.csv'
LR_CAL_PATH      = DATA_DIR / 'long_run_calibration.csv'

# Route-era mislogged-distance rules: MISLOGGED_ROUTES now lives in
# recovery_model.py (imported above) so the recovery fit can apply the same
# rules to the routes' recovery rows without a circular import. The log
# snaps to a constant 17.6 / 20.7 mi at the baseline log/watch ratio from
# 2022-04-28 / 2022-05-13 on; see the constant's comment for the fitted
# factors. Watch enrichment, when present, wins over the rule (it measures
# the actual day; the rule is the route-era median).


def _load_lr_calibration():
    """(intercept_mi, slope) of the profile's log-vs-watch distance curve,
    or None when the artifact is absent (no watch corpus)."""
    if not LR_CAL_PATH.exists():
        return None
    cal = pd.read_csv(LR_CAL_PATH)
    if cal.empty:
        return None
    r = cal.iloc[0]
    return float(r['intercept_mi']), float(r['slope'])


def _lr_watch_corrections(lr):
    """Watch/rule corrections for long-run rows (see long_run_measured /
    MISLOGGED_ROUTES). Adds, NaN/False where not applicable:

      lr_watch     : watch-enriched (measured, complete, calibration
                     present, paved terrain — see the paved-route gate)
      lr_rule      : corrected via a route-era mislogged-distance rule
      corr_miles, corr_time_s, corr_pace_sec_per_mi : corrected values on
                     the honest-log scale (distance = watch through the
                     calibration curve + watch moving time; rule days keep
                     the logged time over the deflated distance)
      d_eff_m      : pause-aware connected-fatigue effective distance
                     (watch days only; rule days have no pause structure)
      watch_miles, watch_moving_s, pause_s, stall_s, n_segs : raw watch
                     measurements for hover lines (watch days only)

    The logged columns (miles, recovery_pace_sec_per_mi) are never touched —
    what Max logged stays what the plots show by default."""
    lr = lr.copy()
    lr['lr_watch'] = False
    lr['lr_rule'] = False
    for col in ('corr_miles', 'corr_time_s', 'corr_pace_sec_per_mi', 'd_eff_m',
                'watch_miles', 'watch_moving_s', 'pause_s', 'stall_s', 'n_segs'):
        lr[col] = np.nan

    cal = _load_lr_calibration()
    if cal is not None and LR_MEASURED_PATH.exists():
        c, m = cal
        meas = pd.read_csv(LR_MEASURED_PATH, parse_dates=['date'])
        meas = meas[meas['complete']].set_index('date')
        # Paved-route gate (Max, June 2026): the calibration curve is fit on
        # paved-outdoor days, and on trail/mixed terrain GPS corner-cutting
        # under tree cover is a route property the curve can't speak for —
        # the watch track isn't trustworthy enough to overrule the log
        # there. Missing terrain_type fails the gate too (conservative:
        # un-typed routes stay logged-as-is until the locations sheet types
        # them). Same .get() guard as elev: watch profiles' daily.csv may
        # lack the column entirely.
        paved = (lr.get('terrain_type', pd.Series(np.nan, index=lr.index))
                 .astype(str).str.strip().str.lower() == 'paved')
        hit = lr['date'].isin(meas.index) & paved
        if hit.any():
            sub = meas.loc[lr.loc[hit, 'date']]
            cal_mi = (sub['watch_miles'] * (1 + m) + c).to_numpy()
            mov_s = sub['watch_moving_s'].to_numpy()
            # Distance bracket (Max, June 2026): true distance is bracketed
            # watch_mi <= true <= hand-logged. The watch UNDER-reads (GPS
            # undercount), so (watch + calibration error term) is our
            # estimate of truth; the hand log is a hard CEILING — Max never
            # under-estimates a run. Clamp the estimate to the logged
            # distance: on days the watch beat its own average error the
            # estimate overshoots the log and the trusted log wins (those
            # days end up effectively uncorrected). TIME is the anchor — it
            # IS the watch time (the log is just that, rounded to the
            # nearest minute), known to the second — so corr_time stays the
            # watch time and corr_pace is a pure DERIVATIVE of
            # (corr_time, corr_mi), never clamped on its own. Replaces the
            # old pace-floor "overread clamp", which governed the derivative
            # (pace) instead of the distance the invariant is about, and so
            # both let corr_mi exceed the log AND fabricated a phantom
            # distance (mov_s/logged_pace) on the days it bound.
            logged_mi = lr.loc[hit, 'miles'].to_numpy(float)
            corr_mi = np.minimum(cal_mi, logged_mi)
            lr.loc[hit, 'lr_watch'] = True
            lr.loc[hit, 'corr_miles'] = corr_mi
            lr.loc[hit, 'corr_time_s'] = mov_s
            lr.loc[hit, 'corr_pace_sec_per_mi'] = mov_s / corr_mi
            lr.loc[hit, 'd_eff_m'] = (sub['d_eff_frac'].to_numpy()
                                      * corr_mi * 1609.344)
            for col in ('watch_miles', 'watch_moving_s', 'pause_s',
                        'stall_s', 'n_segs'):
                lr.loc[hit, col] = sub[col].to_numpy()

    loc = lr['location'].astype(str).str.strip().str.lower()
    for route, start, end, factor in MISLOGGED_ROUTES:
        rule = (~lr['lr_watch'] & (loc == route)
                & (lr['date'] >= pd.Timestamp(start))
                & (lr['date'] < pd.Timestamp(end)))
        if not rule.any():
            continue
        corr_mi = lr.loc[rule, 'miles'] / factor
        t_log = (lr.loc[rule, 'recovery_pace_sec_per_mi']
                 * lr.loc[rule, 'miles'])
        lr.loc[rule, 'lr_rule'] = True
        lr.loc[rule, 'corr_miles'] = corr_mi
        lr.loc[rule, 'corr_time_s'] = t_log
        lr.loc[rule, 'corr_pace_sec_per_mi'] = t_log / corr_mi
    return lr

# ---------- CS basis ----------
def load_cs():
    """Load bayes_cs_summary.csv and derive the CS-implied 5K pace per day.

    Returns (cs_df, epoch_date) where cs_df has columns date, day (days since
    epoch), p5k_implied_min, dp_med, cs_mps_med, ...
    """
    cs = pd.read_csv(CS_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    cs['t5k_pred_sec'] = (5000.0 - cs['dp_med']) / cs['cs_mps_med']
    cs['p5k_implied_min'] = 1609.344 * cs['t5k_pred_sec'] / 5000.0 / 60.0
    epoch = cs['date'].min()
    cs['day'] = (cs['date'] - epoch).dt.days.astype(float)
    return cs, epoch


def add_cs(df, cs, epoch):
    """Add per-row CS context: day-since-epoch, p5k_cs_min, dp_t, dp3_t, year."""
    df = df.copy()
    df['day'] = (df['date'] - epoch).dt.days.astype(float)
    df['p5k_cs_min'] = np.interp(df['day'], cs['day'].values, cs['p5k_implied_min'].values)
    df['dp_t']       = np.interp(df['day'], cs['day'].values, cs['dp_med'].values)
    dp3 = cp3_dprime(cs['dp_med'].values, cs['cs_mps_med'].values,
                     WORKOUT_VMAX_MPS)
    df['dp3_t']      = np.interp(df['day'], cs['day'].values, dp3)
    df['year']       = df['date'].dt.year
    return df


def dp3_at_date():
    """date -> D′₃ (the CP3 anaerobic reservoir) interpolated from the
    profile's CS summary, for parse-time consumers (the connected
    accumulator's effort-aware deflation). Returns None when no CS fit
    exists yet — callers then skip the deflation, which is fine: a profile
    without a CS fit has no projection layer to feed anyway."""
    if not CS_PATH.exists():
        return None
    cs, epoch = load_cs()
    dp3 = cp3_dprime(cs['dp_med'].values, cs['cs_mps_med'].values,
                     WORKOUT_VMAX_MPS)
    days = cs['day'].to_numpy(float)

    def at(dt):
        d = float((pd.Timestamp(dt) - epoch).days)
        return float(np.interp(d, days, dp3))
    return at


# ---------- hill loop metadata (snapshot-driven) ----------
# HC_LOOPS is a distance-resolution fallback only (surveyed meters for the
# core loops, checked before the snapshot). Training-plot eligibility is NOT
# keyed on membership here — it's the runtime n>7 session count in
# project_hill_continuous, so a new route anywhere qualifies by itself.
HC_LOOPS = {
    'lc':   {'distance_m': 1290},
    'rc':   {'distance_m':  850},
    'pwr1': {'distance_m':  620},
}


def load_hill_loop_meta():
    """Return {abbrev: {display_name, city_state, terrain_type, elev_up,
    elev_down, distance_m, type, elev_per_min}} from the snapshot's `hills` +
    `locations` sections. Loop abbrev → location via the hills sheet;
    location → display_name + city_state + terrain_type via the locations
    sheet. Falls back to empty dict if snapshot missing.

    Includes ALL loops from the snapshot, not just HC_LOOPS — the Workouts
    plot needs metadata for hill_rep loops (evst/ev/pwr2) too.
    """
    snapshot_path = find_snapshot([str(DATA_DIR / 'drive_snapshot.csv')])
    if snapshot_path is None:
        return {}
    sections, _ = read_snapshot(snapshot_path)
    hills_df = sections.get('hills', pd.DataFrame())
    locs_df = sections.get('locations', pd.DataFrame())

    loc_lookup = {}
    if 'log_location' in locs_df.columns:
        for _, r in locs_df.iterrows():
            ll = str(r.get('log_location', '')).strip().lower()
            if ll:
                loc_lookup[ll] = (r.get('display_name'), r.get('city_state'),
                                  r.get('terrain_type'))

    out = {}
    for _, r in hills_df.iterrows():
        for ab in [a.strip() for a in str(r.get('abbrev', '')).split(',')]:
            if not ab:
                continue
            loc = str(r.get('location', '')).strip().lower()
            display_name, city_state, terrain_type = loc_lookup.get(
                loc, (None, None, None))
            out[ab] = {
                'display_name': display_name,
                'city_state':   city_state,
                'terrain_type': terrain_type,
                'elev_up':      r.get('elev_gain_up'),
                'elev_down':    r.get('elev_gain_down'),
                'distance_m':   r.get('distance_m'),
                'type':         r.get('type'),
                'elev_per_min': r.get('elev_per_min'),
            }
    return out


HILL_LOOP_META = load_hill_loop_meta()


# ---------- workouts (interval / tempo / rep / fartlek) ----------
def project_workouts(cs, epoch):
    """Return ALL decomposed quality workouts with p5k_min populated and an
    `excluded_reason` flag. Reps and snow sessions are NOT dropped; they're
    flagged so the Workouts plot can show them.

    Always-applied transforms (regardless of exclusion):
      - 2016-07..10 fall and any tempo with quality_distance_m == 5000:
        XC pace correction (-6%), `xc_corrected` flag set

    Type is logged effort INTENT and is preserved end-to-end (tempo stays
    tempo, etc.). The only reclassification anywhere in the pipeline is for
    `f@`-coded fartleks — Max's log uses that letter for both continuous
    fartlek and varied-distance reps/intervals — handled in
    parse_workouts.decompose (continuous -> continuous_fartlek; non-continuous
    -> interval, or rep when short enough to be varied repetitions). A former
    "long non-Nx tempo -> interval" heuristic was a pre-watch band-aid (no
    logged rest meant a continuous 10k tempo got read as one rep faster than
    the PR); watch structure closes that gap, so it was removed.
    """
    w = pd.read_csv(WORKOUTS_PATH, parse_dates=['date'])
    daily = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    daily_cols = ['date', 'workout_raw', 'conditions', 'quality_distance_m',
                  'quality_pace_sec_per_mi', 'display_name', 'city_state',
                  'temp_c']
    for opt in ('partners', 'terrain_type'):
        if opt in daily.columns:
            daily_cols.append(opt)
        else:
            w[opt] = np.nan
    w = w.merge(daily[daily_cols], on='date', how='left')
    w['is_track'] = w['terrain_type'].astype(str).str.lower() == 'track'

    # XC correction (always applied — same convention as XC race correction).
    # Track locations are categorically exempt: the rules target XC-course
    # efforts (fall season window; HS 5K course tempos run as 5×segments),
    # and a workout on an actual track is neither (2021-04-26, a 5000m track
    # tempo, was the one mis-hit).
    fall_2016 = (w['date'] >= pd.Timestamp('2016-07-01')) & (w['date'] <= pd.Timestamp('2016-10-31'))
    hs_5k = (w['type'] == 'tempo') & (w['quality_distance_m'] == 5000)
    xc_mask = (fall_2016 | hs_5k) & ~w['is_track']
    w.loc[xc_mask, 'pace_per_mile'] = w.loc[xc_mask, 'pace_per_mile'] / 1.06
    w['xc_corrected'] = xc_mask

    w = add_cs(w, cs, epoch)
    decay = np.maximum(np.exp(-w['rest_per_mile'] / TAU), DECAY_FLOOR)
    w['D_eff']    = w['rep_dist'] * (1 + (w['rep_count'] - 1) * decay)
    w['t_eff']    = w['pace_per_mile'] * w['D_eff'] / 1609.344
    # Watch-enriched days carry whole-workout D_eff/t_eff computed from the
    # measured per-rep structure (parse_workouts._measured_d_eff) — the
    # uniform effective-rep formula above mis-serves varied-length days
    # (ladders, closers). t_eff_s is raw measured time, so the XC pace
    # correction has to be re-applied to it here.
    if 'd_eff_m' in w.columns:
        has = w['d_eff_m'].notna() & w['t_eff_s'].notna()
        w.loc[has, 'D_eff'] = w.loc[has, 'd_eff_m']
        w.loc[has, 't_eff'] = (w.loc[has, 't_eff_s']
                               / np.where(w.loc[has, 'xc_corrected'], 1.06, 1.0))
    cs_imp = cp3_implied_cs(w['D_eff'], w['t_eff'], w['dp3_t'],
                            WORKOUT_VMAX_MPS)
    w['t_5k_hyp'] = cp3_time(5000.0, cs_imp, w['dp3_t'], WORKOUT_VMAX_MPS)
    w['p5k_min']  = w['t_5k_hyp'] * 1609.344 / 5000 / 60.0
    w['raw_resid'] = (w['p5k_min'] - w['p5k_cs_min']) * 60
    w['category'] = w['type']

    # Exclusion flags (rows are flagged, never dropped, so the Workouts plot
    # still shows them while Training skips them). Reps are NOT excluded —
    # they all carry a connected projection now and rejoin Training
    # scatter-weighted; sub-threshold/outlier workouts are removed by
    # Training's own track-relative prune, not a category flag.
    #
    # Course-verification gate (Max, June 2026 — sign-blind): the projection
    # is only as trustworthy as the course measurement, and mismeasurement
    # cuts both ways (a short course reads fast, a long one slow — solo 2020
    # Powerline intervals were measured short: gravel yet not slower than
    # surrounding workouts, and never replicated once the watch arrived).
    # A workout needs watch verification, a track location, or partners
    # (non-solo) to be trusted outright. Within the unverified remainder,
    # two rescues:
    #   - continuous efforts (0 rest): mismeasurement bites on back-and-
    #     forth reps with badly marked start/finish lines, not on a single
    #     unbroken course;
    #   - pre-2018 well-understood staples (5000t / 6400t / 4800f strings —
    #     these were likely rhs track under the education-hill catch-all).
    # Everything else is flagged 'uncertain accuracy' (shown on the
    # Workouts plot, dropped from Training).
    verified = _watch_verified_dates()
    partners = w['partners'].astype(str).str.strip().str.lower()
    non_solo = partners.ne('solo') & partners.ne('') & partners.ne('nan')
    # Mismatch-demoted days don't count as verified — see watch_log_demotions.
    watch = w['date'].dt.strftime('%Y-%m-%d').isin(verified - watch_log_demotions())
    unverified = ~watch & ~w['is_track'] & ~non_solo
    continuous = w['rest_per_mile'].fillna(-1) == 0
    staple = ((w['date'].dt.year <= 2017)
              & w['workout_raw'].astype(str).str.contains(
                  r'5000t@|6400t@|4800f@', regex=True, na=False))
    suspect = unverified & ~continuous & ~staple

    # Implausibility ceiling (Max, June 2026): the watch-verified corpus
    # bounds how much a genuine workout can beat SAME-DAY CS (currently
    # 8.6 s/mi, 2022-12-05, mid-peak — the real "workouts lead the smoothed
    # CS curve" effect). A non-verified day beating CS by more is a bad
    # decomposition the string can't recover (reps that included 100s/150s,
    # intervals that included 800s) — only watch verification shields here,
    # NOT track/partners/staple trust (2017-03-28, varsity, read 4:46/mi —
    # faster than any capability ever demonstrated). Margin is data-derived
    # and self-adjusts as the verified corpus grows; skipped when no
    # verified rows exist to establish the bound. (Demoted days already left
    # the watch mask above — rejected watch data can't anchor the bound.)
    if watch.any():
        vmax = float(-w.loc[watch, 'raw_resid'].min())
        suspect |= ~watch & (w['raw_resid'] < -vmax)
    snow_w = w['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = w['conditions'].astype(str).str.contains('snow', case=False, na=False)
    w['excluded_reason'] = None
    w.loc[suspect, 'excluded_reason'] = 'uncertain accuracy'
    w.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'

    return w


# ---------- long runs ----------
def _load_beta_long():
    """(beta_long, d_thresh_long) from the profile's bayes_cs_params.csv —
    the same long-distance fade the race plots un-bias with. Falls back to
    (0, 10000) when the params artifact is absent (no CS fit yet)."""
    path = DATA_DIR / 'bayes_cs_params.csv'
    if not path.exists():
        return 0.0, 10000.0
    p = pd.read_csv(path)
    return (float(p['beta_long_med'].iloc[0]),
            float(p['d_thresh_long'].iloc[0]))


def project_long_runs(cs, epoch):
    """Return ALL `run_type == 'long'` rows with absolute pace + 5K-equiv
    projection + `excluded_reason` flag. No filtering by miles or snow.
    Rows missing recovery_pace_sec_per_mi or miles are dropped (no
    decomposable signal).

    The projection treats the run AS IF IT WERE A RACE of that distance
    (Max's contract, June 2026): time is un-biased by the CS fit's
    long-distance fade — divide by 1 + β_long·log(d/d_thresh) for
    d > d_thresh, exactly what cs_projection does for HM/marathon races —
    then projected through the hyperbola. No fitted offset of any kind:
    the former long-run model intercept (a constant ~+48 subtracted from
    every run) claimed a long run predicts a faster 5K than a race at the
    same distance/pace, which is physically indefensible. With β_long the
    hardest long runs (true race-effort simulations) project at/near CS on
    their own, and easy ones honestly slow.

    Watch-enriched and route-rule-corrected rows (see _lr_watch_corrections)
    feed the projection their corrected distance/time and, for watch days, a
    pause-aware connected-fatigue D_eff — so p5k_min/raw_resid (and every
    downstream consumer: the TQ smoother, the dashboard's long-run
    prediction) are correction-aware. β uses the run's full corrected
    distance, not D_eff: the fade is total-work/glycogen depletion, which
    a stoplight pause doesn't reset. The slice gate stays on LOGGED values:
    the 26.2 ceiling is a logging convention (training marathons are logged
    as exactly 26.2) and corrections never move a run across either bound.
    """
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    lr = d[d['run_type'] == 'long'].copy().dropna(subset=['recovery_pace_sec_per_mi', 'miles'])

    snow_w = lr['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = lr['conditions'].astype(str).str.contains('snow', case=False, na=False)
    dur_min = lr['recovery_pace_sec_per_mi'] * lr['miles'] / 60.0
    out_of_slice = (dur_min < LONG_MIN_MINUTES) | (lr['miles'] >= LONG_CEIL_MILES)
    # Partner exclusion (June 2026): same population logic as the recovery
    # fit — a long run paced by someone else's targets isn't Max's effort
    # policy. Solo and varsity admitted (ADMITTED_PARTNERS), matching the
    # recovery prune; fillna('') for profiles with no partners column data.
    partners = (lr.get('partners', pd.Series(np.nan, index=lr.index))
                .fillna('').astype(str).str.strip().str.lower())
    is_partner = ~partners.isin(ADMITTED_PARTNERS)

    lr['excluded_reason'] = None
    lr.loc[out_of_slice, 'excluded_reason'] = 'long_out_of_slice'
    lr.loc[is_partner, 'excluded_reason'] = 'partners'
    # Snow takes priority over slice/partners (an out-of-slice snow run gets
    # 'snow' — the ringed reasons win so the chart explains the exclusion).
    lr.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'

    lr = add_cs(lr, cs, epoch)
    lr = _lr_watch_corrections(lr)
    lr['t_run'] = np.where(lr['corr_time_s'].notna(), lr['corr_time_s'],
                           lr['recovery_pace_sec_per_mi'] * lr['miles'])
    lr['d_m']   = np.where(lr['corr_miles'].notna(),
                           lr['corr_miles'] * 1609.344,
                           lr['miles'] * 1609.344)
    # Connected D_eff (watch days; full distance elsewhere) enters the
    # hyperbola the same way enriched workouts do: t_eff = D_eff / v with v
    # the run's mean moving speed. At long-run distances the hyperbola is
    # nearly flat in D_eff so this is a small, one-directional refinement
    # (~+1-3 s/mi) — the distance calibration does the heavy lifting.
    d_eff = np.where(lr['d_eff_m'].notna(), lr['d_eff_m'], lr['d_m'])
    t_eff = d_eff / (lr['d_m'] / lr['t_run'])
    # Race-equivalent un-bias (see docstring): β at the run's full distance.
    beta_long, d_thresh = _load_beta_long()
    beta = np.where(lr['d_m'] > d_thresh,
                    1.0 + beta_long * np.log(np.maximum(lr['d_m'], d_thresh)
                                             / d_thresh),
                    1.0)
    t_eff = t_eff / beta
    cs_imp = cp3_implied_cs(d_eff, t_eff, lr['dp3_t'], WORKOUT_VMAX_MPS)
    lr['t_5k_hyp'] = cp3_time(5000.0, cs_imp, lr['dp3_t'], WORKOUT_VMAX_MPS)
    lr['p5k_min']  = lr['t_5k_hyp'] * 1609.344 / 5000.0 / 60.0
    lr['raw_resid'] = (lr['p5k_min'] - lr['p5k_cs_min']) * 60
    return lr


# ---------- hill continuous ----------
_HC_PARSE_RX = re.compile(r'(\d+)hc-')
_HC_NREPS_RX = re.compile(r'hc-(\d+)x')
_HC_LOOP_RX  = re.compile(r'hc-\d+x\s+([a-zA-Z0-9]+)')


def _watch_verified_dates():
    """Dates whose structure the watch verified (workout_measured.csv,
    statuses exact / watch-only). Empty set when no enrichment exists."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return set()
    m = pd.read_csv(path)
    return set(m.loc[m['status'].isin(['exact', 'watch-only']), 'date'])


def watch_log_demotions():
    """Dates whose watch enrichment fails the per-rep watch-log mismatch gate
    (WATCH_LOG_MISMATCH_PER_REP_S above). Consumers treat these days as
    NON-ENRICHED: parse_workouts keeps the string-parser row, project_workouts
    drops them from the verified set, the Workouts plot shows no Watch line.
    Empty set when no enrichment or no hand log exists."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return set()
    m = pd.read_csv(path)
    m = m[(m['rep_idx'] > 0) & m['status'].isin(['exact', 'watch-only'])]
    if m.empty or not DAILY_PATH.exists():
        return set()
    daily = pd.read_csv(DAILY_PATH)
    if 'quality_pace_sec_per_mi' not in daily.columns:
        return set()
    qp_map = dict(zip(daily['date'],
                      pd.to_numeric(daily['quality_pace_sec_per_mi'],
                                    errors='coerce')))
    out = set()
    for date, day in m.groupby('date'):
        qp = qp_map.get(date)
        if qp is None or pd.isna(qp) or qp <= 0:
            continue
        watch_pace = day['time_s'].sum() / (day['dist_m'].sum() / 1609.344)
        if abs(watch_pace - qp) / len(day) > WATCH_LOG_MISMATCH_PER_REP_S:
            out.add(date)
    return out


def _load_hill_measured_t():
    """date -> watch-measured moving seconds for the hill block, from
    workout_measured.csv (statuses hill-exact / hill-total; loop rows sum to
    the block total in both). Empty dict when nothing is measured."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m = m[(m['rep_idx'] > 0)
          & (m['status'].isin(['hill-exact', 'hill-total']))]
    if m.empty:
        return {}
    return m.groupby('date')['time_s'].sum().to_dict()


def hc_loop_distance(loop):
    """Surveyed meters for a hill-loop abbrev (HC_LOOPS first, snapshot
    fallback). None when the loop has no measured distance."""
    if loop in HC_LOOPS:
        return HC_LOOPS[loop]['distance_m']
    meta = HILL_LOOP_META.get(loop, {})
    dm = meta.get('distance_m')
    if dm in (None, '') or (isinstance(dm, float) and np.isnan(dm)):
        return None
    return float(dm)


def parse_hc(row):
    s = str(row['workout_raw'])
    m_min = _HC_PARSE_RX.search(s)
    minutes = int(m_min.group(1)) if m_min else None
    m_n = _HC_NREPS_RX.search(s)
    nreps = int(m_n.group(1)) if m_n else None
    m_loc = _HC_LOOP_RX.search(s)
    loop = m_loc.group(1) if m_loc else None
    if loop is None:
        loc_col = str(row['location']).lower()
        if 'rollercoaster' in loc_col: loop = 'rc'
        elif 'powerline west' in loc_col: loop = 'pwr1'
    return pd.Series([minutes, nreps, loop])


def project_hill_continuous(cs, epoch):
    """Return ALL hill_cont sessions with p5k projection + `excluded_reason`.

    Loops outside HC_LOOPS get `excluded_reason='hc_loop_other'` and their
    p5k uses the snapshot's distance_m (best effort). 2016-09 hc/rep hybrids
    are flagged `hc_rep_hybrid`. Snow flagged `snow`.

    The Training plot consumes only rows where excluded_reason is None.
    The Workouts plot keeps everything that has a parsed (minutes, nreps,
    loop, distance_m).
    """
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    h = d[d['run_type'] == 'hill_cont'].copy()
    if h.empty:
        # No hill-continuous sessions (e.g. a watch profile). A 0-row
        # apply(axis=1) returns a column-less frame, so seed the parsed columns
        # explicitly; the rest of the function then computes on empty and
        # returns the full column schema (incl. excluded_reason) that callers
        # like plot_training_quality require.
        h['session_min'] = pd.Series(dtype=float)
        h['nreps'] = pd.Series(dtype=float)
        h['loop'] = pd.Series(dtype=object)
    else:
        h[['session_min', 'nreps', 'loop']] = h.apply(parse_hc, axis=1)
    h = h.dropna(subset=['session_min', 'nreps', 'loop']).copy()

    # distance_m: prefer HC_LOOPS hardcoded constant, fall back to snapshot.
    h['loop_distance_m'] = h['loop'].map(hc_loop_distance)
    h = h.dropna(subset=['loop_distance_m']).copy()

    h['quality_dist_m'] = h['nreps'] * h['loop_distance_m']
    h['actual_pace_s'] = (h['session_min'] * 60.0) / (h['quality_dist_m'] / 1609.344)
    h['d_m']     = h['quality_dist_m']
    h['t_eff']   = h['actual_pace_s'] * h['d_m'] / 1609.344

    # Watch-measured override: exact moving seconds for the loop block
    # (reps.py extract_hill_day) replace the hand log's whole-minute session
    # time, which also silently includes standing rest. Distance stays
    # authoritative — only time moves; everything downstream computes from
    # the overridden t_eff.
    meas = _load_hill_measured_t()
    h['watch_t_eff'] = h['date'].dt.strftime('%Y-%m-%d').map(meas)
    has_watch = h['watch_t_eff'].notna()
    h.loc[has_watch, 't_eff'] = h.loc[has_watch, 'watch_t_eff']
    h.loc[has_watch, 'actual_pace_s'] = (h.loc[has_watch, 't_eff']
                                         / (h.loc[has_watch, 'd_m'] / 1609.344))
    h['watch_measured'] = has_watch

    h = add_cs(h, cs, epoch)
    cs_imp = cp3_implied_cs(h['d_m'], h['t_eff'], h['dp3_t'],
                            WORKOUT_VMAX_MPS)
    h['t_5k_hyp'] = cp3_time(5000.0, cs_imp, h['dp3_t'], WORKOUT_VMAX_MPS)
    h['p5k_min']  = h['t_5k_hyp'] * 1609.344 / 5000.0 / 60.0
    h['raw_resid'] = (h['p5k_min'] - h['p5k_cs_min']) * 60
    # Informational grouping only — corrections come from the hill model
    # (gain + terrain), never from per-loop categories.
    h['category'] = 'hill_' + h['loop'].astype(str)

    def _meta(loop, key):
        return HILL_LOOP_META.get(loop, {}).get(key)
    h['loop_display_name'] = h['loop'].map(lambda l: _meta(l, 'display_name'))
    h['loop_city_state']   = h['loop'].map(lambda l: _meta(l, 'city_state'))
    h['loop_elev_up']      = h['loop'].map(lambda l: _meta(l, 'elev_up'))
    h['loop_elev_down']    = h['loop'].map(lambda l: _meta(l, 'elev_down'))
    # Both `elev_up` and `elev_down` are climbing — `down` is the climbing
    # encountered on the return portion of the loop (rollercoaster style).
    # Total amount gained per session = (elev_up + elev_down) × nreps.
    h['ft_gained'] = (h['loop_elev_up'].fillna(0) + h['loop_elev_down'].fillna(0)) * h['nreps']

    # Hill-model covariates: total gain per quality mile, and the binary
    # terrain class from the locations sheet (paved vs trail; anything
    # non-paved counts as trail).
    h['ft_per_mi'] = h['ft_gained'] / (h['quality_dist_m'] / 1609.344)
    terrain = h['loop'].map(lambda l: _meta(l, 'terrain_type'))
    h['is_trail'] = (terrain.astype(str).str.lower()
                     .map(lambda t: np.nan if t in ('nan', 'none', '')
                          else float(t != 'paved')))

    # Pinned Minetti net up+down cost at each loop's grade (multiplicative
    # on t_eff — see src/shared/hill_model.py). p5k_min_hillcorr is the
    # gain-corrected projection both plots build on; the trail term is
    # fitted downstream and applied on top.
    per_loop_climb = h['loop_elev_up'].fillna(0) + h['loop_elev_down'].fillna(0)
    h['minetti_factor'] = minetti_net_factor(per_loop_climb, h['loop_distance_m'])
    t_eff_corr = h['t_eff'] / h['minetti_factor']
    cs_imp_corr = cp3_implied_cs(h['d_m'], t_eff_corr, h['dp3_t'],
                                 WORKOUT_VMAX_MPS)
    t5k_corr = cp3_time(5000.0, cs_imp_corr, h['dp3_t'], WORKOUT_VMAX_MPS)
    h['p5k_min_hillcorr'] = t5k_corr * 1609.344 / 5000.0 / 60.0
    h['minetti_resid'] = (h['p5k_min_hillcorr'] - h['p5k_cs_min']) * 60

    # Flag (don't drop) hybrids, snow, and loops the model can't cover.
    # TQ scope is covariate-based: any loop with surveyed distance (required
    # above), elevation data, and a terrain class is correctable by the hill
    # model — no per-loop session-count gate, so a brand-new route qualifies
    # on its first session.
    no_cov = (h['loop_elev_up'].isna() & h['loop_elev_down'].isna()) | h['is_trail'].isna()
    hybrid = h['workout_raw'].astype(str).str.contains(r'hc/rep', regex=True, na=False)
    snow_w = h['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = h['conditions'].astype(str).str.contains('snow', case=False, na=False)
    h['excluded_reason'] = None
    h.loc[no_cov, 'excluded_reason'] = 'hc_no_covariates'
    h.loc[hybrid, 'excluded_reason'] = 'hc_rep_hybrid'
    h.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'
    return h


# ---------- hill repeats (NEW) ----------
# Tight regex: `<minutes>[:seconds]hr-<count>x <loop>`. The existing loose
# `_HILL_LOOP_RX` from running_log_parser picks up the post-comma jog
# (e.g. `12j`) on one of the 25 sessions, so use a tighter pattern here.
# Tight pattern: `<minutes>[:seconds]hr-<count>x` followed by a loop token.
# The full pattern requires the loop word; the partial pattern catches the one
# 2018-05-31 session that omits it (loop is recovered from `location` instead).
_HR_FULL_RX    = re.compile(r'(\d+(?::\d+)?)hr-(\d+)x\s+([a-zA-Z0-9]+)', re.IGNORECASE)
_HR_PARTIAL_RX = re.compile(r'(\d+(?::\d+)?)hr-(\d+)x', re.IGNORECASE)


def _parse_hr(row):
    """Return (rep_time_min, rep_count, loop) or (None, None, None)."""
    s = str(row.get('workout_raw', ''))
    m = _HR_FULL_RX.search(s)
    if m:
        t_str, n_str, loop = m.group(1), m.group(2), m.group(3).lower()
    else:
        m = _HR_PARTIAL_RX.search(s)
        if not m:
            return (None, None, None)
        t_str, n_str = m.group(1), m.group(2)
        # Recover loop from the daily location field.
        loc = str(row.get('location', '')).lower()
        if 'everest' in loc:        loop = 'evst'
        elif 'powerline east' in loc: loop = 'pwr2'
        else:                         return (None, None, None)
    if ':' in t_str:
        mm, ss = t_str.split(':')
        rep_time_min = int(mm) + int(ss) / 60.0
    else:
        rep_time_min = float(t_str)
    return (rep_time_min, int(n_str), loop)


def project_hill_reps():
    """Return all hill_rep sessions with rep_time/rep_count/loop parsed and
    elevation joined. Hill reps lack quality data needed for a CS+D'
    projection (no per-rep distance), so this returns no `p5k_min` — the
    Workouts plot positions hill_rep markers at the persisted TQ smoother
    track instead.

    Columns: date, loop, rep_time_min, rep_count, total_elev_ft (ft),
             workout_raw, display_name, city_state, conditions, location,
             excluded_reason.
    """
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    h = d[d['run_type'] == 'hill_rep'].copy()
    if h.empty:
        return h

    parsed = h.apply(_parse_hr, axis=1).tolist()
    h['rep_time_min'] = [p[0] for p in parsed]
    h['rep_count']    = [p[1] for p in parsed]
    h['loop']         = [p[2] for p in parsed]
    h = h.dropna(subset=['rep_time_min', 'rep_count', 'loop']).copy()

    def _meta(loop, key):
        return HILL_LOOP_META.get(loop, {}).get(key)
    h['loop_display_name'] = h['loop'].map(lambda l: _meta(l, 'display_name'))
    h['loop_city_state']   = h['loop'].map(lambda l: _meta(l, 'city_state'))
    h['elev_per_min']      = h['loop'].map(lambda l: _meta(l, 'elev_per_min'))

    epm = pd.to_numeric(h['elev_per_min'], errors='coerce')
    h['total_elev_ft'] = h['rep_time_min'] * h['rep_count'] * epm  # NaN if epm missing

    snow_w = h['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = h['conditions'].astype(str).str.contains('snow', case=False, na=False)
    h['excluded_reason'] = None
    h.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'
    return h
