"""Shared workout-decomposition + 5K-equivalent projection helpers.

Used by both the Training plot (which filters and smooths) and the Workouts /
Long Runs plots (which display every session). Each projection function
returns ALL rows it can decompose; rows that the Training plot would normally
prune are flagged via the ``excluded_reason`` column (None when in-scope) so
both consumers can share the same upstream pipeline.

Excluded reasons used:
    - 'snow'              : snow in workout_raw or conditions (quality workouts)
    - 'long_out_of_slice' : long run under LONG_MIN_MINUTES or at/over
                            LONG_CEIL_MILES
    - 'hc_rep_hybrid'     : 2016-09 hybrid 'hc/rep' sessions
    - 'hc_loop_other'     : hill_cont on a loop outside HC_LOOPS (n<7)
(Workouts have no category-based exclusion beyond snow; sub-threshold quality
days are removed by the Training plot's residual-cutoff outlier prune instead.)

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
RECON_TAU_S = 540.0
# Short-distance anaerobic correction g(d) = ANAEROBIC_K*(1/d - 1/ANAEROBIC_D0)+,
# a per-rep pace add (s/mi) applied INSIDE the connected accumulator. It removes
# the 2-param CP hyperbola's short-effort CS overshoot, which biases sub-~800m
# reps fast (uniform 400m days read ~16 s/mi too fit; 800m ~3; >=1600m ~0). Fit
# June 2026 scatter-weighted with a free offset (the offset = the workout-vs-race
# sub-max effort gap, absorbed downstream by the TQ category offset): K=8000,
# d0=1400 -> g(400)=+14.3, g(800)=+4.3, g(>=1400)=0 s/mi. Distance-only, so it
# shifts the level the hyperbola misplaces without touching responsiveness (gain
# stays ~1). Because it depends on rep distance, a ladder's 400 segments get the
# correction and its 1600 doesn't.
# NOT for races: calibrated on rep-PACE 400s (~72s); all-out race 400s (~57s) sit
# far deeper in the anaerobic regime and need ~4x more correction (the effect
# scales with speed-above-CS, not distance), so make_race_plots keeps its own
# BETA_SHORT display term. See [[project-workout-enrichment]].
ANAEROBIC_K = 8000.0
ANAEROBIC_D0 = 1400.0


def g_anaerobic(d_m):
    """Per-rep anaerobic pace correction (s/mi) added to a rep of distance d_m
    metres; 0 for d_m >= ANAEROBIC_D0. Accepts scalars or numpy arrays."""
    return ANAEROBIC_K * np.maximum(0.0, 1.0 / d_m - 1.0 / ANAEROBIC_D0)
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
    """Add per-row CS context: day-since-epoch, p5k_cs_min, dp_t, year."""
    df = df.copy()
    df['day'] = (df['date'] - epoch).dt.days.astype(float)
    df['p5k_cs_min'] = np.interp(df['day'], cs['day'].values, cs['p5k_implied_min'].values)
    df['dp_t']       = np.interp(df['day'], cs['day'].values, cs['dp_med'].values)
    df['year']       = df['date'].dt.year
    return df


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
                  'display_name', 'city_state', 'temp_c']
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
    w['t_5k_hyp'] = (5000 - w['dp_t']) * w['t_eff'] / (w['D_eff'] - w['dp_t'])
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
    watch = w['date'].dt.strftime('%Y-%m-%d').isin(verified)
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
    # verified rows exist to establish the bound.
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
def project_long_runs(cs, epoch):
    """Return ALL `run_type == 'long'` rows with absolute pace + 5K-equiv
    projection + `excluded_reason` flag. No filtering by miles or snow.
    Rows missing recovery_pace_sec_per_mi or miles are dropped (no
    decomposable signal).
    """
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    lr = d[d['run_type'] == 'long'].copy().dropna(subset=['recovery_pace_sec_per_mi', 'miles'])

    snow_w = lr['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = lr['conditions'].astype(str).str.contains('snow', case=False, na=False)
    dur_min = lr['recovery_pace_sec_per_mi'] * lr['miles'] / 60.0
    out_of_slice = (dur_min < LONG_MIN_MINUTES) | (lr['miles'] >= LONG_CEIL_MILES)

    lr['excluded_reason'] = None
    lr.loc[out_of_slice, 'excluded_reason'] = 'long_out_of_slice'
    # Snow takes priority over slice (an out-of-slice snow run gets 'snow').
    lr.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'

    lr = add_cs(lr, cs, epoch)
    lr['t_run']    = lr['recovery_pace_sec_per_mi'] * lr['miles']
    lr['d_m']      = lr['miles'] * 1609.344
    lr['t_5k_hyp'] = (5000 - lr['dp_t']) * lr['t_run'] / (lr['d_m'] - lr['dp_t'])
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
    h['t_5k_hyp'] = (5000 - h['dp_t']) * h['t_eff'] / (h['d_m'] - h['dp_t'])
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
    t5k_corr = (5000 - h['dp_t']) * t_eff_corr / (h['d_m'] - h['dp_t'])
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
