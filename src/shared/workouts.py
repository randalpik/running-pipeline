"""Shared workout-decomposition + 5K-equivalent projection helpers.

Used by both the Training plot (which filters and smooths) and the Workouts /
Long Runs plots (which display every session). Each projection function
returns ALL rows it can decompose; rows that the Training plot would normally
prune are flagged via the ``excluded_reason`` column (None when in-scope) so
both consumers can share the same upstream pipeline.

Excluded reasons used:
    - 'snow'              : snow in workout_raw or conditions
    - 'rep_anaerobic'     : type == 'rep' (Training drops; Workouts shows w/ offset)
    - 'long_out_of_slice' : long run with miles outside [LONG_FLOOR, LONG_CEIL]
    - 'hc_rep_hybrid'     : 2016-09 hybrid 'hc/rep' sessions
    - 'hc_loop_other'     : hill_cont on a loop outside HC_LOOPS (n<7)

Training plot consumes ``project_*(...)`` and immediately filters to
``df[df['excluded_reason'].isna()]`` (with the long-run slice + outlier prune
on top). Workouts plot keeps all rows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.parsers.snapshot import find_snapshot, read_snapshot


WORKOUTS_PATH = DATA_DIR / 'workout_decomposed.csv'
DAILY_PATH    = DATA_DIR / 'daily.csv'
CS_PATH       = DATA_DIR / 'bayes_cs_summary.csv'

# ---------- pipeline parameters ----------
TAU = 210.0
LONG_FLOOR      = 15.1
LONG_CEIL       = 25.3
LR_INTERNAL_BIN = 21.0

# Route-agnostic elevation cost coefficient for hill_cont 5K-equivalent
# projection on the Workouts plot. Subtracts time saved by climbing (sec) =
# HILL_ELEV_COST_SEC_PER_FT × total feet climbed, then runs the hyperbolic
# projection on the corrected effective time.
# Training plot does not use this — TQ uses per-loop offsets to absorb
# loop-specific systematics including elevation, so a uniform coefficient
# would over-correct on top of those.
HILL_ELEV_COST_SEC_PER_FT = 0.20


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
# HC_LOOPS captures the loops with sufficient training data (n>=7) for the
# Training plot's per-loop offset calibration. Sessions outside this set are
# flagged ``hc_loop_other`` so the Workouts plot can still show them.
HC_LOOPS = {
    'lc':   {'distance_m': 1290},
    'rc':   {'distance_m':  850},
    'pwr1': {'distance_m':  620},
}


def load_hill_loop_meta():
    """Return {abbrev: {display_name, city_state, elev_up, elev_down,
    distance_m, type, elev_per_min}} from the snapshot's `hills` + `locations`
    sections. Loop abbrev → location via the hills sheet; location →
    display_name + city_state via the locations sheet. Falls back to empty
    dict if snapshot missing.

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
                loc_lookup[ll] = (r.get('display_name'), r.get('city_state'))

    out = {}
    for _, r in hills_df.iterrows():
        for ab in [a.strip() for a in str(r.get('abbrev', '')).split(',')]:
            if not ab:
                continue
            loc = str(r.get('location', '')).strip().lower()
            display_name, city_state = loc_lookup.get(loc, (None, None))
            out[ab] = {
                'display_name': display_name,
                'city_state':   city_state,
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
      - Implicit-Nx tempo at >=1600m -> reclassified as interval
      - 2016-07..10 fall and any tempo with quality_distance_m == 5000:
        XC pace correction (-6%), `xc_corrected` flag set
    """
    w = pd.read_csv(WORKOUTS_PATH, parse_dates=['date'])
    daily = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    w = w.merge(daily[['date', 'workout_raw', 'conditions', 'quality_distance_m',
                       'display_name', 'city_state', 'temp_c']],
                on='date', how='left')

    # Tempo @ rep_dist>=1600 with no explicit Nx -> interval (catches 2024-05-04).
    has_nx = w['workout_raw'].astype(str).str.contains(r'\d+x\d+', regex=True, na=False)
    mask = (w['type'] == 'tempo') & (w['rep_dist'] >= 1600) & (~has_nx)
    w.loc[mask, 'type'] = 'interval'

    # XC correction (always applied — same convention as XC race correction).
    fall_2016 = (w['date'] >= pd.Timestamp('2016-07-01')) & (w['date'] <= pd.Timestamp('2016-10-31'))
    hs_5k = (w['type'] == 'tempo') & (w['quality_distance_m'] == 5000)
    xc_mask = fall_2016 | hs_5k
    w.loc[xc_mask, 'pace_per_mile'] = w.loc[xc_mask, 'pace_per_mile'] / 1.06
    w['xc_corrected'] = xc_mask

    w = add_cs(w, cs, epoch)
    decay = np.exp(-w['rest_per_mile'] / TAU)
    w['D_eff']    = w['rep_dist'] * (1 + (w['rep_count'] - 1) * decay)
    w['t_eff']    = w['pace_per_mile'] * w['D_eff'] / 1609.344
    w['t_5k_hyp'] = (5000 - w['dp_t']) * w['t_eff'] / (w['D_eff'] - w['dp_t'])
    w['p5k_min']  = w['t_5k_hyp'] * 1609.344 / 5000 / 60.0
    w['raw_resid'] = (w['p5k_min'] - w['p5k_cs_min']) * 60
    w['category'] = w['type']

    # Flag (don't drop) reps and snow.
    snow_w = w['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = w['conditions'].astype(str).str.contains('snow', case=False, na=False)
    w['excluded_reason'] = None
    w.loc[w['type'] == 'rep', 'excluded_reason'] = 'rep_anaerobic'
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
    out_of_slice = ~lr['miles'].between(LONG_FLOOR, LONG_CEIL)

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


def _parse_hc(row):
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

    h[['session_min', 'nreps', 'loop']] = h.apply(_parse_hc, axis=1)
    h = h.dropna(subset=['session_min', 'nreps', 'loop']).copy()

    # distance_m: prefer HC_LOOPS hardcoded constant, fall back to snapshot.
    def _loop_distance(loop):
        if loop in HC_LOOPS:
            return HC_LOOPS[loop]['distance_m']
        meta = HILL_LOOP_META.get(loop, {})
        dm = meta.get('distance_m')
        return float(dm) if dm not in (None, '') and not (isinstance(dm, float) and np.isnan(dm)) else None

    h['loop_distance_m'] = h['loop'].map(_loop_distance)
    h = h.dropna(subset=['loop_distance_m']).copy()

    h['quality_dist_m'] = h['nreps'] * h['loop_distance_m']
    h['actual_pace_s'] = (h['session_min'] * 60.0) / (h['quality_dist_m'] / 1609.344)
    h['d_m']     = h['quality_dist_m']
    h['t_eff']   = h['actual_pace_s'] * h['d_m'] / 1609.344
    h = add_cs(h, cs, epoch)
    h['t_5k_hyp'] = (5000 - h['dp_t']) * h['t_eff'] / (h['d_m'] - h['dp_t'])
    h['p5k_min']  = h['t_5k_hyp'] * 1609.344 / 5000.0 / 60.0
    h['raw_resid'] = (h['p5k_min'] - h['p5k_cs_min']) * 60
    h['category'] = 'hill_' + h['loop'].astype(str)

    # Flag (don't drop) hybrids, snow, and out-of-scope loops.
    hybrid = h['workout_raw'].astype(str).str.contains(r'hc/rep', regex=True, na=False)
    snow_w = h['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = h['conditions'].astype(str).str.contains('snow', case=False, na=False)
    h['excluded_reason'] = None
    h.loc[~h['loop'].isin(HC_LOOPS.keys()), 'excluded_reason'] = 'hc_loop_other'
    h.loc[hybrid, 'excluded_reason'] = 'hc_rep_hybrid'
    h.loc[snow_w | snow_c, 'excluded_reason'] = 'snow'

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

    # Route-agnostic elevation-corrected 5K-equivalent for the Workouts plot.
    # Effort cost of climbing is removed from t_eff via a uniform coefficient
    # (HILL_ELEV_COST_SEC_PER_FT × ft_gained), then the corrected effective
    # time is projected hyperbolically. Same formula for all loops — no
    # per-loop calibration. Training plot ignores this column and continues
    # to use the uncorrected p5k_min plus per-loop offsets.
    h['t_eff_elev_corr']    = h['t_eff'] - HILL_ELEV_COST_SEC_PER_FT * h['ft_gained']
    h['t_5k_hyp_elev_corr'] = (5000 - h['dp_t']) * h['t_eff_elev_corr'] / (h['d_m'] - h['dp_t'])
    h['p5k_min_elev_corr']  = h['t_5k_hyp_elev_corr'] * 1609.344 / 5000.0 / 60.0
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
