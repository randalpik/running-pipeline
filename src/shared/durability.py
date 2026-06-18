"""Durability + W'-balance long-run pause model (June 2026).

The pause penalty for a long run is the MARGINAL effect of that run's own
stops: the gap between the fastest constant pace it could have sustained
*without* the stops and *with* them. Effective critical speed declines over
time on feet (durability — an accelerating, delayed-onset polynomial, Stevenson
2024); running above it draws down the finite D' reservoir; a stop reconstitutes
D' (Vassallo running W'bal). A continuous run (no stops) has identical
with/without sims, so its penalty is exactly zero AT ANY durability — continuous
runs are ground truth and are never touched. Each run is self-contained: its own
day's CS/D' and its own stop timing. Nothing about one run constrains another
(no cross-era contamination).

Inputs are the RELIABLE quantities only: the corrected average pace and the
button-press pause TIMESTAMPS from the rich detail cache. The noisy per-second
pace is not used; the run's pace fade is represented by the declining effective
CS, not by the (GPS-corrupted) instantaneous trace.

The penalty's three drivers fall out of the mechanism: pause MAGNITUDE (more /
longer stops reconstitute more), PROXIMITY TO THE END (late stops land when CS
has declined and W' is being drawn — early stops, with W' full, buy nothing),
and PROXIMITY TO CS (a run far below effective CS never draws W', so its stops
are worthless). DUR_LOSS_2H is the conservatism dial (slowest projection = more
aggressive decline); since the penalty is the with/without *difference*, raising
it only moves paused runs, never continuous ones.
"""
from __future__ import annotations
import functools
import json
import math

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE

# --- durability: effective-CS decline over time on feet ---
DUR_LOSS_2H = 0.12      # fractional CS loss at 2 h (the conservatism dial)
DUR_ONSET_H = 0.75      # near-flat below this, then accelerates
DUR_EXP     = 1.6       # accelerating polynomial exponent (Stevenson 2024)
DUR_CAP     = 0.25      # max fractional CS loss (ultra-duration floor)
# --- D' (W') decline + reconstitution ---
WPRIME_LOSS_2H = 0.20   # fractional D' loss at 2 h (rides the same curve, scaled)
DP_FLOOR       = 0.30   # D' never falls below this fraction of fresh
TAU_STOP_S     = 110.0  # W' reconstitution time const at a full stop (Vassallo running)
TAU_SLOW_S     = 200.0  # ... while moving below CS
_DT = 2.0               # simulation step (s)

_A = DUR_LOSS_2H / (2.0 - DUR_ONSET_H) ** DUR_EXP
_DECLINE = 0.0 if DUR_LOSS_2H <= 0 else WPRIME_LOSS_2H / DUR_LOSS_2H

_DETAIL_DIRS = [DATA_DIR / 'profiles' / 'coros' / 'details',
                DATA_DIR / 'profiles' / 'maddy' / 'details',
                DATA_DIR / 'details']
_WATCH_DAILY = DATA_DIR / 'watch_daily.csv'
_DAILY = DATA_DIR / 'daily.csv'


def _dfrac(tof_hr):
    """Durability fraction at `tof_hr` hours on feet (0 below onset, then the
    accelerating polynomial, capped)."""
    return min(DUR_CAP, _A * max(tof_hr - DUR_ONSET_H, 0.0) ** DUR_EXP)


def _find_detail(label):
    for d in _DETAIL_DIRS:
        p = d / f'{label}.json'
        if p.exists():
            return p
    return None


def _record_segments(path):
    """[(seg_dist_m, rest_after_s)] from a rich record's freq stream + button
    pauses (raw watch distance; rescaled to the run's corrected distance by the
    caller). [] if not a usable rich record."""
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return []
    pts = [(f[0] / 100.0, f[1] / 100.0) for f in rec.get('freq') or [] if f[1]]
    if len(pts) < 10:
        return []
    pauses = sorted((ps / 100.0, dur / 100.0)
                    for ps, _e, dur in rec.get('pauses') or [] if ps and dur)
    t = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts])
    t0 = t[0]
    segs = []; prev = 0.0
    for ps, dur in pauses:
        dat = float(np.interp(ps - t0, t - t0, d))
        segs.append([dat - prev, dur]); prev = dat
    segs.append([float(d[-1]) - prev, 0.0])
    return [s for s in segs if s[0] > 20.0]


@functools.lru_cache(maxsize=1)
def load_segments():
    """{date_str: [(seg_dist_m, rest_after_s)]} for every LONG-RUN day with a
    rich record carrying at least one stop. Empty dict on a log-only build (no
    watch_daily / detail cache) -> callers then apply no pause penalty (honest:
    no stop data). Only long-run dates are parsed (fast)."""
    if not _WATCH_DAILY.exists() or not _DAILY.exists():
        return {}
    long_dates = set(
        pd.read_csv(_DAILY, usecols=['date', 'run_type'])
        .query("run_type == 'long'")['date'].astype(str).str[:10])
    wd = pd.read_csv(_WATCH_DAILY, dtype={'label_ids': str})
    out = {}
    for _, r in wd.iterrows():
        ds = str(r['date'])[:10]
        if ds not in long_dates:
            continue
        labs = str(r.get('label_ids', '')).replace('|', ',').split(',')
        day = []
        for lab in (l.strip() for l in labs):
            if lab and lab != 'nan':
                p = _find_detail(lab)
                if p is not None:
                    day.extend(_record_segments(p))
        if len(day) > 1:          # >1 segment <=> at least one stop
            out[ds] = day
    return out


# Pre-watch runs have no measured stops. Per Max (2026-06-17) they get an
# aggressive UNIFORM imputed pause structure — the PRE_WATCH_PCTILE (95th) of
# the watch-era stop count + total pause, distributed across thirds by the
# corpus-median split (late-loaded) — applied to EVERY pre-watch run regardless
# of route. The goal is to push old, route-uncorrected long runs off the
# demonstrated-capability frontier, not to analyze them accurately. Dial the
# percentile with this one constant.
PRE_WATCH_PCTILE = 0.90


@functools.lru_cache(maxsize=1)
def _pre_watch_profile():
    """(n_segs, total_pause_s, thirds) at PRE_WATCH_PCTILE of the watch-era
    long-run corpus — the single aggressive stop structure imputed onto pre-watch
    runs. None when no watch corpus. Derived from segments alone (no moving-time
    join): total pause = sum of measured rests."""
    segmap = load_segments()
    if not segmap:
        return None
    nsegs, pauses, thirds = [], [], []
    for segs in segmap.values():
        if len(segs) < 2:
            continue
        dists = np.array([s[0] for s in segs]); rests = np.array([s[1] for s in segs])
        tot = dists.sum(); pause = rests.sum()
        if tot <= 0 or pause <= 0:
            continue
        pos = np.cumsum(dists) / tot
        thirds.append([rests[(pos >= a) & (pos < b)].sum() / pause
                       for a, b in [(0, 1/3), (1/3, 2/3), (2/3, 1.0001)]])
        nsegs.append(len(segs)); pauses.append(pause)
    if not nsegs:
        return None
    t = np.median(np.array(thirds), axis=0); t = t / t.sum()
    return {'n_segs': int(round(np.quantile(nsegs, PRE_WATCH_PCTILE))),
            'total_pause_s': float(np.quantile(pauses, PRE_WATCH_PCTILE)),
            'thirds': tuple(t)}


def _impute_segments(d_m):
    """Synthesize [(seg_dist_m, rest_after_s)] for a PRE-WATCH run from the
    uniform P95 profile: equal-distance segments with the profile's stop count,
    total pause distributed across thirds (late-loaded, so the heavy late stops
    land where they carry W' leverage). Returns None when no profile exists."""
    p = _pre_watch_profile()
    if not p:
        return None
    n = max(int(p['n_segs']), 2)
    total = p['total_pause_s']; thirds = p['thirds']
    seg = d_m / n
    pos = [i / n for i in range(1, n)]                  # n-1 internal stops
    tb = [0 if x < 1/3 else (1 if x < 2/3 else 2) for x in pos]
    cnt = [tb.count(t) for t in (0, 1, 2)]
    rests = [thirds[t] * total / cnt[t] if cnt[t] else 0.0 for t in tb]
    return [[seg, rests[i]] for i in range(n - 1)] + [[seg, 0.0]]


def _min_wbal(phases, cs0, dp, dt=_DT):
    """Minimum W'-balance (m) over a run given [(speed_mps, seconds), ...].
    Durability clock = cumulative MOVING time (stops add no fatigue)."""
    dbal = dp; mn = dp; tof = 0.0
    for v, secs in phases:
        for _ in range(int(round(secs / dt))):
            if v > 0.05:
                tof += dt
            D = _dfrac(tof / 3600.0)
            cs_e = cs0 * (1.0 - D)
            dp_e = dp * (1.0 - _DECLINE * D)
            if dp_e < DP_FLOOR * dp:
                dp_e = DP_FLOOR * dp
            if v > cs_e:
                dbal -= (v - cs_e) * dt
            else:
                tau = TAU_STOP_S if v < 0.5 else TAU_SLOW_S
                dbal += (dp_e - dbal) * (1.0 - math.exp(-dt / tau))
            if dbal > dp_e:
                dbal = dp_e
            if dbal < mn:
                mn = dbal
    return mn


def _fastest_feasible(segs, with_stops, d_m, cs0, dp):
    """Fastest constant speed (m/s) that keeps W'bal >= 0 over the run, either
    with this run's stop structure or run straight through."""
    lo, hi = cs0 * 0.4, cs0 * 1.15
    for _ in range(18):
        v = (lo + hi) / 2.0
        if with_stops:
            phases = []
            for sd, rest in segs:
                phases.append((v, sd / v))
                if rest > 0:
                    phases.append((0.0, rest))
        else:
            phases = [(v, d_m / v)]
        if _min_wbal(phases, cs0, dp) >= 0:
            lo = v
        else:
            hi = v
    return lo


def pause_advantage_s_per_mi(date_str, d_m, avg_speed_mps, cs0_mps, dp_m,
                             is_watch=True):
    """Marginal pace (s/mi) the run's stops bought over running it continuously:
    pace(fastest feasible WITHOUT stops) - pace(fastest feasible WITH this run's
    stops). Zero for a continuous run (no stops) at any durability, and zero
    when the run sits far enough below effective CS that the stops reconstitute
    nothing.

    Watch runs use their REAL measured stops (is_watch True; a continuous watch
    run with <2 segments returns 0 — ground truth, untouched). PRE-WATCH runs
    (is_watch False — no segment data) impute the aggressive uniform P95 stop
    structure via _impute_segments(d_m). Returns 0 with no usable segments /
    invalid CS-D'."""
    if not (cs0_mps > 0 and dp_m > 0 and d_m > 0):
        return 0.0
    if is_watch:
        segs = load_segments().get(date_str)
    else:
        segs = _impute_segments(d_m)
    if not segs or len(segs) < 2:
        return 0.0
    tot = sum(s[0] for s in segs)
    if tot <= 0:
        return 0.0
    k = d_m / tot                                  # rescale to corrected distance
    segs = [(s[0] * k, s[1]) for s in segs]
    v_wp = _fastest_feasible(segs, True, d_m, cs0_mps, dp_m)
    v_np = _fastest_feasible(segs, False, d_m, cs0_mps, dp_m)
    adv = METERS_PER_MILE / v_np - METERS_PER_MILE / v_wp
    return max(0.0, adv)
