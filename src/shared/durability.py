"""Long-run pause handling: stop structure, pre-watch imputation, and the
PAUSE-UNCERTAINTY erosion that the 5K-equiv projection actually uses.

>>> Full model, rationale, and rejected alternatives:
>>> docs/long-run-pause-uncertainty-reference.md  (READ before changing this).

THE MODEL (`eroded_deff`, consumed by workouts.project_long_runs): a paused long
run is less trustworthy as proof of continuous capability, so each pause erodes
all *subsequent* confirmed distance by exp(-gate·RATE·pause_sec·lateness) — driven
by pause LENGTH (not count) and LATENESS, gated by an uncapped effort function so
the easy cloud is untouched. It is an UNCERTAINTY model, not a physical recovery
model. Watch runs use measured stops (`load_segments`); pre-watch runs impute the
global P90 stop structure (`_pre_watch_profile` / `_impute_segments`).

LEGACY — DO NOT REVIVE (`pause_advantage_s_per_mi` + the W'-balance sim
`_min_wbal` / `_fastest_feasible` and the DUR_*/TAU_* constants): the original
"physical" pause penalty — the marginal value of a run's stops at the W'-limited
redline. It over-credited EVERY long run (priced at a redline the easy long-run
pace never reached), violated conservation, and was non-monotone; it was REPLACED
by `eroded_deff`. The code survives only to feed the Long Runs plot's display
toggle and is slated for removal — see the reference doc for why it failed.
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
    watch_daily / detail cache) -> callers then apply no pause erosion (honest:
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
# aggressive imputed pause structure scaled to the run's DISTANCE — the
# PRE_WATCH_PCTILE of the watch-era PER-MILE stop count and PER-MILE total
# pause, distributed across thirds by the corpus-median split (late-loaded).
# Both the stop count and total pause scale with run length (a fixed absolute
# pause made no sense — a 10-mi run and a 28-mi run would impute the same
# stoppage). The goal is to push old, route-uncorrected long runs off the
# demonstrated-capability frontier, not to analyze them accurately. Dial the
# aggressiveness with this one percentile constant.
PRE_WATCH_PCTILE = 0.90


@functools.lru_cache(maxsize=1)
def _pre_watch_profile():
    """(segs_per_mi, pause_s_per_mi, thirds) at PRE_WATCH_PCTILE of the watch-era
    long-run corpus — the aggressive PER-MILE stop structure imputed onto
    pre-watch runs, scaled to each run's distance in _impute_segments. None when
    no watch corpus. Derived from segments alone (no moving-time join): pause =
    sum of measured rests, per mile of measured segment distance."""
    segmap = load_segments()
    if not segmap:
        return None
    segs_pm, pause_pm, thirds = [], [], []
    for segs in segmap.values():
        if len(segs) < 2:
            continue
        dists = np.array([s[0] for s in segs]); rests = np.array([s[1] for s in segs])
        tot = dists.sum(); pause = rests.sum()
        if tot <= 0 or pause <= 0:
            continue
        miles = tot / METERS_PER_MILE
        pos = np.cumsum(dists) / tot
        thirds.append([rests[(pos >= a) & (pos < b)].sum() / pause
                       for a, b in [(0, 1/3), (1/3, 2/3), (2/3, 1.0001)]])
        segs_pm.append(len(segs) / miles); pause_pm.append(pause / miles)
    if not segs_pm:
        return None
    t = np.median(np.array(thirds), axis=0); t = t / t.sum()
    return {'segs_per_mi': float(np.quantile(segs_pm, PRE_WATCH_PCTILE)),
            'pause_s_per_mi': float(np.quantile(pause_pm, PRE_WATCH_PCTILE)),
            'thirds': tuple(t)}


def eroded_deff(date_str, d_m, gate, rate, is_watch=True):
    """Demonstrated effective distance after PAUSE-UNCERTAINTY erosion.

    Not a physical recovery model — it encodes that a paused long run is less
    trustworthy as proof of continuous capability. Every SECOND of a pause erodes
    ALL subsequent moving distance, weighted by how LATE the pause is: a pause of
    P seconds at fraction L of the run multiplies the credit of everything after
    it by exp(-gate·rate·P·L). Consequences, by design:
      • pause LENGTH drives erosion, not count — a 20s stop is ~nil, a 5-min stop
        is large, and it can't be gamed by resuming and re-pausing;
      • a LATE pause (high L) bites far harder per mile than an early one (the
        factor is linear in run-fraction-before-the-pause);
      • no lower bound — every second contributes, infinitesimally.
    `gate` is the effort-gated scale (0 for easy runs → full distance, rising to 1
    as the run pace nears the CS-predicted race pace at its distance), `rate` the
    per-(second·lateness) erosion rate. Watch runs use measured stops; pre-watch
    the imputed structure."""
    if not (gate > 0 and rate > 0 and d_m > 0):
        return d_m
    segs = load_segments().get(date_str) if is_watch else _impute_segments(d_m)
    if not segs:
        return d_m
    tot = sum(s[0] for s in segs)
    if tot <= 0:
        return d_m
    k = d_m / tot                            # rescale measured segs to corrected distance
    cum = 0.0
    expo = 0.0
    d_eff = 0.0
    for sd, rest in segs:
        d_eff += sd * k * math.exp(-expo)
        cum += sd * k
        if rest > 0:
            expo += gate * rate * rest * (cum / d_m)   # lateness-weighted, per second
    return d_eff


def imputed_pause_total_s(d_m):
    """Total imputed pause (s) for a PRE-WATCH run of distance `d_m` — the
    per-mile PRE_WATCH_PCTILE pause rate scaled to the run's length (matches the
    total _impute_segments distributes). None when there's no watch corpus."""
    p = _pre_watch_profile()
    return None if not p else p['pause_s_per_mi'] * (d_m / METERS_PER_MILE)


def _impute_segments(d_m):
    """Synthesize [(seg_dist_m, rest_after_s)] for a PRE-WATCH run by scaling the
    per-mile profile to the run's distance: stop count and total pause both grow
    with length, distributed across thirds (late-loaded, so the heavy late stops
    land where they carry W' leverage). Returns None when no profile exists."""
    p = _pre_watch_profile()
    if not p:
        return None
    miles = d_m / METERS_PER_MILE
    n = max(int(round(p['segs_per_mi'] * miles)), 2)
    total = p['pause_s_per_mi'] * miles; thirds = p['thirds']
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
    """LEGACY (display-only; see module header) — superseded by `eroded_deff`.
    Marginal pace (s/mi) the run's stops bought over running it continuously:
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
