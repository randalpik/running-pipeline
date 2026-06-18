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

HISTORY — the original "physical" pause model (a durability + W'-balance
marginal-stop-value simulation, `pause_advantage_s_per_mi`) was REMOVED. Priced at
the W'-limited redline it over-credited every long run, violated conservation, and
was non-monotone. Do not reintroduce it; the reference doc records why it failed.
"""
from __future__ import annotations
import functools
import json
import math

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE

_DETAIL_DIRS = [DATA_DIR / 'profiles' / 'coros' / 'details',
                DATA_DIR / 'profiles' / 'maddy' / 'details',
                DATA_DIR / 'details']
_WATCH_DAILY = DATA_DIR / 'watch_daily.csv'
_DAILY = DATA_DIR / 'daily.csv'


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


