"""Performance frontier — the demonstrated-5K-capability envelope (red line).

Semantics (Max, June 2026): every kept Training-quality point and eligible
race 5K-equivalent is PROOF of 5K capability at its date. The frontier
answers "how fast COULD I have run a 5K on a given day, given the
performances surrounding it" — a race-PREDICTION line, deliberately more
responsive than CS. It is an envelope over evidence, not a posterior: no CI;
accuracy is owned by upstream point selection (TQ gates + race eligibility).

Formulation (Max, June 2026): the frontier is bounded below by the
**CS-implied 5K prediction** — the baseline assumption is "I'm at least as
fast as CS predicts", and CS's decay structure is far more rigorously
verified than anything derivable from demonstrations alone. Demonstrations
contribute only their EXCESS above that floor:

    excess_i = log(cs5k_pace(t_i) / pace_i)          (clipped at 0)

so the red line rides the CS-5K curve through quiet stretches and bulges
faster wherever something super-CS was demonstrated, relaxing back to the
floor afterward. Earlier absolute-cone designs are dead: linear arms (a
zigzag — no physiological process is linear), then exponential arms with a
bounded total loss (a 2017 peak's saturated tail clamped all of 2018 flat —
"immortal tails"). Anchoring decay to distance-from-CS removes both.

The two arms are different processes (Max, June 2026):
  FORWARD (decay) — first-order relaxation in ABSOLUTE pace space, run as
  a single-state recursion along the grid: each day the frontier loses
  the fraction (1 - exp(-dt/TAU_FWD)) of its current gap to the floor
  ("the farther away from the line, the faster we decay"), asymptotic
  into the reference line. Crucially the relaxation acts on the absolute
  pace, not the floor-relative excess: absent a new demonstration the
  frontier can NEVER get absolutely faster day-over-day. (The pre-July-
  2026 closed form frontier = floor * exp(-excess * e^(-dt/TAU_FWD))
  decayed the RATIO and let the moving floor carry the bulge — on
  Maddy's steeply improving novice curve the "decay" inverted into an
  upward swing that out-ran every race ever completed. Dead.) Under a
  static or IMPROVING floor, merged stretches sit on the fitness line
  exactly and follow it up — that improvement is the race-pinned CS
  fit's to give. Under a DECLINING-fitness floor the frontier lags at
  the relaxation's standing gap (~TAU_FWD x floor slope): capability
  decays at the detraining constant, not at the fit's pace (Max, July
  2026 — replaces the earlier "rides the CS curve through quiet
  stretches, declines included" rule).

  Floors must be FITNESS-LINE-derived, never another frontier: a frontier
  used as a floor injects its own bulge decay into the carry (double-
  counting TAU and, on a dip-then-rise floor, latching values no
  demonstration supports). Cross-frontier influence composes at the
  consumer via min(), e.g. the per-bin prediction channel.
  BACKWARD (build) — a Gaussian shoulder in excess space,
  w = exp(-(dt/TAU_BWD)^2): zero slope at the peak ("you were nearly this
  fit just before a peak" — fitness gains saturate approaching a peak),
  smooth steepening further back, conceptually constrained by how fast
  fitness can physiologically increase. Plain P=2 — smooth everywhere,
  unlike the rejected P=4 hold-then-cliff shoulder.

No gap machinery: a proof's influence dies within ~8 weeks by shape, and
the CS baseline itself carries long absences (e.g. the 2020-21 labrum
crash — the race fit already knows). Demonstrations at or below the floor
contribute nothing; only super-CS efforts shape the line, and a point that
binds the envelope DEFINES it locally (auditable via the `binding` flag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE
from src.shared.cs_projection import (project_races_to_5k_pace,
                                      pace5k_series_to_anchor, load_cs_outputs,
                                      admit_best_per_day)

# FORWARD: first-order relaxation into the floor — the frontier sheds a
# fixed fraction of its current gap to the reference line per day. tau from
# the Banister-range detraining literature + the June 2026 pure-decay fit.
# BACKWARD: Gaussian build shoulder — flat at the peak, steepening back.
# tau calibrated so a typical modern peak (~10 s/mi excess) implies a max
# build rate matching the measured q90 gain rate (~1.74 s/mi/wk):
# max slope of E*exp(-(dt/tau)^2) is E*sqrt(2/e)/tau -> tau ~= 34d.
# (History: linear arms -> zigzag; bounded-loss exponentials -> immortal
# tails; super-Gaussian P=4 both ways -> hold-then-cliff-then-stop, not a
# physical process; symmetric exponential relaxation -> spiky builds;
# ratio-space forward decay -> moving-floor inversion on Maddy's novice
# curve, July 2026.)
TAU_FWD = 45.0   # days — forward (detraining) relaxation
TAU_BWD = 34.0   # days — backward (build) Gaussian shoulder width

# De-minimis excess (log units; ~0.05 s/mi at 8:00 pace — below display
# resolution). Contributions at or below this don't mark the envelope. This
# is annotation hygiene, not a value guard: the Gaussian shoulder has
# infinite support, so an epsilon-excess demo would otherwise claim the
# `binder`/`binding` labels across the whole grid while denting the curve
# by nothing (mislabelled tooltips on the Fitness tab).
MIN_EXCESS = 1e-4

CORPUS_PATH = DATA_DIR / 'training_quality_corpus.csv'


def load_corpus_demos():
    """Kept TQ points as demonstrations: date, pace_min, src, detail.

    Reads the artifact persisted by plot_training_quality.py. Returns an
    empty frame (same columns) when the artifact is missing or empty — a
    watch profile early in its history, or a run ordering where the TQ plot
    hasn't been built yet; the frontier then rests on races alone.
    """
    cols = ['date', 'pace_min', 'src', 'category', 'display_name',
            'city_state', 'detail']
    if not CORPUS_PATH.exists():
        return pd.DataFrame(columns=cols)
    try:
        c = pd.read_csv(CORPUS_PATH, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    if c.empty:
        return pd.DataFrame(columns=cols)
    # display_name / city_state feed the Fitness tooltip title; tolerate older
    # corpus artifacts that predate the columns.
    return pd.DataFrame({'date': c['date'], 'pace_min': c['p5k_corr_min'],
                         'src': c['src'], 'category': c['category'],
                         'display_name': c.get('display_name', ''),
                         'city_state': c.get('city_state', ''),
                         'detail': c['detail']})


def standard_demos(daily_summary, beta_long, d_thresh, xc_correction,
                   races_path=None, exclusions_path=None, corpus=None):
    # exclusions_path is accepted and IGNORED (Aug 2026): the fit no longer
    # prunes races, it weights them continuously by causal shortfall
    # (bayes_cs_fit.causal_race_weights). Kept in the signature so existing
    # callers — and a stale artifact restored from a CI cache — are harmless.
    """The canonical demonstration set: eligible races (fit conventions —
    hard eligibility, XC correction, beta un-bias) plus the kept TQ corpus. Every frontier consumer builds from this so the line is
    identical across tabs.

    corpus: pass a prebuilt corpus frame (plot_training_quality has one
    in-memory) to skip the artifact read; default reads the CSV.
    """
    race_demos = _race_demos(daily_summary, beta_long, d_thresh, xc_correction,
                             races_path)
    if corpus is None:
        corpus = load_corpus_demos()
    return pd.concat([race_demos, corpus], ignore_index=True)


def _race_demos(daily_summary, beta_long, d_thresh, xc_correction,
                races_path=None):
    """Race demonstrations (fit conventions: hard eligibility, XC correction,
    projection) as (date, pace_min, src, category, detail). Shared by
    standard_demos and the structure gate.

    EVERY eligible race is a demonstration — there is no exclusion step. The
    fit weights races by causal shortfall rather than pruning them, so a race
    the model largely discounts still appears here and on the chart; it simply
    does not bind the frontier, because a slow race never could.

    time ≥ 120 s: every race of 2+ minutes is frontier evidence — 800s included
    and may bind (a ≥1500 m cutoff was tried and reverted June 2026: the
    2017-03-30 800 was a coherent lifetime-best peak, not an aberration). Only
    sub-120 s 400s stay out — pure sprints, displayed but never defining
    aerobic capability.
    """
    races_path = races_path or (DATA_DIR / 'races.csv')
    races = pd.read_csv(races_path, parse_dates=['date'])
    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'
    # Downhill admission matches build_eligible in bayes_cs_fit: a Downhill
    # race with measured grade coverage is corrected to its flat-equivalent
    # time and may inform the frontier; without coverage it stays excluded.
    # (bin_frontier deliberately differs — it runs on RAW times, where an
    # aided time would inflate a per-distance card.)
    from src.shared.recovery_model import race_physical_correction
    has_measured = race_physical_correction(races)['has_measured'].to_numpy()
    elig = races[((races['surface'] != 'Downhill') | has_measured)
                 & (races['time_sec'] >= 120)].copy()
    # One race per multi-race day, the best 5K-equivalent — the same rule the
    # fit and the Fitness plot apply. Kept in step deliberately: the frontier is
    # drawn on the Fitness tab over those same diamonds, so a race admitted
    # there and ignored here would put the red line under a visible point.
    elig = admit_best_per_day(elig)
    elig = project_races_to_5k_pace(
        elig, daily_summary, beta_long, d_thresh,
        apply_xc_correction=True, xc_correction=xc_correction)
    return pd.DataFrame({
        'date': elig['date'],
        'pace_min': elig['pace_norm_min'],
        'src': 'race',
        'category': 'race',
        'detail': (elig['event'].fillna('(no event)').astype(str)
                   if 'event' in elig.columns else '(race)'),
    })


def gate_estimated_binders(workouts):
    """Gate 2 (Max, June 2026): flag ESTIMATED-structure, course-OK workouts
    that would BIND the frontier as 'uncertain structure'. Returns a boolean
    mask aligned to ``workouts.index`` (True = flag).

    The frontier floor is built from VERIFIED-structure demos ONLY — kept races
    plus verified-structure, course-trusted workouts. Any estimated-structure
    workout whose 5K-equiv pokes above that floor is distrusted: a guessed
    decomposition (e.g. a bare "2800f" the parser split into 7×400) cannot
    claim a frontier-defining result — it probably hid shorter reps. Estimated
    workouts sitting below the floor ride along untouched (the accepted bulk).
    Long runs are omitted from this floor — they don't bind at workout dates.
    """
    mask = pd.Series(False, index=workouts.index)
    if 'structure_verified' not in workouts.columns or workouts.empty:
        return mask
    sv = workouts['structure_verified'].astype(bool)
    ok = workouts['excluded_reason'].isna() & workouts['p5k_min'].notna()
    est = (~sv) & ok
    if not est.any():
        return mask
    # Load the canonical CS frame ourselves (the caller's may be stripped of
    # derived columns like dp3_evid_med / p5k_implied_min).
    daily, bl, dt, xc = load_cs_outputs(str(DATA_DIR))
    verw = workouts[sv & ok]
    ver_demos = pd.concat([
        _race_demos(daily, bl, dt, xc)[['date', 'pace_min']],
        pd.DataFrame({'date': verw['date'], 'pace_min': verw['p5k_min']}),
    ], ignore_index=True)
    grid = pd.DatetimeIndex(daily['date'])
    vfront, _ = build_frontier(ver_demos, grid,
                               daily['p5k_implied_min'].to_numpy(float))
    gd = grid.to_numpy('datetime64[D]').astype(float)
    fp = vfront['frontier_pace_min'].to_numpy(float)
    for i in workouts.index[est]:
        d = np.datetime64(pd.Timestamp(workouts.at[i, 'date']), 'D').astype(float)
        floor = float(np.interp(d, gd, fp))
        if workouts.at[i, 'p5k_min'] < floor - 1e-9:   # faster than floor → would bind
            mask.at[i] = True
    return mask


def frontier_at_anchor(frontier, daily_summary, anchor_m, beta_long, d_thresh):
    """Frontier-implied total time (seconds) at anchor_m for each grid date.

    Mirror of cs_projection.cs_line_at_anchor, sourced from the frontier:
    back out the frontier-implied CS at each date (CP3 inversion of the 5K
    frontier time against D′₃(t)), then forward-solve the anchor time on
    that curve, with the β_long factor for anchors above d_thresh. NaN
    where the frontier is NaN.

    frontier and daily_summary must share the same grid (both are built on
    the summary's dates by every caller). Forward direction → the
    prediction-edge v_max (conservative-low: short anchors never get a
    time faster than the demonstrated capability warrants).
    """
    return pace5k_series_to_anchor(frontier['frontier_pace_min'].to_numpy(float),
                                   daily_summary, anchor_m, beta_long, d_thresh)


def p5k_from_asym(daily_summary, pace_col):
    """5K-implied pace (min/mi) from an asymptotic-CS pace column of the
    summary, holding dp_med — the same bridge load_cs_outputs applies for
    the median (`p5k_implied_min`)."""
    dp = daily_summary['dp_med'].to_numpy(float)
    cs_mps = METERS_PER_MILE / (daily_summary[pace_col].to_numpy(float) * 60.0)
    return (5000.0 - dp) / cs_mps / 5000.0 * METERS_PER_MILE / 60.0


def bin_frontier(bin_races, anchor_m, grid_dates, daily_summary,
                 beta_long, d_thresh, p5k_pace_min):
    """Per-bin demonstrated-capability channel (Max, July 2026): a frontier
    over the ACTUAL race times in one distance bin, relaxing toward the
    fitness line converted to the bin anchor. Composes with the cross-
    distance channel at the consumer — card = min(cross, bin) — so a race
    at a distance proves that time at that distance regardless of what any
    cross-distance conversion says, all the way down to 400 m.

    bin_races: this bin's races (caller snaps distances to bins). Downhill
    is excluded here (aided times would inflate proof); fatigued races and
    time trials stay — a time actually run is proven capability. No XC
    correction (raw-time proof can only under-claim). Near-anchor races
    (3200 in the 2-Mile bin) are projected to the anchor with the same
    evidence-direction conversion the race tabs use.

    The floor is the FITNESS line at the anchor, never another frontier —
    see the module docstring. Returns total-time seconds at anchor_m per
    grid date; an empty bin returns the converted fitness line unchanged.
    """
    floor_t = np.asarray(pace5k_series_to_anchor(
        p5k_pace_min, daily_summary, anchor_m, beta_long, d_thresh), float)
    miles = anchor_m / METERS_PER_MILE
    # Downhill stays categorically excluded HERE even when watch-covered
    # (unlike _race_demos / build_eligible): the bin frontier runs on RAW
    # times, so an aided time would inflate a per-distance card.
    keep = bin_races[bin_races.get(
        'surface', pd.Series('', index=bin_races.index)).fillna('') != 'Downhill']
    if keep.empty:
        return floor_t
    proj = project_races_to_5k_pace(keep.copy(), daily_summary, beta_long,
                                    d_thresh, apply_xc_correction=False,
                                    apply_physical_correction=False,
                                    norm_dist_m=anchor_m)
    demos = pd.DataFrame({
        'date': proj['date'],
        'pace_min': proj['time_norm_sec'] / miles / 60.0,
        'src': 'race', 'category': 'race',
        'detail': (proj['event'].fillna('').astype(str)
                   if 'event' in proj.columns else ''),
    })
    bf, _ = build_frontier(demos, grid_dates, floor_t / miles / 60.0)
    return bf['frontier_pace_min'].to_numpy(float) * miles * 60.0


def build_frontier_band(demos, grid_dates, daily_summary):
    """Frontier swept across the CS 95% CrI ("what would the frontier be if
    the floor sat anywhere in the CS interval"). Returns (frontier_med,
    frontier_lo95, frontier_hi95, demos_med) — lo/hi are frontier frames
    whose floors are the CS lo95/hi95 5K predictions (lo95 = faster floor).
    Where a demonstration binds, all three collapse onto the demo's pace
    (proof pins the prediction); on the floor the band equals the CS CrI.
    """
    med, demos_med = build_frontier(demos, grid_dates,
                                    daily_summary['p5k_implied_min'])
    lo, _ = build_frontier(demos, grid_dates,
                           p5k_from_asym(daily_summary, 'cs_pace_lo95'))
    hi, _ = build_frontier(demos, grid_dates,
                           p5k_from_asym(daily_summary, 'cs_pace_hi95'))
    # The sweeps can bind on DIFFERENT demos (a slower floor deepens every
    # excess, so e.g. a race's build shoulder reaches further back on the
    # hi sweep), and the recursive forward arm relaxes each line toward its
    # own floor — so raw sweeps can cross the median and put the line
    # outside its own band. A sweep with a slower floor must never claim a
    # faster capability: clamp the edges onto the median (no-op at binding
    # demos, where all three already collapse onto the demo's pace).
    lo['frontier_pace_min'] = np.minimum(lo['frontier_pace_min'],
                                         med['frontier_pace_min'])
    hi['frontier_pace_min'] = np.maximum(hi['frontier_pace_min'],
                                         med['frontier_pace_min'])
    return med, lo, hi, demos_med


def build_frontier(demos, grid_dates, cs5k_pace_min):
    """Excess-above-CS envelope on `grid_dates`.

    Parameters
    ----------
    demos : DataFrame with date, pace_min (5K-equivalent pace, min/mi),
        src, detail. Races and TQ points alike.
    grid_dates : DatetimeIndex.
    cs5k_pace_min : array-like aligned with grid_dates — the CS-implied 5K
        pace (min/mi), i.e. load_cs_outputs' `p5k_implied_min`. This is the
        frontier's floor.

    Returns
    -------
    frontier : DataFrame(date, frontier_pace_min, binder) — never NaN where
        cs5k is defined (floor = CS-5K); binder is the index into the
        returned demos of the excess that defines the bulge (-1 on the floor).
    demos : input sorted/reindexed, plus `excess` (log, >=0 above floor) and
        `binding` (defines the frontier somewhere).
    """
    demos = (demos.dropna(subset=['pace_min', 'date'])
                  .sort_values('date').reset_index(drop=True).copy())
    gd = grid_dates.to_numpy('datetime64[D]').astype(float)
    cs5k = np.asarray(cs5k_pace_min, dtype=float)
    if demos.empty or len(gd) == 0:
        return (pd.DataFrame({'date': grid_dates,
                              'frontier_pace_min': cs5k if len(gd) else [],
                              'binder': -1}),
                demos.assign(excess=pd.Series(dtype=float),
                             binding=pd.Series(dtype=bool)))

    dd = demos['date'].to_numpy('datetime64[D]').astype(float)
    cs5k_at = np.interp(dd, gd, cs5k)
    excess = np.log(cs5k_at / demos['pace_min'].to_numpy(float))
    excess = np.maximum(excess, 0.0)
    demos['excess'] = excess

    # Backward (build) shoulders + demo-date pins, in excess space. dt == 0
    # lands here so every super-floor demo pins the frontier on its own date.
    env = np.zeros(len(gd))
    binder = np.full(len(gd), -1)
    for i in np.nonzero(excess > MIN_EXCESS)[0]:
        dt = gd - dd[i]
        w = np.where(dt <= 0, np.exp(-(dt / TAU_BWD) ** 2), 0.0)
        contrib = excess[i] * w
        take = (contrib > env) & (contrib > MIN_EXCESS)
        env[take] = contrib[take]
        binder[take] = i
    fp = cs5k * np.exp(-env)     # seed: floor, dented by builds/demo pins

    # Demos BEFORE the grid start (windowed plot grids — workouts/TQ/long
    # runs floor their x-axis mid-history) must still project their forward
    # tails into the window: seed the carry at gd[0] with each pre-grid
    # demo's relaxed claim. Constant-floor closed form against cs5k[0] —
    # the same floor their excess saw (np.interp clamps there).
    pre = np.nonzero((excess > MIN_EXCESS) & (dd < gd[0]))[0]
    if len(pre) and np.isfinite(cs5k[0]):
        pace_pre = demos['pace_min'].to_numpy(float)[pre]
        cand = cs5k[0] - (cs5k[0] - pace_pre) * np.exp(-(gd[0] - dd[pre]) / TAU_FWD)
        j = int(np.argmin(cand))
        # De-minimis guard: only a claim meaningfully below the floor may arm
        # the carry (mirror of the MIN_EXCESS cut in the envelope loop).
        if cand[j] < fp[0] and np.log(fp[0] / cand[j]) > MIN_EXCESS:
            fp[0] = cand[j]
            binder[0] = int(pre[j])

    # Forward (decay) arm — first-order relaxation in absolute pace space,
    # carried recursively along the grid: yesterday's frontier value sheds
    # the fraction (1 - exp(-step/TAU_FWD)) of its gap to today's floor.
    # SINGLE-STATE (Max, July 2026): the carry is always live. Under a
    # static or improving floor the min() keeps merged stretches on the
    # fitness line exactly; under a declining-fitness (slowing) floor the
    # frontier lags at the relaxation's standing gap (~TAU_FWD x slope) —
    # capability decays at the detraining constant, not at the fit's pace.
    # (Replaces the two-state binder-gated carry, whose arm/expire
    # machinery existed only to force declines to be tracked exactly — a
    # rule Max retired.) A NaN floor breaks the carry chain.
    prev_pace = prev_day = np.nan
    prev_binder = -1
    for t in range(len(gd)):
        if not np.isfinite(cs5k[t]):
            prev_pace = np.nan
            continue
        if np.isfinite(prev_pace):
            alpha = 1.0 - np.exp(-(gd[t] - prev_day) / TAU_FWD)
            carry = prev_pace + alpha * (cs5k[t] - prev_pace)
            if carry < fp[t]:
                fp[t] = carry
                binder[t] = prev_binder
        prev_pace, prev_day, prev_binder = fp[t], gd[t], binder[t]

    frontier = pd.DataFrame({
        'date': grid_dates,
        'frontier_pace_min': fp,
        'binder': binder,
    })
    bound_ids = set(int(b) for b in np.unique(binder) if b >= 0)
    demos['binding'] = [i in bound_ids for i in range(len(demos))]
    return frontier, demos
