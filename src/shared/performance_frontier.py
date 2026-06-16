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
    frontier(t) = cs5k_pace(t) * exp(-max_i [ excess_i * w(t - t_i) ])

so the red line rides the CS-5K curve through quiet stretches and bulges
faster wherever something super-CS was demonstrated, relaxing back to the
floor afterward. Earlier absolute-cone designs are dead: linear arms (a
zigzag — no physiological process is linear), then exponential arms with a
bounded total loss (a 2017 peak's saturated tail clamped all of 2018 flat —
"immortal tails"). Anchoring decay to distance-from-CS removes both.

The two arms are different processes (Max, June 2026):
  FORWARD (decay) — first-order relaxation, w = exp(-dt/TAU_FWD): loss
  rate proportional to current distance from the floor ("the farther away
  from the line, the faster we decay"), asymptotic into the reference line.
  BACKWARD (build) — a Gaussian shoulder, w = exp(-(dt/TAU_BWD)^2): zero
  slope at the peak ("you were nearly this fit just before a peak" —
  fitness gains saturate approaching a peak), smooth steepening further
  back, conceptually constrained by how fast fitness can physiologically
  increase. Plain P=2 — smooth everywhere, unlike the rejected P=4
  hold-then-cliff shoulder.

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
from src.shared.cs_projection import (project_races_to_5k_pace,
                                      pace5k_series_to_anchor)

# FORWARD: first-order relaxation into the floor — excess decays at a rate
# proportional to its current distance from the reference line. tau from
# the Banister-range detraining literature + the June 2026 pure-decay fit.
# BACKWARD: Gaussian build shoulder — flat at the peak, steepening back.
# tau calibrated so a typical modern peak (~10 s/mi excess) implies a max
# build rate matching the measured q90 gain rate (~1.74 s/mi/wk):
# max slope of E*exp(-(dt/tau)^2) is E*sqrt(2/e)/tau -> tau ~= 34d.
# (History: linear arms -> zigzag; bounded-loss exponentials -> immortal
# tails; super-Gaussian P=4 both ways -> hold-then-cliff-then-stop, not a
# physical process; symmetric exponential relaxation -> spiky builds.)
TAU_FWD = 45.0   # days — forward (detraining) relaxation
TAU_BWD = 34.0   # days — backward (build) Gaussian shoulder width

CORPUS_PATH = DATA_DIR / 'training_quality_corpus.csv'


def load_corpus_demos():
    """Kept TQ points as demonstrations: date, pace_min, src, detail.

    Reads the artifact persisted by plot_training_quality.py. Returns an
    empty frame (same columns) when the artifact is missing or empty — a
    watch profile early in its history, or a run ordering where the TQ plot
    hasn't been built yet; the frontier then rests on races alone.
    """
    cols = ['date', 'pace_min', 'src', 'category', 'detail']
    if not CORPUS_PATH.exists():
        return pd.DataFrame(columns=cols)
    try:
        c = pd.read_csv(CORPUS_PATH, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    if c.empty:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame({'date': c['date'], 'pace_min': c['p5k_corr_min'],
                         'src': c['src'], 'category': c['category'],
                         'detail': c['detail']})


def standard_demos(daily_summary, beta_long, d_thresh, xc_correction,
                   races_path=None, exclusions_path=None, corpus=None):
    """The canonical demonstration set: kept races (fit conventions —
    hard eligibility, auto-exclusions, XC correction, beta un-bias) plus the
    kept TQ corpus. Every frontier consumer builds from this so the line is
    identical across tabs.

    corpus: pass a prebuilt corpus frame (plot_training_quality has one
    in-memory) to skip the artifact read; default reads the CSV.
    """
    races_path = races_path or (DATA_DIR / 'races.csv')
    exclusions_path = exclusions_path or (DATA_DIR / 'bayes_cs_auto_exclusions.csv')
    races = pd.read_csv(races_path, parse_dates=['date'])
    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'
    # time ≥ 120 s: every race of 2+ minutes is frontier evidence — 800s
    # INCLUDED, and they may bind. A distance ≥ 1500 m cutoff was tried and
    # reverted the same day (June 2026): Max initially read his 800s
    # binding the frontier as the model overrating his short distances,
    # but the 2017-03-30 800 case resolved it — it genuinely was his best
    # lifetime effort to that point, superseded three weeks later by three
    # 1600s and an interval workout (a coherent peak, not an aberration).
    # The "3200s set me apart in HS" perception was competition density
    # (more proficient HS milers than 2-milers), not absolute capability.
    # Only sub-120 s races (400s) stay out: pure-sprint efforts read via
    # the conservative evidence-edge v_max, displayed but never defining
    # aerobic capability.
    elig = races[(~races['fatigued'].astype(bool))
                 & (races['surface'] != 'Downhill')
                 & (races['time_sec'] >= 120)].copy()
    if exclusions_path and exclusions_path.exists():
        try:
            excl = pd.read_csv(exclusions_path, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            excl = pd.DataFrame()
        if len(excl):
            keys = set(zip(excl['date'].dt.date, excl['distance_m'].astype(int)))
            elig = elig[~pd.Series(
                list(zip(elig['date'].dt.date, elig['distance_m'].astype(int))),
                index=elig.index).isin(keys)]
    elig = project_races_to_5k_pace(
        elig, daily_summary, beta_long, d_thresh,
        apply_xc_correction=True, xc_correction=xc_correction)
    race_demos = pd.DataFrame({
        'date': elig['date'],
        'pace_min': elig['pace_norm_min'],
        'src': 'race',
        'category': 'race',
        'detail': (elig['event'].fillna('(no event)').astype(str)
                   if 'event' in elig.columns else '(race)'),
    })
    if corpus is None:
        corpus = load_corpus_demos()
    return pd.concat([race_demos, corpus], ignore_index=True)


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


def build_frontier_band(demos, grid_dates, daily_summary):
    """Frontier swept across the CS 95% CrI ("what would the frontier be if
    the floor sat anywhere in the CS interval"). Returns (frontier_med,
    frontier_lo95, frontier_hi95, demos_med) — lo/hi are frontier frames
    whose floors are the CS lo95/hi95 5K predictions (lo95 = faster floor).
    Where a demonstration binds, all three collapse onto the demo's pace
    (proof pins the prediction); on the floor the band equals the CS CrI.
    """
    dp = daily_summary['dp_med'].to_numpy(float)

    def p5k_from_asym(pace_col):
        # 5K-implied pace from an asymptotic-CS pace column, holding dp_med:
        # cs_mps = 1609.344/(pace*60); p5k = (5000-dp)/cs_mps /5000*1609.344/60
        cs_mps = 1609.344 / (daily_summary[pace_col].to_numpy(float) * 60.0)
        return (5000.0 - dp) / cs_mps / 5000.0 * 1609.344 / 60.0

    med, demos_med = build_frontier(demos, grid_dates,
                                    daily_summary['p5k_implied_min'])
    lo, _ = build_frontier(demos, grid_dates, p5k_from_asym('cs_pace_lo95'))
    hi, _ = build_frontier(demos, grid_dates, p5k_from_asym('cs_pace_hi95'))
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

    env = np.zeros(len(gd))
    binder = np.full(len(gd), -1)
    for i in np.nonzero(excess > 0)[0]:
        dt = gd - dd[i]
        w = np.where(dt >= 0,
                     np.exp(-np.abs(dt) / TAU_FWD),          # decay: relaxation
                     np.exp(-(dt / TAU_BWD) ** 2))           # build: shoulder
        contrib = excess[i] * w
        take = contrib > env
        env[take] = contrib[take]
        binder[take] = i

    frontier = pd.DataFrame({
        'date': grid_dates,
        'frontier_pace_min': cs5k * np.exp(-env),
        'binder': binder,
    })
    bound_ids = set(int(b) for b in np.unique(binder) if b >= 0)
    demos['binding'] = [i in bound_ids for i in range(len(demos))]
    return frontier, demos
