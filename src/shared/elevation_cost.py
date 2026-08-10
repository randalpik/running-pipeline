"""Physical hill cost — the two-channel model (Aug 2026 overhaul;
docs/route-normalization-reference.md is the canonical home).

    cost_fraction = c(g_up)·gain_pm − b(g_dn)·loss_pm
    c(g) = c0 + c1·g          climb cost per ft/mi of GROSS gain
    b(g) = b0 + b1·g          descent benefit per ft/mi of GROSS loss (b1 < 0)
    cost_s_per_mi = cost_fraction · flat_equivalent_pace

Inputs are pure course geometry on the fused baro+DEM substrate
(src/coros/elevation.py). MAGNITUDE is gross gain/loss IN EXCESS of the
measured flat-ground floor (~15 ft/mi of TV-noise + micro-undulation that the
day-demeaned calibration absorbs and therefore never priced) — all real
vertical is priced (a 1% grade climbed for 10 miles is 500 real feet; a
magnitude keyed on detected hills would delete it, and flickered ±18% across
reruns of one course, vs ±5-8% for gross). STEEPNESS enters as the hills'
vert·grade mass spread over the excess: g_eff = Σ_hills(vert·grade)/E_gain —
the exact statistic under which the run-level closed form equals the
continuous fit's window-sum (hill-less vertical enters at grade 0). The hills
machinery measures how steep the real climbs and descents are (reproducible
to ~0.1% across course reruns) and contributes nothing else. Nothing about
execution enters, so a correction cannot be moved by running differently.

Why two channels and not a refund ratio (the earlier form): the ratio forces
one parameter to carry two physically separate slopes, and every unconstrained
fit pushed it unphysical (>1 — gently rolling courses reading net-fast) when
the data wanted a small climb cost against a large descent benefit. It also
assumed the ups and downs of a run share an effort, which is what kept hill
workouts on a separate Minetti model. What the Aug 2026 fit established
(mile-grain day-FE, quintile drift, paved+mixed+trail, ~7k miles / ~1k days):

  * climb cost is near grade-invariant (c1 small — Minetti-consistent); the
    steepness effect is a DESCENT phenomenon (b1 negative: at hill grades the
    per-ft return shrinks toward the braking regime);
  * earlier steeper-looking slopes were measurement artifacts (80 m smoothing
    + merge-diluted grade denominators inflated the descent slope ~80%).

Constants are LIVE: scripts/calibrate_climb.py refits them from the profile's
own mile splits each build (data/elevation_calibration.csv), with validation
warnings; the defaults here are the committed fallback for thin corpora (new
profiles, CI without artifacts).
"""
import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR

# Fallback = the Aug 2026 continuous (1.5 mi sliding-window) fit on Max's
# fused corpus. floor_* = the flat-ground pedestal (ft/mi) the day-demeaned
# fit absorbs and application must therefore subtract.
DEFAULT_PARAMS = {'c0': 0.49e-3, 'c1': 0.052e-3, 'b0': 0.46e-3,
                  'b1': -0.036e-3, 'floor_g': 14.8, 'floor_l': 14.5}

# Model-input domain: the calibration observes hill grades to ~p99 10%; beyond
# that both channels are extrapolation, so grades are clamped.
GRADE_DOMAIN_PCT = 12.0

# The two steepness inputs vary per run, so any single number quoted to a
# reader has to name the grade it belongs to. This is the corpus-typical hill
# grade — the one place it is defined, so the plots agree with each other.
DISPLAY_GRADE = 3.0

CALIBRATION_PATH = DATA_DIR / 'elevation_calibration.csv'
_PARAMS_CACHE = {}


def engine_params():
    """c0/c1/b0/b1 from the live calibration artifact
    (scripts/calibrate_climb.py), falling back to DEFAULT_PARAMS. Cached per
    data dir (same lifecycle as physical_route_betas)."""
    key = str(DATA_DIR)
    if key not in _PARAMS_CACHE:
        p = dict(DEFAULT_PARAMS)
        if CALIBRATION_PATH.exists():
            try:
                row = pd.read_csv(CALIBRATION_PATH).iloc[0]
                keys = ('c0', 'c1', 'b0', 'b1', 'floor_g', 'floor_l')
                vals = {k: float(row[k]) for k in keys if k in row.index}
                if (len(vals) == len(keys)
                        and all(np.isfinite(v) for v in vals.values())):
                    p = vals
            except Exception:
                pass
        _PARAMS_CACHE[key] = p
    return _PARAMS_CACHE[key]


def climb_cost(g_up_pct, params=None):
    """c(g): fractional pace cost per ft/mi of hill climb at grade g (%)."""
    p = params or engine_params()
    g = np.clip(np.asarray(g_up_pct, float), 0.0, GRADE_DOMAIN_PCT)
    return p['c0'] + p['c1'] * g


def descent_benefit(g_dn_pct, params=None):
    """b(g): fractional pace return per ft/mi of gross descent at hill grade
    g (%). Declines with steepness toward the braking regime (zero near the
    domain edge on the gross basis)."""
    p = params or engine_params()
    g = np.clip(np.asarray(g_dn_pct, float), 0.0, GRADE_DOMAIN_PCT)
    return p['b0'] + p['b1'] * g


def hill_cost_frac(gain_pm, g_up_pct, loss_pm, g_dn_pct, params=None):
    """Grade cost as a FRACTION of flat pace. ``gain_pm``/``loss_pm`` are
    GROSS fused vertical per mile (ft/mi — all of it, not just detected
    hills); ``g_*_pct`` the vertical-weighted mean segment grades (0 when the
    run has no hills — the base rates price pure undulation). Vectorized."""
    gain_pm = np.asarray(gain_pm, float)
    loss_pm = np.asarray(loss_pm, float)
    return (climb_cost(g_up_pct, params) * gain_pm
            - descent_benefit(g_dn_pct, params) * loss_pm)


def hill_cost(gain_pm, g_up_pct, loss_pm, g_dn_pct, pace_s_per_mi, params=None):
    """Hill pace cost (s/mi). ``pace_s_per_mi`` is the run's MEASURED pace;
    the fractional cost applies to the flat-equivalent pace, whose closed form
    is ``cost = frac·pace/(1+frac)`` (from cost = frac·(pace − cost)) — so a
    run slowed by its own hills doesn't inflate its own correction. Vectorized
    over array inputs."""
    frac = hill_cost_frac(gain_pm, g_up_pct, loss_pm, g_dn_pct, params)
    pace = np.asarray(pace_s_per_mi, float)
    return frac * pace / (1.0 + frac)
