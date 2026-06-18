"""Recovery-pace factor model: OLS on era-detrended residual vs CS.

Lifted out of ``src/plots/make_recovery_plots.py`` so both the Recovery plot
and the Long Runs plot can import the same fit. ``run_plots.sh`` builds Long
Runs *before* Recovery, so a file artifact written by the recovery plot would
be stale or missing — an import is build-order-independent. Same precedent as
``long_run_model.py`` (lifted out of ``plot_training_quality.py``).

Model (fit on the era-detrended residual = pace − CS − era_trend):

  residual_detrended ~ β_temp · temp_heat_hinge
                     + β_marathon · fatigue_marathon
                     + β_race    · fatigue_race_short
                     + β_tod     · tod_is_pm
                     + (pinned) is_offroad·β + alt_kft·β + grade-aware elev_cost
                     + (pinned) wind_mph · β_wind

The temperature term (``temp_centered_feature``, key still ``temp_centered``)
is a one-sided heat hinge ``max(0, air_temp − TEMP_HEAT_ONSET_C)``: cold
contributes zero, only heat above ~6°C slows pace. It replaced a symmetric
apparent-temp-centered-at-12 term (June 2026) whose sub-12°C arm credited a
phantom cold speedup mirroring the heat penalty. Long runs reuse the same SHAPE
with a free (steeper) slope. Humidity was tested as a separate regressor and
dropped — weak, and the heat index (its physical encoding) never beat plain air
temp, so this is air temperature, not feels-like. Wind enters as a pooled,
pinned per-mph cost (``wind_beta``) applied where a watch wind reading exists —
both estimated on the watch-enriched subset but applied as fixed offsets so the
main fit keeps the full corpus (June 2026).

Three classes of rows are pruned from BOTH the era-trend window contents AND
the OLS fit (flags are independent — a row can be in multiple classes):
bad conditions (snow/icy or "snow" in the workout string), partner runs
(any partners entry outside ADMITTED_PARTNERS — solo and varsity are
admitted), and LOO outliers (|residual from leave-one-out 28-day local
mean| > 45 sec/mi against the clean neighbor pool). See the recovery plot docstring for the
full rationale and ``docs/recovery-runs-reference.md`` for the analysis
history (sleep, distance, shoes, non-race quality efforts tested and
excluded as non-factors).

The fit runs on the watch/rule-corrected pace where one exists
(``add_watch_corrections`` below — calibrated watch distance + moving time
on paved days, route-era deflation rules elsewhere, pace-only on
trailing-strides days). The logged columns are never rewritten.

``TRANSFERABLE_FEATURES`` names the per-day factors that are meaningful for
runs other than recovery runs (temperature, recent-race fatigue, time of
day). Route and era are recovery-specific: long-run route betas are fit
inside the TQ model instead (altitude/era confounding — see
``docs/route-normalization-reference.md``), and era trend is a smoothed
track of recovery residuals themselves.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE
from src.shared.elevation_cost import elevation_cost, paved_refund, REFUND_RECOVERY

# ---------- conditions / pruning excluded from fit ----------
# Time-of-day binary indicator. Recovery pace runs ~4.5 sec/mi slower in
# the early/morning window than the afternoon/late window (p<1e-13 in
# fully-normalized residuals), driven by overnight fueling state and
# joint stiffness. Encoded as 1 = PM (afternoon|late), 0 = AM (early|
# morning). `early` (n≈30) and `late` (n≈90) are too sparse to keep as
# their own levels; their means align with morning and afternoon
# respectively to within their CIs.
TOD_PM_VALUES = {'afternoon', 'late'}

EXCLUDED_CONDITIONS = {'snow', 'icy'}

# Workout-string snow detection. Catches "[2\" snow]" and similar annotations
# in workout_raw that the conditions field missed.
SNOW_IN_WORKOUT_RE = re.compile(r'\bsnow\b', re.IGNORECASE)

# Partner-run detection: any partners entry outside ADMITTED_PARTNERS counts
# as a partner run. Partner runs are a different population (the occasional
# Maddy run, one-off training partners) — their pace targets and route
# choices differ from solo recovery, and including them inflates
# within-period variance. They're pruned from the fit alongside
# snow/ice/inside. 'varsity' is ADMITTED (June 2026, Max): in the 2016-17
# era the varsity group's recovery pace WAS Max's pace strategy — those runs
# are his own effort policy, not someone else's — so they belong in the fit
# pool. Individual named partners stay excluded.
ADMITTED_PARTNERS = {'', 'nan', 'solo', 'none', 'varsity'}

# Outlier detection: residual from leave-one-out 28-day local mean exceeds
# OUTLIER_THRESHOLD_SEC. Catches "clearly something happened" days
# (travel/jet-lag, illness, extreme post-marathon fatigue) that no feature
# in the model is set up to capture. ±45 prunes ~16 of ~2,200 eligible
# rows (Apr 2026); about 3σ on the LOO residual distribution.
OUTLIER_THRESHOLD_SEC      = 45.0
OUTLIER_WINDOW_HALF_DAYS   = 14
OUTLIER_MIN_NEIGHBORS      = 5


# ---------- model parameters ----------
TEMP_REFERENCE_C       = 12.0
# Heat-onset threshold for the one-sided temperature hinge (June 2026). Below
# this, temperature contributes zero; above it, a heat penalty accrues. Pinned
# from 2356 well-normalized recovery runs (flat cold plateau, monotonic rise
# from ~6C). Recovery and long runs share this SHAPE; only the slope differs.
TEMP_HEAT_ONSET_C      = 6.0
# Wind enters as a pooled, pinned per-mph cost (wind_beta) applied symmetrically
# around the MEDIAN observed recovery wind (wind_reference_mph): unlogged days
# are filled with that median (a neutral, zero-offset assumption — not dead
# calm), below-median wind is a small credit and above-median a small cost.
# Recent-race fatigue uses exponential decay: contribution = exp(−t/τ) at
# day t post-race, fitted via OLS for amplitude. Per-category τ in days,
# from empirical curve fits on days-since-last-race vs marathon-revealed
# residual (see April 2026 analysis).
FATIGUE_TAU_DAYS = {'marathon': 6.0, 'race_short': 5.0}
ERA_WINDOW_HALF_DAYS   = 182
ERA_WINDOW_MIN_POINTS  = 30
MIN_ROUTE_N            = 13
MARATHON_DISTANCE_M    = 42000

QUALITY_CATS = ['marathon', 'race_short']

# Per-day factors whose betas are meaningful beyond recovery runs (applied
# by the Long Runs plot's normalization toggle and the TQ long-run model
# experiments). Route and era are intentionally absent — see module docstring.
TRANSFERABLE_FEATURES = ['temp_centered', 'fat_marathon', 'fat_race_short',
                         'tod_is_pm']


def apparent_temp_c(temp_c, humidity_pct):
    """Apparent ("feels-like") temperature in Celsius — the NWS heat index.

    Clamped: equals the air temperature below 80°F (humidity is physiologically
    inert when cool — and the heat-index regression isn't defined there) and is
    never below air temp. Where humidity is missing the air temp is returned
    unchanged, so pre-watch rows (no humidity) and CI without a watch cache both
    fall back gracefully to plain temperature.

    Accepts scalars / arrays / Series; returns a float ndarray. NaN air temp
    propagates to NaN so callers' existing NaN handling is preserved.
    """
    t = np.asarray(temp_c, dtype=float)
    rh = (np.asarray(humidity_pct, dtype=float) if humidity_pct is not None
          else np.full(t.shape, np.nan))
    tf = t * 9.0 / 5.0 + 32.0
    full = (-42.379 + 2.04901523 * tf + 10.14333127 * rh
            - 0.22475541 * tf * rh - 0.00683783 * tf**2 - 0.05481717 * rh**2
            + 0.00122874 * tf**2 * rh + 0.00085282 * tf * rh**2
            - 0.00000199 * tf**2 * rh**2)
    adj_low = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(tf - 95.0)) / 17,
                                                0, None))
    full = np.where((rh < 13) & (tf > 80) & (tf < 112), full - adj_low, full)
    adj_high = ((rh - 85) / 10) * ((87 - tf) / 5)
    full = np.where((rh > 85) & (tf > 80) & (tf < 87), full + adj_high, full)
    # Use the heat index only where it's defined (≥80°F) AND humidity is known;
    # elsewhere the air temperature stands, never reduced below it.
    hi_f = np.where((tf >= 80) & np.isfinite(rh), np.maximum(full, tf), tf)
    return (hi_f - 32.0) * 5.0 / 9.0


def temp_centered_feature(df):
    """One-sided heat hinge: ``max(0, air_temp_C - TEMP_HEAT_ONSET_C)``.

    The single place temperature is encoded for the model — every consumer
    (recovery fit, transferable contributions, the physical-betas pool, the
    long-run model) sees the same one-sided heat term, so recovery and long
    runs share the SHAPE and differ only in fitted slope.

    Replaced the old symmetric apparent-temp-centered-at-12 form (June 2026).
    That form was bidirectional: every sub-12C day was credited with a speedup
    mirroring the heat penalty — a phantom cold benefit (it predicted ~-8 s/mi
    at -19C where the data is a flat ~-2). The true recovery shape is a flat
    cold plateau and a monotonic heat rise from ~6C, pinned from 2356
    well-normalized runs. Humidity was tested as a separate term and dropped
    (weak, undefendable; the heat index — its physical encoding — never beat
    plain air temp), so this is air temperature, not feels-like.

    Preserves NaN where ``temp_c`` is NaN; callers ``.fillna(0)`` for a zero
    (sub-onset) contribution. The column/key name ``temp_centered`` is retained
    to limit churn — it is now a hinge, not a centered value.
    """
    t = pd.to_numeric(df['temp_c'], errors='coerce').to_numpy(dtype=float)
    return pd.Series(np.maximum(0.0, t - TEMP_HEAT_ONSET_C), index=df.index)


# ---------- watch / route-rule distance corrections ----------
# Recovery runs get the same log-vs-watch honesty treatment as long runs
# (June 2026): calibrated watch distance + watch moving time replace the
# logged figures in the FIT (the logged columns are never rewritten), gated
# to paved terrain — the calibration curve is fit on paved-outdoor days and
# trail GPS corner-cutting is a route property it can't speak for.
#
# The measurement artifact (recovery_measured.csv) is written by
# src/coros/long_runs.py alongside the long-run one; the calibration curve
# (long_run_calibration.csv) is shared — it was already fit on the pooled
# paved recovery + long corpus.
REC_MEASURED_PATH = DATA_DIR / 'recovery_measured.csv'
REC_CAL_PATH      = DATA_DIR / 'long_run_calibration.csv'
ELEV_MEASURED_PATH = DATA_DIR / 'elevation_measured.csv'
ALT_DAILY_PATH = DATA_DIR / 'altitude_daily.csv'

# Elevation-enrichment constants (June 2026, watch enrichment —
# docs/route-normalization-reference.md, the elevation engine). The grade-cost model itself
# lives in shared.elevation_cost; here we keep only the per-run failure guard.
ELEV_GUARD_FT_PER_MI = 100.0   # extreme watch-failure floor (see per_run_elevation)

# Backfit iterations: the era smoother and the parametric factors are
# estimated jointly (so era-correlated factors — footing, altitude — aren't
# absorbed by the era trend). Converges in a few passes.
BACKFIT_ITERS = 8

# Per-day watch-failure guard. The time-completeness gate can't catch a GPS
# failure that loses DISTANCE while keeping time intact (downtown
# multipath: sporadic lakefront days read 20-30% short against a route
# median of 1.008, implying +50-550 s/mi "corrections"). The
# disambiguation from genuine route mislogging — the thing the corrections
# exist to fix — is route structure: real mislogs are systematic (every
# pre-2022 belle meade day deviates together, moving the route median),
# GPS failures are sporadic one-day spikes against an honest route median.
# So a day is only corrected when its implied inflation (logged miles over
# calibrated watch miles) sits within WATCH_FAIL_DEV of its route's median
# inflation (routes with < WATCH_FAIL_MIN_ROUTE_N clean watch days compare
# against the profile-global median instead). 0.06 is the empirical break:
# the deviation distribution's 95th percentile is 0.0585, with the failure
# tail starting just above (57 of ~1245 days skipped, June 2026); the
# fit's ±45 s/mi LOO prune backstops anything that slips through.
WATCH_FAIL_DEV         = 0.06
WATCH_FAIL_MIN_ROUTE_N = 5

# Trailing strides/sprints in a recovery string: 'rec@7:33/400st',
# 'rec@7:25/6x100st', 'rec@5:50/6sp', 'rec@6:04/2st'. Max PAUSES the watch
# for the strides (verified June 2026 against the 22 watch-covered paved
# stride days vs 1,227 normal: watch moving time runs −1.3% of logged vs
# +1.0% normal; recorded pause time is 4.5 min vs 1.75; and logged miles
# sit +0.32 mi ABOVE the calibrated watch distance vs −0.01 normal — that
# 0.32 mi is the stride distance the paused watch never recorded). So on a
# stride day the watch measures the RECOVERY PORTION ONLY. Two consequences:
#  (1) excluded from the calibration fit corpus (long_runs.fit_calibration)
#      — the +0.32 mi stride excess would bias the slope/intercept up and
#      over-correct every day;
#  (2) no watch correction applied (the ``ok`` mask below folds in ~strid):
#      ``recovery_pace_sec_per_mi`` is the explicit logged ``rec@M:SS``
#      (extract_recovery_pace), already the clean recovery-segment pace, and
#      a recovery-only watch measurement run through a whole-run-fit
#      calibration would at best just reproduce it. Keep the logged pace.
# (A stride day on a mislogged ROUTE still takes the route-rule pace
# ×factor below — a distance-estimate error biases the logged @pace itself.)
STRIDE_SUFFIX_RX = re.compile(r'/\s*(?:\d+\s*x\s*)?\d+\s*(?:st|sp)\b',
                              re.IGNORECASE)

# Route-era mislogged-distance rules (Max-specific carve-out). The two Nashville
# staples were over-logged pre-2022 until Max re-measured them in April 2022.
# Factor PINNED at 1.05 flat (2026-06-17). The real over-logging is watch-
# confirmed at ~5% for BOTH routes (pre-2022 logged/watch median ÷ the post-2022
# accurate-logging baseline: belle meade 1.069/1.011 ≈ 1.057, greenway
# 1.057/1.001 ≈ 1.055), essentially flat across intensity — the steeper raw
# trend was a watch GPS-undercount artifact that doesn't apply to pre-watch
# runs. Per-route (not flat across ALL routes): Lake Samm's watch ratio is
# 1.007 (accurately logged), so it gets NO correction — a blanket pre-watch
# haircut would wrongly deflate it. This correction handles only DISTANCE; the
# pre-watch pause-uncertainty erosion is separate (durability.py imputes the
# global P90 watch-era stop structure onto pre-watch runs; see
# docs/long-run-pause-uncertainty-reference.md). Together they pull the pre-watch
# Nashville long runs off the demonstrated-capability frontier. Distance error is a route
# property, so the factor also applies to the routes' recovery rows; applied
# back to 2018; watch enrichment, when present, wins over the rule. Lives here
# (not workouts.py) so both the long-run projection and the recovery fit consume
# it without a circular import.
MISLOGGED_ROUTES = (
    ('belle meade', '2018-01-01', '2022-04-15', 1.05),
    ('greenway',    '2018-01-01', '2022-04-15', 1.05),
)


def load_distance_calibration(path=REC_CAL_PATH):
    """(intercept_mi, slope) of the profile's log-vs-watch distance curve,
    or None when the artifact is absent (no watch corpus). THE single loader
    for long_run_calibration.csv — shared by the recovery and long-run models
    (workouts imports this one), which both read the same pooled curve."""
    if not path.exists():
        return None
    cal = pd.read_csv(path)
    if cal.empty:
        return None
    r = cal.iloc[0]
    return float(r['intercept_mi']), float(r['slope'])


def add_watch_corrections(rec):
    """Watch/rule corrections for recovery rows, mirroring the long-run
    treatment (workouts._lr_watch_corrections). Adds, NaN/False where not
    applicable:

      rec_watch    : watch-enriched (measured, complete, calibration
                     present, paved terrain, NOT a strides day)
      rec_rule     : corrected via a route-era mislogged-distance rule
      has_strides  : trailing strides/sprints logged (STRIDE_SUFFIX_RX)
      corr_miles, corr_time_s, corr_pace_sec_per_mi : corrected values on
                     the honest-log scale (NaN where uncorrected)
      pause_s      : watch-recorded paused time (s) on watch-enriched rows
                     (NaN elsewhere) — surfaced in the plot tooltip

    Trailing-strides days get NO watch correction: Max pauses the watch for
    the strides (see STRIDE_SUFFIX_RX), so the watch measures the recovery
    portion only — it's short of the logged run, not long. The logged
    ``recovery_pace_sec_per_mi`` is the explicit ``rec@M:SS``
    (extract_recovery_pace), already the clean recovery-segment pace, so
    it's kept as-is; a recovery-only watch measurement through the
    whole-run-fit calibration would only reproduce it. The watch's role on
    these days is the calibration-corpus exclusion (in
    long_runs.fit_calibration), where the unrecorded stride distance would
    otherwise bias the curve. A strides day on a mislogged ROUTE still takes
    the route-rule pace ×factor below: that's a distance-estimate error,
    which biases the logged @pace itself regardless of strides.

    The distance bracket matches the long-run guard (workouts): true
    distance is bracketed watch_mi <= true <= hand-logged. (watch +
    calibration error term) is the estimate; the hand log is a hard ceiling
    Max never under-estimates, so corr_mi = min(estimate, logged). Time is
    the anchor (it IS the watch time, the log just rounds it to the minute);
    corr_pace is the pure derivative corr_time/corr_mi. On routes Max
    happens to log tightly (centennial, mccabe, boulder creek, hopewell
    junction all sit at 0.97-0.99 logged/honest) the estimate overshoots the
    log, so they clamp to the logged distance — effectively uncorrected,
    never longer than logged.

    The logged columns are never touched — plots show what Max logged."""
    rec = rec.copy()
    rec['has_strides'] = (rec.get('workout_raw',
                                  pd.Series(np.nan, index=rec.index))
                          .fillna('').astype(str)
                          .apply(lambda s: bool(STRIDE_SUFFIX_RX.search(s))))
    rec['rec_watch'] = False
    rec['rec_rule'] = False
    for col in ('corr_miles', 'corr_time_s', 'corr_pace_sec_per_mi', 'pause_s'):
        rec[col] = np.nan

    cal = load_distance_calibration()
    if cal is not None and REC_MEASURED_PATH.exists():
        c, m = cal
        try:
            meas = pd.read_csv(REC_MEASURED_PATH, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            # A profile with watch details but no recovery days writes a
            # column-less artifact.
            meas = pd.DataFrame(columns=['date', 'complete'])
        meas = meas[meas['complete'].astype(bool)] if len(meas) else meas
        meas = meas.set_index('date')
        paved = (rec.get('terrain_type', pd.Series(np.nan, index=rec.index))
                 .astype(str).str.strip().str.lower() == 'paved')
        hit = rec['date'].isin(meas.index) & paved
        if hit.any():
            sub = meas.loc[rec.loc[hit, 'date']]
            cal_mi = (sub['watch_miles'] * (1 + m) + c).to_numpy()
            mov_s = sub['watch_moving_s'].to_numpy()
            strid = rec.loc[hit, 'has_strides'].to_numpy(bool)
            # Watch-failure guard (WATCH_FAIL_DEV above): day inflation vs
            # route-median inflation, medians computed on stride-free hit
            # days only (stride miles are nominally padded). Stride days are
            # excluded from correction entirely (see docstring), so the
            # guard's ``ok`` mask folds in ~strid.
            logged_mi = rec.loc[hit, 'miles'].to_numpy(float)
            infl = logged_mi / cal_mi
            locs = (rec.loc[hit, 'location'].astype(str)
                    .str.strip().str.lower().to_numpy())
            med_df = pd.DataFrame({'loc': locs[~strid],
                                   'infl': infl[~strid]})
            med = med_df.groupby('loc')['infl'].agg(['median', 'size'])
            glob_med = float(np.median(infl[~strid])) if (~strid).any() else 1.0
            route_med = np.array([
                float(med.loc[l, 'median'])
                if l in med.index and med.loc[l, 'size'] >= WATCH_FAIL_MIN_ROUTE_N
                else glob_med
                for l in locs])
            ok = (np.abs(infl - route_med) <= WATCH_FAIL_DEV) & ~strid
            # Distance bracket — see docstring (corr_mi never exceeds logged).
            corr_mi = np.minimum(cal_mi, logged_mi)
            idx = rec.index[hit]
            pause_s = sub['pause_s'].to_numpy()
            rec.loc[idx[ok], 'rec_watch'] = True
            rec.loc[idx[ok], 'corr_time_s'] = mov_s[ok]
            rec.loc[idx[ok], 'corr_pace_sec_per_mi'] = (mov_s / corr_mi)[ok]
            rec.loc[idx[ok], 'corr_miles'] = corr_mi[ok]
            rec.loc[idx[ok], 'pause_s'] = pause_s[ok]

    loc = (rec.get('location', pd.Series(np.nan, index=rec.index))
           .astype(str).str.strip().str.lower())
    for route, start, end, factor in MISLOGGED_ROUTES:
        rule = (~rec['rec_watch'] & (loc == route)
                & (rec['date'] >= pd.Timestamp(start))
                & (rec['date'] < pd.Timestamp(end)))
        if not rule.any():
            continue
        # corr_pace reduces to logged_pace x factor (the polluted logged
        # miles cancel), so it's valid on stride days too; the distance/
        # time figures are only claimed for stride-free days.
        rec.loc[rule, 'rec_rule'] = True
        rec.loc[rule, 'corr_pace_sec_per_mi'] = (
            rec.loc[rule, 'recovery_pace_sec_per_mi'] * factor)
        clean = rule & ~rec['has_strides']
        rec.loc[clean, 'corr_miles'] = rec.loc[clean, 'miles'] / factor
        rec.loc[clean, 'corr_time_s'] = (
            rec.loc[clean, 'recovery_pace_sec_per_mi']
            * rec.loc[clean, 'miles'])
    return rec


# ---------- helpers ----------

def per_run_elevation(rec):
    """Per-run gain/loss (ft per mile) for the physical route model, as fitted
    features (NOT a pre-combined penalty — the recovery fit estimates the
    slopes itself, era-free, now that the per-route dummies are gone).

    Returns a DataFrame [elev_gain_pm, elev_loss_pm] aligned to ``rec.index``,
    populated for every row via the fallback chain:
      per-run MEASURED (elevation_measured.csv, watch enrichment), then
      route-median measured (pre-watch runs on a watch-covered route), then
      the route's ``elev_per_mile`` constant (balanced gain≈loss — covers
      pre-watch-only routes like education hill and CI builds with no
      elevation artifact), then 0.
    Extreme watch failures (``gain/mi > max(ELEV_GUARD_FT_PER_MI,
    3x route median)`` — suburbia 2023-07-27, east boulder 2024-04-24) revert
    to the route median."""
    gain = pd.Series(np.nan, index=rec.index, dtype=float)
    loss = pd.Series(np.nan, index=rec.index, dtype=float)
    if ELEV_MEASURED_PATH.exists():
        try:
            em = pd.read_csv(ELEV_MEASURED_PATH, parse_dates=['date'])
        except (pd.errors.EmptyDataError, FileNotFoundError):
            em = None
        if em is not None and not em.empty and 'elev_gain_ft' in em.columns:
            em = em.merge(rec[['date', 'location']], on='date', how='left')
            # Prefer DEM-along-GPS gain/loss where present (long-run + race rows;
            # see dem_elevation.py). Long runs are loops, so the barometric net
            # is a phantom morning-drift descent — DEM removes it. Recovery rows
            # carry no dem_* and fall through to barometric, so this is scoped to
            # whatever the backfill DEM-filled without a run_type branch here.
            gain_src = (em['dem_gain_ft'].where(em['dem_gain_ft'].notna(),
                                                em['elev_gain_ft'])
                        if 'dem_gain_ft' in em.columns else em['elev_gain_ft'])
            loss_src = (em['dem_loss_ft'].where(em['dem_loss_ft'].notna(),
                                                em['elev_loss_ft'])
                        if 'dem_loss_ft' in em.columns else em['elev_loss_ft'])
            em['gpm'] = gain_src / em['corr_miles']
            em['lpm'] = loss_src / em['corr_miles']
            rmed = em.groupby('location')['gpm'].median()
            lmed = em.groupby('location')['lpm'].median()
            bad = (em['gpm'] > ELEV_GUARD_FT_PER_MI) & (
                em['gpm'] > 3 * em['location'].map(rmed))
            em.loc[bad, 'gpm'] = em.loc[bad, 'location'].map(rmed)
            em.loc[bad, 'lpm'] = em.loc[bad, 'location'].map(lmed)
            per_run = (em.dropna(subset=['date'])
                       .drop_duplicates('date').set_index('date'))
            gain = rec['date'].map(per_run['gpm'])
            loss = rec['date'].map(per_run['lpm'])
            gain = gain.fillna(rec['location'].map(rmed))
            loss = loss.fillna(rec['location'].map(lmed))
    epm = (rec.get('elev_per_mile', pd.Series(np.nan, index=rec.index))
           .astype(float))
    gain = gain.fillna(epm)
    loss = loss.fillna(epm)
    return pd.DataFrame({'elev_gain_pm': gain.fillna(0.0),
                         'elev_loss_pm': loss.fillna(0.0)})


def per_run_altitude(df):
    """Per-run altitude (thousands of feet above sea level) for the hypoxia
    term: the run's MEASURED mean elevation — midpoint of the watch's smoothed
    daily min/max (``altitude_daily.csv``, the same per-day layer that feeds
    the Altitude trend) — where available, falling back to the location's
    base-elevation constant (the daily ``altitude`` join column) on pre-watch /
    unmeasured days, then 0 (sea level). Measured per-run beats the
    per-location constant: it adds within-location resolution and fixes
    hand-set constant errors (e.g. watershed's 580 ft constant is ~255 ft
    measured). Aligned to ``df.index``; both sources are in feet."""
    const = pd.to_numeric(df.get('altitude', pd.Series(np.nan, index=df.index)),
                          errors='coerce')
    mid = pd.Series(np.nan, index=df.index, dtype=float)
    if ALT_DAILY_PATH.exists():
        try:
            a = pd.read_csv(ALT_DAILY_PATH, parse_dates=['date'])
        except (pd.errors.EmptyDataError, FileNotFoundError):
            a = None
        if a is not None and not a.empty and 'min_elev_ft' in a.columns:
            a_mid = ((a['min_elev_ft'] + a['max_elev_ft']) / 2.0)
            a_mid.index = a['date']
            a_mid = a_mid[~a_mid.index.duplicated()]
            mid = df['date'].map(a_mid)
    return (mid.fillna(const).fillna(0.0) / 1000.0)


# Altitude (hypoxia) is ~zero below ~914 m (3000 ft), then declines roughly
# linearly with altitude in endurance athletes (Wehrlin & Hallén 2006; the
# clinical heuristic is ~8-11% VO2max per 1000 m *above* ~3000 ft). Max's data
# is bimodal (sea level + Boulder ~5400 ft + 3 Magnolia runs ~8400 ft) so it
# can't identify a SHAPE — we pin the shape (threshold + linear) from the
# literature and fit only the SLOPE (physical_route_betas). A line through the
# origin both invents a phantom effect at low altitude and under-slopes the
# high end; the threshold fixes both (Boulder, the bulk, barely moves; Magnolia
# steepens ~+37%; everything < 3000 ft → exactly 0).
ALTITUDE_THRESHOLD_KFT = 3.0


def altitude_regressor(alt_kft):
    """Science-pinned hypoxia regressor: ``max(0, alt_kft − 3.0)``. The fit
    estimates the pace slope ON this (so the Boulder data anchors the scale),
    and every consumer (recovery, long-run, race) must build the altitude term
    through this same transform so fit and application stay consistent."""
    return np.maximum(0.0, np.asarray(alt_kft, dtype=float) - ALTITUDE_THRESHOLD_KFT)


def _race_dem_elevation(races):
    """Per-race gain/loss (ft/mi) and mean elevation (kft) from the DEM-along-
    GPS layer (``elevation_measured.csv`` race rows, columns ``dem_*``; see
    src/coros/dem_elevation.py). The watch's barometric net is per-race noise;
    DEM resampled along the reliable GPS track is the race source of truth.
    Returns (gain_pm, loss_pm, mean_kft, grade_available) Series aligned to
    ``races.index`` — NaN / False where no DEM row exists (track races, which
    the elevation backfill skips, and pre-watch races)."""
    n = races.index
    gain = pd.Series(np.nan, index=n, dtype=float)
    loss = pd.Series(np.nan, index=n, dtype=float)
    mean_kft = pd.Series(np.nan, index=n, dtype=float)
    if not ELEV_MEASURED_PATH.exists():
        return gain, loss, mean_kft, pd.Series(False, index=n)
    try:
        em = pd.read_csv(ELEV_MEASURED_PATH, parse_dates=['date'])
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return gain, loss, mean_kft, pd.Series(False, index=n)
    if 'dem_gain_ft' not in em.columns:
        return gain, loss, mean_kft, pd.Series(False, index=n)
    em = em[(em['run_type'] == 'race') & em['dem_gain_ft'].notna()].copy()
    em = em.drop_duplicates('date').set_index(em['date'].dt.date)
    dist_mi = races['distance_m'].astype(float) / METERS_PER_MILE
    key = pd.to_datetime(races['date']).dt.date
    gain = key.map(em['dem_gain_ft']) / dist_mi
    loss = key.map(em['dem_loss_ft']) / dist_mi
    mean_kft = key.map(em['dem_mean_elev_ft']) / 1000.0
    return gain, loss, mean_kft, gain.notna()


def race_physical_correction(races, daily=None):
    """Per-race physical TIME correction — grade + off-road footing + altitude —
    converting each watch-covered race to its flat / sea-level / smooth-
    equivalent time BEFORE it informs CS, so the demonstrated-capability
    frontier measures fitness, not the course (docs/watch-stream-enrichment-
    plan.md §B). Reuses §A's machinery: the shared ``elevation_cost`` engine,
    footing/altitude pinned from ``physical_route_betas`` (the SAME constants
    recovery and long runs use), per-race grade/altitude from the DEM-along-GPS
    layer (``elevation_measured.csv`` ``dem_*`` — barometric net is per-race
    noise, so races use DEM, unlike recovery/long which average out on baro).

    Conventions (identical to §A): subtract the cost from the race time. A
    net-DOWNHILL race (cost<0) gets time ADDED → projects SLOWER → correctly
    discounts the assisted time (Boston). Net-uphill / altitude races are
    credited faster. Race effort ≈ 1.0 by construction (a race is run at its
    own-distance ceiling), so paved descents refund at the race edge
    (``paved_refund(1.0)`` ≈ 0.85) — grade bites hardest here.

    Gating (the categorical XC ×1.08 / Downhill-exclusion stay as the pre-watch
    fallback — replaced by the measured correction only WHERE WATCH DATA
    EXISTS):
      * ``has_measured`` (full grade+footing applies, and the caller turns OFF
        the categorical for that race): a watch-covered, non-track race with a
        DEM row.
      * Track races: grade gated OFF (flat; the backfill skips them anyway), but
        altitude hypoxia still applies via the per-run altitude chain.
      * Altitude applies wherever a per-run altitude is available (DEM mean for
        DEM races; the watch/location chain for track-at-altitude).

    Returns a DataFrame aligned to ``races.index``:
        grade_s_per_mi, footing_s_per_mi, alt_s_per_mi, total_s_per_mi
        dt_sec       : total_s_per_mi × distance_mi (subtract from time_sec)
        has_measured : bool (see gating above)
    """
    df = races.copy()
    if 'surface' not in df.columns:
        df['surface'] = ''
    if daily is None:
        daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    # Join location metadata (terrain/altitude) from the daily race rows —
    # races.csv carries none. One daily row per race day (race_seq=1 back-prop).
    # A watch-only profile (Coros) has no location-metadata join, so daily
    # carries none of these columns; select only what's present and let the
    # downstream defaults (terrain -> 'paved', altitude -> per-run/0) apply.
    meta_cols = [c for c in ('terrain_type', 'altitude', 'elev_per_mile', 'location')
                 if c in daily.columns]
    meta = (daily[daily['run_type'] == 'race']
            [['date'] + meta_cols]
            .drop_duplicates('date')).copy()
    for c in meta_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    # ``date`` may be datetime64 (races.csv load) or python date (build_eligible
    # normalizes it); merge on a common date key so either dtype works.
    df['_dk'] = pd.to_datetime(df['date']).dt.date
    meta['_dk'] = pd.to_datetime(meta['date']).dt.date
    df = df.merge(meta.drop(columns=['date']), on='_dk', how='left').drop(columns=['_dk'])
    # terrain_type is accessed directly below; materialize it as blank when the
    # profile has no terrain metadata so it maps to the 'paved' (flat) default.
    if 'terrain_type' not in df.columns:
        df['terrain_type'] = np.nan

    gain_pm, loss_pm, dem_mean_kft, grade_avail = _race_dem_elevation(df)
    is_track = df['surface'].fillna('').astype(str).str.lower() == 'track'
    grade_avail = grade_avail & ~is_track

    terr = (df['terrain_type'].astype(str).str.strip().str.lower())
    terr = terr.where(terr.isin(('paved', 'mixed', 'trail')), 'paved')
    # Race effort ≈ 1.0: paved descents refund at the race edge; mixed/trail
    # keep their (effort-flat) rough-footing refund.
    refund = terr.map(REFUND_RECOVERY).astype(float).to_numpy(copy=True)
    refund[(terr == 'paved').to_numpy()] = paved_refund(1.0)
    grade = elevation_cost(gain_pm.fillna(0.0).to_numpy(),
                           loss_pm.fillna(0.0).to_numpy(),
                           terr.to_numpy(), refund=refund)
    grade = np.where(grade_avail.to_numpy(), grade, 0.0)

    pb = physical_route_betas()
    is_offroad = terr.isin(('mixed', 'trail')).astype(float).to_numpy()
    footing = np.where(grade_avail.to_numpy(), pb['is_offroad'] * is_offroad, 0.0)
    # Altitude: DEM mean where present, else the per-run altitude chain (covers
    # track-at-altitude, whose grade is off but hypoxia is real — Boulder track).
    # Through the science-pinned threshold regressor: a sub-3000 ft race (sea
    # level, Nashville's 400 ft constant) gets no hypoxia term, so it neither
    # fabricates a phantom correction nor flips has_measured (which had wrongly
    # admitted the pre-watch Downhill TT). Boulder/Magnolia get the real effect.
    alt_eff = altitude_regressor(dem_mean_kft.fillna(per_run_altitude(df)))
    alt_cost = pb['alt_kft'] * alt_eff

    total = grade + footing + alt_cost
    dist_mi = df['distance_m'].astype(float).to_numpy() / METERS_PER_MILE
    out = pd.DataFrame({
        'grade_s_per_mi': grade,
        'footing_s_per_mi': footing,
        'alt_s_per_mi': alt_cost,
        'total_s_per_mi': total,
        'dt_sec': total * dist_mi,
        'has_measured': grade_avail.to_numpy() | (alt_eff > 0),
    }, index=races.index)
    return out


def days_since(daily, source_dates_sorted):
    if not source_dates_sorted:
        return np.full(len(daily), np.nan)
    sd = np.array([d.value for d in source_dates_sorted])
    out = []
    for d in daily['date']:
        idx = np.searchsorted(sd, d.value, side='left')
        if idx == 0:
            out.append(np.nan)
        else:
            prev = pd.Timestamp(int(sd[idx - 1]))
            out.append((d - prev).days)
    return np.array(out, dtype=float)


def centered_rolling_mean(target_dates, source_dates, source_values,
                           half_days, min_points):
    target_ms = np.array([d.value // 10**6 for d in target_dates])
    source_ms = np.array([d.value // 10**6 for d in source_dates])
    source_v = np.asarray(source_values, dtype=float)
    half_ms = half_days * 86_400_000

    out = np.full(len(target_ms), np.nan)
    lo = hi = 0
    n_src = len(source_ms)
    for i, t in enumerate(target_ms):
        while lo < n_src and source_ms[lo] < t - half_ms:
            lo += 1
        while hi < n_src and source_ms[hi] <= t + half_ms:
            hi += 1
        if hi - lo >= min_points:
            out[i] = source_v[lo:hi].mean()
    return out


def sanitize_route(name):
    s = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
    return f'rt_{s}'


def quality_category_dates(daily, races):
    """Sorted race-day dates per fatigue category, derived the same way the
    recovery fit derives them: daily race rows, categorized as 'marathon'
    when the date appears in races.csv at marathon distance or beyond."""
    marathon_dates = set(races.loc[races['distance_m'] >= MARATHON_DISTANCE_M,
                                    'date'].tolist())
    is_race = daily['run_type'] == 'race'
    out = {}
    out['marathon'] = sorted(
        daily.loc[is_race & daily['date'].isin(marathon_dates), 'date'].tolist())
    out['race_short'] = sorted(
        daily.loc[is_race & ~daily['date'].isin(marathon_dates), 'date'].tolist())
    return out


def add_quality_features(df, quality_dates):
    """Add ``dsq_<cat>`` (days since most recent <cat> race) and ``fat_<cat>``
    (exponential fatigue decay, exp(−days/τ)) columns. Returns a copy."""
    df = df.copy()
    for cat in QUALITY_CATS:
        df[f'dsq_{cat}'] = days_since(df, quality_dates.get(cat, []))
        df[f'fat_{cat}'] = np.exp(-df[f'dsq_{cat}'].fillna(np.inf)
                                  / FATIGUE_TAU_DAYS[cat])
    return df


def tod_is_pm(df):
    """1.0 for afternoon/late, 0.0 for early/morning or missing (AM baseline)."""
    tod_clean = df['time_of_day'].astype(str).str.strip().str.lower()
    return tod_clean.isin(TOD_PM_VALUES).astype(float)


def transferable_contributions(df, betas, quality_dates, temp_ref=0.0):
    """Per-row modeled pace contribution (sec/mi) of the transferable
    factors — temperature, recent-race fatigue, time of day — for any
    daily-frame subset (e.g. long runs). Missing temp/TOD contribute 0,
    as does any factor whose beta key is absent from ``betas`` (the TQ
    long-run model passes a dict without ``tod_is_pm``, so TOD drops out).
    Subtracting the result from observed pace normalizes those factors out.

    ``temp_ref`` is the median hinge value the temperature slope was centered
    on (the fit's ``temp_ref``); subtracting it references the temperature
    adjustment to a typical day rather than the cold floor of the hinge, so
    normalizing temp doesn't move every run one way. Pass the corresponding
    fit's ``temp_ref`` so application matches how the slope was centered.
    """
    df = add_quality_features(df, quality_dates)
    contrib = betas.get('temp_centered', 0.0) * (
        temp_centered_feature(df).fillna(0.0) - temp_ref)
    for cat in QUALITY_CATS:
        contrib = contrib + betas.get(f'fat_{cat}', 0.0) * df[f'fat_{cat}']
    contrib = contrib + betas.get('tod_is_pm', 0.0) * tod_is_pm(df)
    return contrib.to_numpy(dtype=float)


# ---------- fit ----------


def _degrade_warn(what, exc):
    """Surface an UNEXPECTED failure in a pinned-beta estimator that is about to
    fall back to a zero (no-op) correction. Missing input files are the expected
    sparse-profile / CI-without-details case and are handled quietly by the
    caller; everything else is printed so a silent zeroing of a physical
    correction can't mask a regression (empty/short data is already guarded
    before the math, so an exception reaching here is genuinely unexpected)."""
    print(f'  WARNING: {what} failed ({type(exc).__name__}: {exc}); '
          f'falling back to no correction')


_PHYS_BETAS_CACHE: dict = {}


def physical_route_betas():
    """Pooled flat-footing (``is_offroad``) + altitude (``alt_kft``) pace costs
    (s/mi), fit jointly on recovery + in-slice long runs on a shared run-pace
    residual scale (``pace − cs_pace``) with a corpus level dummy (``is_long``)
    and the recovery era-backfit. THE single source of truth for these two
    physical constants: the recovery model pins them (``fit_recovery_model``)
    and the long-run 5K conversion credits them
    (``workouts.project_long_runs``), so one constant per channel applies
    everywhere. Pooling lets the large recovery corpus + shared era control
    discipline the era-confounded long-run off-road rows (which alone read a
    spurious +12.7); the betas are corpus-stable (footing +4.1→+4.7, altitude
    +0.87→+0.78 when long runs join).

    Cached per data dir. Degrades gracefully: recovery-only when there are no
    long-run rows, zeros when the recovery fit is unavailable (sparse profiles,
    CI without a details cache) — a zero cost is just no correction."""
    key = str(DATA_DIR)
    if key in _PHYS_BETAS_CACHE:
        return _PHYS_BETAS_CACHE[key]
    from src.shared.workouts import long_run_fit_rows
    out = {'is_offroad': 0.0, 'alt_kft': 0.0}
    try:
        daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
        races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
        cs = pd.read_csv(DATA_DIR / 'bayes_cs_summary.csv', parse_dates=['date'])
        if 'cs_pace_sec' not in cs.columns:
            cs['cs_pace_sec'] = cs['cs_pace_med'] * 60.0
        # Recovery side: self-fit (pin_physical=False) ONLY to build the frame
        # — this is the call that breaks the recursion.
        fr = fit_recovery_model(daily, races, cs, verbose=False,
                                pin_physical=False, apply_wind=False)
        if fr is None:
            _PHYS_BETAS_CACHE[key] = out
            return out
        rec = fr.rec[~fr.rec['is_pruned']].copy()
        rec['resid'] = rec['residual_raw']
        rec['is_long'] = 0.0
        rows = [rec]

        lr = long_run_fit_rows()
        if lr is not None and not lr.empty:
            lr = add_quality_features(lr, quality_category_dates(daily, races))
            lr['temp_centered'] = temp_centered_feature(lr).fillna(0.0)
            lr['tod_is_pm'] = tod_is_pm(lr)
            _epoch = pd.Timestamp('1970-01-01')
            cs_pace = np.interp((lr['date'] - _epoch).dt.days.to_numpy(),
                                (cs['date'] - _epoch).dt.days.to_numpy(),
                                cs['cs_pace_sec'].values)
            lr['resid'] = lr['pace_for_fit'] - cs_pace
            terr = (lr.get('terrain_type', pd.Series('', index=lr.index))
                    .astype(str).str.strip().str.lower())
            lr['is_offroad'] = terr.isin(('mixed', 'trail')).astype(float)
            lr['alt_kft'] = altitude_regressor(per_run_altitude(lr))
            terr_c = terr.where(terr.isin(('paved', 'mixed', 'trail')), 'paved')
            lr['elev_cost'] = elevation_cost(
                lr['elev_gain_pm'].fillna(0).to_numpy(),
                lr['elev_loss_pm'].fillna(0).to_numpy(), terr_c.to_numpy())
            lr['is_long'] = 1.0
            rows.append(lr)

        base = ['temp_centered'] + [f'fat_{c}' for c in QUALITY_CATS] + ['tod_is_pm']
        # is_long only when both corpora are present (else collinear w/ const).
        fit_cols = base + ['is_offroad', 'alt_kft'] + (
            ['is_long'] if len(rows) > 1 else [])
        rcols = ['date', 'resid', 'elev_cost', 'is_long'] + base + \
            ['is_offroad', 'alt_kft']
        pool = pd.concat([r.reindex(columns=rcols, fill_value=0.0)
                          for r in rows], ignore_index=True)
        pool = pool.dropna(subset=['resid', 'elev_cost'] + fit_cols)
        # Identifiability guard: with NO off-road terrain labels (watch-only
        # profiles have none), footing can't be separated from altitude — a
        # slow trail run has nowhere to load but altitude, skewing it wildly.
        # Drop both terrain channels; a zero physical cost is just no
        # correction. (The run/walk ceiling on long-run rows has already
        # removed the hikes that made this acute.)
        if pool['is_offroad'].abs().sum() == 0:
            fit_cols = [c for c in fit_cols if c not in ('is_offroad', 'alt_kft')]
        if len(pool) > len(fit_cols) + 1:
            X = np.hstack([np.ones((len(pool), 1)),
                           pool[fit_cols].fillna(0.0).to_numpy(float)])
            elev = pool['elev_cost'].to_numpy(float)
            raw = pool['resid'].to_numpy(float)
            dates = pool['date'].tolist()
            coef = np.zeros(X.shape[1])
            for _ in range(BACKFIT_ITERS):
                cleaned = raw - elev - (X @ coef)
                era = np.asarray(centered_rolling_mean(
                    dates, dates, cleaned, ERA_WINDOW_HALF_DAYS,
                    ERA_WINDOW_MIN_POINTS), dtype=float)
                coef, *_ = np.linalg.lstsq(X, raw - elev - era, rcond=None)
            cmap = dict(zip(['const'] + fit_cols, coef))
            out = {'is_offroad': float(cmap.get('is_offroad', 0.0)),
                   'alt_kft': float(cmap.get('alt_kft', 0.0))}
    except FileNotFoundError:
        pass
    except Exception as exc:
        _degrade_warn('physical_route_betas', exc)
    _PHYS_BETAS_CACHE[key] = out
    return out


_WIND_BETA_CACHE: dict = {}


def wind_beta():
    """Pooled wind pace cost (s/mi per mph) estimated on the watch-enriched
    recovery rows (those carrying a ``wind_mph`` reading, ~2021+).

    Wind is near-orthogonal to temp/fatigue/TOD, so — like the physical route
    betas — it's pinned and applied as an additive offset only where wind is
    known (0 elsewhere). That keeps the main fit on the full corpus instead of
    collapsing it to the ~60% of rows with a watch wind value. Estimated by
    re-fitting the base features + wind on the watch subset against the model's
    own fit target (``residual_raw − era − route``), so the slope is net of the
    other factors (matches the validated diagnostic, +0.29 s/mi per mph).

    Cached per data dir. Degrades to 0.0 (a no-op correction) when there's no
    watch wind data — e.g. CI without a details cache — so the model still fits,
    just without the wind channel."""
    key = str(DATA_DIR)
    if key in _WIND_BETA_CACHE:
        return _WIND_BETA_CACHE[key]
    out = 0.0
    try:
        daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
        races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
        cs = pd.read_csv(DATA_DIR / 'bayes_cs_summary.csv', parse_dates=['date'])
        if 'cs_pace_sec' not in cs.columns:
            cs['cs_pace_sec'] = cs['cs_pace_med'] * 60.0
        # Standard fit WITHOUT wind — breaks the wind→fit→wind recursion and
        # gives a residual that's the genuine leftover after every other factor.
        fr = fit_recovery_model(daily, races, cs, verbose=False,
                                apply_wind=False)
        if fr is None:
            _WIND_BETA_CACHE[key] = out
            return out
        rec = fr.rec.copy()
        rec['target'] = (rec['residual_raw'] - rec['era_trend']
                         - rec['contrib_route'])
        base = (['temp_centered'] + [f'fat_{c}' for c in QUALITY_CATS]
                + ['tod_is_pm'])
        m = (~rec['is_pruned']) & rec['wind_mph'].notna() & rec['target'].notna()
        for c in base:
            m = m & rec[c].notna()
        s = rec[m]
        if len(s) > len(base) + 2:
            X = np.column_stack(
                [np.ones(len(s))]
                + [s[c].to_numpy(float) for c in base]
                + [s['wind_mph'].to_numpy(float) - wind_reference_mph()])
            coef, *_ = np.linalg.lstsq(X, s['target'].to_numpy(float),
                                       rcond=None)
            out = float(coef[-1])
    except FileNotFoundError:
        pass
    except Exception as exc:
        _degrade_warn('wind_beta', exc)
    _WIND_BETA_CACHE[key] = out
    return out


_WIND_REF_CACHE: dict = {}


def wind_reference_mph():
    """Median observed recovery-run wind (mph) — the symmetric center of the
    pinned wind term. The applied offset is ``wind_b·(wind_mph − ref)`` with
    unlogged days filled with this ``ref``, so a missing reading is neutral
    (zero offset), calmer-than-typical days get a small credit, and windier
    days a small cost. Centering is reference-invariant for the fitted slope
    (a constant shift of the regressor is absorbed by the intercept); it only
    sets the fill value and the sign convention of the applied correction.

    Cached per data dir; degrades to 0.0 (the old calm baseline, a harmless
    no-op center) when there's no watch wind data — e.g. CI without a details
    cache."""
    key = str(DATA_DIR)
    if key in _WIND_REF_CACHE:
        return _WIND_REF_CACHE[key]
    out = 0.0
    try:
        daily = pd.read_csv(DATA_DIR / 'daily.csv')
        w = daily.loc[daily['run_type'] == 'recovery', 'wind_mph'].dropna()
        if len(w):
            out = float(w.median())
    except FileNotFoundError:
        pass
    except Exception as exc:
        _degrade_warn('wind_reference_mph', exc)
    _WIND_REF_CACHE[key] = out
    return out


def fit_recovery_model(daily, races, cs_summary, verbose=True,
                       pin_physical=True, apply_wind=True):
    """Fit the recovery factor model on loaded frames.

    ``daily`` is the (already era-filtered) daily log; ``races`` is
    races.csv; ``cs_summary`` needs ``date`` plus ``cs_pace_sec`` (computed
    from ``cs_pace_med`` if absent). Returns a SimpleNamespace with
    everything the recovery plot consumes — ``rec`` (recovery rows with
    feature/flag/contrib columns), ``betas``, ``intercept``, ``r2_detrended``,
    ``r2_raw``, ``n_fit``, ``qualifying_routes``, ``route_col_map``,
    ``route_counts``, ``global_mean_residual``, ``quality_dates`` — or
    ``None`` when there isn't enough data to fit (sparse profiles).
    """
    if 'cs_pace_sec' not in cs_summary.columns:
        cs_summary = cs_summary.copy()
        cs_summary['cs_pace_sec'] = cs_summary['cs_pace_med'] * 60.0

    quality_dates = quality_category_dates(daily, races)

    # Recovery subset
    rec = daily[daily['run_type'] == 'recovery'].copy()
    rec = rec.dropna(subset=['recovery_pace_sec_per_mi'])
    rec = rec.sort_values('date').reset_index(drop=True)
    if rec.empty:
        if verbose:
            print('  0 recovery days with valid pace — skipping fit')
        return None
    # CS belief at each run's date via np.interp, which HOLDS the endpoint value
    # beyond the CS summary's range — the same extrapolation add_cs() uses for
    # every other plot. A run is therefore never dropped for lack of an exact
    # CS-date match, so recovery stays current even if the Bayesian fit is
    # stale. With the fit now spanning the full run history (bayes_cs_fit grid),
    # the lookup is in-range in the common case; the hold-last is the safety net
    # that previously caused recent runs to vanish (when this was a merge+dropna).
    _epoch = pd.Timestamp('1970-01-01')
    rec['cs_pace_sec'] = np.interp(
        (rec['date'] - _epoch).dt.days.to_numpy(),
        (cs_summary['date'] - _epoch).dt.days.to_numpy(),
        cs_summary['cs_pace_sec'].values,
    )
    if verbose:
        print(f'  {len(rec)} recovery days with valid pace')

    # Watch/rule distance corrections (add_watch_corrections above). The
    # FIT runs on the corrected pace where one exists — honest distances
    # make the route betas physical — while the logged columns stay
    # untouched for display.
    rec = add_watch_corrections(rec)
    rec['pace_for_fit'] = np.where(rec['corr_pace_sec_per_mi'].notna(),
                                   rec['corr_pace_sec_per_mi'],
                                   rec['recovery_pace_sec_per_mi'])
    if verbose:
        n_w = int(rec['rec_watch'].sum())
        n_r = int(rec['rec_rule'].sum())
        if n_w or n_r:
            moved = (rec['pace_for_fit']
                     - rec['recovery_pace_sec_per_mi'])
            moved = moved[moved.abs() > 0]
            print(f'  corrections: {n_w} watch, {n_r} route-rule '
                  f'(median shift {moved.median():+.1f} s/mi over '
                  f'{len(moved)} moved rows)')

    # Features (sleep_centered and miles_centered are intentionally NOT computed)
    # Temperature: one-sided heat hinge max(0, air_temp - 6C) — see
    # temp_centered_feature. Cold contributes zero; only heat above ~6C slows
    # recovery pace. Slope is fit below; long runs reuse the same shape.
    rec['temp_centered'] = temp_centered_feature(rec)
    rec['residual_raw'] = rec['pace_for_fit'] - rec['cs_pace_sec']

    # Time-of-day binary indicator: 1 for PM (afternoon/late), 0 for AM
    # (early/morning). Recovery rows have 100% TOD coverage; missing or
    # unknown values fall to 0 (AM baseline).
    rec['tod_is_pm'] = tod_is_pm(rec)

    rec['conditions_clean'] = rec['conditions'].astype(str).str.strip().str.lower()
    cond_excluded = rec['conditions_clean'].isin(EXCLUDED_CONDITIONS)
    workout_snow = rec['workout_raw'].fillna('').astype(str).apply(
        lambda w: bool(SNOW_IN_WORKOUT_RE.search(w)))
    rec['is_bad_cond'] = cond_excluded | workout_snow
    rec['workout_snow_only'] = workout_snow & ~cond_excluded

    # Partner-run detection
    # fillna('') before astype(str): pandas 3.0's astype(str) leaves NaN as NaN
    # (not the string 'nan'), which would fall outside ADMITTED_PARTNERS and
    # flag every blank-partner run as a partner run — catastrophic for a
    # profile with no partners data at all (the whole fit gets pruned away).
    rec['partners_clean'] = (rec['partners'].fillna('')
                             .astype(str).str.strip().str.lower())
    rec['is_partner_run'] = ~rec['partners_clean'].isin(ADMITTED_PARTNERS)

    # Outlier detection — leave-one-out 28-day rolling mean of recovery
    # pace. The neighbor pool (the source of "normal" pace) is restricted
    # to rows that are neither bad-cond nor partner-run, so the local
    # baseline reflects typical solo recovery state. The LOO residual is
    # computed for EVERY recovery row, so a bad-cond or partner-run day
    # can also receive the outlier flag if its pace is anomalous against
    # the clean local mean. This means the three flags overlap, and a
    # single point can be in multiple classes.
    neighbor_mask = ~rec['is_bad_cond'] & ~rec['is_partner_run']
    rec['loo_resid'] = np.nan
    if neighbor_mask.sum() > 0:
        nbr_dates_ms = np.array(
            [d.value // 10**6 for d in rec.loc[neighbor_mask, 'date']])
        nbr_pace = rec.loc[neighbor_mask, 'pace_for_fit'].to_numpy()
        all_dates_ms = np.array([d.value // 10**6 for d in rec['date']])
        all_pace = rec['pace_for_fit'].to_numpy()
        in_pool = neighbor_mask.to_numpy()
        half_ms = OUTLIER_WINDOW_HALF_DAYS * 86_400_000
        loo_resid = np.full(len(rec), np.nan)
        for i, d_ms in enumerate(all_dates_ms):
            lo = np.searchsorted(nbr_dates_ms, d_ms - half_ms, side='left')
            hi = np.searchsorted(nbr_dates_ms, d_ms + half_ms, side='right')
            n_in = hi - lo
            if in_pool[i]:
                # Leave one out: this row IS a neighbor of itself.
                if n_in < OUTLIER_MIN_NEIGHBORS + 1:
                    continue
                local_mean = (nbr_pace[lo:hi].sum() - all_pace[i]) / (n_in - 1)
            else:
                # Bad-cond / partner row — not in pool, no leave-out needed.
                if n_in < OUTLIER_MIN_NEIGHBORS:
                    continue
                local_mean = nbr_pace[lo:hi].mean()
            loo_resid[i] = all_pace[i] - local_mean
        rec['loo_resid'] = loo_resid
    rec['is_outlier_loo'] = (
        rec['loo_resid'].abs() > OUTLIER_THRESHOLD_SEC).fillna(False)

    # Combined "pruned from fit" flag — drives both OLS exclusion and the
    # three independent visibility toggles on the chart. Flags are NOT
    # mutually exclusive; a single row can have more than one set.
    rec['is_pruned'] = (rec['is_bad_cond']
                       | rec['is_partner_run']
                       | rec['is_outlier_loo'])

    # Outlier breakdown by which other flags also apply
    n_outlier_clean = int((rec['is_outlier_loo']
                           & ~rec['is_bad_cond']
                           & ~rec['is_partner_run']).sum())
    n_outlier_bad = int((rec['is_outlier_loo'] & rec['is_bad_cond']).sum())
    n_outlier_partner = int((rec['is_outlier_loo'] & rec['is_partner_run']).sum())
    if verbose:
        print(f'  pruned from fit: {rec["is_pruned"].sum()} unique '
              f'({rec["is_bad_cond"].sum()} bad-cond '
              f'[{cond_excluded.sum()} cond + {workout_snow.sum()} workout-snow], '
              f'{rec["is_partner_run"].sum()} partner-runs, '
              f'{rec["is_outlier_loo"].sum()} outliers '
              f'[{n_outlier_clean} clean + {n_outlier_bad} also bad-cond + '
              f'{n_outlier_partner} also partner])')

    rec = add_quality_features(rec, quality_dates)

    # Physical route model (June 2026): per-run MEASURED elevation + terrain
    # type REPLACE the per-route dummies entirely. Elevation and terrain are
    # mechanical / era-invariant, so unlike free per-route betas they aren't
    # confounded by which era a route was run in (the same reason route betas
    # were removed from the long-run model). Whatever they DON'T explain stays
    # as visible residual — either era confound or a genuine effort-policy
    # difference in how Max approached that route — intentionally NOT
    # normalized out.
    elev = per_run_elevation(rec)
    rec['elev_gain_pm'] = elev['elev_gain_pm']
    rec['elev_loss_pm'] = elev['elev_loss_pm']
    terr = (rec.get('terrain_type', pd.Series('', index=rec.index))
            .astype(str).str.strip().str.lower())
    # Off-road footing (mixed+trail combined): the FLAT surface penalty, fit
    # below. Altitude (stored in FEET) → thousands of ft, fit below. Both are
    # era-correlated in Max's history (mixed = early/sea-level, paved = Boulder
    # era/altitude), so they're isolated via the backfit, not a sequential
    # era-detrend that would absorb them.
    rec['is_offroad'] = terr.isin(('mixed', 'trail')).astype(float)
    rec['alt_kft'] = altitude_regressor(per_run_altitude(rec))
    # PINNED grade-aware elevation cost (shared.elevation_cost): climbing
    # costs, descents refund a terrain-dependent fraction (paved ~full at
    # recovery effort, mixed ~1/3 — the downhill-braking asymmetry). It SCALES
    # with the route's actual gain/loss, so a hilly mixed route costs more than
    # a flat one. Pinned (not fitted) because gain/loss are collinear at run
    # level — the slopes are imported from the per-mile data where they ARE
    # identifiable. is_offroad then carries only the residual FLAT-footing cost.
    terr_for_cost = terr.where(terr.isin(('paved', 'mixed', 'trail')), 'paved')
    rec['elev_cost'] = elevation_cost(rec['elev_gain_pm'].fillna(0).to_numpy(),
                                      rec['elev_loss_pm'].fillna(0).to_numpy(),
                                      terr_for_cost.to_numpy())

    # Route counts kept for reporting/UI only — routes are no longer fit as
    # free dummies (route_col_map intentionally empty).
    route_counts = (rec.loc[~rec['is_pruned'], 'location']
                     .dropna().value_counts())
    qualifying_routes = (route_counts[route_counts >= MIN_ROUTE_N]
                         .index.tolist())
    qualifying_routes = sorted(qualifying_routes, key=lambda x: -route_counts[x])
    route_col_map = {}

    # ---------- features ----------
    base_features = (['temp_centered']
                     + [f'fat_{c}' for c in QUALITY_CATS]
                     + ['tod_is_pm'])
    # Physical route terms: flat-footing (is_offroad) + altitude. Identified
    # on the POOLED recovery+long-run corpus (physical_route_betas — the
    # single source of truth, shared with the long-run 5K conversion) and
    # PINNED here alongside the grade-aware elev_cost, NOT re-fit per call, so
    # there's one physical constant per channel everywhere it's applied.
    # pin_physical is False only when physical_route_betas itself calls in to
    # build the recovery side of the pool — it self-fits is_offroad/alt_kft
    # then (that's the recovery estimate the pool reads), which is what breaks
    # the recursion.
    if pin_physical:
        pooled = physical_route_betas()
        pinned_phys = (pooled['is_offroad'] * rec['is_offroad'].fillna(0.0)
                       + pooled['alt_kft'] * rec['alt_kft'].fillna(0.0)
                       ).to_numpy(float)
        physical_features = []
    else:
        pooled = None
        pinned_phys = np.zeros(len(rec))
        physical_features = ['is_offroad', 'alt_kft']
    feature_cols = base_features + physical_features

    # Wind: pooled, pinned per-mph cost applied symmetrically around the median
    # observed recovery wind (wind_reference_mph). Unlogged days are filled with
    # that median, so they're neutral (zero offset) rather than assumed calm;
    # calmer-than-median days get a small credit and windier days a small cost.
    # The full corpus stays in the fit. apply_wind is False only on the
    # bootstrap calls (wind_beta / physical_route_betas) that build the
    # pools — breaks recursion.
    if apply_wind:
        wind_b = wind_beta()
        wind_ref = wind_reference_mph()
        wind_off = (wind_b * (rec['wind_mph'].fillna(wind_ref)
                              - wind_ref)).to_numpy(float)
    else:
        wind_b = 0.0
        wind_off = np.zeros(len(rec))

    rec_fit = rec[~rec['is_pruned']].dropna(
        subset=feature_cols + ['residual_raw', 'elev_cost']).copy()
    if len(rec_fit) <= len(feature_cols) + 1:
        if verbose:
            print(f'  only {len(rec_fit)} fit-eligible rows for '
                  f'{len(feature_cols)} features — skipping fit')
        return None

    # ---------- backfit: era smoother <-> parametric factors ----------
    # The era trend is a temporal smoother, so a factor varying on the same
    # slow (era) timescale would be absorbed if era were computed once on the
    # raw residual. Footing and altitude are both era-correlated in Max's
    # history (mixed/sea-level early, paved/Boulder-altitude late), so we
    # estimate them JOINTLY with era by backfitting: recompute era on the
    # residual with the current factors AND the pinned elev_cost removed, refit
    # the factors on the era-removed residual, iterate. Fast factors
    # (temp/fatigue/TOD) are era-immune either way; the backfit is what lets
    # footing and altitude separate from era and from each other.
    pool_mask = (~rec['is_pruned']).to_numpy()
    fit_mask = rec.index.isin(rec_fit.index)
    Xall = np.hstack([np.ones((len(rec), 1)),
                      rec[feature_cols].fillna(0.0).to_numpy(float)])
    # Pinned offset: grade-aware elev_cost + (when pinned) pooled
    # footing/altitude + the pinned wind cost, all subtracted as fixed terms so
    # the backfit isolates the era trend + fast factors.
    elev_all = rec['elev_cost'].to_numpy(float) + pinned_phys + wind_off
    raw_all = rec['residual_raw'].to_numpy(float)
    dates_all = rec['date'].tolist()
    pool_dates = rec.loc[pool_mask, 'date'].tolist()
    Xf = Xall[fit_mask]
    coef = np.zeros(Xf.shape[1])
    era_all = np.zeros(len(rec))
    for _ in range(BACKFIT_ITERS):
        cleaned = raw_all - elev_all - (Xall @ coef)
        era_all = np.asarray(centered_rolling_mean(
            dates_all, pool_dates, cleaned[pool_mask],
            ERA_WINDOW_HALF_DAYS, ERA_WINDOW_MIN_POINTS), dtype=float)
        target = (raw_all - elev_all - era_all)[fit_mask]
        coef, *_ = np.linalg.lstsq(Xf, target, rcond=None)
    intercept = float(coef[0])
    betas = {f: float(b) for f, b in zip(feature_cols, coef[1:])}
    if pin_physical:
        betas['is_offroad'] = float(pooled['is_offroad'])
        betas['alt_kft'] = float(pooled['alt_kft'])
    betas['wind_mph'] = wind_b

    rec['era_trend'] = era_all
    rec['residual_detrended'] = rec['residual_raw'] - rec['era_trend']
    # Center era contributions on the mean era level (physical already removed,
    # so this is the fitness baseline).
    global_mean_residual = float(np.nanmean(era_all[pool_mask]))
    if verbose:
        print(f'  global mean residual: {global_mean_residual:+.2f} sec/mi '
              f'(mean era level; era fit on the physically-cleaned residual)')

    rec_fit = rec[fit_mask].copy()
    # Includes the pinned footing/altitude AND wind offsets so yhat predicts the
    # full model when those terms are pinned rather than fit.
    elev_fit = (rec_fit['elev_cost'].to_numpy(float) + pinned_phys[fit_mask]
                + wind_off[fit_mask])
    era_fit = rec_fit['era_trend'].to_numpy(float)
    resid_target = rec_fit['residual_raw'].to_numpy(float) - era_fit
    yhat = Xf @ coef + elev_fit  # predicts the era-detrended residual
    ss_res = float(np.sum((resid_target - yhat) ** 2))
    ss_tot = float(np.sum((resid_target - resid_target.mean()) ** 2))
    r2_detrended = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    raw_y = rec_fit['residual_raw'].to_numpy()
    raw_yhat = rec_fit['era_trend'].to_numpy() + yhat
    ss_res_raw = float(np.sum((raw_y - raw_yhat) ** 2))
    ss_tot_raw = float(np.sum((raw_y - raw_y.mean()) ** 2))
    r2_raw = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw > 0 else 0.0

    if verbose:
        print(f'\nOLS results (n={len(rec_fit)}):')
        print(f'  R² on detrended:    {r2_detrended:.3f}')
        print(f'  R² on raw residual: {r2_raw:.3f}')
        print(f'  intercept: {intercept:+.2f}')
        print(f'\n  Per-day factors:')
        for f in base_features:
            print(f'    {f:18s}  β = {betas[f]:+.3f}')
        if apply_wind:
            print(f'    {"wind_mph":18s}  β = {betas["wind_mph"]:+.3f} s/mi per '
                  f'mph (pinned; applied where a watch reading exists)')
        print(f'\n  Physical route model (replaces per-route dummies):')
        print(f'    elevation grade: PINNED grade-aware cost '
              f'(shared.elevation_cost); mean {rec_fit["elev_cost"].mean():+.1f} '
              f's/mi over the fit set (scales with each route\'s gain/loss)')
        print(f'    off-road footing: β = {betas["is_offroad"]:+.1f} s/mi (flat '
              f'surface penalty)')
        print(f'    altitude: β = {betas["alt_kft"]:+.2f} s/mi per 1000 ft above '
              f'{ALTITUDE_THRESHOLD_KFT:.0f}k threshold '
              f'(= {betas["alt_kft"] * (5.4 - ALTITUDE_THRESHOLD_KFT):+.1f} at '
              f'Boulder, {betas["alt_kft"] * (8.4 - ALTITUDE_THRESHOLD_KFT):+.1f} '
              f'at Magnolia)')

    # ---------- per-point contributions ----------
    # Reference the temperature ADJUSTMENT to the median clean-day hinge (like
    # wind around its median): the one-sided hinge is >= 0 everywhere, so an
    # uncentered contribution would move EVERY run one way when normalized
    # (temp touches every run, unlike altitude/footing). Centering makes hot
    # days move faster and cool days slower, the median day unchanged. This
    # re-references only the displayed/applied contribution — the fitted slope
    # betas['temp_centered'] is untouched (centering shifts the intercept only).
    temp_ref = float(rec.loc[~rec['is_pruned'], 'temp_centered'].median())
    rec['contrib_temp'] = betas['temp_centered'] * (
        rec['temp_centered'].fillna(0) - temp_ref)
    rec['contrib_quality'] = sum(
        betas[f'fat_{c}'] * rec[f'fat_{c}'].fillna(0) for c in QUALITY_CATS)
    rec['contrib_tod'] = betas['tod_is_pm'] * rec['tod_is_pm'].fillna(0)
    # Wind cost (s/mi); 0 where no watch wind reading exists or on calm days.
    rec['contrib_wind'] = wind_off

    # Physical route cost, split into independently-toggleable channels:
    #   elevation = NET grade at the paved/full-refund baseline,
    #               c_up·(gain−loss) — ZERO for loops, net up/down for
    #               point-to-point.
    #   terrain   = paved-vs-not FLAT-footing (is_offroad β) — zero on paved,
    #               the flat surface penalty on mixed/trail.
    #   terrain_descent = the refund asymmetry (elev_cost − paved-equivalent):
    #               the mixed/trail downhill-braking penalty, which SCALES with
    #               descent. It's an elevation×terrain interaction — it exists
    #               only because the route both descends AND is rough — so it is
    #               applied only when BOTH the Elevation and Terrain toggles are
    #               on (gated in make_recovery_plots.js), not as its own
    #               checkbox. Folded into contrib_route below so the pinned
    #               physical offset (wind_beta target) stays complete.
    #   altitude  = the altitude penalty.
    paved_equiv = elevation_cost(rec['elev_gain_pm'].fillna(0).to_numpy(),
                                 rec['elev_loss_pm'].fillna(0).to_numpy(),
                                 np.full(len(rec), 'paved'))
    rec['contrib_elevation'] = paved_equiv
    rec['contrib_terrain'] = betas['is_offroad'] * rec['is_offroad'].fillna(0)
    rec['contrib_terrain_descent'] = rec['elev_cost'].to_numpy() - paved_equiv
    rec['contrib_altitude'] = betas['alt_kft'] * rec['alt_kft'].fillna(0)
    rec['contrib_route'] = (rec['contrib_elevation'] + rec['contrib_terrain']
                            + rec['contrib_terrain_descent']
                            + rec['contrib_altitude'])

    # Era contribution centered on global mean
    rec['contrib_era'] = rec['era_trend'].fillna(global_mean_residual) - global_mean_residual

    return SimpleNamespace(
        rec=rec,
        betas=betas,
        intercept=intercept,
        r2_detrended=r2_detrended,
        r2_raw=r2_raw,
        n_fit=len(rec_fit),
        qualifying_routes=qualifying_routes,
        route_col_map=route_col_map,
        route_counts=route_counts,
        global_mean_residual=global_mean_residual,
        quality_dates=quality_dates,
    )
