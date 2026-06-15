"""Recovery-pace factor model: OLS on era-detrended residual vs CS.

Lifted out of ``src/plots/make_recovery_plots.py`` so both the Recovery plot
and the Long Runs plot can import the same fit. ``run_plots.sh`` builds Long
Runs *before* Recovery, so a file artifact written by the recovery plot would
be stale or missing — an import is build-order-independent. Same precedent as
``long_run_model.py`` (lifted out of ``plot_training_quality.py``).

Model (fit on the era-detrended residual = pace − CS − era_trend):

  residual_detrended ~ β_temp · temp_centered
                     + Σ β_r · route_dummy_r       (n ≥ MIN_ROUTE_N)
                     + β_marathon · fatigue_marathon
                     + β_race    · fatigue_race_short
                     + β_tod     · tod_is_pm

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
from src.shared.elevation_cost import elevation_cost

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
# docs/watch-stream-enrichment-plan.md thread 1). The grade-cost model itself
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

# Route-era mislogged-distance rules (Max-specific carve-out, June 2026 —
# same precedent as the race plots' handdrawn curve: hand-derived knowledge
# the data can't reconstruct). The two Nashville staples were logged with
# inflated distances until Max re-measured them in April 2022. Factors are
# the median logged-over-honest inflation across the routes'
# pre-2022-04-15 watch-covered LONG runs (honest = watch distance through
# the profile's calibration curve): belle meade 1.068 (n=18), greenway
# 1.056 (n=11). Distance error is a route property, so the same factors
# apply to the routes' recovery rows (9 pre-watch days). Applied per Max's
# call back to 2018. Watch enrichment, when present, wins over the rule.
# Lives here (not workouts.py) so both the long-run projection and the
# recovery fit can consume it without a circular import.
MISLOGGED_ROUTES = (
    ('belle meade', '2018-01-01', '2022-04-15', 1.068),
    ('greenway',    '2018-01-01', '2022-04-15', 1.056),
)


def _load_calibration():
    """(intercept_mi, slope) of the profile's log-vs-watch distance curve,
    or None when the artifact is absent (no watch corpus). Duplicates
    workouts._load_lr_calibration — that module imports this one."""
    if not REC_CAL_PATH.exists():
        return None
    cal = pd.read_csv(REC_CAL_PATH)
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
    for col in ('corr_miles', 'corr_time_s', 'corr_pace_sec_per_mi'):
        rec[col] = np.nan

    cal = _load_calibration()
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
            rec.loc[idx[ok], 'rec_watch'] = True
            rec.loc[idx[ok], 'corr_time_s'] = mov_s[ok]
            rec.loc[idx[ok], 'corr_pace_sec_per_mi'] = (mov_s / corr_mi)[ok]
            rec.loc[idx[ok], 'corr_miles'] = corr_mi[ok]

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
            em['gpm'] = em['elev_gain_ft'] / em['corr_miles']
            em['lpm'] = em['elev_loss_ft'] / em['corr_miles']
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


def transferable_contributions(df, betas, quality_dates):
    """Per-row modeled pace contribution (sec/mi) of the transferable
    factors — temperature, recent-race fatigue, time of day — for any
    daily-frame subset (e.g. long runs). Missing temp/TOD contribute 0,
    as does any factor whose beta key is absent from ``betas`` (the TQ
    long-run model passes a dict without ``tod_is_pm``, so TOD drops out).
    Subtracting the result from observed pace normalizes those factors out.
    """
    df = add_quality_features(df, quality_dates)
    contrib = betas.get('temp_centered', 0.0) * (
        (df['temp_c'] - TEMP_REFERENCE_C).fillna(0.0))
    for cat in QUALITY_CATS:
        contrib = contrib + betas.get(f'fat_{cat}', 0.0) * df[f'fat_{cat}']
    contrib = contrib + betas.get('tod_is_pm', 0.0) * tod_is_pm(df)
    return contrib.to_numpy(dtype=float)


# ---------- fit ----------

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
                                pin_physical=False)
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
            lr['temp_centered'] = (lr['temp_c'] - TEMP_REFERENCE_C).fillna(0.0)
            lr['tod_is_pm'] = tod_is_pm(lr)
            _epoch = pd.Timestamp('1970-01-01')
            cs_pace = np.interp((lr['date'] - _epoch).dt.days.to_numpy(),
                                (cs['date'] - _epoch).dt.days.to_numpy(),
                                cs['cs_pace_sec'].values)
            lr['resid'] = lr['pace_for_fit'] - cs_pace
            terr = (lr.get('terrain_type', pd.Series('', index=lr.index))
                    .astype(str).str.strip().str.lower())
            lr['is_offroad'] = terr.isin(('mixed', 'trail')).astype(float)
            lr['alt_kft'] = per_run_altitude(lr)
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
            out = {'is_offroad': float(cmap['is_offroad']),
                   'alt_kft': float(cmap['alt_kft'])}
    except Exception:
        pass
    _PHYS_BETAS_CACHE[key] = out
    return out


def fit_recovery_model(daily, races, cs_summary, verbose=True, pin_physical=True):
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
    rec['temp_centered'] = rec['temp_c'] - TEMP_REFERENCE_C
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
    rec['alt_kft'] = per_run_altitude(rec)
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
    # footing/altitude, both subtracted as fixed terms so the backfit isolates
    # the era trend + fast factors.
    elev_all = rec['elev_cost'].to_numpy(float) + pinned_phys
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

    rec['era_trend'] = era_all
    rec['residual_detrended'] = rec['residual_raw'] - rec['era_trend']
    # Center era contributions on the mean era level (physical already removed,
    # so this is the fitness baseline).
    global_mean_residual = float(np.nanmean(era_all[pool_mask]))
    if verbose:
        print(f'  global mean residual: {global_mean_residual:+.2f} sec/mi '
              f'(mean era level; era fit on the physically-cleaned residual)')

    rec_fit = rec[fit_mask].copy()
    # Includes the pinned footing/altitude offset so yhat predicts the full
    # model when physical terms are pinned rather than fit.
    elev_fit = rec_fit['elev_cost'].to_numpy(float) + pinned_phys[fit_mask]
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
        print(f'\n  Physical route model (replaces per-route dummies):')
        print(f'    elevation grade: PINNED grade-aware cost '
              f'(shared.elevation_cost); mean {rec_fit["elev_cost"].mean():+.1f} '
              f's/mi over the fit set (scales with each route\'s gain/loss)')
        print(f'    off-road footing: β = {betas["is_offroad"]:+.1f} s/mi (flat '
              f'surface penalty)')
        print(f'    altitude: β = {betas["alt_kft"]:+.2f} s/mi per 1000 ft '
              f'(= {betas["alt_kft"] * 5.4:+.1f} at Boulder)')

    # ---------- per-point contributions ----------
    rec['contrib_temp'] = betas['temp_centered'] * rec['temp_centered'].fillna(0)
    rec['contrib_quality'] = sum(
        betas[f'fat_{c}'] * rec[f'fat_{c}'].fillna(0) for c in QUALITY_CATS)
    rec['contrib_tod'] = betas['tod_is_pm'] * rec['tod_is_pm'].fillna(0)

    # Physical route cost, split into 3 independently-toggleable channels:
    #   elevation = NET grade at the paved/full-refund baseline,
    #               c_up·(gain−loss) — ZERO for loops, net up/down for
    #               point-to-point.
    #   terrain   = paved-vs-not: flat-footing (is_offroad β) + the refund
    #               asymmetry (elev_cost − paved-equivalent) — zero on paved,
    #               the mixed downhill-braking penalty (scales with descent)
    #               on mixed/trail.
    #   altitude  = the altitude penalty.
    paved_equiv = elevation_cost(rec['elev_gain_pm'].fillna(0).to_numpy(),
                                 rec['elev_loss_pm'].fillna(0).to_numpy(),
                                 np.full(len(rec), 'paved'))
    rec['contrib_elevation'] = paved_equiv
    rec['contrib_terrain'] = ((rec['elev_cost'].to_numpy() - paved_equiv)
                              + betas['is_offroad'] * rec['is_offroad'].fillna(0))
    rec['contrib_altitude'] = betas['alt_kft'] * rec['alt_kft'].fillna(0)
    rec['contrib_route'] = (rec['contrib_elevation'] + rec['contrib_terrain']
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
