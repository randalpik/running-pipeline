"""Dashboard tab — text-only summary of key running data.

Sections:
1. Stats (streak, mileage totals, current shoes)
2. Personal Records (best race per FILTER_BINS distance)
3. Race Predictions (predicted time per distance, direct from the current
   PERFORMANCE FRONTIER — demonstrated capability; the band is the frontier
   swept across the CS 95% CrI, which collapses where a recent demonstration
   pins the frontier and equals the CS CrI on the floor)
4. Workout Pace Predictions ("fastest I could physically run this workout
   given the current frontier" — direct hyperbolic projection, no empirical
   residual offsets)
+ footer with snapshot last-updated timestamp.

Self-contained HTML — bypasses ``render_plot`` so the dashboard doesn't pull
in the Plotly bundle for nothing. Reuses ``base.css`` from
``src/plotting/_scaffold/`` for visual consistency with other tabs, and the
tab-key forwarder from ``render.py`` so Alt+←/→ tab cycling works.
"""
from __future__ import annotations

import calendar
import datetime as dt
import math
import os
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.units import METERS_PER_MILE
from src.shared.effective_mileage import effective_daily_miles
from src.shared.workouts import (
    TAU,
    load_cs, project_long_runs,
)
from src.shared.long_run_model import fit_long_run_model
from src.shared.cs_projection import (load_cs_outputs, t5k_to_anchor_time,
                                      cp3_dprime, cp3_implied_cs, cp3_time,
                                      vmax_predict)
from src.shared.performance_frontier import (standard_demos,
                                              build_frontier_band,
                                              frontier_at_anchor)
from src.plotting.formatters import sec_to_mss, sec_to_mss_full
from src.plotting.markers import PR_EXCLUDED_SURFACES
from src.plotting.render import _TAB_KEY_FORWARDER_JS

# Authoritative distance list (matches make_race_plots.FILTER_BINS).
FILTER_BINS = [
    ('400m',     400),
    ('800m',     800),
    ('1500m',    1500),
    ('Mile',     METERS_PER_MILE),
    ('3000m',    3000),
    ('2 Mile',   3218.688),
    ('5K',       5000),
    ('10K',      10000),
    ('HM',       21097.5),
    ('Marathon', 42195),
]

# Unlogged training miles per year, Max-only: hand-verified cumulative totals
# from paper logs that exist in no CSV, with race miles EXCLUDED — those races
# are snapshot:additions stubs in daily.csv and are summed from there, so
# including them here would double-count. 2014: cumulative log total 1312
# minus 38.92 stub race miles, rounded (the 2014 log was mile-granular).
UNLOGGED_MILES = {
    2014: 1273,
    2015: 0,  # pending: confirmed legacy-log total minus its race miles
}
TRAINING_SHOE_RUN_THRESHOLD = 3  # consecutive recovery runs to qualify
# The current pair's mileage block ends once this many recovery runs in a row
# use a different shoe. Counting *runs* (not days) means a no-running gap can't
# split a pair, and it works whether or not asterisks disambiguate the model.
SHOE_BLOCK_DIFF_RUNS = 14

# Short distances are handled structurally by the CP3 projection layer
# (cs_projection.cp3_*) — the former β_short stretch is gone; see
# docs/cs-model-reference.md ("Projection method: CP3").

OUT_HTML = OUTPUT_DIR / 'dashboard.html'
SCAFFOLD_DIR = Path(__file__).resolve().parents[1] / 'plotting' / '_scaffold'
_DASHBOARD_DIR = Path(__file__).resolve().parent
_DASHBOARD_CSS = (_DASHBOARD_DIR / 'dashboard.css').read_text()
_DASHBOARD_JS = (_DASHBOARD_DIR / 'dashboard.js').read_text()


# ----- helpers -----
def scrub_asterisk(name):
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    s = str(name).strip()
    if not s or s.lower() in ('nan', 'none'):
        return None
    return s.rstrip('*').strip()


def is_streak_active(last_log_date, now_utc):
    today_utc = now_utc.date()
    threshold_days = 2 if now_utc.hour >= 12 else 3
    return last_log_date >= today_utc - dt.timedelta(days=threshold_days)


def fmt_race_time(sec):
    """Race time: M:SS.x for sub-1hr (always shows tenths, even if .0),
    H:MM:SS for 1hr+."""
    if sec is None or pd.isna(sec):
        return '—'
    sec = float(sec)
    if sec >= 3600:
        return sec_to_mss(sec)
    # Sub-1hr — always show tenths, including '.0' to designate precision.
    tenths_total = int(round(sec * 10))
    whole = tenths_total // 10
    frac = tenths_total % 10
    m, ss = divmod(whole, 60)
    return f'{m}:{ss:02d}.{frac}'


def fmt_pace_per_mi(sec_per_mi):
    if sec_per_mi is None or pd.isna(sec_per_mi):
        return '—'
    return f'{sec_to_mss(sec_per_mi)}/mi'


def _fmt_cri_part(x):
    """Ns for sub-1-minute, MM:SS otherwise."""
    return f'{int(round(x))}s' if x < 60 else sec_to_mss(x)


def fmt_cri_offsets(t_med, lo_sec, hi_sec):
    """Asymmetric 95% CrI as offsets from the prediction: '−Δfast / +Δslow'.
    The frontier band is NOT symmetric about the median (its width depends on
    where the frontier sits relative to CS, and it collapses onto a binding
    demonstration), so a single ± would misstate both bounds. Tiny negatives
    from numerical noise are clamped to 0."""
    if any(v is None or pd.isna(v) for v in (t_med, lo_sec, hi_sec)):
        return '—'
    down = max(0.0, float(t_med) - float(lo_sec))   # faster bound
    up = max(0.0, float(hi_sec) - float(t_med))      # slower bound
    return f'−{_fmt_cri_part(down)} / +{_fmt_cri_part(up)}'


# PR values are tinted gold, shaded by recency (most recent PR = brightest,
# oldest = darkest/muted) so the freshest demonstrated capability reads loudest.
_PR_GOLD_BRIGHT = (0xFF, 0xD7, 0x4A)   # bright gold — most recent
_PR_GOLD_DARK   = (0x8A, 0x66, 0x12)   # muted dark gold — oldest
# Recency is log-warped before shading: PRs typically bunch up in the recent
# few years with one or two old outliers, so a linear date scale leaves the
# cluster nearly the same shade. The log curve fixes both endpoints (most- and
# least-recent PR still hit the bright/dark extremes) but expands the recent
# cluster across more of the gold range. Larger k = more spread near recent.
_PR_RECENCY_LOG_K = 9.0


def _recency_brightness(frac):
    """Log-warp a linear recency fraction (1 = most recent) into a brightness in
    [0,1] with the endpoints pinned. age = 1−frac; brightness =
    1 − ln(1+k·age)/ln(1+k), so recent (small-age) PRs spread out while old
    outliers compress toward the dark end."""
    age = 1.0 - max(0.0, min(1.0, float(frac)))
    return 1.0 - math.log1p(_PR_RECENCY_LOG_K * age) / math.log1p(_PR_RECENCY_LOG_K)


def _pr_gold(frac):
    """Hex gold for a recency fraction in [0,1] (1 = most recent = brightest),
    log-warped so clustered recent PRs stay visually distinct."""
    b = _recency_brightness(frac)
    rgb = tuple(round(d + (br - d) * b)
                for d, br in zip(_PR_GOLD_DARK, _PR_GOLD_BRIGHT))
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def fmt_friendly_date(d):
    """e.g. '17 Apr 2023'."""
    if d is None or pd.isna(d):
        return '—'
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    return f'{d.day} {d.strftime("%b")} {d.year}'


# ----- Stats -----
def compute_stats(daily, races, now_utc):
    contemporary = daily[daily['source_file'] != 'snapshot:additions'].copy()
    contemporary = contemporary.sort_values('date').reset_index(drop=True)
    last_log_date = contemporary['date'].max().date()

    # --- streak ---
    dates_set = set(contemporary['date'].dt.date)
    streak_len = 0
    cur = last_log_date
    while cur in dates_set:
        streak_len += 1
        cur -= dt.timedelta(days=1)
    streak_start = last_log_date - dt.timedelta(days=streak_len - 1)
    streak_active = is_streak_active(last_log_date, now_utc)

    # --- miles logged ---
    # All of daily counts, including pre-floor race-addition stubs (their
    # mileage is real). UNLOGGED_MILES adds Max's off-CSV paper-log years;
    # it must not be added to other profiles, whose data starts when their
    # device history does.
    unlogged = (sum(UNLOGGED_MILES.values())
                if os.environ.get('RP_PROFILE', 'max') == 'max' else 0)
    miles_logged = int(math.floor(daily['miles'].sum() + unlogged))

    # --- miles projected ---
    today_utc = now_utc.date()
    if streak_active:
        proj_year = last_log_date.year
        divisor_date = last_log_date
    else:
        proj_year = today_utc.year
        divisor_date = today_utc
    days_in_year = 366 if calendar.isleap(proj_year) else 365
    year_miles = daily[daily['date'].dt.year == proj_year]['miles'].sum()
    divisor = divisor_date.timetuple().tm_yday
    if divisor > 0 and year_miles > 0:
        projected = year_miles * (days_in_year / divisor)
    else:
        projected = 0.0

    # --- past 7 days ---
    window_start = pd.Timestamp(last_log_date) - pd.Timedelta(days=6)
    window_end = pd.Timestamp(last_log_date)
    past7 = daily[(daily['date'] >= window_start) & (daily['date'] <= window_end)]
    past_7_miles = past7['miles'].sum()

    # --- current training shoe ---
    rec = contemporary[contemporary['run_type'] == 'recovery'].copy()
    rec['shoes_clean'] = rec['shoes'].map(scrub_asterisk)
    rec = rec.dropna(subset=['shoes_clean']).sort_values('date').reset_index(drop=True)
    training_shoe = _find_training_shoe(rec)
    if training_shoe is not None:
        training_miles = _current_shoe_block_miles(
            daily, rec, training_shoe, last_log_date)
    else:
        training_miles = None

    # --- current racing shoe ---
    racing_shoe = _find_racing_shoe(races)

    return {
        'last_log_date':   last_log_date,
        'streak_len':      streak_len,
        'streak_start':    streak_start,
        'streak_active':   streak_active,
        'miles_logged':    miles_logged,
        'projected_year':  proj_year,
        'projected_miles': projected,
        'past_7_miles':    past_7_miles,
        'training_shoe':   training_shoe,
        'training_miles':  training_miles,
        'racing_shoe':     racing_shoe,
    }


def _find_training_shoe(rec):
    """Walk recovery rows latest-first; return the shoe that is the same on
    the latest run and at least the prior 2 recovery runs (3 in a row over
    recovery-only sequence). Non-recovery days were already excluded."""
    if len(rec) < TRAINING_SHOE_RUN_THRESHOLD:
        return None
    shoes = rec['shoes_clean'].values
    n = len(shoes)
    # Walk from the end backwards looking for a run of length >= 3.
    end = n - 1
    while end >= TRAINING_SHOE_RUN_THRESHOLD - 1:
        candidate = shoes[end]
        run_len = 1
        i = end - 1
        while i >= 0 and shoes[i] == candidate:
            run_len += 1
            i -= 1
            if run_len >= TRAINING_SHOE_RUN_THRESHOLD:
                return candidate
        # Not enough; jump to the start of this run (i+1) - 1, i.e. continue
        # at index i, looking for a different candidate.
        end = i
    return None


def _current_shoe_block_miles(daily, rec, shoe, last_log_date):
    """Miles on the *current physical pair* of `shoe`.

    Walk recovery runs latest-first (`rec` is sorted ascending with a
    `shoes_clean` column). The current block extends back until we hit
    SHOE_BLOCK_DIFF_RUNS recovery runs in a row in a different shoe — counting
    runs, not days, so a stretch of not running never splits a pair. Then sum
    every daily mile in `shoe` from the block's first day through the last log
    date, which naturally excludes earlier pairs of the same model."""
    block_start = None
    diff_streak = 0
    for r in rec.iloc[::-1].itertuples(index=False):
        if r.shoes_clean == shoe:
            diff_streak = 0
            block_start = r.date.date() if hasattr(r.date, 'date') else r.date
        else:
            diff_streak += 1
            if diff_streak >= SHOE_BLOCK_DIFF_RUNS:
                break
    if block_start is None:
        return 0.0
    df = daily.copy()
    df['shoes_clean'] = df['shoes'].map(scrub_asterisk)
    mask = ((df['shoes_clean'] == shoe)
            & (df['date'] >= pd.Timestamp(block_start))
            & (df['date'] <= pd.Timestamp(last_log_date)))
    return float(df.loc[mask, 'miles'].sum())


def _find_racing_shoe(races):
    rs = races[(races['surface'] == 'Road') &
               (races['distance_m'] >= 10000)].copy()
    rs = rs.dropna(subset=['shoes', 'date']).sort_values('date')
    if rs.empty:
        return None
    return scrub_asterisk(rs.iloc[-1]['shoes'])


# ----- Personal Records -----
def compute_prs(races):
    races = races.dropna(subset=['distance_m', 'time_sec', 'date']).copy()
    races = races[~races['surface'].isin(PR_EXCLUDED_SURFACES)]
    # Time trials aren't real races — they're benchmark efforts solo or in
    # training. Restrict PRs to genuine competition.
    is_tt = races['event'].fillna('').astype(str).str.contains(
        'time trial', case=False, regex=False)
    races = races[~is_tt]

    # Snap each race to the closest FILTER_BINS distance (no tolerance cap).
    def snap(d):
        return min(FILTER_BINS, key=lambda nt: abs(nt[1] - float(d)))[0]
    races['bin'] = races['distance_m'].map(snap)

    out = []
    for name, _ in FILTER_BINS:
        sub = races[races['bin'] == name]
        if sub.empty:
            out.append({'distance': name, 'time_sec': None, 'date': None,
                        'event': None, 'location': None})
            continue
        # PR = lowest pace_sec_per_mi (faster = smaller = better).
        sub = sub.copy()
        sub['pace'] = sub['pace_sec_per_mi'].fillna(
            sub['time_sec'] * METERS_PER_MILE / sub['distance_m'])
        best = sub.sort_values('pace').iloc[0]
        out.append({
            'distance': name,
            'time_sec': float(best['time_sec']),
            'date':     best['date'],
            'event':    best.get('event') if pd.notna(best.get('event')) else None,
            'location': best.get('location') if pd.notna(best.get('location')) else None,
        })
    return out


# ----- Race Predictions -----
def compute_race_predictions(daily_summary, beta_long, d_thresh,
                             front_med, front_lo, front_hi):
    """Per FILTER_BIN distance: predicted time direct from today's frontier
    ("the fastest I could physically race this distance"), with a band from
    the frontier swept across the CS 95% CrI. Where a recent demonstration
    binds, the three sweeps collapse onto it (proof pins the prediction);
    on the floor the band equals the CS CrI. Short distances ride the CP3
    bend inside frontier_at_anchor — no β_short."""
    out = []
    for name, d in FILTER_BINS:
        t_med = frontier_at_anchor(front_med, daily_summary, d, beta_long,
                                   d_thresh)[-1]
        t_fast = frontier_at_anchor(front_lo, daily_summary, d, beta_long,
                                    d_thresh)[-1]
        t_slow = frontier_at_anchor(front_hi, daily_summary, d, beta_long,
                                    d_thresh)[-1]
        # Asymmetric band: t_med is generally NOT the midpoint of
        # [t_fast, t_slow], so we keep both bounds rather than a half-width.
        out.append({'distance': name, 'time_sec': float(t_med),
                    'lo_sec': float(t_fast), 'hi_sec': float(t_slow)})
    return out


# ----- Workout Pace Predictions -----
# Recency weighting for the long-run prediction ground. 365d half-life
# balances "current era's effort policy" against sample size (effective
# n ≈ 60, vs 12 at a 90d half-life); the June 2026 sanity check showed
# 90/180/365d half-lives all land within 2 s/mi, so the choice isn't
# load-bearing.
LR_PRED_HALFLIFE_DAYS = 365
# Duration the card's long-run pace is projected to (and labeled as). A time
# target (vs a fixed distance) self-scales per runner — no assumption that the
# profile runs any particular distance — which matters for newer profiles that
# have no grounds for a 20-mile projection yet.
LR_PRED_HOURS = 2


def _long_run_residual(lr_in_aug):
    """Recency-weighted (exponential, ``LR_PRED_HALFLIFE_DAYS`` half-life)
    mean raw_resid over non-outlier long runs on familiar routes
    (``route != 'other'``). Distance-unconditioned: the former lr_lo/lr_hi
    empirical means differed by 0.3 s/mi (June 2026 sanity check) — the
    per-bin split contributed nothing — while recency moves the estimate
    ~+7 s/mi (current-era long runs sit further off CS than the all-time
    mean).

    None when there's no usable history: a 0 residual would collapse the
    long-run prediction onto the bare CS curve (i.e. ~interval pace) — far
    too fast. Callers omit the prediction instead of fabricating one.
    """
    keep = lr_in_aug[~lr_in_aug['is_outlier']]
    # Max grounds the residual on familiar routes so the route-mix skew is
    # captured. Other profiles have no repeated routes yet — every run is
    # 'other' — so requiring one would drop them all; use every non-outlier
    # long run instead.
    if os.environ.get('RP_PROFILE', 'max') == 'max':
        keep = keep[keep['route'] != 'other']
    if keep.empty:
        return None
    age_days = (keep['date'].max() - keep['date']).dt.days.astype(float)
    w = np.exp(-np.log(2) * age_days / LR_PRED_HALFLIFE_DAYS)
    return float((w * keep['raw_resid']).sum() / w.sum())


def _invert_projection(make_efforts, t5k_target, dp3, vmax, lo=200.0, hi=600.0):
    """Find the pace (s/mi) at which a structured workout's connected
    projection equals the frontier's 5K capability. make_efforts(pace) must
    return (d_eff, t_eff) via THE SAME machinery the TQ corpus uses
    (parse_workouts._connected_core / _cf_structure) — predictions and
    plotted points are projections of one another by construction (Max,
    June 2026: 'I assumed those were aligned'). Bisection; the projection
    is monotone in pace. vmax = workout_vmax(CS) at the prediction date."""
    def t5k_of(pace):
        d_eff, t_eff = make_efforts(pace)
        return float(cp3_time(5000.0,
                              cp3_implied_cs(d_eff, t_eff, dp3, vmax),
                              dp3, vmax))
    for _ in range(60):
        mid = (lo + hi) / 2
        if t5k_of(mid) < t5k_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_workout_predictions(daily_summary, front_med):
    """Direct frontier projections (Max, June 2026): "the fastest I could
    physically run this workout given the current frontier". Each structured
    prediction INVERTS the exact projection the TQ corpus applies to that
    workout shape (connected accumulator with the effort-aware deflation,
    CF 500/300 reconstruction, CP3 projection), so a workout run at the
    predicted pace would plot exactly ON the frontier. No empirical
    residual offsets anywhere."""
    from src.parsers.parse_workouts import _connected_core, _cf_structure
    from src.shared.workouts import workout_vmax
    latest = daily_summary.iloc[-1]
    # Workout predictions invert the TQ corpus machinery. v_max is the single
    # CS-multiple at the latest date (workout_vmax == evidence edge).
    vmax = workout_vmax(float(latest['cs_mps_med']))
    dp3 = float(cp3_dprime(latest['dp_med'], latest['cs_mps_med'], vmax))
    t5k_front = (float(front_med['frontier_pace_min'].iloc[-1])
                 * 60.0 * 5000.0 / METERS_PER_MILE)

    # --- Intervals 6x1600m, 3:00 rest (actual seconds, connected core) ---
    def _intervals(pace):
        dists = [1600.0] * 6
        times = [pace * 1600.0 / METERS_PER_MILE] * 6
        rests = [180.0] * 5
        return _connected_core(dists, times, rests, dp3=dp3, vmax=vmax)
    pace_intervals = _invert_projection(_intervals, t5k_front, dp3, vmax)

    # --- Continuous fartlek 8000m: the 500/300 reconstruction, inverted.
    #     The blended pace is what the log would read; the hard-500 pace
    #     (blended / structure ratio) is the actual prescription.
    def _cf(pace):
        d_eff, t_eff, _ = _cf_structure(8000.0, pace, dp3=dp3, vmax=vmax)
        return d_eff, t_eff
    pace_fartlek = _invert_projection(_cf, t5k_front, dp3, vmax)
    from src.parsers.parse_workouts import CF_HARD_M, CF_FLOAT_M, CF_FLOAT_HARD_RATIO
    hards = 8000.0 // (CF_HARD_M + CF_FLOAT_M) * CF_HARD_M + 8000.0 % (CF_HARD_M + CF_FLOAT_M)
    floats = 8000.0 - hards
    pace_cf_hard = (pace_fartlek * 8000.0
                    / (hards + CF_FLOAT_HARD_RATIO * floats))

    # --- Long run = the fastest LR_PRED_HOURS-hour effort: the canonical
    #     5K-equiv → anchor projection (t5k_to_anchor_time — World Athletics at
    #     this >5K range, same as the HM/marathon race predictions and the long
    #     runs' own frontier curves), inverted for a TIME target instead of a
    #     fixed distance. Bisect for the distance whose projected time equals the
    #     target (the projection is monotone in distance), then report its pace.
    #     Self-scales per runner (no fixed-distance assumption). The former
    #     cp3_time × β_long path predicted faster-than-CS at multi-hour efforts
    #     once β_long was retired to 0 — bare CP3 asymptotes to CS from above.
    vp = vmax_predict(float(latest['cs_mps_med']))
    dp3_p = float(latest['dp3_pred_med'])

    def _t_long(d):
        return float(t5k_to_anchor_time(t5k_front, dp3_p, d, vp))

    target_long = LR_PRED_HOURS * 3600.0
    lo_d, hi_d = 1000.0, 80000.0      # 1–80 km brackets any 2-hr effort
    for _ in range(60):
        mid = (lo_d + hi_d) / 2
        if _t_long(mid) < target_long:
            lo_d = mid
        else:
            hi_d = mid
    d_long = (lo_d + hi_d) / 2
    pace_long = target_long * METERS_PER_MILE / d_long

    return {
        'intervals_6x1600': pace_intervals,
        'fartlek_8000':     pace_fartlek,
        'fartlek_8000_hard': pace_cf_hard,
        'long':             pace_long,
    }


# ----- HTML rendering -----
def render_html(stats, prs, race_preds, workout_preds, last_updated_str, last_updated_iso):
    base_css = (SCAFFOLD_DIR / 'base.css').read_text()

    # Build stats rows
    streak_value = (
        f"<b>{stats['streak_len']:,}</b> days, "
        f"since {fmt_friendly_date(stats['streak_start'])}"
        if stats['streak_active'] else "<b>0</b> days"
    )
    miles_logged_html = f"<b>{stats['miles_logged']:,}</b> miles"
    projected_html = f"<b>{stats['projected_miles']:,.0f}</b> miles"
    past7_html = f"<b>{stats['past_7_miles']:.1f}</b> miles"

    stat_rows = [
        ('Streak:',                streak_value),
        ('Lifetime logged:',       miles_logged_html),
        (f"Projected for {stats['projected_year']}:", projected_html),
        ('Past 7 days:',           past7_html),
    ]
    # Shoe rows are omitted entirely (not shown as a "—" line) when the profile
    # logs no shoes — e.g. watch imports. Each row appears only if present, so
    # a profile that tracks training but not racing shoes shows just the one.
    if stats['training_shoe']:
        stat_rows.append((
            'Current training shoe:',
            f"<b>{escape(stats['training_shoe'])}</b> "
            f"<span class=\"dim\">({stats['training_miles']:.1f} miles)</span>"))
    if stats['racing_shoe']:
        stat_rows.append((
            'Current racing shoe:',
            f"<b>{escape(stats['racing_shoe'])}</b>"))
    stats_html = ''.join(
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        for label, value in stat_rows
    )

    # PRs table — Distance left, PR/Date right-aligned numeric cells, Event/Location left.
    # PR times are gold, shaded by recency across the set of dated PRs.
    pr_dates = [pd.Timestamp(r['date']) for r in prs
                if r['time_sec'] is not None and r['date'] is not None]
    d_lo = min(pr_dates).value if pr_dates else 0
    d_span = (max(pr_dates).value - d_lo) if pr_dates else 0
    pr_rows = []
    for r in prs:
        if r['time_sec'] is None:
            pr_rows.append(
                f'<tr><td>{escape(r["distance"])}</td>'
                f'<td colspan="4" class="dim">—</td></tr>'
            )
            continue
        loc = escape(r['location']) if r['location'] else ''
        evt = escape(r['event']) if r['event'] else ''
        frac = ((pd.Timestamp(r['date']).value - d_lo) / d_span
                if r['date'] is not None and d_span else 1.0)
        gold = _pr_gold(frac)
        pr_rows.append(
            f'<tr>'
            f'<td>{escape(r["distance"])}</td>'
            f'<td class="num"><b style="color:{gold}">{fmt_race_time(r["time_sec"])}</b></td>'
            f'<td class="num">{fmt_friendly_date(r["date"])}</td>'
            f'<td>{evt}</td>'
            f'<td>{loc}</td>'
            f'</tr>'
        )
    prs_html = '\n'.join(pr_rows)

    # Race Predictions table — both numeric cols right-aligned.
    rp_rows = []
    for r in race_preds:
        if r['time_sec'] is None or pd.isna(r['time_sec']):
            rp_rows.append(
                f'<tr><td>{escape(r["distance"])}</td>'
                f'<td colspan="2" class="dim">—</td></tr>'
            )
            continue
        rp_rows.append(
            f'<tr>'
            f'<td>{escape(r["distance"])}</td>'
            f'<td class="num"><b class="pred-value">{fmt_race_time(r["time_sec"])}</b></td>'
            f'<td class="num">{fmt_cri_offsets(r["time_sec"], r["lo_sec"], r["hi_sec"])}</td>'
            f'</tr>'
        )
    race_pred_html = '\n'.join(rp_rows)

    # Workout pace predictions
    wp_rows = [
        ('Intervals (6×1600m):', fmt_pace_per_mi(workout_preds['intervals_6x1600'])),
        ('Fartlek (8000m continuous):',
         f"{fmt_pace_per_mi(workout_preds['fartlek_8000'])} "),
    ]
    wp_rows.append((f'Long ({LR_PRED_HOURS} hours):',
                    fmt_pace_per_mi(workout_preds['long'])))
    workout_html = ''.join(
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value"><b>{value}</b></div>'
        for label, value in wp_rows
    )

    head_css = base_css + '\n' + _DASHBOARD_CSS

    body = f"""
<div class="rp-title-bar">
  <div class="rp-title-main">Dashboard</div>
  <div class="rp-title-sub">Key data, by the numbers</div>
</div>
<div class="dash-body">
  <div class="dash-main">

    <section class="dash-section" id="sec-stats">
      <h2>Stats</h2>
      <div class="kv">
{stats_html}
      </div>
    </section>

    <section class="dash-section" id="sec-prs">
      <h2 title="Fastest time at each distance in an official race.">Personal Records</h2>
      <table class="dash">
        <thead><tr>
          <th>Distance</th>
          <th class="num">PR</th>
          <th class="num">Date</th>
          <th>Event</th><th>Location</th>
        </tr></thead>
        <tbody>
{prs_html}
        </tbody>
      </table>
    </section>

    <section class="dash-section" id="sec-predictions">
      <h2 title="Predicted race time at each distance from today's performance frontier (demonstrated capability).">Race Predictions</h2>
      <table class="dash">
        <thead><tr>
          <th>Distance</th>
          <th class="num">Prediction</th>
          <th class="num" title="95% credible interval. Real ability is very likely to land somewhere between these two bounds.">95% CrI</th>
        </tr></thead>
        <tbody>
{race_pred_html}
        </tbody>
      </table>
    </section>

    <section class="dash-section" id="sec-workouts">
      <h2 title="Predicted paces for representative workouts at current fitness.">Workout Pace Predictions</h2>
      <div class="kv">
{workout_html}
      </div>
    </section>

  </div>
  <div class="dash-footer">
    Created by Max Randal. Last updated <time id="rp-last-updated" datetime="{escape(last_updated_iso)}">{escape(last_updated_str)}</time>.
    <div class="dash-signout">
      <button id="rp-signout" type="button">Sign out</button>
    </div>
  </div>
</div>
<script>
{_DASHBOARD_JS}
</script>
{_TAB_KEY_FORWARDER_JS}
"""

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Dashboard — running pipeline</title>
<meta name="rp-slug" content="dashboard">
<style>
{head_css}
</style>
</head>
<body>
{body}
</body></html>
"""


# ----- main -----
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = dt.datetime.now(dt.timezone.utc)

    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    # Source of truth: watch/route distance-corrected mileage (decrease-only;
    # corr <= logged). On-disk daily.csv 'miles' is untouched; all mileage
    # totals below (lifetime, year, past-7, shoe blocks) read the corrected
    # value. Pace/run_type/date logic is unaffected.
    daily['miles'] = effective_daily_miles(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])

    daily_summary, beta_long, d_thresh, _xc = load_cs_outputs(str(DATA_DIR))

    # Performance frontier (median + CS-CrI sweep) — the source for every
    # prediction below. Predictions are AS OF TODAY, not the fit grid's
    # margin-extended end: CS barely moves in the extrapolated tail, but
    # frontier excess decays on the ~6-week scale, so iloc[-1] on the full
    # grid would price the predictions months after the last demonstration.
    demos = standard_demos(daily_summary, beta_long, d_thresh, _xc)
    asof_mask = daily_summary['date'] <= pd.Timestamp(now_utc.date())
    summary_asof = (daily_summary[asof_mask] if asof_mask.any()
                    else daily_summary).reset_index(drop=True)
    front_med, front_lo, front_hi, _ = build_frontier_band(
        demos, pd.DatetimeIndex(summary_asof['date']), summary_asof)

    stats = compute_stats(daily, races, now_utc)
    prs = compute_prs(races)
    race_preds = compute_race_predictions(summary_asof, beta_long, d_thresh,
                                          front_med, front_lo, front_hi)
    workout_preds = compute_workout_predictions(summary_asof, front_med)

    snapshot_path = DATA_DIR / 'drive_snapshot.csv'
    # Anchor the timestamp in UTC at build time. The dashboard JS hydrates
    # it into the viewer's local time at render; the fallback string below
    # is what users with JS disabled (or before hydration) see, so it's
    # explicitly labeled UTC to avoid ambiguity.
    mtime_utc = dt.datetime.fromtimestamp(
        os.path.getmtime(snapshot_path), tz=dt.timezone.utc
    )
    last_updated_iso = mtime_utc.isoformat()
    last_updated = (
        f'{mtime_utc.day} {mtime_utc.strftime("%b")} {mtime_utc.year} '
        f'at {mtime_utc.strftime("%H:%M")} UTC'
    )

    html = render_html(stats, prs, race_preds, workout_preds,
                       last_updated, last_updated_iso)
    OUT_HTML.write_text(html)
    print(f'Wrote {OUT_HTML}')


if __name__ == '__main__':
    main()
