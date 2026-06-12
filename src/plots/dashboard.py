"""Dashboard tab — text-only summary of key running data.

Sections:
1. Stats (streak, mileage totals, current shoes)
2. Personal Records (best race per FILTER_BINS distance)
3. Race Predictions (predicted time + 95% CrI per distance, from latest CS posterior)
4. Workout Pace Predictions (intervals / fartlek / long 20 / long 24)
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
from src.shared.plot_window import daily_floor
from src.shared.workouts import (
    TAU,
    load_cs, project_long_runs,
)
from src.shared.long_run_model import fit_long_run_model
from src.shared.cs_projection import load_cs_outputs
from src.plotting.formatters import sec_to_mss, sec_to_mss_full
from src.plotting.markers import PR_EXCLUDED_SURFACES
from src.plotting.render import _TAB_KEY_FORWARDER_JS

# Authoritative distance list (matches make_race_plots.FILTER_BINS).
FILTER_BINS = [
    ('400m',     400),
    ('800m',     800),
    ('1500m',    1500),
    ('Mile',     1609.344),
    ('3000m',    3000),
    ('2 Mile',   3218.688),
    ('5K',       5000),
    ('10K',      10000),
    ('HM',       21097.5),
    ('Marathon', 42195),
]

PRE_2016_VERIFIED_MILES = 1419
TRAINING_SHOE_RUN_THRESHOLD = 3  # consecutive recovery runs to qualify
# The current pair's mileage block ends once this many recovery runs in a row
# use a different shoe. Counting *runs* (not days) means a no-running gap can't
# split a pair, and it works whether or not asterisks disambiguate the model.
SHOE_BLOCK_DIFF_RUNS = 14

# Short-distance correction matching make_race_plots.py — track distances
# below 800m get stretched because the CS+D' model under-predicts time
# (peak speed limits and anaerobic capacity dominate, not sustained CS).
BETA_SHORT     = 0.35
D_THRESH_SHORT = 800.0

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


def fmt_cri_halfwidth(half_sec):
    """±Ns for sub-1-minute, ±MM:SS otherwise."""
    if half_sec is None or pd.isna(half_sec):
        return '—'
    half = float(half_sec)
    if half < 60:
        return f'±{int(round(half))}s'
    return f'±{sec_to_mss(half)}'


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
    # PRE_2016_VERIFIED_MILES is a Max-specific hand-verified pre-2016 total
    # (his paper logs aren't in the CSV); it must not be added to other
    # profiles, whose data starts when their device history does.
    since_2016 = daily[daily['date'] >= daily_floor()]
    pre_floor_miles = (PRE_2016_VERIFIED_MILES
                       if os.environ.get('RP_PROFILE', 'max') == 'max' else 0)
    miles_logged = int(math.floor(since_2016['miles'].sum() + pre_floor_miles))

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
            sub['time_sec'] * 1609.344 / sub['distance_m'])
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
def _beta_factor(d, beta_long, d_thresh_long, beta_short=BETA_SHORT,
                  d_thresh_short=D_THRESH_SHORT):
    """Same shape as cs_projection._beta_factor — long-distance fade for
    d > d_thresh_long, short-distance stretch for d < d_thresh_short."""
    if d > d_thresh_long and beta_long > 0:
        return 1.0 + beta_long * np.log(d / d_thresh_long)
    if d < d_thresh_short and beta_short > 0:
        return 1.0 + beta_short * np.log(d_thresh_short / d)
    return 1.0


def _time_at(d, dp, cs_mps, beta_long, d_thresh):
    if d <= dp or cs_mps <= 0:
        return float('nan')
    return (d - dp) / cs_mps * _beta_factor(d, beta_long, d_thresh)


def compute_race_predictions(daily_summary, beta_long, d_thresh):
    """For each FILTER_BIN distance, compute predicted time + 95% CrI half-width."""
    latest = daily_summary.iloc[-1]
    cs_mps_med = float(latest['cs_mps_med'])
    dp_med = float(latest['dp_med'])
    cs_pace_lo95 = float(latest['cs_pace_lo95'])
    cs_pace_hi95 = float(latest['cs_pace_hi95'])

    # Back out cs_mps consistent with the cs_pace bounds, holding dp at dp_med.
    def cs_mps_from_pace(pace_min):
        return 1609.344 * (5000.0 - dp_med) / pace_min / 5000.0 / 60.0

    cs_mps_lo95 = cs_mps_from_pace(cs_pace_hi95)  # slow pace -> low cs
    cs_mps_hi95 = cs_mps_from_pace(cs_pace_lo95)  # fast pace -> high cs

    out = []
    for name, d in FILTER_BINS:
        t_med = _time_at(d, dp_med, cs_mps_med, beta_long, d_thresh)
        t_lo = _time_at(d, dp_med, cs_mps_hi95, beta_long, d_thresh)  # fastest
        t_hi = _time_at(d, dp_med, cs_mps_lo95, beta_long, d_thresh)  # slowest
        half = (t_hi - t_lo) / 2.0
        out.append({'distance': name, 'time_sec': t_med, 'half_sec': half})
    return out


# ----- Workout Pace Predictions -----
# Recency weighting for the long-run prediction ground. 365d half-life
# balances "current era's effort policy" against sample size (effective
# n ≈ 60, vs 12 at a 90d half-life); the June 2026 sanity check showed
# 90/180/365d half-lives all land within 2 s/mi, so the choice isn't
# load-bearing.
LR_PRED_HALFLIFE_DAYS = 365
# Distance the card's long-run pace is projected to (and labeled as).
LR_PRED_MILES = 20


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


def compute_workout_predictions(daily_summary, lr_in_aug,
                                beta_long=0.0, d_thresh=10000.0):
    latest = daily_summary.iloc[-1]
    cs_mps = float(latest['cs_mps_med'])
    dp = float(latest['dp_med'])
    p5k_cs_min = float(latest.get('p5k_implied_min',
                                   1609.344 * (5000 - dp) / cs_mps / 5000 / 60))
    p5k_cs_sec = p5k_cs_min * 60.0  # 5K-equivalent CS pace in sec/mi.

    # --- Intervals 6x1600m, 3:00 rest ---
    rep_dist = 1600.0
    rep_count = 6
    rest_sec = 180.0
    rest_per_mile = rest_sec / (rep_dist / 1609.344)
    decay = math.exp(-rest_per_mile / TAU)
    d_eff_int = rep_dist * (1 + (rep_count - 1) * decay)
    t_eff_int = (d_eff_int - dp) / cs_mps
    pace_intervals = t_eff_int * 1609.344 / d_eff_int

    # --- Fartlek 8000m continuous ---
    d_far = 8000.0
    t_far = (d_far - dp) / cs_mps
    pace_fartlek = t_far * 1609.344 / d_far

    # --- Long runs: recency-weighted mean raw_resid over familiar-route,
    #     non-pruned long runs. Treat it as the expected 5K-equivalent pace
    #     residual; project from CS to the card's distance. raw_resid is the
    #     race-equivalent projection (β_long un-biased going in), so the
    #     round trip back to a long-run PACE must re-apply β at the card's
    #     distance — otherwise the fade we removed going in stays removed
    #     and the predicted pace reads too fast.
    lr_resid = _long_run_residual(lr_in_aug)

    def project_long(miles, resid_sec_per_mi):
        d = miles * 1609.344
        p5k_pred_sec = p5k_cs_sec + resid_sec_per_mi
        t_5k_pred_sec = p5k_pred_sec * 5000.0 / 1609.344
        cs_mps_lr = (5000.0 - dp) / t_5k_pred_sec
        t_d = (d - dp) / cs_mps_lr * _beta_factor(d, beta_long, d_thresh)
        return t_d * 1609.344 / d

    pace_long = (project_long(LR_PRED_MILES, lr_resid)
                 if lr_resid is not None else None)

    return {
        'intervals_6x1600': pace_intervals,
        'fartlek_8000':     pace_fartlek,
        'long':             pace_long,
        '_debug':           {'lr_resid': lr_resid},
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
        pr_rows.append(
            f'<tr>'
            f'<td>{escape(r["distance"])}</td>'
            f'<td class="num"><b>{fmt_race_time(r["time_sec"])}</b></td>'
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
            f'<td class="num"><b>{fmt_race_time(r["time_sec"])}</b></td>'
            f'<td class="num">{fmt_cri_halfwidth(r["half_sec"])}</td>'
            f'</tr>'
        )
    race_pred_html = '\n'.join(rp_rows)

    # Workout pace predictions
    wp_rows = [
        ('Intervals (6×1600m):', fmt_pace_per_mi(workout_preds['intervals_6x1600'])),
        ('Fartlek (8000m continuous):', fmt_pace_per_mi(workout_preds['fartlek_8000'])),
    ]
    # Long-run prediction needs an empirical slowdown residual; omit it when
    # the profile has no long-run history (see _long_run_residual).
    if workout_preds['long'] is not None:
        wp_rows.append((f'Long ({LR_PRED_MILES} miles):',
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
      <h2>Personal Records</h2>
      <table class="dash">
        <thead><tr>
          <th>Distance</th><th class="num">PR</th><th class="num">Date</th>
          <th>Event</th><th>Location</th>
        </tr></thead>
        <tbody>
{prs_html}
        </tbody>
      </table>
    </section>

    <section class="dash-section" id="sec-predictions">
      <h2>Race Predictions</h2>
      <table class="dash">
        <thead><tr>
          <th>Distance</th>
          <th class="num">Prediction</th>
          <th class="num">95% CrI</th>
        </tr></thead>
        <tbody>
{race_pred_html}
        </tbody>
      </table>
    </section>

    <section class="dash-section" id="sec-workouts">
      <h2>Workout Pace Predictions</h2>
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
    OUTPUT_DIR.mkdir(exist_ok=True)
    now_utc = dt.datetime.now(dt.timezone.utc)

    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])

    daily_summary, beta_long, d_thresh, _xc = load_cs_outputs(str(DATA_DIR))

    # Long-run model fit (in-slice runs, not snow). We need the augmented
    # frame (with `route` and `is_outlier`) for the recency-weighted
    # long-run residual on familiar routes — the model fit itself is only
    # used here for its outlier flags.
    cs, epoch = load_cs()
    lr_all = project_long_runs(cs, epoch)
    lr_in = lr_all[lr_all['excluded_reason'].isna()].copy()
    lr_in_aug, _lr_fit, _qual_routes = fit_long_run_model(lr_in)

    stats = compute_stats(daily, races, now_utc)
    prs = compute_prs(races)
    race_preds = compute_race_predictions(daily_summary, beta_long, d_thresh)
    workout_preds = compute_workout_predictions(daily_summary, lr_in_aug,
                                                beta_long, d_thresh)

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
