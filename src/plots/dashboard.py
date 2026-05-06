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
from src.shared.workouts import (
    LONG_FLOOR, LONG_CEIL, LR_INTERNAL_BIN, TAU,
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
SHOE_GAP_DAYS = 180  # 6 months for the training-shoe mileage cutoff
TRAINING_SHOE_RUN_THRESHOLD = 3  # consecutive recovery runs to qualify

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
    since_2016 = daily[daily['date'] >= pd.Timestamp('2016-01-01')]
    miles_logged = int(math.floor(since_2016['miles'].sum() + PRE_2016_VERIFIED_MILES))

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
        training_miles = _shoe_mileage_with_gap(daily, training_shoe, last_log_date)
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


def _shoe_mileage_with_gap(daily, shoe, last_log_date):
    """Sum miles for the given shoe walking backwards from last_log_date,
    stopping when last_seen - row.date > SHOE_GAP_DAYS (i.e. we've crossed
    a 6-month gap during which this shoe wasn't worn — we're in a previous
    physical pair's era)."""
    df = daily.copy()
    df['shoes_clean'] = df['shoes'].map(scrub_asterisk)
    df = df[df['date'] <= pd.Timestamp(last_log_date)].sort_values(
        'date', ascending=False).reset_index(drop=True)
    total = 0.0
    last_seen = None
    for row in df.itertuples(index=False):
        miles = float(row.miles) if not pd.isna(row.miles) else 0.0
        rdate = row.date.date() if hasattr(row.date, 'date') else row.date
        if row.shoes_clean == shoe:
            total += miles
            last_seen = rdate
        else:
            if last_seen is not None and (last_seen - rdate).days > SHOE_GAP_DAYS:
                break
    return total


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
def _bin_residuals(lr_in_aug):
    """Return {'lr_lo': mean_raw_resid, 'lr_hi': mean_raw_resid} where the
    mean is over long runs that are (a) on a qualifying route — i.e.
    ``route != 'other'`` (have a route beta) — and (b) not flagged as
    outliers by the iterative MAD prune.

    This empirical mean captures the per-bin residual on routes Max
    actually runs in each bin, including the route-mix skew (e.g. lr_lo
    runs on belle meade/greenway pull lr_lo's mean down).
    """
    keep = lr_in_aug[(lr_in_aug['route'] != 'other')
                     & (~lr_in_aug['is_outlier'])]
    out = {}
    for b in ('lr_lo', 'lr_hi'):
        sub = keep[keep['bin'] == b]
        out[b] = float(sub['raw_resid'].mean()) if len(sub) else 0.0
    return out


def compute_workout_predictions(daily_summary, lr_in_aug):
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

    # --- Long runs: empirical mean raw_resid filtered to qualifying-route,
    #     non-pruned long runs in each bin. Treat that mean as the expected
    #     5K-equivalent pace residual; project from CS to the workout distance.
    bin_resid = _bin_residuals(lr_in_aug)

    def project_long(miles, resid_sec_per_mi):
        d = miles * 1609.344
        p5k_pred_sec = p5k_cs_sec + resid_sec_per_mi
        t_5k_pred_sec = p5k_pred_sec * 5000.0 / 1609.344
        cs_mps_lr = (5000.0 - dp) / t_5k_pred_sec
        t_d = (d - dp) / cs_mps_lr
        return t_d * 1609.344 / d

    pace_long_24 = project_long(24, bin_resid['lr_hi'])

    return {
        'intervals_6x1600': pace_intervals,
        'fartlek_8000':     pace_fartlek,
        'long_24':          pace_long_24,
        '_debug':           bin_resid,
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
    if stats['training_shoe']:
        ts = (f"<b>{escape(stats['training_shoe'])}</b> "
              f"<span class=\"dim\">({stats['training_miles']:.1f} miles)</span>")
    else:
        ts = '<span class="dim">—</span>'
    if stats['racing_shoe']:
        rs = f"<b>{escape(stats['racing_shoe'])}</b>"
    else:
        rs = '<span class="dim">—</span>'

    stat_rows = [
        ('Streak:',                streak_value),
        ('Lifetime logged:',       miles_logged_html),
        (f"Projected for {stats['projected_year']}:", projected_html),
        ('Past 7 days:',           past7_html),
        ('Current training shoe:', ts),
        ('Current racing shoe:',   rs),
    ]
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
        ('Long (24 miles):', fmt_pace_per_mi(workout_preds['long_24'])),
    ]
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
    # frame (with `route`, `bin`, `is_outlier`) to compute empirical bin
    # residuals on qualifying routes — the model fit itself is only used
    # here for its outlier flags.
    cs, epoch = load_cs()
    lr_all = project_long_runs(cs, epoch)
    lr_in = lr_all[lr_all['excluded_reason'].isna()].copy()
    lr_in_aug, _lr_fit, _qual_routes = fit_long_run_model(lr_in)

    stats = compute_stats(daily, races, now_utc)
    prs = compute_prs(races)
    race_preds = compute_race_predictions(daily_summary, beta_long, d_thresh)
    workout_preds = compute_workout_predictions(daily_summary, lr_in_aug)

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
