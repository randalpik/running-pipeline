"""make_recovery_plots.py - Recovery runs analyzed against CS pace.

Two side-by-side panels in one HTML:
  - Left: absolute recovery pace + CS gold reference curve
  - Right: residual (pace - CS) + flat zero gold reference

Both panels are anchored to the watch/route-corrected pace (``pace_for_fit``
in recovery_model — calibrated watch measurement where one exists, route-era
deflation for over-measured pre-watch routes, logged pace otherwise), the same
canonical source of truth the fit and the Long Runs plot use. Corrected pace is
the only pace shown; the originally-logged value lives in the raw logs. The
residual panel therefore equals (absolute corrected pace − CS) point-for-point.

A right-sidebar checkbox UI applies normalization factors. Most factors
adjust BOTH panels simultaneously (subtracting β × x from each point's
y-value), but the "Era trend" toggle is special and applies ONLY to the
residual panel — the absolute pace panel always shows the corrected
pace minus only the cross-sectional factor adjustments. This lets you
ask "how did my absolute pace compare to CS over time, controlling for
these factors?" without the chart's baseline shifting underneath you.

Era trend is a centered ±182-day rolling mean of residual. Toggling it
on subtracts (era_trend − global_mean_residual) from each residual,
which centers the points around the global mean rather than collapsing
them toward zero. The gold zero line on the residual panel remains the
CS reference.

Per-day factors are fit via OLS on the era-detrended residual:

  residual_detrended ~ β_temp · temp_heat_hinge
                     + β_marathon · fatigue_marathon
                     + β_race    · fatigue_race_short
                     + β_tod     · tod_is_pm
                     + (pinned) footing + altitude + grade-aware elevation
                     + (pinned) wind_mph · β_wind

The temperature term is a one-sided heat hinge, max(0, air_temp − 6°C): cold
contributes zero, only heat above ~6°C slows recovery (see
recovery_model.temp_centered_feature; long runs reuse the same shape with a
free slope). It replaced a symmetric apparent-temp term whose sub-12°C arm
credited a phantom cold speedup. Humidity was tested as a separate regressor
and dropped (weak; the heat index never beat plain air temp). Wind is a pinned
per-mph cost applied where a watch wind reading exists. See the model module
for the pooled/pinned mechanics.

Sleep cycles and recovery distance were tested in earlier model versions
and produced near-zero coefficients with no useful explanatory power. They
are excluded.

Hill workouts (continuous + reps) and tempo/interval/rep/fartlek workouts
were also tested as fatigue features and showed near-zero or noise-level
βs. Only races and long runs leave detectable next-day pace effects.

Training-quality residual (the smoothed training-derived deviation from CS
that plot_training_quality.py produces) was tested as a feature and found
to be redundant with era trend — the two are highly collinear over their
shared range, since both are smoothed long-term tracks of fitness vs CS.
Era trend wins because it's data-driven from recovery itself.

Three classes of rows are pruned from BOTH the era-trend window contents
AND the OLS fit, but remain on the chart with hover notes. **A single row
can fall into multiple classes** — the three flags are independent.

  1. Bad conditions — conditions ∈ {snow, icy}, OR the workout string
     contains "snow" (catches "[2" snow]" annotations the conditions
     field missed). HAND-LOGGED inside/treadmill/indoor-track runs are
     kept (still valid pace data on a stable surface); WATCH-derived
     indoor runs are dropped from the corpus upstream, before this plot
     sees them — their distance is the watch's uncalibrated stride
     estimate (recovery_model.is_watch_indoor).
  2. Partner runs — any partners entry outside ADMITTED_PARTNERS
     (blank/solo/none/varsity). Varsity is admitted (June 2026): in the
     2016-17 era the varsity group's recovery pace WAS Max's own pace
     strategy. Individual named partners remain a different population.
  3. Outliers — |residual from leave-one-out 28-day local mean| > 45
     sec/mi, where the local mean is computed against the *clean*
     neighbor pool (rows that are neither bad-cond nor partner-run).
     Removes "clearly something happened" days (travel, illness,
     extreme post-marathon fatigue). Computed for every recovery row,
     so a partner-run or bad-cond day can also be flagged outlier if
     its pace is anomalous against the clean local mean.

The visibility section has three independent toggles ("Hide bad
conditions", "Hide partner runs", "Hide outliers"), each with its own count
and an All/None group. Because flags can overlap, the counts may sum
to more than the unique pruned total.

Hidden points have y=null on the trace (not just opacity=0) so hover
events don't fire on them.

Hover annotates each point with display_name and city_state read directly
from columns on daily.csv (populated by build_dataset.py from the locations
sheet of the adjustments doc). These are purely informational — they don't
feed the model. Daily rows whose location isn't in the locations sheet
fall back to the raw log_location string.

Workflow:
  python src/plots/make_recovery_plots.py

Defaults read CS fit outputs, daily.csv, and races.csv from data/ and write
the HTML into output/. Override with --in-dir, --daily, --races, --out-dir.

Dependencies: pandas, numpy, plotly.
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import clip_to_daily_floor, axis_pad_entry
from src.plotting import (render_plot, CursorTooltip, MobileLayout,
                            apply_default_layout,
                            reshape_patch, assert_reshape_compatible,
                            sec_to_mss, FG, route_label, marker_half_px,
                            CS_LINE, CS_LINE_WIDTH, TREND_LINE, TREND_WIDTH,
                            GRID, gaussian_rolling_trend,
                            yearly_x_axis_kwargs, nice_time_ticks,
                            nice_time_interval, time_ticks_at_interval, tt_kv)
from src.plotting import widgets
from src.shared.paths import DEBUG_DIR
from src.shared.cs_projection import load_cs_outputs
from src.shared.recovery_model import (fit_recovery_model, TEMP_HEAT_ONSET_C,
                                       FATIGUE_TAU_DAYS, MIN_ROUTE_N,
                                       QUALITY_CATS, ALTITUDE_THRESHOLD_KFT)

_PLOTS_DIR = Path(__file__).resolve().parent
_RECOVERY_JS = _PLOTS_DIR / 'make_recovery_plots.js'


DEFAULT_IN_DIR = str(DATA_DIR)
DEFAULT_DAILY  = str(DATA_DIR / 'daily.csv')
DEFAULT_RACES  = str(DATA_DIR / 'races.csv')
DEFAULT_OUT    = str(OUTPUT_DIR)


# Model constants and the fit itself live in src/shared/recovery_model.py
# (shared with the Long Runs plot's normalization toggle). What stays here
# is presentation-only.

# Hover threshold — only show the "Recent race" line if within this many
# days of the most recent race. Cosmetic only; doesn't affect the model.
FATIGUE_HOVER_DAYS = 14

TREND_SMOOTH_SIGMA_DAYS = 28  # Gaussian σ; FWHM ≈ 66d, ~95% mass within ±56d


# ---------- helpers ----------

def signed_sec(s):
    """Plot-local signed-seconds formatter — emits ``+30`` / ``-30`` (no
    M:SS conversion). Differs from formatters.signed_sec, which emits
    ``+0:30`` / ``-0:30``. Recovery's residual ticks and hover already
    carry the ``sec/mi`` label so a count is clearer than a duration."""
    if s is None or pd.isna(s):
        return ''
    s = int(round(float(s)))
    return f'{"+" if s >= 0 else "−"}{abs(s)}'


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    ap.add_argument('--in-dir', default=DEFAULT_IN_DIR)
    ap.add_argument('--daily', default=DEFAULT_DAILY)
    ap.add_argument('--races', default=DEFAULT_RACES)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--diagnostics', action='store_true',
                    help='Also write route_betas.csv (per-route pace '
                         'coefficients) into output/debug/. Off by default '
                         '— the file is informational and not consumed by '
                         'anything downstream.')
    args = ap.parse_args()

    print(f'Loading daily.csv from {args.daily}...')
    daily = pd.read_csv(args.daily, parse_dates=['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    # Drop the pre-2016 race-addition stubs the world map uses; recovery
    # plots stay anchored to the 2016+ logging era.
    daily = clip_to_daily_floor(daily).reset_index(drop=True)

    print(f'Loading races.csv from {args.races}...')
    races = pd.read_csv(args.races, parse_dates=['date'])

    print(f'Loading CS outputs (tag={args.tag}) from {args.in_dir}...')
    cs_summary, beta_long, d_thresh, xc = load_cs_outputs(args.in_dir, args.tag)
    cs_summary['cs_pace_sec'] = cs_summary['cs_pace_med'] * 60.0

    res = fit_recovery_model(daily, races, cs_summary)
    if res is None:
        raise SystemExit('Not enough recovery data to fit the model; '
                         'no recovery plot written.')
    rec = res.rec
    betas, intercept = res.betas, res.intercept
    r2_detrended, r2_raw = res.r2_detrended, res.r2_raw
    qualifying_routes = res.qualifying_routes
    route_col_map, route_counts = res.route_col_map, res.route_counts
    global_mean_residual = res.global_mean_residual
    _epoch = pd.Timestamp('1970-01-01')

    # ---------- build figure ----------
    # Daily CS reference series across the full recovery range, hold-last
    # extrapolated past the summary's end (np.interp) so the CS line and the JS
    # per-day grid cover recent runs even if the fit ended before the last run.
    cs_grid = pd.date_range(rec['date'].min(), rec['date'].max(), freq='D')
    cs_for_plot = pd.DataFrame({
        'date': cs_grid,
        'cs_pace_sec': np.interp((cs_grid - _epoch).days.to_numpy(),
                                 (cs_summary['date'] - _epoch).dt.days.to_numpy(),
                                 cs_summary['cs_pace_sec'].values),
    })

    def build_hover(row):
        # Per-run HTML for the smart-spikeline scaffold's snap mode and the
        # smooth-mode "Nearest run" section. The cursor scaffold renders the
        # date and the CS-pace/Trend-pace/CS-residual/Trend-residual rows
        # itself, so this content focuses on what's run-specific.
        # Watch/route corrected pace is the displayed source of truth and the
        # only pace shown (matching the Long Runs tooltip); corr_pace_sec_per_mi
        # is the value the FIT used and what the marker is plotted at
        # (pace_for_fit). The originally-logged pace is recoverable from the raw
        # logs and is intentionally not shown here. A [watch-measured] /
        # [corrected route] tag flags the source when the correction meaningfully
        # differs (>= 1 s/mi); the distance bracket pins many under-logged days
        # to corr_pace ≈ logged, where no tag is warranted. A corrected day on a
        # strides route carries pace only (corr_miles NaN — distance polluted by
        # both the route over-estimate and the strides).
        logged_pace = row['recovery_pace_sec_per_mi']
        corr_p = row.get('corr_pace_sec_per_mi')
        has_corr = pd.notna(corr_p)
        disp_pace = float(corr_p) if has_corr else logged_pace
        meaningful = has_corr and abs(float(corr_p) - logged_pace) >= 1.0

        if has_corr and pd.notna(row.get('corr_miles')):
            disp_dist = f"  ({row['corr_miles']:.1f} mi)"
        elif has_corr and row.get('has_strides'):
            disp_dist = '  (pace only — strides)'
        else:
            disp_dist = f"  ({row['miles']:.1f} mi)"

        tag = ''
        if meaningful:
            if row.get('rec_watch'):
                tag = ' <span style="color:#888">[watch-measured]</span>'
            elif row.get('rec_rule'):
                tag = ' <span style="color:#888">[corrected route]</span>'
        parts = [tt_kv('Pace', f"{sec_to_mss(disp_pace)}/mi{disp_dist}{tag}")]

        # Measured vertical, shown when the grade contribution moves this
        # run's adjusted pace notably (>= 1 s/mi).
        ce = row.get('contrib_elevation')
        if pd.notna(ce) and abs(float(ce)) >= 1.0:
            mi = float(row['corr_miles']) if pd.notna(row.get('corr_miles')) \
                else float(row['miles'])
            gain = float(row.get('disp_gain_pm')
                         or row.get('elev_gain_pm') or 0) * mi
            loss = float(row.get('disp_loss_pm')
                         or row.get('elev_loss_pm') or 0) * mi
            if gain or loss:
                from src.plotting.hover import signed_mss
                cs, ds = row.get('climb_s_mi'), row.get('desc_s_mi')
                if pd.notna(cs) and pd.notna(ds):
                    parts.append(tt_kv(
                        'Elevation',
                        f'+{gain:.0f} ft ({signed_mss(float(cs) * mi)}) / '
                        f'−{loss:.0f} ft ({signed_mss(float(ds) * mi)})'))
                else:
                    parts.append(tt_kv('Elevation',
                                       f'+{gain:.0f}/−{loss:.0f} ft'))

        if pd.notna(row.get('temp_c')):
            parts.append(tt_kv('Temp', f"{row['temp_c']:.0f}°C"))
        if pd.notna(row.get('wind_mph')):
            parts.append(tt_kv('Wind', f"{row['wind_mph']:.0f} mph"))
        loc = row.get('location')
        if pd.notna(loc) and str(loc) != 'nan':
            # Shared dedup formatter (watch profiles: display_name == city_state).
            label = route_label(row.get('display_name'), row.get('city_state')) or str(loc)
            parts.append(f"<i>{label}</i>")
        if row.get('is_bad_cond'):
            if row.get('workout_snow_only'):
                parts.append(f"<i>Snow noted in workout (excluded from fit)</i>")
            else:
                parts.append(f"<i>Conditions: {row['conditions_clean']} "
                             f"(excluded from fit)</i>")
        elif (pd.notna(row.get('conditions_clean'))
              and str(row['conditions_clean']) not in ('nan', '')):
            parts.append(tt_kv('Conditions', row['conditions_clean']))
        if row.get('is_partner_run'):
            parts.append(f"<i>Partners: {row['partners']} "
                         f"(excluded from fit)</i>")
        elif pd.notna(row.get('partners')) and str(row['partners']) != 'nan':
            parts.append(tt_kv('With', row['partners']))
        if row.get('is_outlier_loo'):
            parts.append(f"<i>Outlier (excluded from fit)</i>")
        # Most recent race within fatigue decay window only
        most_recent = None  # (cat, days_ago)
        for cat in QUALITY_CATS:
            d = row.get(f'dsq_{cat}')
            if pd.notna(d) and d <= FATIGUE_HOVER_DAYS:
                if most_recent is None or d < most_recent[1]:
                    most_recent = (cat, int(d))
        if most_recent is not None:
            cat, d = most_recent
            label = "Recent marathon" if cat == 'marathon' else "Recent race"
            parts.append(tt_kv(label, f"{d}d ago"))
        tod_raw = row.get('time_of_day')
        if pd.notna(tod_raw) and str(tod_raw).strip():
            parts.append(tt_kv('Time of day', str(tod_raw).strip().lower()))
        return '<br>'.join(parts)

    rec['_hover'] = rec.apply(build_hover, axis=1)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=('Absolute pace', 'Residual vs 5K fitness'),
                         horizontal_spacing=0.07)

    fig.add_trace(go.Scatter(
        x=cs_for_plot['date'], y=cs_for_plot['cs_pace_sec'],
        mode='lines', line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
        name='5K fitness', hoverinfo='skip', showlegend=True,
        meta={'role': 'reference'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[rec['date'].min(), rec['date'].max()], y=[0, 0],
        mode='lines', line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
        name='Zero', hoverinfo='skip', showlegend=False,
        meta={'role': 'reference'},
    ), row=1, col=2)

    # Pre-compute the initial (no-normalization-applied, no-hides-applied)
    # trend so the line renders on first paint. The JS rollingTrend reruns
    # with the same algorithm whenever the user toggles a factor or filter.
    rec_dates_ms = np.array([d.value // 10**6 for d in rec['date']],
                            dtype=np.int64)
    init_pace_x, init_pace_y = gaussian_rolling_trend(
        rec_dates_ms, rec['pace_for_fit'].values,
        sigma_days=TREND_SMOOTH_SIGMA_DAYS)
    init_resid_x, init_resid_y = gaussian_rolling_trend(
        rec_dates_ms, rec['residual_raw'].values,
        sigma_days=TREND_SMOOTH_SIGMA_DAYS)
    init_pace_dates  = pd.to_datetime(init_pace_x,  unit='ms')
    init_resid_dates = pd.to_datetime(init_resid_x, unit='ms')

    fig.add_trace(go.Scatter(
        x=init_pace_dates, y=init_pace_y, mode='lines',
        line=dict(color=TREND_LINE, width=TREND_WIDTH),
        name='Trend', hoverinfo='skip', showlegend=True,
        meta={'role': 'trend_pace'},
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=init_resid_dates, y=init_resid_y, mode='lines',
        line=dict(color=TREND_LINE, width=TREND_WIDTH),
        name='Trend', hoverinfo='skip', showlegend=False,
        meta={'role': 'trend_resid'},
    ), row=1, col=2)

    # Customdata channels — channels 0-7 ORDER MUST MATCH FACTOR_ORDER in JS:
    # 0=temp, 1=elevation, 2=terrain, 3=altitude, 4=recent_effort, 5=tod,
    # 6=era, 7=wind. Channel 8 (terrain_descent) is NOT a FACTOR_ORDER entry
    # and has no checkbox: it's the mixed/trail descent-braking refund, an
    # elevation×terrain interaction the JS applies only when BOTH the Elevation
    # and Terrain toggles are on.
    contrib_arr = np.stack([
        rec['contrib_temp'].to_numpy(),
        rec['contrib_elevation'].to_numpy(),
        rec['contrib_terrain'].to_numpy(),
        rec['contrib_altitude'].to_numpy(),
        rec['contrib_quality'].to_numpy(),
        rec['contrib_tod'].to_numpy(),
        rec['contrib_era'].to_numpy(),
        rec['contrib_wind'].to_numpy(),
        rec['contrib_terrain_descent'].to_numpy(),
    ], axis=1).tolist()

    # Snap HTML lives on the residual trace's text field — same content as
    # the pace trace's, since both panels represent the same run. (The
    # per-trace hover html lookup falls back to text when customdata is a
    # structured array, which is the case here — customdata holds the
    # factor-contribution vector used by update().)
    hover_html = rec['_hover'].tolist()
    fig.add_trace(go.Scatter(
        x=rec['date'],
        y=rec['pace_for_fit'],
        mode='markers',
        marker=dict(
            size=4.5,
            color=rec['temp_c'],
            coloraxis='coloraxis',
            opacity=0.6,
            line=dict(width=0),
        ),
        customdata=contrib_arr,
        text=hover_html,
        hoverinfo='skip',
        name='Recovery (pace)',
        showlegend=False,
        meta={'role': 'pace',
              'snap_eligible': True,
              'raw_y': rec['pace_for_fit'].tolist()},
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rec['date'],
        y=rec['residual_raw'],
        mode='markers',
        marker=dict(
            size=4.5,
            color=rec['temp_c'],
            coloraxis='coloraxis',
            opacity=0.6,
            line=dict(width=0),
        ),
        customdata=contrib_arr,
        text=hover_html,
        hoverinfo='skip',
        name='Recovery (residual)',
        showlegend=False,
        meta={'role': 'residual',
              'snap_eligible': True,
              'raw_y': rec['residual_raw'].tolist(),
              'date_ms': [int(d.value // 10**6) for d in rec['date']],
              'is_bad_cond': rec['is_bad_cond'].astype(bool).tolist(),
              'is_partner_run': rec['is_partner_run'].astype(bool).tolist(),
              'is_outlier': rec['is_outlier_loo'].astype(bool).tolist()},
    ), row=1, col=2)

    # ---------- axes ----------
    # Tight data range; the half-marker pixel gutter (so edge dots aren't
    # clipped) is added at render time by axis_pad.js and re-applied on resize.
    # Both panels share the date axis, so pad xaxis and xaxis2 alike.
    x_lo, x_hi = rec['date'].min(), rec['date'].max()
    _rec_half_px = marker_half_px(4.5, symbol='circle')
    axis_pad_rec = [axis_pad_entry(x_lo, x_hi, _rec_half_px, axis='xaxis'),
                    axis_pad_entry(x_lo, x_hi, _rec_half_px, axis='xaxis2')]

    # Absolute-pace axis: nice ticks over the central 99.8% of recovery paces,
    # clamped to a sane [3:00, 12:00] window. target=7 → the former 30s spacing
    # over Max's span, adapting the interval to any profile's range. The fast
    # bound also takes in the 5K-fitness line's peak (faster than any recovery
    # run) so that reference curve never clips off the top of the panel.
    _llo = max(180.0, min(float(rec['pace_for_fit'].quantile(0.001)),
                          float(cs_for_plot['cs_pace_sec'].min())))
    _lhi = min(720.0, float(rec['pace_for_fit'].quantile(0.999)))
    left_ticks, _ = nice_time_ticks(_llo, _lhi, target=7)
    left_y_lo, left_y_hi = left_ticks[0], left_ticks[-1]

    # Residual axis (signed): same density via the shared interval picker.
    res_lo = float(rec['residual_raw'].quantile(0.005))
    res_hi = float(rec['residual_raw'].quantile(0.995))
    _riv = nice_time_interval(res_lo, res_hi, target=7)
    right_ticks, _ = time_ticks_at_interval(res_lo, res_hi, _riv)
    right_ticks = [int(round(t)) for t in right_ticks]
    right_y_lo, right_y_hi = right_ticks[0], right_ticks[-1]

    for col in (1, 2):
        fig.update_xaxes(**yearly_x_axis_kwargs(x_lo, x_hi),
                          row=1, col=col)

    fig.update_yaxes(tickvals=left_ticks,
                      ticktext=[sec_to_mss(t) for t in left_ticks],
                      range=[left_y_hi, left_y_lo],
                      title_text='Recovery pace (per mile)',
                      gridcolor=GRID,
                      row=1, col=1)
    fig.update_yaxes(tickvals=right_ticks,
                      ticktext=[signed_sec(t) if t != 0 else '0' for t in right_ticks],
                      range=[right_y_hi, right_y_lo],
                      title_text='Residual (sec/mi above 5K fitness)',
                      gridcolor=GRID,
                      zeroline=False,
                      row=1, col=2)

    apply_default_layout(
        fig,
        font=dict(color=FG, size=12),
        margin=dict(l=70, r=300, t=40, b=28),
        hoverlabel=dict(bgcolor='#222', bordercolor='#555',
                        font=dict(color=FG, size=12)),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.005, xanchor='left',
            y=0.10, yanchor='top',  # just below the colorbar (which ends at y≈0.15)
            bgcolor='rgba(26,26,26,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color=FG, size=11),
            itemsizing='constant',
        ),
        coloraxis=dict(
            # Match the Running Log conditional formatting rule (column G temp):
            # -10°C #00B0F0 (light blue) → 22°C #92D050 (green) → 40°C #FF0000 (red)
            colorscale=[
                [0.00, 'rgb(0,176,240)'],     # -10°C
                [0.64, 'rgb(146,208,80)'],    # 22°C  ((22−(−10))/(40−(−10)) = 0.64)
                [1.00, 'rgb(255,0,0)'],       # 40°C
            ],
            cmin=-10, cmax=40,
            colorbar=dict(
                title=dict(text='°C', side='right',
                            font=dict(color=FG, size=11)),
                orientation='v', x=1.005, xanchor='left',
                y=0.45, yanchor='top',  # below the toolbar's typical footprint
                len=0.30, thickness=10,
                tickvals=[-10, 0, 10, 22, 30, 40],
                tickfont=dict(color=FG, size=10),
                outlinewidth=0,
            ),
        ),
    )
    fig.update_annotations(font=dict(color=FG, size=14))

    # Mobile reshape: stack the two panels (2x1) on a scrollable page instead
    # of squeezing them side by side. Traces untouched — same two axis pairs
    # in the same fill order. Margins, legend, colorbar and the #norm-filter
    # rail all stay at their desktop spots on the right — panels are narrower
    # for it, by design, so mobile reads like desktop. The scratch fig omits
    # shared_xaxes on purpose: a purely geometric patch, and the top panel's
    # own tick labels read as a useful separator between panels.
    M_ROW_H, M_ROW_GAP = 300, 64
    mob_h = 2 * M_ROW_H + M_ROW_GAP + 40 + 28  # t/b margins as desktop
    fig_m = make_subplots(rows=2, cols=1,
                          subplot_titles=('Absolute pace',
                                          'Residual vs 5K fitness'),
                          vertical_spacing=M_ROW_GAP / (mob_h - 40 - 28))
    assert_reshape_compatible(fig, fig_m)
    mobile_rec = MobileLayout(patch=reshape_patch(fig_m, height_px=mob_h))

    # ---------- spikeline tooltip payload ----------
    # Per-day arrays the JS uses for the trend section, plus a sorted list
    # of sessions for nearest-run lookup. Both smooth and snap modes read
    # from the same payload — the only difference is the snap path uses
    # the snapped point's day directly (so trend rows reflect that exact
    # day) and skips the date header + "Nearest run" caption.
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start = x_lo.normalize()
    plot_end   = x_hi.normalize()
    grid_dates = pd.date_range(plot_start, plot_end, freq='D')
    grid_day_idx = ((grid_dates - js_epoch).days).astype(int)

    def _interp_to_grid(xs_ms, ys):
        if len(xs_ms) == 0:
            return [None] * len(grid_day_idx)
        xs_day = xs_ms.astype(np.float64) / 86_400_000.0
        out = np.interp(grid_day_idx.astype(float), xs_day, ys,
                        left=np.nan, right=np.nan)
        return [None if np.isnan(v) else round(float(v), 2) for v in out]

    cs_pace_per_day    = _interp_to_grid(
        np.array([d.value // 10**6 for d in cs_for_plot['date']], dtype=np.int64),
        cs_for_plot['cs_pace_sec'].values)
    trend_pace_per_day = _interp_to_grid(init_pace_x,  init_pace_y)
    trend_resid_per_day = _interp_to_grid(init_resid_x, init_resid_y)

    sessions = []
    for _, r in rec.iterrows():
        sessions.append({
            'day':   int((r['date'] - js_epoch).days),
            'resid': float(r['residual_raw']) if pd.notna(r['residual_raw']) else None,
            'html':  r['_hover'],
        })
    sessions.sort(key=lambda s: s['day'])

    payload = {
        'first_day':   int(grid_day_idx[0]),
        'cs_pace':     cs_pace_per_day,
        'trend_pace':  trend_pace_per_day,
        'trend_resid': trend_resid_per_day,
        'sessions':    sessions,
        'nearest_window_days': 60,
    }
    smooth_build_js = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.cs_pace.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function paceMSS(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    var mn = Math.floor(x / 60), sc = x % 60;
    return mn + ':' + (sc < 10 ? '0' : '') + sc;
  }
  function signedSec(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    if (x === 0) return '0';
    return (x > 0 ? '+' : '−') + Math.abs(x);
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  // Date header in both modes — in snap mode it identifies the snapped
  // point's day, in smooth mode it identifies the cursor's date.
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: trend info for the absolute-pace panel.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>5K fitness</span><b>' + paceMSS(P.cs_pace[idx]) + '/mi</b></div>';
  html += '<div class="tt-row"><span>Trend pace</span><b>' + paceMSS(P.trend_pace[idx]) + '/mi</b></div>';
  html += '</div>';

  // Section 2: residual + run details. In smooth mode, "Nearest run [+/-d]"
  // header points at a run within ±nearest_window_days; in snap mode the
  // section is unlabeled and references the snapped point directly.
  var run = null;
  var s = P.sessions;
  if (isSnap) {
    // Find session at the snapped day (recovery has at most one per day).
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    if (s[lo] && s[lo].day === day) run = s[lo];
  } else if (s.length) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    var cands = [s[lo]];
    if (lo > 0) cands.push(s[lo - 1]);
    var best = null, bestAbs = 9999;
    for (var k = 0; k < cands.length; k++) {
      var ad = Math.abs(cands[k].day - day);
      if (ad < bestAbs) { bestAbs = ad; best = cands[k]; }
    }
    if (best && bestAbs <= P.nearest_window_days) run = best;
  }
  if (run) {
    var residIdx = run.day - P.first_day;
    var trResid = (residIdx >= 0 && residIdx < P.trend_resid.length)
                  ? P.trend_resid[residIdx] : null;
    html += '<div class="tt-section">';
    if (!isSnap) {
      var dd2 = run.day - day;
      var lbl = dd2 === 0 ? 'same day'
              : (dd2 > 0 ? '+' + dd2 + ' day' + (dd2 === 1 ? '' : 's')
                         :  dd2 + ' day' + (dd2 === -1 ? '' : 's'));
      html += '<div class="tt-section-title">Nearest run [' + lbl + ']</div>';
    }
    html += '<div class="tt-row"><span>Fitness residual</span><b>' + signedSec(run.resid) + ' sec/mi</b></div>';
    html += '<div class="tt-row"><span>Trend residual</span><b>' + signedSec(trResid) + ' sec/mi</b></div>';
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    first_day = int(grid_day_idx[0])
    last_day  = int(grid_day_idx[-1])

    out_path = os.path.join(args.out_dir, 'recovery_pace.html')
    render_plot(
        fig, out_path,
        title_slug='recovery_pace',
        page_title='Recovery',
        title='Recovery runs vs. race fitness',
        subtitle='Absolute pace and relative fitness signal, '
                 'controlling for combinations of factors',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=smooth_build_js,
            first_day=first_day,
            last_day=last_day,
        ),
        overlay_html=(
            widgets.js_globals({'TREND_SIGMA_DAYS': TREND_SMOOTH_SIGMA_DAYS})
            + '\n'
            + build_normalization_ui(
                betas, intercept, r2_detrended, r2_raw, res.n_fit,
                int(rec['is_bad_cond'].sum()),
                int(rec['is_partner_run'].sum()),
                int(rec['is_outlier_loo'].sum()),
                int(rec['is_pruned'].sum()),
                qualifying_routes, route_col_map, route_counts,
                global_mean_residual,
                available_norm=_available_norm_factors(rec, qualifying_routes))
        ),
        overlay_js_files=[_RECOVERY_JS],
        axis_pad=axis_pad_rec,
        mobile_layout=mobile_rec,
        extra_head_css=(
            # Suppress Plotly's built-in hover label — recovery has
            # hoverlabel-configured traces that would otherwise double
            # up with the smart spikeline tooltip. NOT global in
            # base.css since world_map relies on native hover.
            '.hovertext { display: none !important; }\n'
            # Mobile: the panel's desktop slot (inline top:48) collides with
            # the subtitle line, which runs the full width on a phone.
            # html.rp-mobile is the single mobile signal (see
            # _scaffold/mobile.js) — never a live viewport query.
            'html.rp-mobile #norm-filter { top: 64px !important; }'
        ),
    )
    print(f'\nWrote {out_path}')

    # Export route betas for downstream consumption (e.g. long-run TQ
    # corrections). The route betas are recovery-only, so they're effort-
    # uncontaminated and represent the pace cost of each route at easy
    # effort. The elev/terrain model parameters from the cross-route
    # exploration are NOT exported here since they're a separate analysis.
    # Currently no downstream code reads this file, so it's gated behind
    # --diagnostics and routed to output/debug/.
    if args.diagnostics:
        suffix = f'_{args.tag}' if args.tag else ''
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        betas_csv = DEBUG_DIR / f'route_betas{suffix}.csv'
        betas_df = pd.DataFrame([
            {'route': r, 'n': int(route_counts[r]), 'beta_sec_per_mi': betas[route_col_map[r]]}
            for r in qualifying_routes
        ]).sort_values('beta_sec_per_mi')
        betas_df.to_csv(betas_csv, index=False)
        print(f'Wrote {betas_csv} ({len(betas_df)} routes)')


def _available_norm_factors(rec, qualifying_routes):
    """Which normalization factors actually have data to subtract.

    A factor is available when its per-row contribution column carries a finite
    non-zero value somewhere — i.e. toggling it would actually move points. This
    is what drives the data-aware UI: a watch profile with no partners, one
    location, etc. simply doesn't show checkboxes that would do nothing.
    """
    def has_signal(col):
        if col not in rec.columns:
            return False
        v = rec[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return bool(len(v) and np.any(np.abs(v) > 1e-9))
    return {
        'era':           has_signal('contrib_era'),
        'temp':          has_signal('contrib_temp'),
        'elevation':     has_signal('contrib_elevation'),
        'terrain':       has_signal('contrib_terrain'),
        'altitude':      has_signal('contrib_altitude'),
        'recent_effort': has_signal('contrib_quality'),
        'time_of_day':   has_signal('contrib_tod'),
        'wind':          has_signal('contrib_wind'),
    }


def build_normalization_ui(betas, intercept, r2_detrended, r2_raw, n_fit,
                            n_bad_cond, n_partner_runs, n_outliers, n_pruned_unique,
                            qualifying_routes, route_col_map,
                            route_counts, global_mean_residual,
                            available_norm=None):
    """Right-sidebar UI. 4 normalization toggles + 3 visibility toggles.

    Normalization order MUST match customdata channel order:
    0=temp, 1=route, 2=recent_effort, 3=time_of_day, 4=era

    Era applies ONLY to residual panel (not absolute pace). Other
    normalization factors apply to both panels. The visibility section
    has three independent toggles (Hide bad conditions, Hide non-solo,
    Hide outliers) each with its own count and a shared All/None group.
    Hidden points have y set to null on the trace so hover events don't
    fire on them; the rolling-mean trend line ignores hidden points.
    """
    q_betas = [betas['fat_marathon'], betas['fat_race_short']]

    # Only show factors/filters that have data — a checkbox that can't change
    # anything (no partners logged, a single location, no bad-condition days)
    # is omitted rather than shown dead.
    av = available_norm or {}
    factors_norm = [(k, label) for (k, label) in (
        ('era',           'Era trend'),
        ('temp',          'Temperature'),
        ('elevation',     'Elevation (net)'),
        ('terrain',       'Terrain'),
        ('altitude',      'Altitude'),
        ('recent_effort', 'Recent race'),
        ('time_of_day',   'Time of day'),
        ('wind',          'Wind'),
    ) if av.get(k, True)]
    norm_rows = widgets.checkbox_rows(
        factors_norm, data_attr='factor', checked=False
    ).replace('data-factor=', 'data-mode="norm" data-factor=')

    filter_items = [(k, label, f'({c})') for (k, label, c) in (
        ('hide_bad_cond', 'Hide bad conditions', n_bad_cond),
        ('hide_partner',  'Hide partner runs',   n_partner_runs),
        ('hide_outlier',  'Hide outliers',       n_outliers),
    ) if c > 0]
    filter_rows = widgets.checkbox_rows(
        filter_items, data_attr='factor', checked=False
    ).replace('data-factor=', 'data-mode="filter" data-factor=')

    # Coefficient details, only for factors that are actually shown.
    detail_rows = []
    if av.get('era', True):
        detail_rows.append(widgets.detail_row(
            'Era trend',
            f': flattens year-over-year changes, on residual panel only'))
    if av.get('temp', True):
        detail_rows.append(widgets.detail_row(
            'Temperature',
            f': β = {betas["temp_centered"]:+.2f} s/mi per °C above '
            f'{int(TEMP_HEAT_ONSET_C)}'))
    if av.get('elevation', True):
        from src.shared.elevation_cost import engine_params
        ep = engine_params()

        def _rate(base, slope):
            sign = '+' if slope >= 0 else '−'
            return (f'{base * 100:.4f} {sign} {abs(slope) * 100:.4f} '
                    f'per 1% grade')

        def _slope(v):
            return f'{"+" if v >= 0 else "−"}{abs(v) * 100:.4f} per 1% grade'

        # One inline block, like every other coefficient row here — base
        # value with the grade slope in parentheses, same shape both sides.
        detail_rows.append(widgets.detail_row(
            'Elevation',
            f': β = {ep["c0"] * 100:.4f}% of pace per ft/mi gained '
            f'({_slope(ep["c1"])}), {ep["b0"] * 100:.4f}% per ft/mi lost '
            f'({_slope(ep["b1"])})'))
    if av.get('terrain', True):
        detail_rows.append(widgets.detail_row(
            'Terrain',
            f': β = {betas.get("trail_frac", 0):+.1f} s/mi on unpaved '
            f'terrain'))
    if av.get('altitude', True):
        detail_rows.append(widgets.detail_row(
            'Altitude',
            f': β = {betas.get("alt_kft", 0):+.2f} s/mi per 1000 ft above '
            f'{ALTITUDE_THRESHOLD_KFT:.0f}k'))
    if av.get('recent_effort', True):
        detail_rows.append(widgets.detail_row(
            'Recent race ',
            f'(exponential decay): marathon {q_betas[0]:+.1f} '
            f'(τ={FATIGUE_TAU_DAYS["marathon"]:.0f}d), '
            f'short race {q_betas[1]:+.1f} '
            f'(τ={FATIGUE_TAU_DAYS["race_short"]:.0f}d)'))
    if av.get('time_of_day', True):
        detail_rows.append(widgets.detail_row(
            'Time of day',
            f': β = {betas["tod_is_pm"]:+.2f} s/mi for afternoon/late '
            '(vs early/morning)'))
    if av.get('wind', True):
        detail_rows.append(widgets.detail_row(
            'Wind',
            f': β = {betas.get("wind_mph", 0):+.2f} sec/mi per mph '))

    details_body = (
        ''.join(detail_rows)
        + widgets.noteworthy(
            'Sleep cycles, run distance, shoes, weather and humidity were tested and '
            'excluded as non-factors.')
    )

    parts = []
    if factors_norm:
        parts += [
            widgets.title('Normalize'),
            widgets.subtitle("Subtract each factor's modeled contribution."),
            widgets.button_row([('nf-norm-all', 'All'), ('nf-norm-none', 'None')]),
            norm_rows,
        ]
    if filter_items:
        if parts:
            parts.append(widgets.divider())
        parts += [
            widgets.title('Hide from chart'),
            widgets.button_row([('nf-hide-all', 'All'), ('nf-hide-none', 'None')]),
            filter_rows,
        ]
    body = (
        ''.join(parts)
        + '\n<details><summary>Coefficient details</summary>\n'
        + details_body
        + '\n</details>\n'
        + widgets.stats_footer([
            f'n={n_fit:,} in fit; {n_pruned_unique} excluded (classes overlap)',
            f'R² = {r2_detrended:.3f} (factors only on detrended)',
            f'R² = {r2_raw:.3f} (era + factors on raw)',
            f'Trend: gaussian kernel, σ={TREND_SMOOTH_SIGMA_DAYS}d',
        ])
    )

    return widgets.sidebar(
        'norm-filter',
        body=body,
        anchor='',
        width_px=240,
        top_px=48,
        right_px=12,
    )


if __name__ == '__main__':
    main()
