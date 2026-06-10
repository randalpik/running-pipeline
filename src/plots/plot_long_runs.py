"""
plot_long_runs.py — Qualitative "every long run" plot at absolute pace.

Shows every `run_type == 'long'` session — including those outside the TQ
model's [LONG_MIN_MINUTES, LONG_CEIL_MILES) slice — at absolute pace (no 5K-equivalent
projection, no per-route correction). Two CS-derived reference curves give
the equivalent half-marathon and marathon paces from the model: how fast a
given fitness predicts you could run those distances.

Marker color encodes distance via a continuous lavender→deep-purple gradient,
bracketed at the dataset's miles min/max.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import daily_floor, axis_pad_entry
from src.shared.workouts import load_cs, project_long_runs
from src.shared.cs_projection import load_cs_outputs, cs_line_at_anchor
from src.shared.recovery_model import transferable_contributions
from src.shared.long_run_model import (fit_long_run_model, load_quality_dates,
                                       MIN_COV_N)
from src.plotting import widgets
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            right_margin_for_anchored_box, route_paren,
                            sec_to_mss, GRID, CAT_COLORS, CS_LINE, rgba,
                            yearly_x_axis_kwargs, nice_time_ticks, marker_half_px)

# Width of the distance-gradient box (#lr-gradient); also used to size margin.r.
# Holds a 160px gradient bar with 10px horizontal padding + 1px border per side.
GRADIENT_BOX_WIDTH = 182


OUTPUT_DIR.mkdir(exist_ok=True)
OUT_HTML = str(OUTPUT_DIR / 'long_runs.html')
_LR_JS = Path(__file__).resolve().parent / 'plot_long_runs.js'

# Distance gradient: 3-stop blue → purple → magenta for high contrast across
# the long-run distance range. cmin/cmax are computed from the dataset's
# miles min/max.
LR_GRAD_BLUE    = '#3498DB'  # bright blue   (short)
LR_GRAD_PURPLE  = '#8E44AD'  # vivid purple  (mid)
LR_GRAD_MAGENTA = '#E91E63'  # magenta       (long)
LR_GRADIENT = [[0.0, LR_GRAD_BLUE], [0.5, LR_GRAD_PURPLE], [1.0, LR_GRAD_MAGENTA]]

# CS reference curve colors. Marathon = full CS_LINE gold (darker because
# longer-distance fade pushes pace slower); HM = a lighter, semi-transparent
# version of the same orange/gold so both read as the same "CS" family.
CS_LINE_HM       = rgba('#ffb450', 0.55)
CS_LINE_MARATHON = CS_LINE


def _y_safe(arr):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else float(v) for v in arr]


def long_run_hover(r):
    title = f"Long run{route_paren(r.get('display_name'), r.get('city_state'))}"
    pace_sec = float(r['recovery_pace_sec_per_mi'])
    parts = [
        f"<b>{title}</b>",
        f"{r['miles']:.1f} mi @ {sec_to_mss(pace_sec)}/mi",
        f"<b>Temp:</b> {r['temp_c']:.0f}°C",
    ]
    return "<br>".join(parts)


def main():
    cs, epoch = load_cs()
    lr = project_long_runs(cs, epoch)
    # Drop implausibly-slow "long runs": these are trail runs / hikes the watch
    # recorded as ordinary runs (Trail Run not selected at start), not running
    # long runs. Real easy long runs for these athletes top out ~10 min/mi;
    # there's a clean gap before the artifacts (13–28 min/mi), so 12:00 cleanly
    # separates them.
    lr = lr[lr['recovery_pace_sec_per_mi'].astype(float) <= 12 * 60].copy()
    if lr.empty:
        raise SystemExit('No long runs to plot.')

    # CS reference curves at HM and marathon. cs_line_at_anchor returns total
    # time in seconds; convert to pace (min/mi) for display.
    daily_summary, beta_long, d_thresh, _ = load_cs_outputs(str(DATA_DIR), '')
    daily_plot = daily_summary[daily_summary['date'] >= daily_floor()].copy()

    hm_dist_m, mar_dist_m = 21097.5, 42195.0
    t_hm  = cs_line_at_anchor(daily_plot, hm_dist_m, beta_long, d_thresh)
    t_mar = cs_line_at_anchor(daily_plot, mar_dist_m, beta_long, d_thresh)
    hm_pace_min  = t_hm  / (hm_dist_m  / 1609.344) / 60.0
    mar_pace_min = t_mar / (mar_dist_m / 1609.344) / 60.0

    pace_min = lr['recovery_pace_sec_per_mi'].astype(float) / 60.0
    miles    = lr['miles'].astype(float)

    # ---------- normalization (long-run-sourced covariates) ----------
    # Betas come from the TQ long-run model (same fit plot_training_quality
    # renders): temperature fit on the in-slice long runs, race fatigue as
    # one fitted scale with the marathon/short contrast pinned from the
    # recovery fit (see long_run_model docstring). Recovery-sourced
    # amplitudes were tried and rejected (June 2026) — long runs are hit
    # ~2.3× harder than recovery runs — and recovery's time-of-day effect
    # is dead on long runs (see docs/training-quality-reference.md).
    # Adjustments apply to every plotted run, including out-of-slice ones
    # the fit itself excludes.
    norm_adj = None
    lr_in = lr[lr['excluded_reason'].isna()]
    if len(lr_in) >= MIN_COV_N:
        quality_dates = load_quality_dates()
        _, lr_fit, _ = fit_long_run_model(lr_in.copy(), quality_dates)
        if lr_fit.cov_coefs:
            adj = transferable_contributions(lr, lr_fit.cov_coefs, quality_dates)
            # Omit a dead checkbox (profile where every adjustment is ~0).
            if np.abs(adj).max() > 0.05:
                norm_adj = adj

    # ---------- figure ----------
    fig = go.Figure()

    # Marathon curve (darker gold, full saturation).
    fig.add_trace(go.Scatter(
        x=daily_plot['date'], y=_y_safe(mar_pace_min),
        mode='lines', name='CS marathon pace',
        line=dict(color=CS_LINE_MARATHON, width=2.2),
        hoverinfo='skip',
    ))
    # HM curve (lighter gold).
    fig.add_trace(go.Scatter(
        x=daily_plot['date'], y=_y_safe(hm_pace_min),
        mode='lines', name='CS half-marathon pace',
        line=dict(color=CS_LINE_HM, width=2.0),
        hoverinfo='skip',
    ))

    # Long-run markers: single trace, continuous purple gradient by distance.
    cd = [long_run_hover(r) for _, r in lr.iterrows()]
    fig.add_trace(go.Scatter(
        x=lr['date'], y=_y_safe(pace_min.values),
        mode='markers', name=f'Long runs (n={len(lr)})',
        marker=dict(
            color=miles.values, size=8,
            colorscale=LR_GRADIENT,
            cmin=float(miles.min()), cmax=float(miles.max()),
            showscale=False,
            line=dict(color='rgba(255,255,255,0.4)', width=0.5),
            opacity=0.9,
        ),
        customdata=cd,
        hoverinfo='skip',
        meta={'role': 'long_runs',
              'snap_eligible': True,
              'raw_y': _y_safe(pace_min.values)},
    ))

    # ---------- layout ----------
    # Absolute pace axis (descending = faster up): closest 30s (0.5 min/mi)
    # marks enclosing all plotted data — the two CS reference curves and the
    # long-run markers. Data-driven so it fits each profile (Max's data already
    # spans the former fixed 4:30–8:30 bounds, so his axis is unchanged).
    _ys = np.concatenate([
        np.asarray(mar_pace_min, dtype=float),
        np.asarray(hm_pace_min, dtype=float),
        np.asarray(pace_min.values, dtype=float),
    ])
    _ys = _ys[np.isfinite(_ys)]
    _lo, _hi = (float(_ys.min()), float(_ys.max())) if len(_ys) else (4.5, 8.5)
    # target=16 reproduces the former 15 s/mi spacing over Max's ~4:30-8:30
    # span; adapts the interval to any profile's range.
    _ticks_sec, ticktext = nice_time_ticks(_lo * 60, _hi * 60, target=16)
    tickvals = [t / 60.0 for t in _ticks_sec]
    y_min, y_max = tickvals[0], tickvals[-1]

    # Tight date range (first daily run → last long run); the half-marker pixel
    # gutter is added at render time by axis_pad.js and re-applied on resize.
    lr_lo, lr_hi = daily_floor(), lr['date'].max()
    axis_pad_lr = [axis_pad_entry(lr_lo, lr_hi, marker_half_px(8, symbol='circle', line_width=0.5))]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70,
                    r=right_margin_for_anchored_box(GRADIENT_BOX_WIDTH, legend_min_px=200),
                    b=60),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    font=dict(size=11)),
        xaxis=yearly_x_axis_kwargs(lr_lo, lr_hi, title='Date'),
        yaxis=dict(title='Pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridcolor=GRID, zeroline=False),
    )

    # ---------- distance-gradient legend overlay ----------
    # No native colorbar (showscale=False) since the encoding is
    # "intuitive: bigger = darker"; provide a small textual key by the legend
    # for explicit distance anchors.
    miles_min_int = int(np.floor(miles.min()))
    miles_max_int = int(np.ceil(miles.max()))
    # Distance gradient strip — three plot-specific bits (gradient colors,
    # min/max labels) are inlined; the panel chrome comes from .rp-sidebar
    # via widgets.sidebar(compact=True).
    gradient_bar = (
        '<div class="rp-sidebar-title" style="font-size:11px;'
        'margin-bottom:5px">Distance (mi)</div>'
        '<div style="width:100%;height:10px;border-radius:2px;'
        'margin-bottom:3px;background:linear-gradient(to right,'
        f'{LR_GRAD_BLUE},{LR_GRAD_PURPLE},{LR_GRAD_MAGENTA})"></div>'
        '<div style="display:flex;justify-content:space-between;'
        f'font-size:10.5px;color:#aaa"><span>{miles_min_int}</span>'
        f'<span>{miles_max_int}</span></div>'
    )
    norm_section = ''
    if norm_adj is not None:
        norm_section = (
            widgets.divider()
            + widgets.checkbox_rows([('normalize', 'Normalize')],
                                    data_attr='lrnorm', checked=False)
            + widgets.subtitle('Subtract modeled temperature and '
                               'recent-race effects (long-run-fit betas).')
        )
    overlay_html = widgets.sidebar(
        'lr-gradient', body=gradient_bar + norm_section,
        compact=True, width_px=GRADIENT_BOX_WIDTH,
    )
    if norm_adj is not None:
        overlay_html = (widgets.js_globals(
            {'LR_NORM_ADJ': [round(float(v), 2) for v in norm_adj]})
            + '\n' + overlay_html)

    # ---------- cursor-tooltip payload ----------
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start, plot_end = lr_lo, lr_hi
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    # Per-day HM and marathon CS pace (min/mi) for the smooth-mode tooltip.
    days_2016 = (all_days - epoch).days.astype(float).values
    daily_days = (daily_plot['date'] - epoch).dt.days.astype(float).values
    hm_per_day  = np.interp(days_2016, daily_days, hm_pace_min)
    mar_per_day = np.interp(days_2016, daily_days, mar_pace_min)

    sessions = []
    for i, (_, r) in enumerate(lr.iterrows()):
        s = {'day': int((r['date'] - js_epoch).days),
             'html': long_run_hover(r)}
        if norm_adj is not None:
            s['adj'] = round(float(norm_adj[i]), 1)
        sessions.append(s)
    sessions.sort(key=lambda s: s['day'])

    first_day = int((all_days[0]  - js_epoch).days)
    last_day  = int((all_days[-1] - js_epoch).days)

    payload = {
        'first_day': first_day,
        'hm_pace':  [round(float(v), 4) for v in hm_per_day],
        'mar_pace': [round(float(v), 4) for v in mar_per_day],
        'sessions': sessions,
        'nearest_window_days': 60,
    }

    build_js = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.hm_pace.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function fmtMin(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    var s = Math.round(v * 60);
    var m = Math.floor(s / 60), r = s % 60;
    return m + ':' + (r < 10 ? '0' : '') + r;
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>CS half-marathon</span><b>' + fmtMin(P.hm_pace[idx]) + '/mi</b></div>';
  html += '<div class="tt-row"><span>CS marathon</span><b>' + fmtMin(P.mar_pace[idx]) + '/mi</b></div>';
  html += '</div>';

  var run = null;
  var s = P.sessions;
  if (isSnap) {
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

  if (run || (isSnap && pointHtml)) {
    html += '<div class="tt-section">';
    if (!isSnap && run) {
      var dd2 = run.day - day;
      var lbl = dd2 === 0 ? 'same day'
              : (dd2 > 0 ? '+' + dd2 + ' day' + (dd2 === 1 ? '' : 's')
                         :  dd2 + ' day' + (dd2 === -1 ? '' : 's'));
      html += '<div class="tt-section-title">Nearest long run [' + lbl + ']</div>';
    }
    html += (isSnap && pointHtml ? pointHtml : run.html);
    if (window.__lrNormOn && run && run.adj != null) {
      // Shift applied to the point by the Normalize toggle: y = raw − adj.
      // One decimal: adjustments are small (often < 5 s/mi), whole seconds
      // would flatten most of them to +0.
      var sh = -run.adj;
      html += '<div class="tt-row"><span>Normalized adjustment</span><b>' +
              (sh >= 0 ? '+' : '−') + Math.abs(sh).toFixed(1) +
              ' s/mi</b></div>';
    }
    html += '</div>';
  }
  return html;
}
"""

    render_plot(
        fig, OUT_HTML,
        title_slug='long_runs',
        page_title='Long Runs',
        title='All long runs at absolute pace',
        subtitle='With marathon and half-marathon pace prediction lines',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
        overlay_html=overlay_html,
        overlay_js_files=[_LR_JS],
        axis_pad=axis_pad_lr,
    )
    print(f'Wrote {OUT_HTML}  ({len(lr)} long runs, '
          f'miles {miles.min():.1f}–{miles.max():.1f})')


if __name__ == '__main__':
    main()
