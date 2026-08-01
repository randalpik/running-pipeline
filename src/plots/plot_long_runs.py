"""
plot_long_runs.py — Qualitative "every long run" plot at absolute pace.

Shows every `run_type == 'long'` session — including those outside the TQ
model's [LONG_MIN_MINUTES, LONG_CEIL_MILES) slice — at absolute pace (no
5K-equivalent projection). Two CS-derived reference curves give the
equivalent half-marathon and marathon paces from the model: how fast a
given fitness predicts you could run those distances.

Displayed distance and pace are the WATCH-CORRECTED source of truth that
project_long_runs computed (calibrated watch measurements; route-era
deflation for pre-watch mislogged routes); the originally-logged value
appears in the tooltip as a secondary line only when it differs materially
(≥0.2 mi); otherwise the two collapse into a single corrected row. The
Normalize toggle (off by default) subtracts route and training-state effects
to the equivalent flat / sea-level pace. The Show-tags toggle (off by default —
the distance gradient is the primary encoding) overlays halo rings: light
blue = snow, yellow = partner-paced (both excluded from Training, matching the
workout/recovery exclusion methodology), gray = watch-enriched. Corrections
are display/projection-side only — the log columns are never rewritten.

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
from src.shared.units import METERS_PER_MILE
from src.shared.plot_window import training_floor, clip_to_daily_floor, axis_pad_entry
from src.shared.workouts import load_cs, project_long_runs, _legacy_verified_dates
from src.shared.cs_projection import load_cs_outputs
from src.shared.performance_frontier import (standard_demos, build_frontier,
                                              frontier_at_anchor)
from src.shared.recovery_model import transferable_contributions
from src.shared.long_run_model import (fit_long_run_model, load_quality_dates,
                                       MIN_COV_N)
from src.plotting import widgets
from src.plotting import (render_plot, CursorTooltip, MobileLayout,
                            apply_default_layout,
                            right_margin_for_anchored_box, route_paren,
                            sec_to_mss, GRID, CAT_COLORS,
                            FRONTIER_LINE, TAG_COLORS,
                            rgba, yearly_x_axis_kwargs, nice_time_ticks,
                            marker_half_px, tt_kv, tt_title, long_run_lines)

# Width of the distance-gradient box (#lr-gradient); also used to size margin.r.
# Holds a 160px gradient bar with 10px horizontal padding + 1px border per side.
GRADIENT_BOX_WIDTH = 182


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
FRONTIER_LINE_HM = rgba(FRONTIER_LINE, 0.55)


def _y_safe(arr):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else float(v) for v in arr]


# Correction-tag halo rings — same construction (and 'enriched' gray) as the
# Workouts plot's condition tags. Marker is 8px + 0.5px outline here: outer
# edge 8/2 + 0.25 = 4.25px, 1px halo gap → inner stroke edge 5.25px →
# size = 2 × (5.25 + 0.5) = 11.5.
TAG_RING_SIZE  = 11.5
TAG_RING_WIDTH = 1
LR_TAG_LEGEND = {
    'snow':               'Snow',
    'partners':           'Partner run',
    'enriched':           'Watch-enriched',
    'legacy':             'Hand-verified (legacy)',
}


def lr_tag(r):
    """Ring tag for a long run, or None. Same priority logic as the Workouts
    plot's session_tag: category exclusions (snow, partner-paced — both
    dropped from Training, matching the workout/recovery methodology) win,
    then the informational 'enriched' tag: the projection used calibrated watch
    values (the size of the adjustment is visible via the Watch-correction
    toggle). Rule-corrected pre-watch days are unringed — the logged-vs-corrected
    tooltip rows already convey the correction. Out-of-slice runs stay unringed —
    showing them at absolute pace is this plot's whole point. (Long runs are not
    subject to the TQ outlier prune — that machinery was removed.)"""
    er = r.get('excluded_reason')
    if er == 'snow':
        return 'snow'
    if er == 'partners':
        return 'partners'
    if r.get('lr_watch'):
        return 'enriched'
    if r.get('legacy_verified'):
        return 'legacy'
    return None


LR_TAG_NOTE = {
    'snow':               'Excluded from Training: snow',
    'partners':           'Excluded from Training: partner-paced',
}


def long_run_hover(r):
    """Tooltip html for one long run. Title + distance/pace/paused lines come
    from the shared :func:`hover.long_run_lines` builder (the canonical form the
    Training and Fitness tabs reuse verbatim). Temp is omitted on log-only days
    (no watch sync → NaN). The descriptive Long Runs tab omits the pause-adjusted
    line (model-driven tabs add it)."""
    title = tt_title('Long run', r.get('display_name'), r.get('city_state'))
    tag = lr_tag(r)
    note = ''
    if tag in LR_TAG_NOTE:
        note = (f'<i style="color:{TAG_COLORS[tag]}">'
                f'{LR_TAG_NOTE[tag]}</i>')
    temp = (tt_kv('Temp', f"{r['temp_c']:.0f}°C")
            if pd.notna(r.get('temp_c')) else '')
    parts = [title, *long_run_lines(r), temp, note]
    return "<br>".join(p for p in parts if p)


def main():
    cs, epoch = load_cs()
    lr = project_long_runs(cs, epoch)
    # Hand-verified legacy long runs (snapshot `training`, injected inside
    # project_long_runs) get their own halo ring.
    lr['legacy_verified'] = (
        lr['date'].dt.strftime('%Y-%m-%d').isin(_legacy_verified_dates()))
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
    # training_floor(): axis + reference curves extend back to the first
    # legacy training entry when the profile has one, else = daily_floor.
    plot_floor = training_floor()
    daily_plot = clip_to_daily_floor(daily_summary, floor=plot_floor).copy()

    hm_dist_m, mar_dist_m = 21097.5, 42195.0

    # Frontier reference curves at HM and marathon (demonstrated capability;
    # src/shared/performance_frontier.py — corpus artifact written by
    # plot_training_quality, which runs earlier in run_plots.sh).
    _xc = load_cs_outputs(str(DATA_DIR), '')[3]
    front_demos = standard_demos(daily_summary, beta_long, d_thresh, _xc)
    frontier, _ = build_frontier(front_demos, pd.DatetimeIndex(daily_plot['date']),
                                 daily_plot['p5k_implied_min'])
    tf_hm  = frontier_at_anchor(frontier, daily_plot, hm_dist_m, beta_long, d_thresh)
    tf_mar = frontier_at_anchor(frontier, daily_plot, mar_dist_m, beta_long, d_thresh)
    front_hm_pace_min  = tf_hm  / (hm_dist_m  / METERS_PER_MILE) / 60.0
    front_mar_pace_min = tf_mar / (mar_dist_m / METERS_PER_MILE) / 60.0

    pace_min = lr['recovery_pace_sec_per_mi'].astype(float) / 60.0
    miles    = lr['miles'].astype(float)
    # Watch-corrected values (NaN where no correction exists). These are the
    # displayed source of truth: display pace/distance fall back to the
    # logged value only where no correction exists, so the gradient stays
    # continuous and every run plots.
    corr_pace_min = lr['corr_pace_sec_per_mi'].astype(float) / 60.0
    corr_miles    = lr['corr_miles'].astype(float).fillna(miles)
    disp_pace = corr_pace_min.fillna(pace_min)
    disp_miles = corr_miles

    # ---------- normalization (full physical + training-state) ----------
    # Normalize subtracts EVERYTHING project_long_runs / the TQ long-run model
    # apply to reach a run's flat / sea-level race-equivalent pace, so the
    # toggle on the Long Runs graph matches the Training page's decomposition
    # exactly (synced — same physical_route_betas + same long-run fit):
    #   PHYSICAL ROUTE (per-run, s/mi, computed in project_long_runs):
    #     grade (measured gain/loss through the elevation engine) +
    #     off-road footing + altitude — all pinned, the SAME constants the
    #     recovery model and the Training adjustments box show.
    #   TRAINING STATE (pooled-pinned betas, transferable_contributions):
    #     temperature + recent-race fatigue (TOD is dead on long runs) — now the
    #     SAME pooled constants the recovery model uses, not a separate fit.
    # Default is OFF → raw logged pace; the effort-level intercept is NEVER
    # applied. Adjustments cover every plotted run, including out-of-slice
    # ones the fit itself excludes.
    norm_adj = None
    lr_in = lr[lr['excluded_reason'].isna()]
    phys_adj = (lr['grade_cost_s_per_mi'].fillna(0).to_numpy()
                + lr['footing_cost_s_per_mi'].fillna(0).to_numpy()
                + lr['alt_cost_s_per_mi'].fillna(0).to_numpy())
    if len(lr_in) >= MIN_COV_N:
        quality_dates = load_quality_dates()
        _, lr_fit, _ = fit_long_run_model(lr_in.copy(), quality_dates)
        state_adj = (transferable_contributions(lr, lr_fit.cov_coefs, quality_dates,
                                                 lr_fit.temp_ref)
                     if lr_fit.cov_coefs else np.zeros(len(lr)))
        adj = phys_adj + state_adj
        # Omit a dead checkbox (profile where every adjustment is ~0).
        if np.abs(adj).max() > 0.05:
            norm_adj = adj

    # ---------- figure ----------
    fig = go.Figure()

    # Frontier curves only (the gold CS pair was removed June 2026):
    # marathon bright purple, HM faint purple — "fastest I could physically
    # race this distance that day". HM is added (and listed) first so the
    # legend order matches both the graph (the faster HM line sits higher on
    # the inverted pace axis) and the tooltip (HM above marathon). The opaque
    # marathon line is added last so it draws on top of the faint HM line.
    fig.add_trace(go.Scatter(
        x=daily_plot['date'], y=_y_safe(front_hm_pace_min),
        mode='lines', name='Frontier half-marathon pace',
        line=dict(color=FRONTIER_LINE_HM, width=2.0),
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=daily_plot['date'], y=_y_safe(front_mar_pace_min),
        mode='lines', name='Frontier marathon pace',
        line=dict(color=FRONTIER_LINE, width=2.2),
        hoverinfo='skip',
    ))

    # Long-run markers: single trace, continuous purple gradient by distance.
    # y/color are the watch-corrected display (the source of truth); meta.disp_y
    # carries that base so plot_long_runs.js can re-derive the Normalize /
    # Adjust-for-pauses toggles without recomputing. Both toggles ship OFF, so
    # the initial y is the raw corrected base; toggling re-derives from disp_y.
    disp_base_y = _y_safe(disp_pace.values)
    init_y = disp_base_y
    cd = [long_run_hover(r) for _, r in lr.iterrows()]
    fig.add_trace(go.Scatter(
        x=lr['date'], y=init_y,
        mode='markers', name=f'Long runs (n={len(lr)})',
        marker=dict(
            color=disp_miles.values, size=8,
            colorscale=LR_GRADIENT,
            cmin=float(disp_miles.min()), cmax=float(disp_miles.max()),
            showscale=False,
            line=dict(color='rgba(255,255,255,0.4)', width=0.5),
            opacity=0.9,
        ),
        customdata=cd,
        hoverinfo='skip',
        meta={'role': 'long_runs',
              'snap_eligible': True,
              'disp_y': disp_base_y},
    ))

    # Condition-tag halo rings (same construction as the Workouts plot).
    # meta.idx maps ring points back to marker-trace positions so the JS
    # toggles can move the rings with the markers. Rings ship VISIBLE but
    # parked at an off-axis sentinel (1900 — same trick as the race plots'
    # legend sentinels; empty-data traces can suppress legend entries):
    # the Tags legend group therefore exists from load with constant size,
    # so toggling Show-tags never reflows the legend, the gradient sidebar
    # anchored below it, or the plot margin. The JS toggle swaps the ring
    # DATA between sentinel and the real points, never trace visibility.
    tags = [lr_tag(r) for _, r in lr.iterrows()]
    has_tags = any(t for t in tags)
    dates = lr['date'].tolist()
    sentinel_x = [pd.Timestamp('1900-01-01')]
    sentinel_y = [6.0]
    for tag, label in LR_TAG_LEGEND.items():
        idx = [i for i, t in enumerate(tags) if t == tag]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=sentinel_x, y=sentinel_y,
            mode='markers', name=f'{label} (n={len(idx)})',
            marker=dict(symbol='circle', size=TAG_RING_SIZE,
                        color='rgba(0,0,0,0)',
                        line=dict(width=TAG_RING_WIDTH, color=TAG_COLORS[tag])),
            hoverinfo='skip',
            legendgroup='tags', legendgrouptitle_text='Tags',
            meta={'role': 'lr_ring', 'idx': idx,
                  'x_real': [str(pd.Timestamp(dates[i]).date()) for i in idx]},
        ))

    # ---------- layout ----------
    # Absolute pace axis (descending = faster up): closest 30s (0.5 min/mi)
    # marks enclosing all plotted data — the two CS reference curves and the
    # long-run markers. Data-driven so it fits each profile (Max's data already
    # spans the former fixed 4:30–8:30 bounds, so his axis is unchanged).
    # Envelope must enclose both reachable views: the raw base and Normalize on
    # (subtracts cost → faster), so neither pushes a point off-axis.
    _base = disp_pace.to_numpy(dtype=float)
    _n = norm_adj / 60.0 if norm_adj is not None else 0.0
    _ys = np.concatenate([
        np.asarray(front_mar_pace_min, dtype=float),
        np.asarray(front_hm_pace_min, dtype=float),
        _base,
        _base - _n,
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
    lr_lo, lr_hi = plot_floor, lr['date'].max()
    axis_pad_lr = [axis_pad_entry(lr_lo, lr_hi, marker_half_px(8, symbol='circle', line_width=0.5))]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70,
                    r=right_margin_for_anchored_box(GRADIENT_BOX_WIDTH, legend_min_px=200),
                    b=28),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        xaxis=yearly_x_axis_kwargs(lr_lo, lr_hi),
        yaxis=dict(title='Pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridcolor=GRID, zeroline=False),
    )

    # ---------- distance-gradient legend overlay ----------
    # No native colorbar (showscale=False) since the encoding is
    # "intuitive: bigger = darker"; provide a small textual key by the legend
    # for explicit distance anchors.
    miles_min_int = int(np.floor(disp_miles.min()))
    miles_max_int = int(np.ceil(disp_miles.max()))
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
    tags_section = ''
    if has_tags:
        tags_section = (
            widgets.divider()
            + widgets.checkbox_rows([('tags', 'Show tags')],
                                    data_attr='lrtags', checked=False)
        )
    norm_section = ''
    if norm_adj is not None:
        norm_section = (
            widgets.divider()
            + widgets.checkbox_rows([('normalize', 'Normalize')],
                                    data_attr='lrnorm', checked=False)
            + widgets.subtitle('Subtract route and training effects to show the equivalent flat, paved, non-fatigued pace at sea level in good conditions.')
        )
    overlay_html = widgets.sidebar(
        'lr-gradient',
        body=gradient_bar + norm_section + tags_section,
        compact=True, width_px=GRADIENT_BOX_WIDTH,
    )
    if norm_adj is not None:
        overlay_html = widgets.js_globals(
            {'LR_NORM_ADJ': [round(float(v), 2) for v in norm_adj]}) + '\n' + overlay_html

    # ---------- cursor-tooltip payload ----------
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start, plot_end = lr_lo, lr_hi
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    # Per-day HM and marathon FRONTIER pace (min/mi) for the tooltip.
    days_2016 = (all_days - epoch).days.astype(float).values
    daily_days = (daily_plot['date'] - epoch).dt.days.astype(float).values
    hm_per_day  = np.interp(days_2016, daily_days, front_hm_pace_min)
    mar_per_day = np.interp(days_2016, daily_days, front_mar_pace_min)

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
  html += '<div class="tt-row"><span>Frontier half marathon pace</span><b>' + fmtMin(P.hm_pace[idx]) + '/mi</b></div>';
  html += '<div class="tt-row"><span>Frontier marathon pace</span><b>' + fmtMin(P.mar_pace[idx]) + '/mi</b></div>';
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
    // Watch-corrected figures are the displayed source of truth and are
    // baked into both the snap-mode pointHtml (customdata) and run.html.
    html += (isSnap && pointHtml ? pointHtml : (run ? run.html : ''));
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
        # Mobile: cap the legend at half the plot height (internal scrollbar)
        # so the legend-anchored #lr-gradient box below splits the short
        # right rail with it roughly evenly.
        mobile_layout=MobileLayout(patch={'legend.maxheight': 0.5},
                                   scroll=False),
    )
    print(f'Wrote {OUT_HTML}  ({len(lr)} long runs, '
          f'miles {miles.min():.1f}–{miles.max():.1f})')


if __name__ == '__main__':
    main()
