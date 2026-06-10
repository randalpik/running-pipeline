"""Annual plot: year-over-year mileage on a shared Jan-Dec calendar axis.

Every year with daily.csv data becomes one trace on a yearless calendar
x-axis (each date is mapped onto dummy leap year 2000 so Feb 29 has a
slot and month boundaries align across years). A toggle switches between:

  - Cumulative: per-year running total of miles, one dot per run day.
  - Average:    28-day centered moving average of daily miles, drawn as
                a line per year. The average is computed contiguously
                across the whole logging span (zero-filled calendar, so
                rest days count as 0) and then sliced into years — the
                first/last ~2 weeks of interior years incorporate the
                adjacent year's training, by design.

Year colors are evenly spaced hues around the full color wheel,
chronological from hue 0, at fixed saturation/lightness tuned for the
dark theme.

Tooltip (smooth cursor mode, no snap): every visible year's value at the
hovered calendar day, sorted by value descending. Cumulative values are
carry-forward filled in the payload so a year without a run on the exact
day still shows its most recent total.
"""
import argparse
import colorsys
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import daily_floor
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                          FG_DIM, GRID)
from src.plotting import widgets

DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT = str(OUTPUT_DIR)

_PLOTS_DIR = Path(__file__).resolve().parent
_ANNUAL_JS = _PLOTS_DIR / 'make_annual_plot.js'
_ANNUAL_CSS = _PLOTS_DIR / 'make_annual_plot.css'

MA_WINDOW_DAYS = 28

# All years overlay on dummy leap year 2000; payload arrays are indexed by
# day-of-year within it (0 = Jan 1 ... 59 = Feb 29 ... 365 = Dec 31).
DUMMY_YEAR = 2000
N_DOY = 366
DUMMY_DAY0 = (pd.Timestamp(DUMMY_YEAR, 1, 1) - pd.Timestamp('1970-01-01')).days

MARKER_SIZE = 7     # cumulative dots — big enough to overlap at full-year zoom
LINE_WIDTH = 2      # average lines

# Custom HTML legend in the reserved right margin (the geography pattern:
# toggle fixed at top:56, legend fixed below it at top:110). Width here
# matches the CSS width in make_annual_plot.css.
LEGEND_WIDTH_PX = 90

PALETTE_SAT = 0.62
PALETTE_LIGHT = 0.60


def annual_palette(n):
    """n hues evenly spaced around the full color wheel, chronological
    from hue 0, at fixed S/L readable on the dark background."""
    out = []
    for i in range(n):
        h = i / max(1, n)
        r, g, b = colorsys.hls_to_rgb(h, PALETTE_LIGHT, PALETTE_SAT)
        out.append('#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255)))
    return out


def doy_index(ts):
    """Day-of-year index in the dummy leap year (0..365). Non-leap years
    skip 59 (Feb 29); leap arithmetic comes from the 2000 calendar."""
    return (pd.Timestamp(DUMMY_YEAR, ts.month, ts.day)
            - pd.Timestamp(DUMMY_YEAR, 1, 1)).days


def dummy_date(ts):
    return f'{DUMMY_YEAR}-{ts.month:02d}-{ts.day:02d}'


def build_cumulative(miles_by_date, years):
    """Per-year cumulative miles.

    Returns (cum_x, cum_y, cum_doy): marker coordinates (one per run day,
    x on the dummy calendar) plus a per-doy carry-forward array for the
    tooltip — every day from a year's first run onward holds the most
    recent total. Non-final years extend their final total to Dec 31;
    the final (possibly partial) year stays None past its last logged
    date so the tooltip drops it there.
    """
    cum_x, cum_y, cum_doy = {}, {}, {}
    for y in years:
        season = miles_by_date[miles_by_date.index.year == y].cumsum()
        cum_x[y] = [dummy_date(d) for d in season.index]
        cum_y[y] = [round(float(v), 1) for v in season.values]
        arr = [None] * N_DOY
        for d, v in zip(season.index, cum_y[y]):
            arr[doy_index(d)] = v
        fill_end = N_DOY if y != years[-1] else doy_index(season.index[-1]) + 1
        prev = None
        for k in range(fill_end):
            if arr[k] is None:
                arr[k] = prev
            else:
                prev = arr[k]
        cum_doy[y] = arr
    return cum_x, cum_y, cum_doy


def build_average(miles_by_date, years):
    """Per-year slices of the contiguous 28-day centered moving average.

    The series is zero-filled onto a complete daily calendar first
    (daily.csv prunes rest days), so interior year boundaries see full
    windows; only the global edges of the span are partial-window
    (min_periods=1, matching qualitative_trends' envelope edges).
    """
    cal = pd.date_range(miles_by_date.index.min(), miles_by_date.index.max(),
                        freq='D')
    full = miles_by_date.reindex(cal, fill_value=0.0)
    ma = full.rolling(MA_WINDOW_DAYS, center=True, min_periods=1).mean()

    avg_doy = {y: [None] * N_DOY for y in years}
    for d, v in ma.items():
        avg_doy[d.year][doy_index(d)] = round(float(v), 2)

    avg_x, avg_y = {}, {}
    for y in years:
        seg = ma[ma.index.year == y]
        avg_x[y] = [dummy_date(d) for d in seg.index]
        avg_y[y] = [round(float(v), 2) for v in seg.values]
    return avg_x, avg_y, avg_doy


def build_legend_html(years, colors):
    """Custom HTML legend: one click-to-hide row per year, positioned by
    make_annual_plot.css below the toggle bar (the geography pattern)."""
    parts = ['<div id="annual-legend">']
    for i, y in enumerate(years):
        parts.append(
            f'<div class="legend-item" data-trace-idx="{i}">'
            f'<span class="legend-box" style="background:{colors[i]}"></span>'
            f'<span class="legend-name">{y}</span></div>')
    parts.append('</div>')
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily', default=DEFAULT_DAILY)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.daily, parse_dates=['date'])
    df = df[df['date'] >= daily_floor()].copy()
    df = df.dropna(subset=['miles'])
    df['date'] = df['date'].dt.normalize()
    miles_by_date = df.groupby('date')['miles'].sum().sort_index()

    years = sorted(miles_by_date.index.year.unique().tolist())
    colors = annual_palette(len(years))

    cum_x, cum_y, cum_doy = build_cumulative(miles_by_date, years)
    avg_x, avg_y, avg_doy = build_average(miles_by_date, years)

    fig = go.Figure()
    for i, y in enumerate(years):
        # Initial mode is Cumulative (markers); the toggle restyles
        # x/y/mode from meta. line.color is preset so switching to line
        # mode picks up the year's hue without a restyle of style props.
        fig.add_trace(go.Scatter(
            name=str(y),
            x=cum_x[y], y=cum_y[y],
            mode='markers',
            marker=dict(size=MARKER_SIZE, color=colors[i]),
            line=dict(color=colors[i], width=LINE_WIDTH),
            hoverinfo='skip',
            meta={'x_cum': cum_x[y], 'y_cum': cum_y[y],
                  'x_avg': avg_x[y], 'y_avg': avg_y[y]},
        ))

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70, r=LEGEND_WIDTH_PX + 40, b=56),
        hovermode=False,
        showlegend=False,
        xaxis=dict(
            type='date',
            # 1-day pad each side so Jan 1 / Dec 31 dots aren't clipped.
            range=[f'{DUMMY_YEAR - 1}-12-31', f'{DUMMY_YEAR + 1}-01-01'],
            dtick='M1',
            tickformat='%b',
            ticklabelmode='period',   # center month names within months
            tickfont=dict(color=FG_DIM, size=11),
            gridcolor=GRID,
        ),
        yaxis=dict(
            title=dict(text='Cumulative miles',
                       font=dict(color=FG_DIM, size=11)),
            rangemode='tozero',
            tickfont=dict(color=FG_DIM, size=11),
            gridcolor=GRID, zerolinecolor=GRID,
        ),
    )

    payload = {
        'doy0': DUMMY_DAY0,
        'years': years,
        'colors': colors,
        'cum': [cum_doy[y] for y in years],
        'avg': [avg_doy[y] for y in years],
    }

    toggle_html = widgets.toggle_bar(
        'annual-toggle',
        [('cum', 'Cumulative'), ('avg', 'Average')],
        default_id='cum',
    )
    # The toggle bar's default position (top:14px) is too high — this plot
    # has the title bar above it, so push the toggle down to 56px. The
    # custom legend sits below it (top:110 in make_annual_plot.css).
    toggle_html = toggle_html.replace(
        'class="rp-toggle-bar"',
        'class="rp-toggle-bar" style="top: 56px"',
    )
    overlay_html = toggle_html + '\n' + build_legend_html(years, colors)

    out_path = os.path.join(args.out_dir, 'annual.html')
    render_plot(
        fig, out_path,
        title_slug='annual',
        page_title='Annual mileage',
        title='Annual Mileage',
        subtitle='Cumulative miles and 28-day average by calendar day, '
                 'per year',
        overlay_html=overlay_html,
        overlay_js_files=[_ANNUAL_JS],
        # Rows are just "swatch year | value" — release base.css's 200px
        # min-width so the box hugs its content.
        extra_head_css='.rp-tooltip { min-width: 0; }',
        extra_head_css_files=[_ANNUAL_CSS],
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=_BUILD_JS,
            first_day=DUMMY_DAY0,
            last_day=DUMMY_DAY0 + N_DOY - 1,
            spike_snap_day=True,
        ),
    )
    print(f'wrote {out_path}')

    print(f'\n{len(years)} years ({years[0]}-{years[-1]}):')
    for i, y in enumerate(years):
        avg_vals = [v for v in avg_doy[y] if v is not None]
        print(f'  {y}  {colors[i]}  total={cum_y[y][-1]:>7.1f} mi  '
              f'run-days={len(cum_y[y]):>3d}  '
              f'avg-ma min={min(avg_vals):.2f} max={max(avg_vals):.2f}')


_BUILD_JS = r"""
function buildTooltip(day) {
  var P = window.__TT_DATA;
  var doy = day - P.doy0;
  if (doy < 0 || doy >= 366) return '';

  var gd = document.querySelector('.plotly-graph-div');
  if (!gd || !gd.data) return '';
  // Mode is whatever the toggle last restyled the traces to.
  var isCum = String(gd.data[0].mode).indexOf('markers') !== -1;

  var dt = new Date(day * 86400000);
  var MON = ['Jan','Feb','Mar','Apr','May','Jun',
             'Jul','Aug','Sep','Oct','Nov','Dec'];
  var dateStr = MON[dt.getUTCMonth()] + ' ' + dt.getUTCDate();

  var rows = [];
  for (var i = 0; i < P.years.length; i++) {
    // Trace i is years[i] (added in order); drop legend-hidden years.
    var vis = gd.data[i] ? gd.data[i].visible : true;
    if (vis === false || vis === 'legendonly') continue;
    var arr = isCum ? P.cum[i] : P.avg[i];
    var v = arr ? arr[doy] : null;
    if (v === null || v === undefined) continue;
    rows.push({ year: P.years[i], val: v, color: P.colors[i] });
  }
  if (!rows.length) return '';
  rows.sort(function (a, b) { return b.val - a.val; });

  var html = '<div class="tt-date">' + dateStr + '</div>';
  for (var j = 0; j < rows.length; j++) {
    var r = rows[j];
    html += '<div class="tt-row">'
          +   '<span><span style="background:' + r.color
          +     ';display:inline-block;width:9px;height:9px;'
          +     'border-radius:2px;margin-right:6px"></span>'
          +     r.year + '</span>'
          +   '<span><b>' + (isCum ? r.val.toFixed(0) : r.val.toFixed(1))
          +     '</b> <span class="tt-mute">'
          +     (isCum ? 'mi' : 'mi/day') + '</span></span>'
          + '</div>';
  }
  return html;
}
"""


if __name__ == '__main__':
    main()
