"""Plot the Bayesian CS timeline with credible-interval ribbons.

Reads the outputs of bayes_cs_fit.py (summary CSV, params CSV) and renders an
interactive HTML chart. Race diamonds are projected to 5K-equivalent pace
via the hyperbolic CS model: for each race at (d_race, t_race), the implied
CS is (d_race - D')/t_race, anchored to the model's posterior-median D' at
the race's date. The 5K-equivalent then evaluates as
t_5K = (5000 - D') * t_race / (d_race - D'). This preserves race-to-race
deviations (fast races project fast, slow races project slow) while handling
short distances correctly via D' rather than Riegel's piecewise exponent.

5K is chosen as the anchor distance because the vast majority of Max's races
are 5Ks (track 5Ks, XC 5Ks, road 5Ks across all eras).

XC race times are pre-corrected by dividing by (1 + xc_correction) before
projection — same correction the fit script applies before fitting — so XC
diamonds visually align with the CS line.

Default behavior: reads CS fit outputs from data/ (the bayes fit's default
output directory) and writes the HTML to output/.

Usage:
  python src/plots/bayes_cs_plot.py
  python src/plots/bayes_cs_plot.py --tag v9
  python src/plots/bayes_cs_plot.py --in-dir /path/to/outputs --tag v9

Required dependencies:
  pip install pandas numpy scipy plotly
"""
import argparse
import json
import os
import sys
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.cs_projection import load_cs_outputs, project_races_to_5k_pace


DEFAULT_IN_DIR = str(DATA_DIR)
DEFAULT_RACES  = str(DATA_DIR / 'races.csv')


def sec_to_mss(s):
    if s is None or pd.isna(s): return ''
    # Round to integer seconds first, then format. Avoids the "4:60" bug
    # where s=299.6 → m=4, round(s%60)=60.
    total = int(round(s))
    if total >= 3600:
        h = total // 3600
        m = (total % 3600) // 60
        ss = total % 60
        return f'{h}:{m:02d}:{ss:02d}'
    m = total // 60
    ss = total % 60
    return f'{m}:{ss:02d}'


def sec_to_mss_full(s):
    """Format seconds preserving subsecond precision for race times.
    Examples: 143.2 → '2:23.2', 18*60+38.5 → '18:38.5', 8756.4 → '2:25:56.4'.
    Trailing .0 is hidden when the value is an integer second.
    """
    if s is None or pd.isna(s): return ''
    s = float(s)
    # Round to tenths first to avoid float-representation issues
    # (e.g. 3599.95 - 3599 = 0.94999... in float).
    tenths_total = int(round(s * 10))
    whole = tenths_total // 10
    frac_tenths = tenths_total % 10  # 0-9
    if whole >= 3600:
        h = whole // 3600
        m = (whole % 3600) // 60
        ss = whole % 60
        body = f'{h}:{m:02d}:{ss:02d}'
    else:
        m = whole // 60
        ss = whole % 60
        body = f'{m}:{ss:02d}'
    if frac_tenths > 0:
        return f'{body}.{frac_tenths}'
    return body


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', default=DEFAULT_IN_DIR,
                   help=f'Directory containing fit outputs (default {DEFAULT_IN_DIR})')
    p.add_argument('--tag', default='',
                   help='Tag suffix matching the fit run (e.g. "v9")')
    p.add_argument('--races', default=DEFAULT_RACES,
                   help='Path to races.csv used by the fit')
    p.add_argument('--out', default=None,
                   help=f'Output HTML path (default: cs_timeline{{tag}}.html in {OUTPUT_DIR})')
    return p.parse_args()


def main():
    args = parse_args()
    suffix = f'_{args.tag}' if args.tag else ''

    # Load summary (already daily-interpolated) and bias parameters via shared module.
    daily_summary, beta_long_med, d_thresh_long, xc_correction = load_cs_outputs(
        args.in_dir, args.tag)
    print(f"Bias parameters: β_long={beta_long_med:.4f}, "
          f"xc_correction={xc_correction:.4f}, d_thresh={d_thresh_long:.0f}m")
    print(f"Summary: {len(daily_summary)} daily points, "
          f"{daily_summary['date'].iloc[0].date()} to {daily_summary['date'].iloc[-1].date()}")

    # ---------- load and filter races ----------
    if not os.path.exists(args.races):
        sys.exit(f"ERROR: races file not found: {args.races}")
    races = pd.read_csv(args.races, parse_dates=['date'])
    races['date_d'] = races['date'].dt.date

    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'

    elig = races[
        (~races['fatigued'].astype(bool)) &
        (races['surface'] != 'Downhill') &
        (races['time_sec'] >= 120)
    ].copy().sort_values('date')
    print(f"Eligible races: {len(elig)}")

    # ---------- truncate to plotting window + project races to 5K-equivalent ----------
    # The CS plot truncates to >= 2013-06-01; pre-2013 has very few datapoints
    # and huge uncertainty. project_races_to_5k_pace handles XC pre-correction
    # (matches the fit) and the β_long un-bias for distances > d_thresh internally;
    # see cs_projection.py for the rationale and formulas.
    summary_plot = (daily_summary[daily_summary['date'] >= '2013-06-01']
                    .copy().reset_index(drop=True))
    elig_plot = elig[elig['date'] >= '2013-06-01'].copy()
    print(f'Plot window: {len(summary_plot)} daily points')

    elig_plot = project_races_to_5k_pace(
        elig_plot, summary_plot, beta_long_med, d_thresh_long,
        apply_xc_correction=True, xc_correction=xc_correction)
    elig_plot = elig_plot[elig_plot['pace_norm_min'].notna()].copy()
    print(f'Hyperbolic projection (5K-equiv): {len(elig_plot)} race diamonds')

    # ---------- figure ----------
    fig = go.Figure()

    # 95% ribbon
    fig.add_trace(go.Scatter(
        x=summary_plot['date'].tolist() + summary_plot['date'].tolist()[::-1],
        y=summary_plot['cs_pace_lo95'].tolist() + summary_plot['cs_pace_hi95'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,180,80,0.10)',
        line=dict(width=0), mode='lines',
        name='95% credible interval', hoverinfo='skip', showlegend=True))
    # 50% ribbon
    fig.add_trace(go.Scatter(
        x=summary_plot['date'].tolist() + summary_plot['date'].tolist()[::-1],
        y=summary_plot['cs_pace_lo50'].tolist() + summary_plot['cs_pace_hi50'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,180,80,0.25)',
        line=dict(width=0), mode='lines',
        name='50% credible interval', hoverinfo='skip', showlegend=True))
    # Posterior median CS
    fig.add_trace(go.Scatter(
        x=summary_plot['date'], y=summary_plot['cs_pace_med'],
        mode='lines', line=dict(color='rgb(255,180,80)', width=2.5),
        name='Posterior median CS', hoverinfo='skip', showlegend=True))
    # Race diamonds (bias-corrected)
    surf_colors = {'Track': 'rgba(255,91,77,0.7)',
                   'Road':  'rgba(74,163,255,0.7)',
                   'XC':    'rgba(74,222,128,0.7)'}
    for surf, col in surf_colors.items():
        sub = elig_plot[elig_plot['surface'] == surf]
        if len(sub) == 0: continue
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pace_norm_min'],
            mode='markers', name=surf,
            marker=dict(color=col, size=8, symbol='diamond',
                        line=dict(width=0.3, color='white')),
            hoverinfo='skip',
            legendgroup='races',
            legendgrouptitle_text='Race pace (5K-equiv)'))

    # ---------- per-day hover content ----------
    hover_text = []
    for _, r in summary_plot.iterrows():
        d = r['date'].date()
        ep = elig_plot.copy()
        ep['days_off'] = (ep['date'].dt.date - d).apply(lambda x: x.days)
        ep['abs_off'] = ep['days_off'].abs()
        if len(ep):
            nearest = ep.loc[ep['abs_off'].idxmin()]
            delta = int(nearest['days_off'])
            ev = nearest.get('event') or '(no event)'
            if pd.isna(ev): ev = '(no event)'
            # Show original (uncorrected) race time/pace — that's the truthful
            # race result. The XC correction is only for fitting/plotting the
            # 5K-equivalent pace; the actual race happened in the original time.
            t_orig = nearest.get('time_sec_original', nearest['time_sec'])
            p_orig = nearest.get('pace_sec_per_mi_original',
                                 nearest.get('pace_sec_per_mi'))
            pace_raw_str = (sec_to_mss(p_orig)
                            if p_orig is not None and not pd.isna(p_orig) else '')
            is_xc_nearest = str(nearest.get('surface', '')).upper() == 'XC'
            is_5k = abs(float(nearest['distance_m']) - 5000.0) < 1.0

            # 5K-equivalent: time + pace pair (matches actual race time format)
            equiv_pace_sec = float(nearest['pace_norm_min']) * 60
            equiv_time_sec = equiv_pace_sec * 5000.0 / 1609.344

            # Four cases — see comments
            #   non-XC 5K:    drop the line entirely (just duplicates race time)
            #   XC 5K:        say "XC-corrected" (XC adjustment is the only difference)
            #   XC non-5K:    say "5K-equiv (XC-corrected)"
            #   non-XC non-5K: say "5K-equiv"
            if is_5k and not is_xc_nearest:
                equiv_line = ''
            else:
                if is_xc_nearest and is_5k:
                    label = 'XC-corrected'
                elif is_xc_nearest:
                    label = '5K-equiv (XC-corrected)'
                else:
                    label = '5K-equiv'
                equiv_line = (f"<div>{label}: <b>{sec_to_mss(equiv_time_sec)}</b> "
                              f"<span class='cs-tt-mute'>({sec_to_mss(equiv_pace_sec)}/mi)</span></div>")

            nr = (f"<div class='cs-tt-section'>"
                  f"<div class='cs-tt-section-title'>Nearest race ({nearest['date'].date()}, {delta:+d}d)</div>"
                  f"<div>{ev} <span class='cs-tt-mute'>({nearest['surface']})</span></div>"
                  f"<div>{int(nearest['distance_m'])}m in <b>{sec_to_mss_full(t_orig)}</b> "
                  f"<span class='cs-tt-mute'>({pace_raw_str}/mi)</span></div>"
                  f"{equiv_line}"
                  f"</div>")
        else:
            nr = ''
        hover_text.append(
            f"<div class='cs-tt-date'>{d}</div>"
            f"<div class='cs-tt-section'>"
            f"<div class='cs-tt-row'><span>CS median</span><b>{sec_to_mss(r['cs_pace_med']*60)}/mi</b></div>"
            f"<div class='cs-tt-row'><span>50% interval</span>{sec_to_mss(r['cs_pace_lo50']*60)}–{sec_to_mss(r['cs_pace_hi50']*60)}/mi</div>"
            f"<div class='cs-tt-row'><span>95% interval</span>{sec_to_mss(r['cs_pace_lo95']*60)}–{sec_to_mss(r['cs_pace_hi95']*60)}/mi</div>"
            f"</div>"
            f"{nr}"
        )

    # ---------- layout ----------
    y_min, y_max = 4.50, 8.00
    ytick_vals = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    ytick_txt = [sec_to_mss(v * 60) for v in ytick_vals]

    bias_str = f"β_long={beta_long_med:.3f}, xc_correction={xc_correction:.3f}"
    fig.update_layout(
        title=dict(
            text=f'Critical Speed fitness trend — Bayesian latent-process model'
                 f'<br><sub style="font-size:13px;color:#bbb">'
                 f'Posterior median (line) with 50% and 95% credible-interval ribbons · '
                 f'Diamonds show race performance projected to 5K-equivalent via hyperbolic CS model · '
                 f'<i>{bias_str}</i>'
                 f'</sub>',
            y=0.965),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#eee'),
        autosize=True,
        margin=dict(t=110, l=70, r=200, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02, groupclick='toggleitem'),
        xaxis=dict(title='Date', showgrid=True, gridcolor='#333',
                   dtick='M12', tickformat='%Y',
                   range=[pd.Timestamp('2013-06-01'),
                          summary_plot['date'].iloc[-1]]),
        yaxis=dict(title='Pace (min/mi)', range=[y_max, y_min],
                   tickmode='array', tickvals=ytick_vals, ticktext=ytick_txt,
                   showgrid=True, gridcolor='#333'))

    out_html = args.out or str(OUTPUT_DIR / f'cs_timeline{suffix}.html')
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True,
                   config={'responsive': True})

    # ---------- inject custom CSS + cursor-following tooltip ----------
    epoch = pd.Timestamp('1970-01-01')
    hover_payload = [
        [(d - epoch).days, h] for d, h in zip(summary_plot['date'], hover_text)
    ]
    hover_payload_json = json.dumps(hover_payload)
    first_day = (summary_plot['date'].iloc[0] - epoch).days
    last_day  = (summary_plot['date'].iloc[-1] - epoch).days

    custom = """
<style>
html, body {
  margin: 0; padding: 0;
  width: 100%; height: 100%;
  background: #1a1a1a;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.plotly-graph-div, #plot-container, .js-plotly-plot {
  width: 100% !important;
  height: 100vh !important;
}

#cs-custom-tooltip {
  position: fixed; top: 0; left: 0;
  background: rgba(26,26,26,0.96);
  color: #eee;
  border: 1px solid #555;
  padding: 10px 14px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 13px;
  line-height: 1.5;
  border-radius: 4px;
  pointer-events: none;
  z-index: 9999;
  max-width: 380px;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
#cs-custom-tooltip .cs-tt-date { font-weight: 600; font-size: 14px; margin-bottom: 6px; color: #fff; }
#cs-custom-tooltip .cs-tt-section { margin-top: 6px; padding-top: 6px; border-top: 1px solid #444; }
#cs-custom-tooltip .cs-tt-section:first-of-type { border-top: 0; margin-top: 0; padding-top: 0; }
#cs-custom-tooltip .cs-tt-section-title { font-weight: 600; margin-bottom: 4px; color: #ddd; }
#cs-custom-tooltip .cs-tt-row { display: flex; justify-content: space-between; gap: 16px; white-space: nowrap; }
#cs-custom-tooltip .cs-tt-row > span:first-child { color: #aaa; }
#cs-custom-tooltip .cs-tt-mute { color: #999; }
#cs-custom-tooltip b { color: #fff; font-weight: 600; }

#cs-spike-line {
  position: fixed; top: 0; left: 0;
  width: 1px; height: 100vh;
  background: rgba(255,255,255,0.3);
  pointer-events: none;
  z-index: 9998;
  display: none;
}
</style>
<div id="cs-custom-tooltip"></div>
<div id="cs-spike-line"></div>
<script>
(function() {
  var hoverData = __HOVER_PAYLOAD__;
  var firstDay = __FIRST_DAY__;
  var lastDay  = __LAST_DAY__;
  var hoverByDay = {};
  for (var i = 0; i < hoverData.length; i++) {
    hoverByDay[hoverData[i][0]] = hoverData[i][1];
  }

  var tt = document.getElementById('cs-custom-tooltip');
  var spike = document.getElementById('cs-spike-line');
  var lastContent = ''; var ttW = 0; var ttH = 0;
  var rafScheduled = false; var pendingX = 0; var pendingY = 0; var pendingContent = '';
  var pendingShow = false;

  function update() {
    rafScheduled = false;
    if (!pendingShow) { tt.style.display = 'none'; spike.style.display = 'none'; return; }
    if (pendingContent !== lastContent) {
      tt.innerHTML = pendingContent;
      lastContent = pendingContent;
      ttW = tt.offsetWidth; ttH = tt.offsetHeight;
    }
    var x = pendingX + 15; var y = pendingY + 10;
    if (x + ttW > window.innerWidth)  x = pendingX - ttW - 15;
    if (y + ttH > window.innerHeight) y = pendingY - ttH - 10;
    tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    tt.style.display = 'block';
    spike.style.transform = 'translateX(' + pendingX + 'px)';
    spike.style.display = 'block';
  }

  function bind() {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (!pdiv || !pdiv._fullLayout) { setTimeout(bind, 100); return; }
    pdiv.addEventListener('mousemove', function(e) {
      var fl = pdiv._fullLayout;
      if (!fl) return;
      var xa = fl.xaxis;
      var rect = pdiv.getBoundingClientRect();
      var bgRect = fl._size;
      var plotLeft  = rect.left + bgRect.l;
      var plotRight = rect.left + bgRect.l + bgRect.w;
      var plotTop   = rect.top  + bgRect.t;
      var plotBot   = rect.top  + bgRect.t + bgRect.h;
      if (e.clientX < plotLeft || e.clientX > plotRight ||
          e.clientY < plotTop  || e.clientY > plotBot) {
        pendingShow = false;
        if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
        return;
      }
      var dataX = xa.p2c(e.clientX - rect.left - bgRect.l);
      var dayIdx = Math.round(dataX / 86400000);
      if (dayIdx < firstDay) dayIdx = firstDay;
      if (dayIdx > lastDay)  dayIdx = lastDay;
      var content = hoverByDay[dayIdx];
      if (!content) { pendingShow = false; }
      else {
        pendingContent = content;
        pendingX = e.clientX; pendingY = e.clientY;
        pendingShow = true;
      }
      if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
    });
    pdiv.addEventListener('mouseleave', function() {
      pendingShow = false;
      if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
    });
    if (window.Plotly) window.Plotly.Plots.resize(pdiv);
  }
  bind();
  window.addEventListener('resize', function() {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (pdiv && window.Plotly) window.Plotly.Plots.resize(pdiv);
  });
})();
</script>
"""
    custom = (custom
              .replace('__HOVER_PAYLOAD__', hover_payload_json)
              .replace('__FIRST_DAY__', str(first_day))
              .replace('__LAST_DAY__', str(last_day)))

    with open(out_html) as f:
        html = f.read()
    html = html.replace('<body>', '<body style="margin:0;padding:0;background:#1a1a1a;">')
    html = html.replace('</body>', custom + '</body>')
    with open(out_html, 'w') as f:
        f.write(html)

    print(f"Wrote {out_html}")


if __name__ == '__main__':
    main()
