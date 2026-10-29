"""Shoes tab: every physical pair as a row of worn-day segments over time.

Pairs, roles and per-pair mileage come from src/shared/shoes.py (the same
object the dashboard's current-shoe row reads); see docs/shoes-reference.md.

Layout
------
One horizontal bar segment per day a pair was worn — runs of consecutive days
merge into one segment, which looks identical and keeps the trace small —
coloured by brand (src/plotting/brand_palette.py). Inferred warm-up days on
spike days count as wear and get segments too. Trainers fill a block of rows
on top and everything else (racers, spikes) a block below, so the training
lineage reads as one cascade (Max, Oct 2026). Within each block rows are
reused: a pair takes the lowest row whose previous pair was last worn more
than ROW_REUSE_DAYS before this one's first wear.

Interaction (plot_shoes.js)
---------------------------
Bars are never thinner than 1px, so every day can be seen and hovered. A
pair's hover target is its whole region — first wear to last, full row
height — so dragging across seams and gaps never drops the tooltip.
Hovering outlines every segment of that pair and shows its tooltip:
name, date range (World Map logic), days worn, miles, PRs set (if it raced)
and the best performance credited to it. Only the LOGGED shoe earns a
performance — a track race credits the spikes, never the trainer inferred for
that day's warm-up.

Only profiles that log shoes get the tab: with no shoe data nothing is
written, and the shell hides tabs whose HTML doesn't exist.
"""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.units import METERS_PER_MILE
from src.shared.plot_window import pad_range
from src.shared.effective_mileage import effective_daily_miles
from src.shared.shoes import build_shoes
from src.shared.cs_projection import load_cs_outputs
from src.shared.performance_frontier import standard_demos
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                          yearly_x_axis_kwargs, sec_to_mss, sec_to_mss_prec,
                          time_decimals, tt_row, tt_title, CAT_LABEL)
from src.plotting import widgets
from src.plotting.formatters import fmt_date_range
from src.plotting.brand_palette import brand_colors, min_separation
from src.plots.dashboard import pr_candidates

OUT_HTML = OUTPUT_DIR / 'shoes.html'
_SHOES_JS = Path(__file__).resolve().parent / 'plot_shoes.js'

# A row is free for a new pair once its last pair is this long retired.
ROW_REUSE_DAYS = 365
# Bar thickness, in row units (1 = touching neighbours).
BAR_WIDTH = 0.72
# Extra space between the trainer block and the rest, in rows.
BLOCK_GAP = 0.5
DAY = pd.Timedelta(days=1)


def pack_rows(pairs: pd.DataFrame) -> list:
    """Row per pair (pairs sorted by first wear): the lowest row whose last
    occupant was last worn more than ROW_REUSE_DAYS before this pair's first
    wear, else a new row."""
    row_last, rows = [], []
    for first, last in zip(pairs['first'], pairs['last']):
        for i, prev in enumerate(row_last):
            if (first - prev).days > ROW_REUSE_DAYS:
                row_last[i] = last
                rows.append(i)
                break
        else:
            row_last.append(last)
            rows.append(len(row_last) - 1)
    return rows


def segments(dates) -> list:
    """Worn dates -> [(start, end)) runs of consecutive days."""
    days = pd.DatetimeIndex(sorted(set(dates)))
    breaks = np.flatnonzero(np.diff(days.values) != np.timedelta64(1, 'D')) + 1
    return [(run[0], run[-1] + DAY) for run in np.split(days, breaks)]


def pr_counts(races: pd.DataFrame, logged: pd.Series) -> tuple[pd.Series, set]:
    """(pair -> races that set a PR at the time, pairs that raced at all).
    Dashboard PR rules (``pr_candidates``); a race is a PR when it beats every
    earlier race in its bin. All of races.csv is scanned, so pre-shoe-era
    bests set the bar."""
    c = pr_candidates(races).sort_values(['date', 'race_seq'], kind='stable')
    prior = c.groupby('bin')['pace'].transform(
        lambda p: p.cummin().shift(fill_value=np.inf))
    prs = c[c['pace'] < prior]
    counts = logged.reindex(prs['date']).value_counts()
    raced = set(logged.reindex(races['date']).dropna())
    return counts, raced


def _demo_title(r) -> str:
    """Session title in the Fitness tab's form (bayes_cs_plot._frontier_title)."""
    if r['src'] == 'long_run':
        label = 'Long run'
    elif r['src'] == 'hill':
        label = CAT_LABEL['hill_cont']
    elif r['src'] == 'hill_rep':
        label = CAT_LABEL['hill_rep']
    else:
        label = CAT_LABEL.get(r['category'], str(r['category']).title())
    return tt_title(label, r.get('display_name'), r.get('city_state'))


def _performance_html(r) -> str:
    pace_sec = float(r['pace_min']) * 60
    lines = []
    # A 5000m race's 5K equivalent only echoes its own time — dropped, as on
    # the Fitness tab.
    is_5k = r['src'] == 'race' and abs(float(r['distance_m']) - 5000.0) < 1.0
    if r['src'] == 'race':
        dist, t = float(r['distance_m']), float(r['time_sec'])
        td = (int(r['time_dec']) if pd.notna(r['time_dec'])
              else time_decimals(t))
        lines.append(f"<b>{escape(str(r['detail']))}</b> "
                     f"<span class='tt-mute'>({r['surface']})</span>")
        lines.append(f"{int(dist)}m in <b>{sec_to_mss_prec(t, td)}</b> "
                     f"<span class='tt-mute'>"
                     f"({sec_to_mss(t * METERS_PER_MILE / dist)}/mi)</span>")
    else:
        lines.append(_demo_title(r))
        lines.append(str(r['detail']))
    if not is_5k:
        lines.append(f"5K equivalent: <b>{sec_to_mss(pace_sec * 5000 / METERS_PER_MILE)}</b> "
                     f"<span class='tt-mute'>({sec_to_mss(pace_sec)}/mi)</span>")
    lines.append(f"<span class='tt-mute'>{r['date'].strftime('%b %-d, %Y')}</span>")
    return ''.join(f'<div>{ln}</div>' for ln in lines)


def best_performances(logged: pd.Series) -> dict:
    """pair -> HTML of its best demonstration (lowest 5K-equivalent pace)
    among everything the Fitness and Training tabs plot: Fitness's races and
    the kept Training corpus (workouts, long runs, hills)."""
    try:
        summary, beta_long, d_thresh, xc = load_cs_outputs(str(DATA_DIR))
    except FileNotFoundError as e:
        print(f'  WARNING: no CS fit outputs ({e}); best performances omitted')
        return {}
    demos = standard_demos(summary, beta_long, d_thresh, xc)
    # Fitness draws no Downhill race (standard_demos admits measured ones for
    # the frontier line only).
    demos = demos[~((demos['src'] == 'race') & (demos['surface'] == 'Downhill'))]
    demos = demos.dropna(subset=['pace_min']).copy()
    demos['pair'] = logged.reindex(demos['date']).to_numpy()
    demos = demos.dropna(subset=['pair'])
    best = demos.loc[demos.groupby('pair')['pace_min'].idxmin()]
    return {r['pair']: _performance_html(r) for _, r in best.iterrows()}


def pair_tooltip(p, prs: int | None, best: str | None) -> str:
    parts = [f'<div class="tt-date">{escape(p["name"])}</div>',
             tt_row('Worn', fmt_date_range(p['first'], p['last'])),
             tt_row('Days', f'{int(p["days"]):,}'),
             tt_row('Miles', f'{p["miles"]:,.1f}')]
    if prs is not None:
        parts.append(tt_row('PRs set', str(prs)))
    if best:
        parts.append('<div class="tt-section"><div class="tt-section-title">'
                     f'Best performance</div>{best}</div>')
    return ''.join(parts)


def main():
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    # Display mileage, the same figure the dashboard's shoe row uses.
    daily['miles'] = effective_daily_miles(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    shoes = build_shoes(daily, races)
    if shoes.empty:
        OUT_HTML.unlink(missing_ok=True)
        print('no shoe data — Shoes tab not written')
        return

    pairs = shoes.pairs.copy()
    # Trainers on top, everything else below; each block packs on its own.
    trainer = pairs['role'] == 'trainer'
    pairs['row'] = 0.0
    pairs.loc[trainer, 'row'] = pack_rows(pairs[trainer])
    below = (pairs.loc[trainer, 'row'].max() + 1 + BLOCK_GAP) if trainer.any() else 0
    pairs.loc[~trainer, 'row'] = [below + r for r in pack_rows(pairs[~trainer])]
    logged = shoes.logged_pair()
    prs, raced = pr_counts(races, logged)
    best = best_performances(logged)
    brands = list(dict.fromkeys(pairs['brand']))   # first-wear order
    colors = brand_colors(brands)

    fig = go.Figure()
    js_pairs = []
    seen_brands = set()
    for pid, p in pairs.iterrows():
        segs = segments(shoes.wear.loc[shoes.wear['pair'] == pid, 'date'])
        fig.add_trace(go.Bar(
            orientation='h',
            y=[p['row']] * len(segs),
            base=[s.strftime('%Y-%m-%d') for s, _ in segs],
            x=[(e - s) / pd.Timedelta(milliseconds=1) for s, e in segs],
            width=BAR_WIDTH,
            marker=dict(color=colors[p['brand']], line=dict(width=0)),
            name=p['brand'], legendgroup=p['brand'],
            legendrank=sorted(brands).index(p['brand']),
            showlegend=p['brand'] not in seen_brands,
            hoverinfo='skip',
        ))
        seen_brands.add(p['brand'])
        js_pairs.append({
            'row': float(p['row']),
            'segs': [[int(s.value // 10**6), int(e.value // 10**6)] for s, e in segs],
            'html': pair_tooltip(p, int(prs.get(pid, 0)) if pid in raced else None,
                                 best.get(pid)),
        })

    y_max = float(pairs['row'].max())
    x_lo, x_hi = pad_range(pairs['first'].min(), pairs['last'].max() + DAY,
                           frac=0.01)
    apply_default_layout(
        fig,
        barmode='overlay',
        hovermode=False,
        margin=dict(t=20, l=30, r=150, b=28),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        xaxis=dict(type='date', **yearly_x_axis_kwargs(x_lo, x_hi)),
        # Rows carry no meaning beyond separating pairs: no labels, no y-zoom.
        yaxis=dict(range=[y_max + 0.5, -0.5], showticklabels=False,
                   showgrid=False, zeroline=False, fixedrange=True),
    )

    render_plot(
        fig, OUT_HTML,
        title_slug='shoes',
        page_title='Shoes',
        title='Shoes',
        subtitle='Days each pair was worn, colored by brand',
        overlay_html=widgets.js_globals({'SHOES': {'half': BAR_WIDTH / 2,
                                                   'pairs': js_pairs}}),
        overlay_js_files=[_SHOES_JS],
        cursor_tooltip=CursorTooltip(spike=False),
    )
    print(f'wrote {OUT_HTML}')
    print(f'{len(pairs)} pairs; {pairs.loc[trainer, "row"].nunique()} trainer rows + '
          f'{pairs.loc[~trainer, "row"].nunique()} other rows; {len(brands)} brands, '
          f'min brand dE {min_separation(colors):.1f}')
    for pid, p in pairs.sort_values(['row', 'first']).iterrows():
        print(f"  row {p['row']:4.1f}  {p['name']:38s} {p['brand']:12s} "
              f"{p['days']:4d}d {p['miles']:7.1f} mi  PRs "
              f"{int(prs.get(pid, 0)) if pid in raced else '-':>2}  "
              f"best={'y' if pid in best else '-'}")


if __name__ == '__main__':
    main()
