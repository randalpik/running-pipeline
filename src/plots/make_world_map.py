"""Minimalist world map of every city Max has run in.

Each city is one partially transparent point. Color matches that city's
color in the Locations plot (make_geography_plot.py). Size encodes total
miles on a log scale. Hover shows city, mileage, and a context-sensitive
date range. A toolbar in the top-right toggles the geographic scope
(World / N. America / Europe) — regional scopes pull in subunit borders
(US states, Canadian provinces) and lake outlines that aren't shipped in
plotly's world topojson.
"""
import argparse
import calendar
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.geocoding import ensure_coords
from src.plotting import render_plot, apply_default_layout
from src.plots.make_geography_plot import build_categories


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT   = str(OUTPUT_DIR)

# Marker sizing: diameter in px = log10(miles + 1) * SIZE_SCALE + SIZE_OFFSET
SIZE_SCALE  = 8.0
SIZE_OFFSET = 4.0
MARKER_OPACITY = 0.85

BG = '#1a1a1a'
LINE_COLOR = '#ffffff'
COASTLINE_WIDTH = 1.2
BORDER_WIDTH    = 0.5
# Subunits (US states / Canadian provinces) ship in plotly's regional topojsons
# (north america, usa, etc.) but NOT in the world tile, even at resolution=50.
# So they're invisible at scope='world' and visible once the user toggles to
# a regional scope.
SUBUNIT_WIDTH = 0.5

# Each entry: (toggle_id, button_label, plotly_scope, lon_range, lat_range).
#
# `plotly_scope` is the value passed to `layout.geo.scope`; plotly only
# accepts a fixed enum ('world', 'usa', 'europe', 'asia', 'africa',
# 'north america', 'south america'), so finer-grained crops like 'PNW' ride
# on top of an existing scope ('north america' here) with tighter lon/lat
# ranges. Two toggle ids can therefore share the same plotly_scope.
#
# List order is the order the buttons render in the toolbar.
SCOPES = [
    ('world',         'World',      'world',         [-180, 180], [-60, 85]),
    ('europe',        'Europe',     'europe',        [-25, 55],   [33, 75]),
    ('north_america', 'N. America', 'north america', [-172, -50], [10, 78]),
    ('pnw',           'PNW',        'north america', [-129, -114], [41, 51]),
]
DEFAULT_SCOPE = 'world'

# Indexed view of SCOPES for lookup by toggle id.
SCOPE_CONFIG = {
    sid: {'label': label, 'plotly_scope': pscope, 'lon': lon, 'lat': lat}
    for sid, label, pscope, lon, lat in SCOPES
}


def fmt_date_range(d_min: pd.Timestamp, d_max: pd.Timestamp) -> str:
    if d_min == d_max:
        return d_min.strftime('%b %-d, %Y')
    if d_min.year == d_max.year and d_min.month == d_max.month:
        return d_min.strftime('%b %Y')
    if d_min.year == d_max.year:
        return (f'{calendar.month_abbr[d_min.month]}-'
                f'{calendar.month_abbr[d_max.month]} {d_min.year}')
    return f'{d_min.year}-{d_max.year}'


def _scope_geo(scope_id: str) -> dict:
    """Build the full `layout.geo` config for a given toggle id.

    Used both to seed the figure with the default scope and to swap to a new
    scope on toggle click — same dict shape on the wire either way, which
    keeps the JS reset trivial (replace the geo block wholesale).
    """
    cfg = SCOPE_CONFIG[scope_id]
    return dict(
        scope=cfg['plotly_scope'],
        projection=dict(type='natural earth'),
        resolution=50,
        bgcolor=BG,
        showcoastlines=True,
        coastlinecolor=LINE_COLOR, coastlinewidth=COASTLINE_WIDTH,
        showcountries=True,
        countrycolor=LINE_COLOR,   countrywidth=BORDER_WIDTH,
        showsubunits=True,
        subunitcolor=LINE_COLOR,   subunitwidth=SUBUNIT_WIDTH,
        showland=False, showocean=False, showlakes=False,
        showrivers=False, showframe=False,
        lonaxis=dict(range=cfg['lon']),
        lataxis=dict(range=cfg['lat']),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--daily',   default=DEFAULT_DAILY)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    df = pd.read_csv(args.daily, parse_dates=['date'])
    df = df.dropna(subset=['city_state'])
    df = df[df['city_state'].astype(str).str.strip() != '']

    agg = (df.groupby('city_state', as_index=False)
             .agg(miles=('miles', 'sum'),
                  d_min=('date', 'min'),
                  d_max=('date', 'max')))
    agg['date_range'] = [fmt_date_range(a, b)
                         for a, b in zip(agg['d_min'], agg['d_max'])]

    label_for, _, label_color, *_ = build_categories(df)
    agg['color'] = agg['city_state'].map(
        lambda cs: label_color.get(label_for.get(cs)))
    agg['color'] = agg['color'].fillna('#888888')

    coords = ensure_coords(agg['city_state'].tolist())
    agg['lat'] = agg['city_state'].map(
        lambda cs: coords.get(cs, (None, None))[0])
    agg['lon'] = agg['city_state'].map(
        lambda cs: coords.get(cs, (None, None))[1])

    missing = agg[agg['lat'].isna()]
    if len(missing):
        print(f'[world-map] {len(missing)} city-state(s) have no coords - '
              f'omitted from map:')
        print(missing[['city_state', 'miles']]
              .sort_values('miles', ascending=False)
              .to_string(index=False))
    agg = agg.dropna(subset=['lat', 'lon']).copy()

    agg['size'] = np.log10(agg['miles'] + 1) * SIZE_SCALE + SIZE_OFFSET

    # Smaller cities render on top of larger ones so they don't get hidden.
    agg = agg.sort_values('miles', ascending=False)

    fig = go.Figure(go.Scattergeo(
        lat=agg['lat'], lon=agg['lon'],
        mode='markers',
        marker=dict(
            size=agg['size'],
            color=agg['color'],
            opacity=MARKER_OPACITY,
            line=dict(width=0.5, color=LINE_COLOR),
            sizemode='diameter',
        ),
        customdata=np.stack([agg['city_state'],
                             agg['miles'].round(1),
                             agg['date_range']], axis=-1),
        hovertemplate=('<b>%{customdata[0]}</b><br>'
                       '%{customdata[1]:,.1f} mi<br>'
                       '%{customdata[2]}<extra></extra>'),
        hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)',
                        bordercolor='#444',
                        font=dict(color='#eee')),
    ))

    fig.update_layout(geo=_scope_geo(DEFAULT_SCOPE))

    apply_default_layout(fig)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        # The plot is intentionally static: drag-pan and scroll-zoom are
        # both off, the modebar is hidden, and the scope toggle is the only
        # interaction that changes the view. Plotly's geo subplot has no
        # native API to clamp pan/zoom to a scope's lon/lat bounds, and a
        # custom relayout-clamp listener is research-grade work we're not
        # taking on here — better static than half-bounded.
        dragmode=False,
    )

    total_cities = agg['city_state'].nunique()
    total_miles = agg['miles'].sum()
    print(f'[world-map] rendered {total_cities} cities, '
          f'{total_miles:,.0f} mi total')

    overlay_html = _build_scope_toggle()

    out_path = os.path.join(args.out_dir, 'world_map.html')
    render_plot(
        fig, out_path,
        title_slug='world_map',
        page_title='World Map',
        title='World map',
        subtitle=f'{total_miles:,.0f} mi across {total_cities} cities',
        overlay_html=overlay_html,
        plotly_config={
            'scrollZoom': False,
            'displayModeBar': False,
        },
    )
    print(f'wrote {out_path}')


def _build_scope_toggle() -> str:
    """Top-right toggle that swaps geo.scope between world / NA / Europe.

    This toolbar is the **only** interaction the plot supports — drag-pan
    and scroll-zoom are disabled in the layout/config (see ``main``), so
    every visible state change goes through here.

    Each click does a wholesale ``layout.geo`` replacement (via
    Plotly.react) rather than a ``Plotly.relayout`` patch, because
    relayout deep-merges and leaves stale center/scale/rotation/axis-range
    values from the previous scope — which is what produced the "Europe
    renders blank" symptom and the glitchy switching.
    """
    scope_geos = {sid: _scope_geo(sid) for sid, *_ in SCOPES}

    css = """
<style>
#scope-toggle {
  position: fixed; top: 14px; right: 20px;
  background: rgba(26,26,26,0.92);
  border: 1px solid #444; border-radius: 6px;
  padding: 4px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 13px; z-index: 1000;
  display: flex; gap: 2px;
}
#scope-toggle .st-btn {
  padding: 6px 14px; cursor: pointer; border-radius: 4px;
  color: #aaa; background: transparent; border: none;
  transition: background 0.12s, color 0.12s;
  font-size: 13px; font-family: inherit;
}
#scope-toggle .st-btn:hover { color: #fff; background: #333; }
#scope-toggle .st-btn.active {
  background: #93f; color: #fff; font-weight: 500;
}
</style>
"""
    btn_html = '\n'.join(
        f'  <button class="st-btn{" active" if sid == DEFAULT_SCOPE else ""}" '
        f'data-scope="{sid}">{label}</button>'
        for sid, label, *_ in SCOPES
    )

    payload = json.dumps(scope_geos, separators=(',', ':'))
    js = f"""
<script>
(function () {{
  var SCOPE_GEOS = {payload};
  function pdiv() {{ return document.querySelector('.plotly-graph-div'); }}
  var btns = document.querySelectorAll('#scope-toggle .st-btn');

  function applyScope(scope) {{
    var gd = pdiv();
    if (!gd || !SCOPE_GEOS[scope]) return;
    btns.forEach(function (b) {{
      b.classList.toggle('active', b.getAttribute('data-scope') === scope);
    }});
    // Build a layout copy with the geo block fully replaced. Plotly.react
    // diffs and re-renders without retaining any of the previous scope's
    // projection state.
    var layout = Object.assign({{}}, gd.layout);
    layout.geo = SCOPE_GEOS[scope];
    Plotly.react(gd, gd.data, layout);
  }}

  btns.forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      applyScope(btn.getAttribute('data-scope'));
    }});
  }});
}})();
</script>
"""
    return css + '<div id="scope-toggle">\n' + btn_html + '\n</div>\n' + js


if __name__ == '__main__':
    main()
