"""Minimalist world map of every city Max has run in.

Each city is one partially transparent point. Color matches that city's
color in the Locations plot (make_geography_plot.py). Size encodes total
miles on a log scale. Hover shows city, mileage, and a context-sensitive
date range. A toolbar in the top-right toggles the geographic scope
(World / Europe / N. America / PNW) — regional scopes pull in subunit
borders (US states, Canadian provinces) that aren't shipped in plotly's
world topojson, and their lon/lat viewports are computed at render time
from a per-scope envelope + the actual data bbox + plot-div aspect ratio
(see _build_scope_toggle's JS).
"""
import argparse
import calendar
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'parsers'))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.geocoding import ensure_coords
from src.shared.country_codes import country_abbrev
from src.shared.effective_mileage import effective_daily_miles
from src.plotting import render_plot, apply_default_layout
from src.plotting import widgets
from src.plots.make_geography_plot import build_categories
from snapshot import (find_snapshot, read_snapshot,  # type: ignore
                      coord_overrides_from_sections)

_PLOTS_DIR = Path(__file__).resolve().parent
_WORLD_MAP_JS = _PLOTS_DIR / 'make_world_map.js'


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT   = str(OUTPUT_DIR)

# Marker sizing: diameter in px = log10(max(miles, SIZE_FLOOR_MILES) + 1)
#   * SIZE_SCALE + SIZE_OFFSET. The floor is 3 miles — both for cities with
#   miles=0 (historical-only entries with no recorded mileage) and for race-
#   only cities whose total is dominated by sub-5K races. Rationale: any
#   city Max actually visited had at least one ~3-mile easy run on the
#   ground, even if it wasn't logged.
SIZE_SCALE  = 8.0
SIZE_OFFSET = 4.0
SIZE_FLOOR_MILES = 3.0
MARKER_OPACITY = 0.95

BG = '#1a1a1a'
# Slightly lighter than BG so land is distinguishable from ocean at a
# glance — important on zoomed-in regional scopes where the city points
# alone don't carry the geographic context.
LAND_COLOR = "#292929"
LINE_COLOR = "#ffffff"
COASTLINE_WIDTH = 1.2
BORDER_WIDTH    = 0.5
# Subunits (US states / Canadian provinces) ship in plotly's regional topojsons
# (north america, usa, etc.) but NOT in the world tile, even at resolution=50.
# So they're invisible at scope='world' and visible once the user toggles to
# a regional scope.
SUBUNIT_WIDTH = 0.5

# Each entry is a dict keyed by:
#   id           — toggle id used in the toolbar / JS
#   label        — button text
#   plotly_scope — passed to `layout.geo.scope`. Plotly only accepts a fixed
#                  enum ('world', 'usa', 'europe', 'asia', 'africa',
#                  'north america', 'south america'); finer crops like PNW
#                  ride on top of an existing scope ('north america'). Two
#                  ids can share the same plotly_scope.
#
# Then either:
#   static       — explicit {'lon': [a,b], 'lat': [c,d]} for layout.geo.
#                  Used by World only.
# OR:
#   envelope     — point filter only. Cities outside this lat/lon box are
#                  hidden in this scope. The envelope does NOT appear in
#                  any layout property and does NOT bound the viewport.
#   pad_lat,
#   pad_lon      — degrees of breathing room around the in-scope cities'
#                  bbox; JS uses this to seed the viewport before
#                  expanding to match the div aspect.
#
# Non-world scopes have their lonaxis.range / lataxis.range computed by
# JS at render and resize: lat range = data ± pad_lat (fixed), lon range
# expanded so the natural-earth-projected aspect of the box equals the
# div aspect. Plotly's geolayer SVG fits the projected box to the div, so
# matching aspects means the geolayer rect equals the div rect — no
# internal letterbox on either side. The whole computation is driven by
# the in-scope cities + padding; nothing about the envelope itself ends
# up in a layout property.
#
# List order is the order the buttons render in the toolbar.
SCOPES = [
    # World shows every city — envelope is the whole globe (it's a no-op
    # filter, just keeps the same code path as other scopes). bbox + JS
    # aspect-fit handle the rest, with lat clamped to lat_clamp (-60..85
    # for world, dropping Antarctica — there's no data there) and a
    # fallback that expands lon when lat alone can't match div_aspect.
    {'id': 'world', 'label': 'World', 'plotly_scope': 'world',
     'envelope': {'lon': [-180, 180], 'lat': [-90, 90]},
     'lat_clamp': [-60, 85],
     'pad_lat': 1.5, 'pad_lon': 3},
    # Europe + Asia are sparse maps (low-mileage cities only, since Max lives
    # in NA). Doubling point sizes makes them visible at a glance — the
    # log-scale sizing on its own gets too small to spot in these scopes.
    # Europe and Asia use scope='world' with JS aspect-fit cropping (same
    # pattern as PNW). Plotly's per-continent scopes clip Russia at the
    # boundary, leaving it as empty white space on both views.
    {'id': 'europe', 'label': 'Europe', 'plotly_scope': 'world',
     'envelope': {'lon': [-30, 60], 'lat': [30, 75]},
     'pad_lat': 1.5, 'pad_lon': 3,
     'point_size_mult': 1.6},
    {'id': 'asia', 'label': 'Asia', 'plotly_scope': 'world',
     'envelope': {'lon': [60, 180], 'lat': [-10, 75]},
     'pad_lat': 1.5, 'pad_lon': 3,
     'point_size_mult': 1.6},
    {'id': 'north_america', 'label': 'N. America', 'plotly_scope': 'north america',
     'envelope': {'lon': [-172, -50], 'lat': [10, 78]},
     'pad_lat': 1.5, 'pad_lon': 3},
    # PNW envelope: WA + OR + ID + BC + AB + western MT + SE-Alaska room.
    # Northern bound 60° leaves room for northern BC / SE Alaska. Today this
    # captures WA + Vancouver BC + Banff + western MT.
    {'id': 'pnw', 'label': 'PNW', 'plotly_scope': 'north america',
     'envelope': {'lon': [-140, -110], 'lat': [41, 60]},
     'pad_lat': 1, 'pad_lon': 2},
]
DEFAULT_SCOPE = 'world'

SCOPE_CONFIG = {s['id']: s for s in SCOPES}

# Per-scope geo-feature visibility. The 'thin coastline border' artifact in
# regional scopes comes from countries' ocean-edge polygons being drawn at
# higher detail than the world coastline layer — the country polygon traces
# the coast and creates a parallel thin line just inside the thick coast.
# Suppressing coastlines in regional scopes hides that artifact; world keeps
# coastlines because there countries don't trace ocean as densely (and
# Antarctica / ocean-only islands need them to render at all).
SCOPE_FEATURES = {
    # World + Europe + Asia: full hierarchy reads cleanly (no spurious
    # dual-trace artifact like NA had).
    'world':         {'show_coastlines': True, 'show_countries': True,  'show_subunits': True},
    'europe':        {'show_coastlines': True, 'show_countries': True,  'show_subunits': True},
    'asia':          {'show_coastlines': True, 'show_countries': True,  'show_subunits': True},
    # NA: country layer carries a spurious curved arc across the northern US
    # (the line tracks well inland of the 49th parallel). Subunit lines
    # cover the actual US/Canada border cleanly, so countries are off here.
    'north_america': {'show_coastlines': True, 'show_countries': False, 'show_subunits': True},
    # PNW: at this zoom there's a visible coastline-resolution mismatch
    # between the world coastline layer and the country layer (both trace
    # the coast at different detail). Subunits trace the coast at the right
    # resolution on their own, so we drop both upper layers in PNW.
    # subunit_width bumped because every shared subunit border is drawn by
    # both regions' polygons (~2× thickness on land), so to keep the *coast*
    # readable at this scale we have to thicken everything slightly.
    'pnw':           {'show_coastlines': False, 'show_countries': False, 'show_subunits': True,
                      'subunit_width': 1.0},
}

# --debug-borders presets — toggle visibility of geo features for artifact
# isolation. Apply by passing --debug-borders=N on the CLI; overrides the
# per-scope SCOPE_FEATURES defaults across all scopes.
DEBUG_BORDER_PRESETS = {
    '1.0': {'show_coastlines': True,  'show_countries': True,  'show_subunits': True},   # baseline
    '1.1': {'show_coastlines': False, 'show_countries': True,  'show_subunits': True},   # coast off
    '1.2': {'show_coastlines': True,  'show_countries': False, 'show_subunits': True},   # countries off
    '1.3': {'show_coastlines': True,  'show_countries': True,  'show_subunits': False},  # subunits off
    '1.4': {'show_coastlines': True,  'show_countries': False, 'show_subunits': False},  # coast only
}


def _load_snapshot_sections() -> dict:
    """Load all sections from the default snapshot path. Returns {} if absent."""
    snapshot_path = find_snapshot([str(DATA_DIR / 'drive_snapshot.csv')])
    if not snapshot_path:
        return {}
    sections, _ = read_snapshot(snapshot_path)
    return sections


def _historical_bounds_from_sections(sections: dict) -> list[dict]:
    """Extract historical city date-bounds from snapshot sections.

    Returns a list of ``{city_state, min_hist, max_hist}`` with the dates
    parsed to ``pd.Timestamp``. Rows missing either bound or with an
    unparseable city_state are skipped silently.
    """
    hist_df = sections.get('historical')
    if hist_df is None or hist_df.empty:
        return []
    out = []
    for _, row in hist_df.iterrows():
        cs = row.get('city_state')
        if not isinstance(cs, str) or not cs.strip():
            continue
        try:
            min_h = pd.Timestamp(row['min_hist'])
            max_h = pd.Timestamp(row['max_hist'])
        except (KeyError, TypeError, ValueError):
            continue
        if pd.isna(min_h) or pd.isna(max_h):
            continue
        out.append({'city_state': cs.strip(),
                    'min_hist': min_h,
                    'max_hist': max_h})
    return out


def fmt_date_range(d_min: pd.Timestamp, d_max: pd.Timestamp) -> str:
    if d_min == d_max:
        return d_min.strftime('%b %-d, %Y')
    if d_min.year == d_max.year and d_min.month == d_max.month:
        return d_min.strftime('%b %Y')
    if d_min.year == d_max.year:
        return (f'{calendar.month_abbr[d_min.month]}-'
                f'{calendar.month_abbr[d_max.month]} {d_min.year}')
    return f'{d_min.year}-{d_max.year}'


def _cities_trace(agg_filtered: pd.DataFrame, size_mult: float = 1.0) -> dict:
    """Cities Scattergeo trace, parameterized by which rows of `agg` to plot.

    ``size_mult`` scales every marker diameter — used by sparse scopes
    (Europe, Asia) where a 2× bump keeps low-mileage points visible.

    Cities with miles==0 (historical-only entries from the snapshot's
    ``historical`` section) hide the mileage line in the hover by emitting
    an empty miles_html string; the template concatenates it directly so
    no leftover blank line appears.
    """
    miles_html = [
        '' if float(m) == 0 else f'{float(m):,.1f} mi<br>'
        for m in agg_filtered['miles']
    ]
    return dict(
        type='scattergeo',
        meta='cities',
        lat=agg_filtered['lat'].tolist(),
        lon=agg_filtered['lon'].tolist(),
        mode='markers',
        marker=dict(
            size=(agg_filtered['size'] * size_mult).tolist(),
            color=agg_filtered['color'].tolist(),
            opacity=MARKER_OPACITY,
            line=dict(width=0.5, color=LINE_COLOR),
            sizemode='diameter',
        ),
        customdata=[
            [cs, mh, dr]
            for cs, mh, dr in zip(agg_filtered['city_state'],
                                  miles_html,
                                  agg_filtered['date_range'])
        ],
        hovertemplate=('<b>%{customdata[0]}</b><br>'
                       '%{customdata[1]}%{customdata[2]}<extra></extra>'),
        hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)',
                        bordercolor='#444',
                        font=dict(color='#eee')),
    )


def _build_scope_traces(scope: dict, agg: pd.DataFrame) -> list[dict]:
    """Cities filtered by the scope envelope. World's envelope spans the
    whole globe so the filter is a no-op — same code path as the regional
    scopes, no special case.
    """
    env = scope['envelope']
    m = ((agg['lon'] >= env['lon'][0]) & (agg['lon'] <= env['lon'][1]) &
         (agg['lat'] >= env['lat'][0]) & (agg['lat'] <= env['lat'][1]))
    return [_cities_trace(agg[m], size_mult=scope.get('point_size_mult', 1.0))]


def _scope_bbox(scope: dict, agg: pd.DataFrame) -> dict:
    """Bbox of the in-scope cities + per-scope padding. JS uses this to
    compute the aspect-fit lon/lat range.
    """
    env = scope['envelope']
    m = ((agg['lon'] >= env['lon'][0]) & (agg['lon'] <= env['lon'][1]) &
         (agg['lat'] >= env['lat'][0]) & (agg['lat'] <= env['lat'][1]))
    in_scope = agg[m]
    if len(in_scope) == 0:
        lon_min, lon_max = env['lon']
        lat_min, lat_max = env['lat']
    else:
        lon_min = float(in_scope['lon'].min()) - scope['pad_lon']
        lon_max = float(in_scope['lon'].max()) + scope['pad_lon']
        lat_min = float(in_scope['lat'].min()) - scope['pad_lat']
        lat_max = float(in_scope['lat'].max()) + scope['pad_lat']
    return {
        'lon_min': lon_min, 'lon_max': lon_max,
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_c': (lon_min + lon_max) / 2,
        'lat_c': (lat_min + lat_max) / 2,
        'lon_span': lon_max - lon_min,
        'lat_span': lat_max - lat_min,
    }


def _scope_geo(scope_id: str, feature_override: dict | None = None) -> dict:
    """Build the full `layout.geo` config for a given toggle id.

    `feature_override` (from --debug-borders) overrides show_coastlines /
    show_countries / show_subunits for all scopes uniformly.

    World scopes get explicit lon/lat range. Non-world scopes leave the
    range unset on the wire and rely on JS to apply ranges that match the
    plot div's aspect ratio. Plotly's geolayer SVG sizes itself by fitting
    the *projected* lon/lat box into the div, so when the projected aspect
    of the box equals the div aspect the geolayer fills the div edge-to-
    edge — no L/R or T/B empty BG inside the div.
    """
    cfg = SCOPE_CONFIG[scope_id]
    feats = dict(SCOPE_FEATURES[scope_id])
    if feature_override:
        feats.update(feature_override)
    subunit_width = feats.get('subunit_width', SUBUNIT_WIDTH)
    geo = dict(
        scope=cfg['plotly_scope'],
        projection=dict(type='natural earth'),
        resolution=50,
        bgcolor=BG,
        showcoastlines=feats['show_coastlines'],
        coastlinecolor=LINE_COLOR, coastlinewidth=COASTLINE_WIDTH,
        showcountries=feats['show_countries'],
        countrycolor=LINE_COLOR,   countrywidth=BORDER_WIDTH,
        showsubunits=feats['show_subunits'],
        subunitcolor=LINE_COLOR,   subunitwidth=subunit_width,
        showland=True, landcolor=LAND_COLOR,
        showocean=False, showlakes=False,
        showrivers=False, showframe=False,
    )
    # All scopes leave lon/lat range unset on the wire — JS writes them
    # in via aspect-fit at render time and on resize.
    return geo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--daily',   default=DEFAULT_DAILY)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--debug-borders', choices=sorted(DEBUG_BORDER_PRESETS),
                   default=None,
                   help='Override show_coastlines/countries/subunits across '
                        'all scopes for visual artifact isolation. Presets '
                        '1.0-1.4 — see DEBUG_BORDER_PRESETS.')
    args = p.parse_args()
    feature_override = (DEBUG_BORDER_PRESETS[args.debug_borders]
                        if args.debug_borders else None)
    if feature_override:
        print(f'[world-map] --debug-borders={args.debug_borders}: '
              f'{feature_override}')

    df = pd.read_csv(args.daily, parse_dates=['date'])
    df = df.dropna(subset=['city_state'])
    df = df[df['city_state'].astype(str).str.strip() != ''].copy()
    # Source of truth: watch/route distance-corrected mileage (decrease-only;
    # corr <= logged) drives per-city totals. On-disk daily.csv 'miles' is
    # untouched; pre-2016 race stubs and uncorrected rows keep logged miles.
    df['miles'] = effective_daily_miles(df)

    agg = (df.groupby('city_state', as_index=False)
             .agg(miles=('miles', 'sum'),
                  d_min=('date', 'min'),
                  d_max=('date', 'max')))

    # Load snapshot once and pull both coordinate overrides and the
    # historical bounds out of it. Historical rows extend a city's date
    # range (or seed a new city with miles=0) so cities Max remembers
    # running in but never logged appear on the map alongside ones with
    # mileage.
    sections = _load_snapshot_sections()
    historical = _historical_bounds_from_sections(sections)
    if historical:
        existing = agg.set_index('city_state')
        new_rows = []
        for h in historical:
            cs = h['city_state']
            if cs in existing.index:
                cur_min = cast(date, existing.at[cs, 'd_min'])
                cur_max = cast(date, existing.at[cs, 'd_max'])
                existing.at[cs, 'd_min'] = min(cur_min, h['min_hist'])
                existing.at[cs, 'd_max'] = max(cur_max, h['max_hist'])
            else:
                new_rows.append({'city_state': cs, 'miles': 0.0,
                                 'd_min': h['min_hist'],
                                 'd_max': h['max_hist']})
        agg = existing.reset_index()
        if new_rows:
            agg = pd.concat([agg, pd.DataFrame(new_rows)], ignore_index=True)
        print(f'[world-map] merged {len(historical)} historical entries '
              f'(extended-or-added to {len(agg)} cities)')

    agg['date_range'] = [fmt_date_range(a, b)
                         for a, b in zip(agg['d_min'], agg['d_max'])]

    # Categorize colors with historical cities included so they pick up
    # their state's color (e.g. Coulee City, WA → WA-other shade) instead
    # of falling through to the global gray. Inject synthetic miles=0.01
    # rows for historical-only cities — well below PEAK_YEAR_THRESHOLD
    # (26.2) so they can't tip a state into qualifying — purely to slot
    # them into the right state bucket inside build_categories.
    if historical:
        hist_synth = pd.DataFrame([
            {'city_state': h['city_state'], 'miles': 0.01,
             'date': h['min_hist']}
            for h in historical
        ])
        df_for_cat = pd.concat([df, hist_synth], ignore_index=True)
    else:
        df_for_cat = df
    label_for, _, label_color, *_ = build_categories(df_for_cat)
    agg['color'] = agg['city_state'].map(
        lambda cs: label_color.get(label_for.get(cs, '')))
    agg['color'] = agg['color'].fillna('#888888')

    # Load any coordinate overrides from the snapshot. The "coordinates"
    # section is a city_state -> (lat, lon) override applied on top of the
    # Nominatim cache; lets us correct geocoding errors reproducibly from
    # source instead of hand-editing data/city_coords.csv.
    coord_overrides = coord_overrides_from_sections(sections)
    coords = ensure_coords(agg['city_state'].tolist(),
                           overrides=coord_overrides)
    agg['lat'] = agg['city_state'].map(
        lambda cs: coords.get(cast(str, cs), (None, None))[0])
    agg['lon'] = agg['city_state'].map(
        lambda cs: coords.get(cast(str, cs), (None, None))[1])

    missing = agg[agg['lat'].isna()]
    if len(missing):
        print(f'[world-map] {len(missing)} city-state(s) have no coords - '
              f'omitted from map:')
        print(missing[['city_state', 'miles']]
              .sort_values('miles', ascending=False)
              .to_string(index=False))
    agg = agg.dropna(subset=['lat', 'lon']).copy()

    # Display only: collapse foreign 'City, Country' to 'City, CC' for the
    # hover label now that geocoding (which needs the full country name) is
    # done. US/CA city_states already carry a 2-letter code and pass through.
    agg['city_state'] = agg['city_state'].map(country_abbrev)

    agg['size'] = (np.log10(np.maximum(agg['miles'], SIZE_FLOOR_MILES) + 1)
                   * SIZE_SCALE + SIZE_OFFSET)

    # Smaller cities render on top of larger ones so they don't get hidden.
    agg = agg.sort_values('miles', ascending=False)

    scope_traces = {}    # sid -> list[trace dict]
    scope_layouts = {}   # sid -> geo dict
    scope_bboxes = {}    # sid -> bbox dict
    for s in SCOPES:
        scope_traces[s['id']]  = _build_scope_traces(s, agg)
        scope_layouts[s['id']] = _scope_geo(s['id'], feature_override)
        scope_bboxes[s['id']]  = _scope_bbox(s, agg)

    fig = go.Figure(data=scope_traces[DEFAULT_SCOPE])
    fig.update_layout(geo=scope_layouts[DEFAULT_SCOPE])

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
    for s in SCOPES:
        if 'envelope' in s:
            n = len([t for t in scope_traces[s['id']]
                     if t.get('meta') == 'cities'][0]['lat'])
            print(f'[world-map]   scope {s["id"]:14s}: {n} cities')

    overlay_html = _build_scope_toggle(scope_traces, scope_layouts, scope_bboxes)

    # World map specifically wants its plot div to fill the iframe edge-to-
    # edge — there's no axis chrome to reserve space for, and at non-world
    # scopes the geolayer hugs the data so closely that any title bar
    # eating top pixels is wasted real estate. Override the shared scaffold
    # so the plot extends to top:0, and restyle the title as a floating
    # inset box (mirroring the scope-toggle's chip styling) so it stays
    # legible over whatever geography it sits on top of.
    inset_css = """
.plotly-graph-div, #plot-container, .js-plotly-plot {
  top: 0 !important;
}
.rp-title-bar {
  position: fixed !important;
  top: 14px !important;
  left: 20px !important;
  right: auto !important;
  height: auto !important;
  background: rgba(26,26,26,0.92) !important;
  border: 1px solid #444 !important;
  border-radius: 6px !important;
  padding: 8px 14px 6px !important;
  z-index: 1000 !important;
}
/* Inside the mobile shell the 44px hamburger floats at the viewport's
   top-left — snug the inset up into that corner and pad its left so the
   button reads as sitting INSIDE the inset box rather than colliding
   with it. (Overrides base.css's html.rp-mobile padding-left, which this
   block's !important padding above would otherwise defeat.) */
html.rp-mobile .rp-title-bar {
  top: 0 !important;
  left: 0 !important;
  min-height: 44px;
  box-sizing: border-box;
  border-top-left-radius: 0 !important;
  border-top-width: 0 !important;
  border-left-width: 0 !important;
  padding: 8px 14px 6px 52px !important;
}
/* Mobile: the scope toggle rides flush with the top edge, matching the
   title inset, to save vertical space over the map. Keyed on the single
   mobile signal (see _scaffold/mobile.js). */
html.rp-mobile #scope-toggle { top: 0; }
"""

    out_path = os.path.join(args.out_dir, 'world_map.html')
    render_plot(
        fig, out_path,
        title_slug='world_map',
        page_title='World Map',
        title='World map',
        subtitle=f'{total_miles:,.0f} mi across {total_cities} cities',
        overlay_html=overlay_html,
        overlay_js_files=[_WORLD_MAP_JS],
        extra_head_css=inset_css,
        plotly_config={
            'scrollZoom': False,
            'displayModeBar': False,
        },
    )
    print(f'wrote {out_path}')


def _build_scope_toggle(scope_traces: dict, scope_layouts: dict,
                        scope_bboxes: dict) -> str:
    """Top-right pill bar that swaps geo.scope.

    HTML structure here; behavior (aspect-fit projection math, scope
    swap, resize refit) lives in make_world_map.js.
    """
    lat_clamps = {s['id']: s.get('lat_clamp', [-85, 85]) for s in SCOPES}
    globals_html = widgets.js_globals({
        'SCOPE_TRACES':   scope_traces,
        'SCOPE_LAYOUTS':  scope_layouts,
        'SCOPE_BBOXES':   scope_bboxes,
        'LAT_CLAMPS':     lat_clamps,
        'DEFAULT_SCOPE':  DEFAULT_SCOPE,
    })
    bar_html = widgets.toggle_bar(
        'scope-toggle',
        [(s['id'], s['label']) for s in SCOPES],
        default_id=DEFAULT_SCOPE,
    )
    return globals_html + '\n' + bar_html


if __name__ == '__main__':
    main()
