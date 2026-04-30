"""Mileage by city-state, stacked bars with yearly/monthly toggle.

A qualitative geography view of where Max ran across the years.

Color logic
-----------
A city-state gets its own color if it logged at least PEAK_YEAR_THRESHOLD
miles in any single calendar year (default 26.2 — one marathon's worth).
A *state* gets a color if either its peak city qualifies, or the cumulative
state miles in any single year hit the threshold.

Three palette tiers (plus targeted overrides):
  - ANCHOR states (WA/TN/CO/IL/VA/NY/UK): bold + saturated + medium-dark.
  - DARK SINGLETON minors: deep, saturated, no need for shading.
  - STANDARD minors (multi-trace minor states): lighter & less saturated.
  - STATE_PALETTE_OVERRIDE: per-state explicit (h, l, s) for special cases
    like MI = royal blue.

(Other) collapse rule
---------------------
If a state's "(other)" bucket contains exactly one city, that city is
promoted to a named entry — the (other) label disappears for that state.
This keeps the legend/hover concise (no "(other)" subcategory holding a
single city). Currently affects HI, JP, MI in Max's data.

Bar pixel uniformity
--------------------
type='category' axis + post-render JS that snaps every bar's SVG path to
integer pixels with a fixed pixel gap between bins (5px yearly, 1px
monthly).
"""
import argparse
import colorsys
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT   = str(OUTPUT_DIR)

PEAK_YEAR_THRESHOLD = 26.2

# ---- Palette ----
ANCHOR_HUES = {
    'WA': 280,   # purple
    'TN': 130,   # green
    'CO':  30,   # orange
    'IL': 190,   # cyan
    'VA': 325,   # rose-magenta
    'NY':  50,   # yellow
    'UK': 350,   # red
}
ANCHOR_BASE_L = 0.42
ANCHOR_BASE_S = 0.78

# Minor hues (no MI here — see STATE_PALETTE_OVERRIDE).
MINOR_HUES = {
    'AB':    165,   # teal-green
    'LA':    250,   # indigo
    'MA':     10,   # red-orange (brick)
    'HI':    110,   # yellow-green
    'China':   0,   # red
    'DE':     65,   # gold
    'FR':    235,   # blue
    'CH':    215,   # blue (paler)
    'JP':    300,   # magenta-pink
}

# Per-state explicit overrides — bypass tier classification entirely.
# Use for hand-picked colors where tier-based generation doesn't fit.
STATE_PALETTE_OVERRIDE = {
    'MI': (220, 0.50, 0.70),   # royal blue (~#265cc4)
}

DARK_SINGLETON_MINORS = {'AB', 'LA', 'MA', 'China', 'DE', 'CH'}
DARK_MINOR_L = 0.32
DARK_MINOR_S = 0.62
STD_MINOR_L  = 0.55
STD_MINOR_S  = 0.40

NAMED_LIGHT_TOP_OFFSET   = 0.18
OTHER_LIGHT_OFFSET_2     = 0.18
OTHER_LIGHT_OFFSET_3PLUS = 0.30
OTHER_SAT_FACTOR         = 0.55

GLOBAL_OTHER_LABEL = 'Other'   # was 'Other (travel)'
GLOBAL_OTHER_KEY   = '__GLOBAL_OTHER__'
GLOBAL_OTHER_COLOR = '#888888'

# Mode-specific bar gap in pixels (set in JS).
GAP_PX_YEAR  = 5
GAP_PX_MONTH = 1


def hsl_to_hex(h_deg, s, l):
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def get_state(city_state):
    if not isinstance(city_state, str):
        return None
    m = re.search(r',\s*([A-Z]{2,3})$', city_state)
    return m.group(1) if m else city_state.strip()


def fmt_miles(v):
    if v >= 100:
        return f'{v:,.0f}'
    return f'{v:.1f}'


def state_palette_base(state):
    if state in STATE_PALETTE_OVERRIDE:
        return STATE_PALETTE_OVERRIDE[state]
    if state in ANCHOR_HUES:
        return ANCHOR_HUES[state], ANCHOR_BASE_L, ANCHOR_BASE_S
    if state in MINOR_HUES:
        if state in DARK_SINGLETON_MINORS:
            return MINOR_HUES[state], DARK_MINOR_L, DARK_MINOR_S
        return MINOR_HUES[state], STD_MINOR_L, STD_MINOR_S
    return (hash(state) % 360), STD_MINOR_L, STD_MINOR_S


def shade_color(state, rank, n_named, is_other, n_total):
    h, base_l, base_s = state_palette_base(state)
    if is_other:
        if n_total == 2:
            l = base_l + OTHER_LIGHT_OFFSET_2
        else:
            l = base_l + OTHER_LIGHT_OFFSET_3PLUS
        s = base_s * OTHER_SAT_FACTOR
        return hsl_to_hex(h, max(0.05, min(0.95, s)), max(0.05, min(0.92, l)))
    if n_named == 1:
        l = base_l
    else:
        l = base_l + rank * (NAMED_LIGHT_TOP_OFFSET / (n_named - 1))
    return hsl_to_hex(h, base_s, max(0.05, min(0.92, l)))


def collapse_singleton_others(label_for):
    """If a state's '(other)' bucket — or the global Other — contains
    exactly one city, promote that city to a named entry."""
    label_to_cities = {}
    for cs, lab in label_for.items():
        label_to_cities.setdefault(lab, []).append(cs)
    new = dict(label_for)
    for lab, cities in label_to_cities.items():
        if (lab.endswith(' (other)') or lab == GLOBAL_OTHER_LABEL) \
                and len(cities) == 1:
            new[cities[0]] = cities[0]
    return new


def build_categories(df):
    df = df[df['miles'] > 0].copy()
    df['state'] = df['city_state'].apply(get_state)
    df['year']  = df['date'].dt.year

    city_year = (df.groupby(['city_state', 'state', 'year'])['miles'].sum()
                   .reset_index())
    city_peak = (city_year.groupby(['city_state', 'state'])['miles'].max()
                          .reset_index()
                          .rename(columns={'miles': 'peak_year_miles'}))
    city_peak['qualifies'] = city_peak['peak_year_miles'] >= PEAK_YEAR_THRESHOLD

    state_year = df.groupby(['state', 'year'])['miles'].sum().reset_index()
    state_peak = state_year.groupby('state')['miles'].max().to_dict()
    states_with_qual_city = set(city_peak.loc[city_peak['qualifies'], 'state'])
    qual_states = states_with_qual_city | {
        st for st, m in state_peak.items() if m >= PEAK_YEAR_THRESHOLD}

    label_for = {}
    for _, r in city_peak.iterrows():
        cs, st = r['city_state'], r['state']
        if r['qualifies']:
            label_for[cs] = cs
        elif st in qual_states:
            label_for[cs] = f'{st} (other)'
        else:
            label_for[cs] = GLOBAL_OTHER_LABEL

    # Apply the singleton-(other) collapse rule
    label_for = collapse_singleton_others(label_for)

    city_total  = df.groupby('city_state')['miles'].sum().to_dict()
    label_total = {}
    for cs, lab in label_for.items():
        label_total[lab] = label_total.get(lab, 0.0) + city_total[cs]

    state_to_labels = {}
    for cs, lab in label_for.items():
        st = get_state(cs)
        state_to_labels.setdefault(st, set()).add(lab)
    group_for_label = {}
    for lab in label_total:
        if lab == GLOBAL_OTHER_LABEL:
            group_for_label[lab] = GLOBAL_OTHER_KEY
            continue
        if lab.endswith(' (other)'):
            st = lab.split(' (other)')[0]
        else:
            st = get_state(lab)
        if len(state_to_labels.get(st, set())) >= 2:
            group_for_label[lab] = st
        else:
            group_for_label[lab] = lab

    group_total = {}
    for lab, total in label_total.items():
        g = group_for_label[lab]
        group_total[g] = group_total.get(g, 0.0) + total
    group_size = {}
    for lab in label_total:
        g = group_for_label[lab]
        group_size[g] = group_size.get(g, 0) + 1

    sorted_groups = sorted(group_total, key=group_total.get, reverse=True)
    if GLOBAL_OTHER_KEY in sorted_groups:
        sorted_groups.remove(GLOBAL_OTHER_KEY)
        sorted_groups.append(GLOBAL_OTHER_KEY)

    ordered_labels = []
    for g in sorted_groups:
        members = [lab for lab, gg in group_for_label.items() if gg == g]
        named  = sorted([m for m in members if not m.endswith(' (other)')
                                            and m != GLOBAL_OTHER_LABEL],
                        key=lambda x: -label_total[x])
        others = [m for m in members if m.endswith(' (other)')
                                        or m == GLOBAL_OTHER_LABEL]
        ordered_labels.extend(named + others)

    label_color = {GLOBAL_OTHER_LABEL: GLOBAL_OTHER_COLOR}
    # Color anchors and minors that have a state-color
    states_with_traces = {get_state(cs) for cs, lab in label_for.items()
                          if lab != GLOBAL_OTHER_LABEL}
    for st in states_with_traces:
        members = [lab for lab in label_total
                   if (not lab.endswith(' (other)') and
                       lab != GLOBAL_OTHER_LABEL and
                       get_state(lab) == st)
                   or (lab == f'{st} (other)')]
        named = sorted([m for m in members if not m.endswith(' (other)')],
                       key=lambda x: -label_total[x])
        has_other = f'{st} (other)' in label_total
        n_named = len(named)
        n_total = n_named + (1 if has_other else 0)
        for i, lab in enumerate(named):
            label_color[lab] = shade_color(st, i, n_named, False, n_total)
        if has_other:
            label_color[f'{st} (other)'] = shade_color(
                st, n_named, n_named, True, n_total)

    # Build per-(other) sub-city lists (city, lifetime miles), sorted desc
    other_sublists = {}
    for cs, lab in label_for.items():
        if lab.endswith(' (other)') or lab == GLOBAL_OTHER_LABEL:
            other_sublists.setdefault(lab, []).append((cs, city_total[cs]))
    for lab in other_sublists:
        other_sublists[lab].sort(key=lambda t: -t[1])

    return (label_for, ordered_labels, label_color, label_total,
            group_for_label, group_total, group_size, other_sublists)


# ---------- aggregation ----------
def yearly_bins(df):
    yrs = sorted(df['date'].dt.year.unique())
    return [str(y) for y in range(yrs[0], yrs[-1] + 1)]


def monthly_bins(df):
    yrs = sorted(df['date'].dt.year.unique())
    bins = []
    for y in range(yrs[0], yrs[-1] + 1):
        for m in range(1, 13):
            bins.append(f'{y}-{m:02d}')
    return bins


def aggregate(df, freq, label_for, ordered_labels, all_bins):
    df = df[df['miles'] > 0].copy()
    df['label'] = df['city_state'].map(label_for)
    if freq == 'Y':
        df['bin'] = df['date'].dt.year.astype(str)
    else:
        df['bin'] = df['date'].dt.to_period('M').astype(str)
    pivot = (df.groupby(['bin', 'label'])['miles'].sum()
               .unstack(fill_value=0.0))
    for c in ordered_labels:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot.reindex(all_bins, fill_value=0.0)
    return pivot[ordered_labels]


def build_subcity_data(df, label_for, freq):
    """Per-bin sub-city breakdown for grouped labels.

    Returns nested dict: {label: {bin: [(city, miles), ...sorted desc]}}.
    Used both for in-tooltip display and for the bin-level collapse rule
    that promotes a single-city (other) bucket to a flat city entry in
    that specific bin's hover.
    """
    df = df[df['miles'] > 0].copy()
    df['label'] = df['city_state'].map(label_for)
    if freq == 'Y':
        df['bin'] = df['date'].dt.year.astype(str)
    else:
        df['bin'] = df['date'].dt.to_period('M').astype(str)
    out = {}
    grouped = {lab for lab in df['label'].unique()
               if lab == GLOBAL_OTHER_LABEL or lab.endswith(' (other)')}
    for lab in grouped:
        sub = df[df['label'] == lab]
        bins_data = (sub.groupby(['bin', 'city_state'])['miles'].sum()
                        .reset_index())
        out[lab] = {}
        for b, g in bins_data.groupby('bin'):
            g = g.sort_values('miles', ascending=False)
            out[lab][b] = [(r['city_state'], r['miles']) for _, r in g.iterrows()]
    return out


def render_subcity_html(cities):
    """Render a list of (city, miles) tuples as indented hover sub-rows."""
    return ''.join(
        f'<div class="hov-subcity">{c}: {fmt_miles(m)} mi</div>'
        for c, m in cities
    )


def fmt_bin_label(b, freq):
    if freq == 'Y':
        return b
    yr, mo = b.split('-')
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    return f'{months[int(mo)-1]} {yr}'


def build_bin_hover(pivot, freq, ordered_labels, label_color,
                    group_for_label, subcity_data):
    """Per-bin hover HTML.

    Per-bin collapse rule: if an (other) entry has exactly one sub-city
    contributing in this bin, render the city directly (with the (other)
    color) instead of an (other) row + sub-list.
    """
    hover = {}
    for b in pivot.index:
        row = pivot.loc[b]

        # Bucket (label, value, color, render_name, sub_cities) by group.
        # render_name and color may differ from `lab`/its color when the
        # bin-level collapse fires.
        group_items = {}
        for lab in ordered_labels:
            v = row[lab]
            if v <= 0:
                continue
            g = group_for_label[lab]
            sub_cities = subcity_data.get(lab, {}).get(b, [])
            is_othery = (lab.endswith(' (other)') or lab == GLOBAL_OTHER_LABEL)
            if is_othery and len(sub_cities) == 1:
                # Collapse: surface the sole city, drop the sub-list
                sole_city, _ = sub_cities[0]
                render_name = sole_city
                render_sub  = []
            else:
                render_name = lab
                render_sub  = sub_cities if is_othery else []
            group_items.setdefault(g, []).append(
                (lab, v, label_color[lab], render_name, render_sub))

        header = f'<div class="hov-day">{fmt_bin_label(b, freq)}</div>'
        if not group_items:
            hover[b] = header
            continue

        group_bin_total = {g: sum(it[1] for it in items)
                           for g, items in group_items.items()}
        sorted_groups = sorted(group_bin_total,
                               key=group_bin_total.get, reverse=True)
        if GLOBAL_OTHER_KEY in sorted_groups:
            sorted_groups.remove(GLOBAL_OTHER_KEY)
            sorted_groups.append(GLOBAL_OTHER_KEY)

        parts = [header]
        for g in sorted_groups:
            items = group_items[g]
            # Within group: named-style first by miles desc, (other)-style at end.
            # An item is "named-style" iff its underlying lab is NOT (other) —
            # this includes bin-collapsed entries because they surface the
            # underlying city even though the lab itself is (other). Keep
            # collapsed entries as named-style for ordering.
            named  = sorted(
                [it for it in items
                 if not (it[0].endswith(' (other)') and it[3] == it[0])
                 and not (it[0] == GLOBAL_OTHER_LABEL and it[3] == it[0])],
                key=lambda t: -t[1])
            others = [it for it in items
                      if (it[0].endswith(' (other)') and it[3] == it[0])
                      or (it[0] == GLOBAL_OTHER_LABEL and it[3] == it[0])]
            items_ordered = named + others

            render_as_group = (len(items_ordered) >= 2 and g != GLOBAL_OTHER_KEY)
            if render_as_group:
                parts.append(
                    f'<div class="hov-group-title"><b>{g}</b>'
                    f'<span class="hov-grouptotal">  ·  '
                    f'{fmt_miles(group_bin_total[g])} mi</span></div>')
                for lab, val, color, name, sub in items_ordered:
                    parts.append(
                        f'<div class="hov-item grouped">'
                        f'<span class="hov-box" style="background:'
                        f'{color}"></span>'
                        f'<span class="hov-name">{name}</span>'
                        f'<span class="hov-val">  '
                        f'<b>{fmt_miles(val)}</b> mi</span></div>')
                    if sub:
                        parts.append(
                            f'<div class="hov-subs">{render_subcity_html(sub)}</div>')
            else:
                for lab, val, color, name, sub in items_ordered:
                    parts.append(
                        f'<div class="hov-item">'
                        f'<span class="hov-box" style="background:'
                        f'{color}"></span>'
                        f'<span class="hov-name">{name}</span>'
                        f'<span class="hov-val">  '
                        f'<b>{fmt_miles(val)}</b> mi</span></div>')
                    if sub:
                        parts.append(
                            f'<div class="hov-subs">{render_subcity_html(sub)}</div>')

        total = sum(group_bin_total.values())
        parts.append(f'<div class="hov-total">total: '
                     f'<b>{fmt_miles(total)}</b> mi</div>')
        hover[b] = ''.join(parts)
    return hover


def build_bar_traces(yearly, monthly, ordered_labels, label_color):
    traces = []
    y_x = list(yearly.index)
    m_x = list(monthly.index)
    for lab in ordered_labels:
        traces.append(go.Bar(
            x=y_x,
            y=yearly[lab].tolist(),
            marker_color=label_color[lab],
            marker_line_width=0,
            hoverinfo='skip',
            showlegend=False,
            meta={'x_year':  y_x,
                  'y_year':  yearly[lab].tolist(),
                  'x_month': m_x,
                  'y_month': monthly[lab].tolist()},
        ))
    return traces


# ---------- legend ----------
def build_legend_html(ordered_labels, label_color, label_total,
                      group_for_label, group_total, group_size,
                      other_sublists):
    """Custom HTML legend.

    Group items are indented; singletons flush. Each (other) item (and the
    global Other) gets a chevron at the start of the row that expands a
    sub-list of contributing cities. Sub-list is hidden by default.
    """
    parts = ['<div id="geo-legend"><div class="legend-inner">']
    last_group = None
    for i, lab in enumerate(ordered_labels):
        g = group_for_label[lab]
        if g != last_group:
            if last_group is not None:
                parts.append('</div>')
            is_real_group = (group_size[g] >= 2 and g != GLOBAL_OTHER_KEY)
            if is_real_group:
                parts.append(f'<div class="legend-group" data-group-key="{g}">')
                parts.append(
                    f'<div class="legend-title">'
                    f'<b>{g}</b>'
                    f'<span class="legend-life">  ·  '
                    f'{fmt_miles(group_total[g])} mi</span></div>')
            else:
                parts.append('<div class="legend-block">')
            last_group = g

        is_grouped   = (group_size[g] >= 2 and g != GLOBAL_OTHER_KEY)
        is_othery    = lab in other_sublists  # has a sub-list to expand
        cls = 'legend-item' + (' grouped' if is_grouped else '')
        chev_html = ('<span class="chev expandable">▶</span>' if is_othery
                     else '<span class="chev"></span>')
        parts.append(
            f'<div class="{cls}" data-trace-idx="{i}">'
            f'{chev_html}'
            f'<span class="legend-box" style="background:'
            f'{label_color[lab]}"></span>'
            f'<span class="legend-name">{lab}</span>'
            f'<span class="legend-life">  ·  '
            f'{fmt_miles(label_total[lab])} mi</span></div>')

        if is_othery:
            sub_cls_inner = ('legend-subcity'
                             + (' grouped' if is_grouped else ' singleton'))
            parts.append('<div class="legend-subgroup">')
            for city, miles in other_sublists[lab]:
                parts.append(
                    f'<div class="{sub_cls_inner}">{city}: '
                    f'{fmt_miles(miles)} mi</div>')
            parts.append('</div>')
    if last_group is not None:
        parts.append('</div>')
    parts.append('</div></div>')
    return ''.join(parts)


def build_tickvals_for_monthly(monthly_bin_list):
    tv = [b for b in monthly_bin_list if b.endswith('-01')]
    tt = [b.split('-')[0] for b in tv]
    return tv, tt


# ---------- write HTML ----------
def write_html(fig, path, legend_html, payload):
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive': True})

    fullview_css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'background:#1a1a1a;color:#eee;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",'
        'Roboto,Arial,sans-serif;}'
        '.plotly-graph-div,.js-plotly-plot{width:100%!important;'
        'height:100vh!important;}'
        '.barlayer path{shape-rendering:crispEdges;}'
        '</style>'
    )

    overlay_css = r"""
<style>
#geo-toggle {
  position: fixed; top: 56px; right: 20px;
  background: rgba(26,26,26,0.92);
  border: 1px solid #444; border-radius: 6px;
  padding: 4px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 13px; z-index: 1000;
  display: flex; gap: 2px;
}
#geo-toggle .gt-btn {
  padding: 6px 16px; cursor: pointer; border-radius: 4px;
  color: #aaa; background: transparent; border: none;
  transition: background 0.12s, color 0.12s;
  font-size: 13px; font-family: inherit;
}
#geo-toggle .gt-btn:hover { color: #fff; background: #333; }
#geo-toggle .gt-btn.active {
  background: #4aa3ff; color: #fff; font-weight: 500;
}

#geo-legend {
  position: fixed; top: 110px; right: 20px; bottom: 80px;
  width: 300px;
  background: rgba(26,26,26,0.94);
  border: 1px solid #3a3a3a; border-radius: 4px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 13px; color: #ccc;
  z-index: 100;
  overflow-y: auto;
  padding: 8px 0;
}
#geo-legend .legend-inner { padding: 0 12px; }
#geo-legend .legend-group { margin-bottom: 8px; }
#geo-legend .legend-block { margin-bottom: 4px; }
#geo-legend .legend-title {
  font-weight: 600; color: #fff; font-size: 14px;
  cursor: pointer; padding: 2px 0;
  user-select: none;
  line-height: 1.25;
}
#geo-legend .legend-title:hover { color: #4aa3ff; }
#geo-legend .legend-item {
  display: flex; align-items: center; gap: 6px;
  padding: 2px 0; cursor: pointer;
  user-select: none;
  white-space: nowrap;
  line-height: 1.25;
}
#geo-legend .legend-item.grouped { padding-left: 14px; }
#geo-legend .legend-item:hover { color: #fff; }
#geo-legend .chev {
  display: inline-block; width: 13px; flex-shrink: 0;
  text-align: center; font-size: 11px; color: #aaa;
  line-height: 1; transition: transform 0.15s;
}
#geo-legend .chev.expandable { cursor: pointer; }
#geo-legend .chev.expandable:hover { color: #fff; }
#geo-legend .chev.expanded { transform: rotate(90deg); }
#geo-legend .legend-box {
  display: inline-block; width: 12px; height: 12px;
  border-radius: 1px; flex-shrink: 0;
}
#geo-legend .legend-name { color: #ddd; }
#geo-legend .legend-life {
  color: #888; font-size: 11px; margin-left: 2px;
}
#geo-legend .legend-item.hidden .legend-box { opacity: 0.18; }
#geo-legend .legend-item.hidden .legend-name {
  color: #666; text-decoration: line-through;
}
#geo-legend .legend-subgroup { display: none; }
#geo-legend .legend-subgroup.expanded { display: block; }
#geo-legend .legend-subcity {
  font-size: 12px; color: #889;
  white-space: nowrap;
  line-height: 1.25;
  padding: 1px 0;
}
#geo-legend .legend-subcity.singleton { padding-left: 22px; }
#geo-legend .legend-subcity.grouped   { padding-left: 36px; }

#geo-tooltip {
  position: fixed; top: 0; left: 0;
  background: rgba(26,26,26,0.96);
  color: #eee;
  border: 1px solid #555;
  padding: 9px 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 12px; line-height: 1.45;
  border-radius: 4px;
  pointer-events: none;
  z-index: 9999; max-width: 360px;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
#geo-tooltip .hov-day {
  font-weight: 600; font-size: 13px; color: #fff;
  margin-bottom: 5px;
}
#geo-tooltip .hov-group-title {
  font-weight: 600; color: #fff; margin-top: 4px;
}
#geo-tooltip .hov-grouptotal {
  color: #999; font-weight: 400; font-size: 11px;
}
#geo-tooltip .hov-item {
  display: flex; align-items: center; gap: 6px;
  white-space: nowrap; padding: 1px 0;
}
#geo-tooltip .hov-item.grouped { padding-left: 14px; }
#geo-tooltip .hov-box {
  display: inline-block; width: 9px; height: 9px;
  border-radius: 1px; flex-shrink: 0;
}
#geo-tooltip .hov-name { color: #ddd; }
#geo-tooltip .hov-val { color: #ddd; }
#geo-tooltip .hov-val b { color: #fff; }
#geo-tooltip .hov-subs { padding-left: 28px; }
#geo-tooltip .hov-subcity {
  font-size: 11px; color: #9aa; line-height: 1.35;
}
#geo-tooltip .hov-total {
  margin-top: 6px; padding-top: 6px;
  border-top: 1px solid #444;
  color: #aaa; font-size: 12px;
}
#geo-tooltip .hov-total b { color: #fff; }

#geo-spike {
  position: fixed; top: 0; left: 0;
  width: 1px; height: 100vh;
  background: rgba(255,255,255,0.25);
  pointer-events: none;
  z-index: 9998;
  display: none;
}
</style>
"""

    toggle_html = (
        '<div id="geo-toggle">'
        '<button class="gt-btn active" data-mode="year">Yearly</button>'
        '<button class="gt-btn" data-mode="month">Monthly</button>'
        '</div>'
    )
    tooltip_html = '<div id="geo-tooltip"></div><div id="geo-spike"></div>'

    js = r"""
<script>
(function() {
  var GEO = __GEO_PAYLOAD__;
  var mode = 'year';

  function pdiv() { return document.querySelector('.plotly-graph-div'); }

  // ===== Bar pixel snap (uniform per-mode pixel gap) =====
  var BAR_PATH_RE = /^M([-\d.]+),([-\d.]+)V([-\d.]+)H([-\d.]+)V([-\d.]+)Z$/;

  function snapBars() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) return;
    var fl = gd._fullLayout;
    var bg = fl._size;
    var plotW = bg.w;
    var nBins = GEO[mode].bins.length;
    if (nBins === 0 || plotW <= 0) return;
    var pitch = plotW / nBins;
    var gap = GEO[mode].gap_px;

    var paths = gd.querySelectorAll('.barlayer .point path');
    paths.forEach(function(path) {
      var d = path.getAttribute('d');
      if (!d) return;
      var m = d.match(BAR_PATH_RE);
      if (!m) return;
      var x1 = parseFloat(m[1]);
      var y1 = parseFloat(m[2]);
      var y2 = parseFloat(m[3]);
      var x2 = parseFloat(m[4]);
      if (Math.abs(y1 - y2) < 0.5) return;
      var center = (x1 + x2) / 2;
      var binIdx = Math.round(center / pitch - 0.5);
      if (binIdx < 0) binIdx = 0;
      if (binIdx >= nBins) binIdx = nBins - 1;
      var newLeft  = Math.round(binIdx * pitch);
      var newRight = Math.round((binIdx + 1) * pitch) - gap;
      if (newRight <= newLeft) newRight = newLeft + 1;
      var newD = 'M' + newLeft + ',' + y1 + 'V' + y2 + 'H' + newRight + 'V' + y1 + 'Z';
      path.setAttribute('d', newD);
    });
  }

  // ===== Mode toggle =====
  function applyMode() {
    var gd = pdiv();
    if (!gd || !gd.data) return;
    var n = gd.data.length;
    var x = [], y = [];
    for (var i = 0; i < n; i++) {
      var m = gd.data[i].meta || {};
      x.push(mode === 'year' ? m.x_year  : m.x_month);
      y.push(mode === 'year' ? m.y_year  : m.y_month);
    }
    Plotly.restyle(gd, {x: x, y: y});
    var modeData = GEO[mode];
    Plotly.relayout(gd, {
      'xaxis.tickvals': modeData.tickvals,
      'xaxis.ticktext': modeData.ticktext,
    });
  }

  document.querySelectorAll('#geo-toggle .gt-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var newMode = btn.getAttribute('data-mode');
      if (newMode === mode) return;
      mode = newMode;
      document.querySelectorAll('#geo-toggle .gt-btn').forEach(function(b) {
        b.classList.toggle('active', b === btn);
      });
      applyMode();
    });
  });

  // ===== Legend interaction =====
  function setVisible(indices, visible) {
    var gd = pdiv();
    if (!gd) return;
    Plotly.restyle(gd, {visible: visible ? true : 'legendonly'}, indices);
  }

  document.querySelectorAll('#geo-legend .legend-item').forEach(function(el) {
    el.addEventListener('click', function() {
      var idx = parseInt(el.getAttribute('data-trace-idx'));
      var hidden = el.classList.contains('hidden');
      setVisible([idx], hidden);
      el.classList.toggle('hidden', !hidden);
    });
  });

  // Chevron click toggles expand/collapse of the next-sibling subgroup.
  // stopPropagation so the row's visibility-toggle handler doesn't fire.
  document.querySelectorAll('#geo-legend .chev.expandable').forEach(function(chev) {
    chev.addEventListener('click', function(e) {
      e.stopPropagation();
      var item = chev.closest('.legend-item');
      if (!item) return;
      var sub = item.nextElementSibling;
      if (!sub || !sub.classList.contains('legend-subgroup')) return;
      var willExpand = !sub.classList.contains('expanded');
      sub.classList.toggle('expanded', willExpand);
      chev.classList.toggle('expanded', willExpand);
    });
  });

  document.querySelectorAll('#geo-legend .legend-title').forEach(function(el) {
    el.addEventListener('click', function() {
      var groupDiv = el.parentElement;
      var items = groupDiv.querySelectorAll('.legend-item');
      var allHidden = Array.from(items).every(function(it) {
        return it.classList.contains('hidden');
      });
      var visible = allHidden;
      var indices = [];
      items.forEach(function(it) {
        indices.push(parseInt(it.getAttribute('data-trace-idx')));
      });
      setVisible(indices, visible);
      items.forEach(function(it) {
        it.classList.toggle('hidden', !visible);
      });
    });
  });

  // ===== Custom hover =====
  function findBinFromPx(plotPx) {
    var nBins = GEO[mode].bins.length;
    if (nBins === 0) return -1;
    var gd = pdiv();
    var plotW = gd._fullLayout._size.w;
    var pitch = plotW / nBins;
    var idx = Math.floor(plotPx / pitch);
    if (idx < 0) idx = 0;
    if (idx >= nBins) idx = nBins - 1;
    return idx;
  }

  function bindHover() {
    var gd = pdiv();
    if (!gd || !gd._fullLayout) { setTimeout(bindHover, 100); return; }
    var tt = document.getElementById('geo-tooltip');
    var spike = document.getElementById('geo-spike');

    gd.addEventListener('mousemove', function(e) {
      var fl = gd._fullLayout;
      if (!fl) return;
      var rect = gd.getBoundingClientRect();
      var bg = fl._size;
      var pl = rect.left + bg.l, pr = rect.left + bg.l + bg.w;
      var pt = rect.top  + bg.t, pb = rect.top  + bg.t + bg.h;
      if (e.clientX < pl || e.clientX > pr ||
          e.clientY < pt || e.clientY > pb) {
        tt.style.display = 'none';
        spike.style.display = 'none';
        return;
      }
      var plotPx = e.clientX - pl;
      var binIdx = findBinFromPx(plotPx);
      if (binIdx < 0 || binIdx >= GEO[mode].hover_html.length) {
        tt.style.display = 'none';
        spike.style.display = 'none';
        return;
      }
      var html = GEO[mode].hover_html[binIdx];
      if (!html) {
        tt.style.display = 'none';
        spike.style.display = 'none';
        return;
      }
      tt.innerHTML = html;
      tt.style.display = 'block';
      var ttW = tt.offsetWidth, ttH = tt.offsetHeight;
      var x = e.clientX + 14, y = e.clientY + 12;
      if (x + ttW > window.innerWidth)  x = e.clientX - ttW - 14;
      if (y + ttH > window.innerHeight) y = e.clientY - ttH - 12;
      tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';

      var pitch = bg.w / GEO[mode].bins.length;
      var binCenterPx = pl + (binIdx + 0.5) * pitch;
      spike.style.transform = 'translateX(' + binCenterPx + 'px)';
      spike.style.display = 'block';
    });

    gd.addEventListener('mouseleave', function() {
      tt.style.display = 'none';
      spike.style.display = 'none';
    });

    if (gd.on) {
      gd.on('plotly_afterplot', function() {
        requestAnimationFrame(snapBars);
      });
    }
    requestAnimationFrame(snapBars);
    requestAnimationFrame(snapBars);
  }

  bindHover();
})();
</script>
"""

    js = js.replace('__GEO_PAYLOAD__', json.dumps(payload))

    with open(path, 'r') as f:
        html = f.read()
    html = html.replace('<head>', '<head>' + fullview_css + overlay_css, 1)
    html = html.replace('</body>',
                        toggle_html + legend_html + tooltip_html + js + '</body>')
    with open(path, 'w') as f:
        f.write(html)


# ---------- main ----------
def main():
    global PEAK_YEAR_THRESHOLD
    p = argparse.ArgumentParser()
    p.add_argument('--daily',   default=DEFAULT_DAILY)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--threshold', type=float, default=PEAK_YEAR_THRESHOLD)
    args = p.parse_args()
    PEAK_YEAR_THRESHOLD = args.threshold

    df = pd.read_csv(args.daily)
    df['date'] = pd.to_datetime(df['date'])

    (label_for, ordered_labels, label_color, label_total,
     group_for_label, group_total, group_size,
     other_sublists) = build_categories(df)

    y_bins = yearly_bins(df)
    m_bins = monthly_bins(df)

    yearly  = aggregate(df, 'Y', label_for, ordered_labels, y_bins)
    monthly = aggregate(df, 'M', label_for, ordered_labels, m_bins)
    yearly_sub  = build_subcity_data(df, label_for, 'Y')
    monthly_sub = build_subcity_data(df, label_for, 'M')
    yearly_hover  = build_bin_hover(yearly,  'Y', ordered_labels, label_color,
                                     group_for_label, yearly_sub)
    monthly_hover = build_bin_hover(monthly, 'M', ordered_labels, label_color,
                                     group_for_label, monthly_sub)

    bar_traces = build_bar_traces(yearly, monthly, ordered_labels, label_color)
    legend_html = build_legend_html(ordered_labels, label_color, label_total,
                                     group_for_label, group_total, group_size,
                                     other_sublists)

    m_tickvals, m_ticktext = build_tickvals_for_monthly(m_bins)
    payload = {
        'year': {
            'bins': y_bins,
            'tickvals': y_bins,
            'ticktext': y_bins,
            'hover_html': [yearly_hover[b] for b in y_bins],
            'gap_px': GAP_PX_YEAR,
        },
        'month': {
            'bins': m_bins,
            'tickvals': m_tickvals,
            'ticktext': m_ticktext,
            'hover_html': [monthly_hover[b] for b in m_bins],
            'gap_px': GAP_PX_MONTH,
        },
    }

    total_miles = df[df['miles'] > 0]['miles'].sum()
    n_cities = df['city_state'].nunique()

    fig = go.Figure(data=bar_traces)
    fig.update_layout(
        title=dict(
            text=('<b>Mileage locations by city</b>'
                  '<br><sub style="font-size:13px;color:#bbb">'
                  f'{total_miles:,.0f} mi across {n_cities} cities, '
                  f'2016-present'
                  '</sub>'),
            x=0.01, xanchor='left',
            y=0.965, yanchor='top',
            font=dict(color='#eee'),
        ),
        barmode='stack',
        bargap=0,
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#eee'),
        autosize=True,
        margin=dict(t=110, l=70, r=340, b=60),
        showlegend=False,
        hovermode=False,
        xaxis=dict(
            type='category',
            gridcolor='#2a2a2a',
            tickmode='array',
            tickvals=y_bins,
            ticktext=y_bins,
        ),
        yaxis=dict(title='Miles', gridcolor='#2a2a2a',
                   zerolinecolor='#2a2a2a'),
    )

    out_path = os.path.join(args.out_dir, 'mileage_by_geography.html')
    write_html(fig, out_path, legend_html, payload)

    print(f'wrote {out_path}')
    print(f'  total: {total_miles:,.0f} mi · {n_cities} cities')
    print(f'  bins: {len(y_bins)} yearly, {len(m_bins)} monthly '
          f'({m_bins[0]} → {m_bins[-1]})')
    print(f'  gap_px: {GAP_PX_YEAR} yearly, {GAP_PX_MONTH} monthly')
    print(f'  legend (top → bottom):')
    last_group = None
    for label in ordered_labels:
        g = group_for_label[label]
        if g != last_group:
            if group_size[g] >= 2 and g != GLOBAL_OTHER_KEY:
                print(f'    [{g} group · {fmt_miles(group_total[g])} mi]')
            else:
                print(f'    [singleton]')
            last_group = g
        marker = ('  ' if label == GLOBAL_OTHER_LABEL
                  else ('· ' if label.endswith(' (other)') else '✓ '))
        indent = '      ' if (group_size[g] >= 2
                              and g != GLOBAL_OTHER_KEY) else '    '
        print(f'{indent}{marker}{label_color[label]}  '
              f'{label:<28s} {label_total[label]:>7,.0f} mi')
        if label in other_sublists:
            for city, miles in other_sublists[label][:5]:
                print(f'{indent}        {city}: {fmt_miles(miles)} mi')
            if len(other_sublists[label]) > 5:
                print(f'{indent}        ... ({len(other_sublists[label])-5} more)')


if __name__ == '__main__':
    main()
