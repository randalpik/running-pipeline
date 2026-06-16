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
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import daily_floor
from src.shared.country_codes import country_abbrev
from src.shared.effective_mileage import effective_daily_miles
from src.plotting import (render_plot, apply_default_layout, GRID)
from src.plotting import widgets

_PLOTS_DIR = Path(__file__).resolve().parent
_GEO_CSS = _PLOTS_DIR / 'make_geography_plot.css'
_GEO_JS = _PLOTS_DIR / 'make_geography_plot.js'


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


def split_label_for_singleton(lab):
    """Return (state_code, city_or_none) for a singleton label.

    'Chicago, IL' -> ('IL', 'Chicago')
    'China'       -> ('China', None)
    'MA (other)'  -> ('MA', '(other)')
    """
    if lab.endswith(' (other)'):
        return lab[:-len(' (other)')], '(other)'
    m = re.match(r'^(.+),\s*([A-Z]{2,3})$', lab)
    if m:
        return m.group(2), m.group(1)
    return lab, None


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

    sorted_groups = sorted(group_total, key=lambda g: group_total[g], reverse=True)
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
                               key=lambda g: group_bin_total[g], reverse=True)
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
                    f'<div class="hov-group-title">'
                    f'<span class="hov-singleton-prefix">'
                    f'<b class="hov-singleton-state">{g}</b>'
                    f'<span class="hov-sep">·</span>'
                    f'</span>'
                    f'<span class="hov-grouptotal">'
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
                # Non-grouped rows: render as compressed singletons (bold
                # state prefix + box + city + miles) so the row reads as a
                # top-level category, matching the legend treatment. Global
                # Other has no state code, so keep its plain hov-item form.
                is_global_other = (g == GLOBAL_OTHER_KEY)
                for lab, val, color, name, sub in items_ordered:
                    if is_global_other:
                        parts.append(
                            f'<div class="hov-item">'
                            f'<span class="hov-box" style="background:'
                            f'{color}"></span>'
                            f'<span class="hov-name">{name}</span>'
                            f'<span class="hov-val">  '
                            f'<b>{fmt_miles(val)}</b> mi</span></div>')
                    else:
                        state, city = split_label_for_singleton(name)
                        display_name = f"{city}, {state}" if city else state
                        parts.append(
                            f'<div class="hov-singleton">'
                            f'<span class="hov-singleton-prefix">'
                            f'<b class="hov-singleton-state">{state}</b>'
                            f'<span class="hov-sep">·</span>'
                            f'</span>'
                            f'<span class="hov-box" style="background:'
                            f'{color}"></span>'
                            f'<span class="hov-name">{display_name}</span>'
                            f'<span class="hov-val">  ·  '
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

    Real groups (>=2 traces) get a bold title row with indented children.
    State-singletons render as a single compressed line: bold state code,
    color box, city, lifetime miles — matching group-title typography so
    they read as top-level categories instead of sub-rows. The Global
    Other singleton keeps the original chev/box/name form so its expand
    affordance stays intact. (other) items (and the global Other) carry
    a chevron that expands a sub-list of contributing cities.
    """
    parts = ['<div id="geo-legend"><div class="legend-inner">']
    last_group = None
    open_wrapper = False
    for i, lab in enumerate(ordered_labels):
        g = group_for_label[lab]
        is_real_group     = (group_size[g] >= 2 and g != GLOBAL_OTHER_KEY)
        is_state_singleton = (not is_real_group and lab != GLOBAL_OTHER_LABEL)

        if g != last_group:
            if open_wrapper:
                parts.append('</div>')
                open_wrapper = False
            if is_real_group:
                parts.append(f'<div class="legend-group" data-group-key="{g}">')
                parts.append(
                    f'<div class="legend-title">'
                    f'<span class="legend-singleton-prefix">'
                    f'<b class="legend-singleton-state">{g}</b>'
                    f'<span class="legend-singleton-dot">·</span>'
                    f'</span>'
                    f'<span class="legend-life">'
                    f'{fmt_miles(group_total[g])} mi</span></div>')
                open_wrapper = True
            elif is_state_singleton:
                pass  # no wrapper; the singleton row stands alone
            else:
                parts.append('<div class="legend-block">')
                open_wrapper = True
            last_group = g

        if is_state_singleton:
            state, city = split_label_for_singleton(lab)
            name = f"{city}, {state}" if city else state
            parts.append(
                f'<div class="legend-singleton" data-trace-idx="{i}">'
                f'<span class="legend-singleton-prefix">'
                f'<b class="legend-singleton-state">{state}</b>'
                f'<span class="legend-singleton-dot">·</span>'
                f'</span>'
                f'<span class="legend-box" style="background:'
                f'{label_color[lab]}"></span>'
                f'<span class="legend-name">{name}</span>'
                f'<span class="legend-life">  ·  '
                f'{fmt_miles(label_total[lab])} mi</span></div>')
            continue  # collapse rule guarantees no sub-list for state singletons

        is_othery = lab in other_sublists  # has a sub-list to expand
        cls = 'legend-item' + (' grouped' if is_real_group else '')
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
                             + (' grouped' if is_real_group else ' singleton'))
            parts.append('<div class="legend-subgroup">')
            for city, miles in other_sublists[lab]:
                parts.append(
                    f'<div class="{sub_cls_inner}">{city}: '
                    f'{fmt_miles(miles)} mi</div>')
            parts.append('</div>')
    if open_wrapper:
        parts.append('</div>')
    parts.append('</div></div>')
    return ''.join(parts)


def build_tickvals_for_monthly(monthly_bin_list):
    tv = [b for b in monthly_bin_list if b.endswith('-01')]
    tt = [b.split('-')[0] for b in tv]
    return tv, tt


# ---------- write HTML ----------
def write_html(fig, path, legend_html, payload, *, title=None, subtitle=None):
    toggle_html = widgets.toggle_bar(
        'geo-toggle',
        [('year', 'Yearly'), ('month', 'Monthly')],
        default_id='year',
    )
    # The toggle bar's default position (top:14px) is too high — geography
    # has the title bar above it, so push the toggle down to 56px.
    toggle_html = toggle_html.replace(
        'class="rp-toggle-bar"',
        'class="rp-toggle-bar" style="top: 56px"',
    )
    globals_html = widgets.js_globals({'GEO': payload})
    tooltip_html = '<div id="geo-tooltip"></div><div id="geo-spike"></div>'
    overlay_html = (toggle_html + '\n' + legend_html
                    + tooltip_html + '\n' + globals_html)
    render_plot(
        fig, path,
        title_slug='mileage_by_geography',
        page_title='Locations',
        title=title,
        subtitle=subtitle,
        overlay_html=overlay_html,
        overlay_js_files=[_GEO_JS],
        extra_head_css_files=[_GEO_CSS],
    )


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
    # Daily.csv now includes pre-2016 stub rows synthesized from race
    # additions (used by the world map). All other daily-driven plots —
    # this one included — standardize on the 2016+ logging era.
    df = df[df['date'] >= daily_floor()]
    # Drop rows with no resolved city_state (e.g. indoor runs in a watch-import
    # profile) — the geography categories are keyed on city_state.
    df = df.dropna(subset=['city_state']).copy()
    # Source of truth: watch/route distance-corrected mileage (decrease-only;
    # corr <= logged) drives every city/state total below. On-disk daily.csv
    # 'miles' is untouched.
    df['miles'] = effective_daily_miles(df)
    # Display only: collapse foreign 'City, Country' -> 'City, CC' (this plot
    # is a treemap, not geocoded, so the abbreviated form is safe to group on).
    df['city_state'] = df['city_state'].map(country_abbrev)

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
    apply_default_layout(
        fig,
        barmode='stack',
        bargap=0,
        margin=dict(t=20, l=70, r=340, b=28),
        showlegend=False,
        hovermode=False,
        xaxis=dict(
            type='category',
            gridcolor=GRID,
            tickmode='array',
            tickvals=y_bins,
            ticktext=y_bins,
            ticks='outside',
            ticklen=8,
            tickcolor='rgba(0,0,0,0)',
        ),
        yaxis=dict(title='Miles', gridcolor=GRID,
                   zerolinecolor=GRID),
    )

    out_path = os.path.join(args.out_dir, 'mileage_by_geography.html')
    write_html(
        fig, out_path, legend_html, payload,
        title='Mileage by location',
        subtitle=f'{total_miles:,.0f} mi across {n_cities} cities, 2016-present',
    )

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
