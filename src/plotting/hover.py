"""Cursor-tooltip composition helpers — the single standardized system for
the performance plots (Races, Race Distances, Fitness, Training, Workouts,
Long Runs, Recovery).

Two layers live here:

* **Primitives** (`tt_row`, `tt_kv`, `tt_title`) mirror the ``.tt-*`` classes
  in ``_scaffold/base.css`` and encode the house style ONCE — a right-justified
  value in the top section is bold (`tt_row`); a label followed by a colon is
  bold (`tt_kv`); a session title is a bold type with a non-bold parenthetical
  route/location (`tt_title`, the canonical Workouts form).

* **Session line builders** (`long_run_lines`) render the exact same lines on
  every tab that shows them, so "match Long Runs exactly" stays true after the
  next iteration instead of re-diverging across hand-rolled hover functions.

Callers do not escape values (every label is a hardcoded constant or a
Python-formatted number), matching the convention in ``widgets.py``.
"""
from __future__ import annotations

import pandas as pd

from src.plotting.formatters import route_paren, sec_to_mss
from src.shared.paths import DATA_DIR
from src.shared.units import METERS_PER_MILE, FT_PER_M


def tt_row(label: str, value: str) -> str:
    """Top-section two-column row: muted label left, **bold** value right.

    The flex layout (``.tt-row`` in base.css) right-justifies the value;
    wrapping it in ``<b>`` is the house rule that every right-justified
    top-section number is bold — ranges (e.g. a 95% band) included.
    """
    return f'<div class="tt-row"><span>{label}</span><b>{value}</b></div>'


def tt_kv(label: str, value: str) -> str:
    """Inline detail line with a **bold label + colon**: ``<b>{label}:</b> v``."""
    return f'<b>{label}:</b> {value}'


def tt_title(label: str, display_name=None, city_state=None) -> str:
    """Canonical session title: **bold** type + non-bold ``(route, city)``.

    This is the Workouts form, adopted as the standard across tabs.
    """
    return f'<b>{label}</b>{route_paren(display_name, city_state)}'


def long_run_lines(r, pause_adjusted: bool = False, show_logged: bool = True) -> list:
    """Canonical Long Runs distance/pace/paused detail lines.

    Shared by the Long Runs, Training, and Fitness tabs so all three read
    identically. Returns a list of HTML line strings (NO title — compose that
    with :func:`tt_title`). Order is plotted → pause-adjusted → original:

    1. ``{dist} mi @ {pace}/mi · {pause} paused`` (corrected distance when a
       watch/route correction applies, else the logged distance).
    2. ``Pause-adjusted distance: {d_eff} mi`` — only when ``pause_adjusted``
       is set (the model-driven Training/Fitness tabs) AND the pause-aware
       effective distance projected onto the plot differs from the shown
       distance by > 0.1 mi. The descriptive Long Runs tab omits it.
    3. ``Logged: {dist} mi @ {pace}/mi`` — only when ``show_logged`` (the
       descriptive Long Runs tab) AND a correction moved the distance
       materially (≥ 0.2 mi). Training/Fitness suppress it (the corrected and
       pause-adjusted distances already carry the projection story).
    """
    has_corr = pd.notna(r.get('corr_miles'))
    logged = (f"{float(r['miles']):.1f} mi @ "
              f"{sec_to_mss(float(r['recovery_pace_sec_per_mi']))}/mi")
    corr = (f"{float(r['corr_miles']):.1f} mi @ "
            f"{sec_to_mss(float(r['corr_pace_sec_per_mi']))}/mi"
            if has_corr else '')
    pause = ''
    if r.get('lr_watch') and pd.notna(r.get('pause_s')) and float(r['pause_s']) >= 30:
        pause = f" · {sec_to_mss(float(r['pause_s']))} paused"
    elif (not r.get('lr_watch') and pd.notna(r.get('est_pause_s'))
          and float(r.get('pause_erosion_gate') or 0) > 0):
        pause = f" · est. {sec_to_mss(float(r['est_pause_s']))} paused"

    if has_corr and abs(float(r['miles']) - float(r['corr_miles'])) < 0.2:
        main, alt = f"{corr}{pause}", ''
    elif has_corr:
        main = f"{corr}{pause}"
        alt = tt_kv('Logged', logged) if show_logged else ''
    else:
        main, alt = f"{logged}{pause}", ''

    padj = ''
    if pause_adjusted:
        shown_mi = float(r['corr_miles']) if has_corr else float(r['miles'])
        d_eff = r.get('d_eff_m')
        if pd.notna(d_eff):
            padj_mi = float(d_eff) / METERS_PER_MILE
            if abs(padj_mi - shown_mi) > 0.1:
                padj = tt_kv('Pause-adjusted distance', f"{padj_mi:.1f} mi")

    return [ln for ln in (main, padj, alt) if ln]


# --- Hill sessions -----------------------------------------------------------
# Watch per-loop / per-rep detail (workout_measured.csv) + the canonical
# continuous-hill and hill-repeat descriptive bodies. Shared so the Workouts
# (source of truth), Training, and Fitness tabs render hill sessions
# identically. The CSV is a watch artifact; missing/absent → empty dict (the
# bodies then render the parser figures alone).

def hill_measured_lines() -> dict:
    """``{date: '<b>Watch:</b> loops a · b · …'}`` — per-loop splits on
    hill-exact days. The moving total is folded into the headline already, so
    this line only carries the per-loop breakdown."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m = m[(m['rep_idx'] > 0) & (m['status'].isin(['hill-exact', 'hill-total']))]
    out = {}
    for date, day in m.groupby('date'):
        if (day['status'] == 'hill-exact').all() and len(day) > 1:
            splits = ' · '.join(sec_to_mss(t) for t in day['time_s'])
            out[str(date)] = f'<b>Watch:</b> loops {splits}'
    return out


def hill_rep_measured_lines() -> dict:
    """``{date: {total_gain_ft, avg_grade, reps_html}}`` — watch per-rep
    distance + elevation and the block's true average grade, on hillrep-exact
    days. Absent days fall back to the parser estimate."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path, dtype={'date': str})
    m = m[(m['rep_idx'] > 0) & (m['status'] == 'hillrep-exact')]
    out = {}
    for date, day in m.groupby('date'):
        total_gain = float(day['gain_ft'].sum())
        total_dist = float(day['dist_m'].sum())
        avg_grade = (total_gain * FT_PER_M / total_dist * 100) if total_dist else 0.0
        parts = [f"{int(round(r['dist_m']))}m/+{int(round(r['gain_ft']))}ft"
                 for _, r in day.iterrows()]
        out[str(date)] = {
            'total_gain_ft': total_gain, 'avg_grade': avg_grade,
            'reps_html': '<b>Watch:</b> ' + ' · '.join(parts),
        }
    return out


def hill_cont_lines(r) -> list:
    """Universal continuous-hills descriptive lines shown on every tab: the
    structure (``N loops, MM:SS total[, F ft gained]``) and actual pace. The
    raw per-loop ``Watch:`` line stays Workouts-only and is placed there
    directly (after the 5K-equivalent), matching the non-hill measured_line."""
    nreps = int(r['nreps'])
    loops_word = 'loop' if nreps == 1 else 'loops'
    ft_gained = int(round(float(r.get('ft_gained') or 0)))
    time_part = (f"{sec_to_mss(r['t_eff'])} total" if r.get('watch_measured')
                 else f"{int(r['session_min'])} min total")
    body = (f"{nreps} {loops_word}, {time_part}"
            + (f", {ft_gained} ft gained" if ft_gained else ''))
    lines = [body]
    if pd.notna(r.get('actual_pace_s')):
        lines.append(tt_kv('Actual pace', f"{sec_to_mss(r['actual_pace_s'])}/mi"))
    return lines


def hill_rep_lines(r, measured_map=None) -> list:
    """Universal hill-repeat structure line: ``N reps × R[, F ft gained]`` (the
    watch total-gain when available, so the headline elevation matches across
    tabs). Average grade + the raw per-rep ``Watch:`` line stay Workouts-only."""
    rep_count = int(r['rep_count'])
    reps_word = 'rep' if rep_count == 1 else 'reps'
    rt = float(r['rep_time_min'])
    if rt == int(rt):
        rt_str = f"{int(rt)} min"
    else:
        mm = int(rt)
        ss = int(round((rt - mm) * 60))
        rt_str = f"{mm}:{ss:02d}"
    d = r['date']
    key = str(d.date()) if hasattr(d, 'date') else str(d)[:10]
    md = (measured_map or {}).get(key)
    body = f"{rep_count} {reps_word} × {rt_str}"
    elev = r.get('total_elev_ft')
    if md:
        body += f", {int(round(md['total_gain_ft']))} ft gained"
    elif pd.notna(elev):
        body += f", {int(round(float(elev)))} ft gained"
    return [body]
