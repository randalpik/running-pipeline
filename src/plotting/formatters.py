"""Time-formatting helpers shared by every plot script.

Resist the urge to one-function-with-flags: separate names communicate
intent at the call site (sec_to_mss vs signed_sec is not the same as
sec_to_mss(signed=True)).
"""
from __future__ import annotations

import math


def _is_missing(s) -> bool:
    return s is None or (isinstance(s, float) and math.isnan(s))


def route_label(display_name, city_state) -> str:
    """Join display_name + city_state into one ', '-separated label, dropping
    blanks/NaN and deduplicating identical parts.

    Shared by every plot's hover/legend so the watch-import case (no route
    names, so display_name == city_state == 'City, ST') renders 'City, ST'
    once instead of 'City, ST, City, ST'. Single source of truth — fix it here,
    not per tab.
    """
    parts = []
    for x in (display_name, city_state):
        if _is_missing(x):
            continue
        s = str(x).strip()
        if s and s.lower() != 'nan' and s not in parts:
            parts.append(s)
    return ', '.join(parts)


def route_paren(display_name, city_state) -> str:
    """``route_label`` wrapped as ' (…)' for inline append, or '' if empty."""
    lbl = route_label(display_name, city_state)
    return f' ({lbl})' if lbl else ''


def sec_to_mss(s) -> str:
    """Format seconds as ``M:SS`` or ``H:MM:SS``.

    Always positive. For negative inputs use :func:`signed_sec`. Returns
    ``''`` on None/NaN.
    """
    if _is_missing(s):
        return ''
    # Round to integer seconds first to avoid the "4:60" bug
    # where s=299.6 -> m=4, round(s%60)=60.
    total = int(round(float(s)))
    if total >= 3600:
        h = total // 3600
        m = (total % 3600) // 60
        ss = total % 60
        return f'{h}:{m:02d}:{ss:02d}'
    m, ss = divmod(total, 60)
    return f'{m}:{ss:02d}'


def sec_to_mss_full(s) -> str:
    """Format seconds preserving subsecond precision for race times.

    Examples: 143.2 -> '2:23.2', 18*60+38.5 -> '18:38.5', 8756.4 -> '2:25:56.4'.
    Trailing .0 is hidden when the value is an integer second.
    """
    if _is_missing(s):
        return ''
    s = float(s)
    # Round to tenths first to avoid float-representation artifacts
    # (e.g. 3599.95 - 3599 = 0.94999... in float).
    tenths_total = int(round(s * 10))
    whole = tenths_total // 10
    frac = tenths_total % 10
    if whole >= 3600:
        h = whole // 3600
        m = (whole % 3600) // 60
        ss = whole % 60
        body = f'{h}:{m:02d}:{ss:02d}'
    else:
        m, ss = divmod(whole, 60)
        body = f'{m}:{ss:02d}'
    return f'{body}.{frac}' if frac > 0 else body


def time_decimals(s) -> int:
    """Decimal places (0-2) demonstrably present in a seconds value.

    Display-side fallback for race rows without an entered-precision
    ``time_dec`` field. Trailing zeros in the entered form are not
    recoverable from a float (an entered ``16:58.0`` infers as ``16:58``) —
    that's exactly what ``time_dec`` exists to preserve; use this only when
    the field is absent.
    """
    if _is_missing(s):
        return 0
    hundredths = int(round(float(s) * 100))
    if hundredths % 100 == 0:
        return 0
    if hundredths % 10 == 0:
        return 1
    return 2


def sec_to_mss_prec(s, decimals) -> str:
    """Format seconds as ``M:SS`` / ``H:MM:SS`` with exactly ``decimals``
    (0-2) decimal places — the entered-precision formatter for race times.

    ``'16:58.0'`` stays ``16:58.0``, ``'4:48.47'`` keeps both digits, and a
    whole-second entry never grows a fake ``.0``. Course corrections format
    with the ORIGINAL time's decimals so a converted time carries the same
    precision as entered — no more, no less.
    """
    if _is_missing(s):
        return ''
    decimals = int(decimals)
    scale = 10 ** decimals
    # Round in the target precision first to avoid float-representation
    # artifacts (same trick as sec_to_mss_full).
    units_total = int(round(float(s) * scale))
    whole, frac = divmod(units_total, scale)
    if whole >= 3600:
        h = whole // 3600
        m = (whole % 3600) // 60
        ss = whole % 60
        body = f'{h}:{m:02d}:{ss:02d}'
    else:
        m, ss = divmod(whole, 60)
        body = f'{m}:{ss:02d}'
    return f'{body}.{frac:0{decimals}d}' if decimals else body


def signed_sec(s) -> str:
    """Signed M:SS / H:MM:SS — shows ``+`` for positive, ``-`` for negative.

    Used for residuals and signed-tick labels. Zero is returned as ``'0:00'``
    (no sign).
    """
    if _is_missing(s):
        return ''
    f = float(s)
    if f == 0:
        return '0:00'
    sign = '+' if f > 0 else '-'
    return sign + sec_to_mss(abs(f))


def fmt_min(m) -> str:
    """Format minutes (float) as ``M:SS``."""
    if _is_missing(m):
        return ''
    return sec_to_mss(float(m) * 60)
