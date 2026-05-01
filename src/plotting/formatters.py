"""Time-formatting helpers shared by every plot script.

Resist the urge to one-function-with-flags: separate names communicate
intent at the call site (sec_to_mss vs signed_sec is not the same as
sec_to_mss(signed=True)).
"""
from __future__ import annotations

import math


def _is_missing(s) -> bool:
    return s is None or (isinstance(s, float) and math.isnan(s))


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
