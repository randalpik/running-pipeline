"""Physical unit conversions — the single source of truth for the distance and
elevation constants used across the pipeline.

Before this module these values were retyped as bare literals (``1609.344``
~50×) and re-declared under four different names (``METERS_PER_MILE``,
``MILE_M``, ``MILE``); a mistyped digit silently corrupts paces or elevations,
and the divisor sense of ``FT_PER_M`` has bitten before. Import from here.
"""
from __future__ import annotations

# Meters in one statute mile (exact). Paces are per-mile, race distances are
# meters, so this is the bridge between the two.
METERS_PER_MILE = 1609.344

# Meters per foot (exact). Named for the multiply direction (ft → m); to go the
# other way — meters → feet, the common case for elevation — DIVIDE by it.
FT_PER_M = 0.3048
