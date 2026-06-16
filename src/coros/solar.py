"""Self-contained sunrise/sunset and time-of-day binning.

Uses the standard "sunrise equation" (NOAA solar model) so we don't take a
dependency on ``astral`` — the rest of the pipeline is stdlib-only and astral
isn't installed. Accuracy is within a couple of minutes, which is far finer
than the four time-of-day bins need.

Reference: https://en.wikipedia.org/wiki/Sunrise_equation
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone


def _julian_day(d: date) -> float:
    """Julian Day Number at 00:00 UTC for a calendar date."""
    y, m, day = d.year, d.month, d.day
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + day + b - 1524.5)


# Sun-center altitude (deg below horizon) at the moment we call it "rise"/"set"
# vs. "twilight". -0.833 = geometric horizon + refraction + solar radius (the
# standard sunrise/sunset angle); -6 = civil twilight (the gradient's purple
# anchors). Generalizing the angle lets one function serve both.
SUNRISE_ANGLE = -0.833
CIVIL_TWILIGHT_ANGLE = -6.0
# Sun altitude at which daylight reads visually "full blue" — the warm
# sunrise/sunset cast has faded well below this. Used as the blue anchor of the
# Time-of-day gradient (astronomically tied, like the -6 twilight anchor).
BLUE_DAY_ANGLE = 18.0


def _sun_events(d: date, lat: float, lon: float, angle_deg: float):
    """Return (rise_utc, set_utc, transit_utc) aware datetimes for the sun
    crossing ``angle_deg`` (solar noon ``transit`` is always returned).

    ``rise``/``set`` are None for polar day/night at this angle (the sun never
    reaches it); ``transit`` is always valid.
    """
    jd = _julian_day(d)
    n = math.ceil(jd - 2451545.0 + 0.0008)   # current Julian day (integer, at noon UTC)
    # mean solar time. lon is east-positive here; the equation's west-positive
    # l_w = -lon, and J* = n - l_w/360, i.e. J* = n - lon/360 in our convention.
    j_star = n - lon / 360.0
    M = (357.5291 + 0.98560028 * j_star) % 360.0   # solar mean anomaly (deg)
    M_rad = math.radians(M)
    C = (1.9148 * math.sin(M_rad)
         + 0.0200 * math.sin(2 * M_rad)
         + 0.0003 * math.sin(3 * M_rad))           # equation of the center
    lam = (M + C + 180.0 + 102.9372) % 360.0        # ecliptic longitude (deg)
    lam_rad = math.radians(lam)
    j_transit = (2451545.0 + j_star
                 + 0.0053 * math.sin(M_rad)
                 - 0.0069 * math.sin(2 * lam_rad))  # solar noon (Julian)
    transit_dt = _julian_to_dt(j_transit)
    sin_decl = math.sin(lam_rad) * math.sin(math.radians(23.44))
    decl = math.asin(sin_decl)
    lat_rad = math.radians(lat)
    cos_omega = ((math.sin(math.radians(angle_deg)) - math.sin(lat_rad) * sin_decl)
                 / (math.cos(lat_rad) * math.cos(decl)))
    if cos_omega >= 1.0 or cos_omega <= -1.0:
        return None, None, transit_dt               # polar night / midnight sun
    omega = math.degrees(math.acos(cos_omega))      # hour angle (deg)
    j_rise = j_transit - omega / 360.0
    j_set = j_transit + omega / 360.0
    return _julian_to_dt(j_rise), _julian_to_dt(j_set), transit_dt


def sun_events_utc(d: date, lat: float, lon: float):
    """Return (sunrise_utc, sunset_utc) as aware datetimes, or (None, None).

    None is returned for polar day/night (sun never crosses the horizon).
    """
    rise, set_, _ = _sun_events(d, lat, lon, SUNRISE_ANGLE)
    return rise, set_


def solar_anchors_local(d: date, lat: float, lon: float, tz_offset_min: int):
    """Local minutes-of-day for the five solar gradient anchors on date ``d``
    at ``(lat, lon)``: twilight_begin (dawn, sun at -6deg), sunrise, solar_noon,
    sunset, twilight_end (dusk, sun at -6deg).

    Returns a dict {anchor: minutes_float}; an entry is None when that event
    doesn't occur (polar day/night). ``solar_noon`` is always present. Minutes
    are taken on the local clock (``hour*60 + minute``), which is what the Time
    panel's y-axis uses; events that fall on an adjacent local date (near the
    poles) are clamped to [0, 1440] by the caller.
    """
    tz = timezone(_minutes(tz_offset_min))

    def mins(dt):
        if dt is None:
            return None
        local = dt.astimezone(tz)
        return local.hour * 60.0 + local.minute + local.second / 60.0

    dawn, dusk, transit = _sun_events(d, lat, lon, CIVIL_TWILIGHT_ANGLE)
    rise, set_, _ = _sun_events(d, lat, lon, SUNRISE_ANGLE)
    blue_rise, blue_set, _ = _sun_events(d, lat, lon, BLUE_DAY_ANGLE)
    return {
        'twilight_begin': mins(dawn),
        'sunrise': mins(rise),
        'blue_begin': mins(blue_rise),   # sun reaches +18deg (full blue)
        'solar_noon': mins(transit),
        'blue_end': mins(blue_set),
        'sunset': mins(set_),
        'twilight_end': mins(dusk),
    }


def _julian_to_dt(jd: float) -> datetime:
    unix = (jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc)


def time_of_day(start_utc: datetime, lat: float | None, lon: float | None,
                tz_offset_min: int) -> str:
    """Bin a UTC start time into early / morning / afternoon / late.

    Rule: before sunrise -> early; before (clock) noon -> morning; before
    sunset -> afternoon; otherwise late. When GPS is unavailable (e.g. indoor
    runs) we fall back to fixed local-clock cutoffs.
    """
    local = start_utc.astimezone(timezone(_minutes(tz_offset_min)))
    if lat is None or lon is None:
        h = local.hour + local.minute / 60.0
        if h < 6:
            return "early"
        if h < 12:
            return "morning"
        if h < 18:
            return "afternoon"
        return "late"

    sunrise, sunset = sun_events_utc(local.date(), lat, lon)
    if sunrise is not None and local < sunrise.astimezone(local.tzinfo):
        return "early"
    if local.hour < 12:
        return "morning"
    if sunset is not None and local < sunset.astimezone(local.tzinfo):
        return "afternoon"
    return "late"


def _minutes(total_min: int):
    from datetime import timedelta
    return timedelta(minutes=total_min)
