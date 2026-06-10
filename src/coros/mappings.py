"""Scaling constants and bin mappings for Coros activity data.

Every numeric field the Coros Training Hub API returns is integer-scaled.
The factors below were verified against real activities (2026-05-29); guessing
them wrong silently corrupts the dataset, so they live in one place.

The weather/wind bin maps are deliberately tunable: we only have a couple of
observed ``weatherType`` values so far, so the importer logs the distinct
values it sees (see build_current_log) and we refine these tables empirically.
"""
from __future__ import annotations

# ---- scaling (raw API integer -> real unit) ----
DISTANCE_DIV = 100.0      # centimetres -> metres
TIME_DIV = 100.0          # centiseconds -> seconds  (time/totalTime/workoutTime)
TIMESTAMP_DIV = 100.0     # centiseconds -> unix epoch seconds
GPS_DIV = 1e7             # scaled int -> decimal degrees
WEATHER_DIV = 10.0        # temperature/humidity/windSpeed/windDirection
TZ_UNIT_MIN = 15          # `timezone` is in quarter-hours

METERS_PER_MILE = 1609.344
SEC_PER_KM_TO_SEC_PER_MILE = METERS_PER_MILE / 1000.0  # avgPace is sec/km

# ---- sport types (Coros sportType codes) ----
SPORT_RUN = 100
SPORT_INDOOR_RUN = 101
SPORT_TRAIL_RUN = 102
SPORT_TRACK_RUN = 103
RUN_SPORTS = {SPORT_RUN, SPORT_INDOOR_RUN, SPORT_TRAIL_RUN, SPORT_TRACK_RUN}

# ---- classification thresholds ----
# Aligned with the long-run model's LONG_MIN_MINUTES (workouts.py): a run
# is "long" once fueling/hydration become a real concern, ~80 min in
# regardless of pace.
LONG_RUN_MIN_SECONDS = 80 * 60   # moving time > 80 min => long run

# ---- weather bins ----
# Coros `weatherType` is the AccuWeather icon code (verified: weatherType N
# corresponds to the icon file N.png; 1=Sunny, 2=Mostly Sunny, 33=Clear-night,
# etc.). We map the full AccuWeather code set onto the log's allowed `weather`
# bins (clear, cloudy, overcast, light rain, heavy rain, showers, drizzle,
# snow, flurries, fog, smoky). Day/night variants collapse to the same bin.
# Unknown codes map to None (blank) rather than a wrong bin.
WEATHER_TYPE_TO_BIN: dict[int, str] = {
    1: "clear", 2: "clear", 30: "clear", 31: "clear", 32: "clear",
    33: "clear", 34: "clear", 37: "clear",            # sunny / clear (day+night)
    3: "cloudy", 4: "cloudy", 35: "cloudy", 36: "cloudy", 38: "cloudy",
    6: "cloudy", 7: "cloudy",                          # partly/mostly cloudy
    5: "smoky",                                        # hazy sunshine
    8: "overcast",                                     # dreary / overcast
    11: "fog",
    12: "showers", 13: "showers", 14: "showers", 39: "showers", 40: "showers",
    18: "light rain", 26: "light rain",                # rain / freezing rain
    15: "heavy rain", 16: "heavy rain", 17: "heavy rain",
    41: "heavy rain", 42: "heavy rain",                # thunderstorms
    19: "flurries", 20: "flurries", 21: "flurries", 43: "flurries",
    22: "snow", 23: "snow", 24: "snow", 25: "snow", 29: "snow", 44: "snow",
}


def weather_bin(weather_type) -> str | None:
    try:
        return WEATHER_TYPE_TO_BIN.get(int(weather_type))
    except (TypeError, ValueError):
        return None


# ---- wind bins ----
# windSpeed comes in (raw/10) units, in m/s. Thresholds are NOT Beaufort —
# they're calibrated against a 21-day overlap between Coros windSpeed and Max's
# own subjective wind labels (Spearman r=0.62; this cut agreed on 81% of days).
# The takeaway: the subjective scale runs much higher than Beaufort — winds up
# to ~8.5 m/s read as "low". The actual second runner labels differently, but
# this is the best anchor we have and far better than a generic scale. Retune
# if/when that runner's own labels become available.
WIND_BINS_MS = [
    (8.75, "low"),
    (11.75, "moderate"),
    (14.0, "high"),
]
WIND_BIN_TOP = "extreme"


def wind_bin(wind_speed_ms) -> str | None:
    if wind_speed_ms is None:
        return None
    for threshold, label in WIND_BINS_MS:
        if wind_speed_ms < threshold:
            return label
    return WIND_BIN_TOP
