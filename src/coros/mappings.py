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
WEATHER_DIV = 10.0        # temperature/humidity/windDirection (raw/10)
# windSpeed: raw/10 is KM/H, NOT m/s. The earlier "m/s" label was wrong — a
# uniform ~3.6x scale error is invisible to the rank-correlation that "validated"
# it (verified 2024-11-05: raw 463 -> 46.3 km/h = 28.8 mph, matching the Coros
# app's "29 mph"; the old code showed 46.3 m/s). Display in mph (Coros app unit).
KMH_TO_MPH = 0.6213712
TZ_UNIT_MIN = 15          # `timezone` is in quarter-hours

from src.shared.units import METERS_PER_MILE  # re-exported (M.METERS_PER_MILE)
SEC_PER_KM_TO_SEC_PER_MILE = METERS_PER_MILE / 1000.0  # avgPace is sec/km

# ---- excluded activities ----
# Phantom / corrupted Coros activities to drop wherever the cache is loaded.
# 440449841644994562: a 2021-12-29 "North Bend Course" duplicate with a
# corrupted GPS track and a wrong (01:52 local) start time — Max confirmed it
# never happened; the day's real run is a separate activity that stays.
EXCLUDED_LABEL_IDS = {"440449841644994562"}

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
# Wind bins in MPH. Thresholds are NOT Beaufort — calibrated against a 21-day
# overlap between Coros windSpeed and Max's subjective labels (Spearman r=0.62,
# 81% agreement); his scale runs low (gentle winds read "low"). Originally fit in
# the raw km/h units (8.75/11.75/14 km/h) back when those were mislabeled m/s;
# converted to mph here (x KMH_TO_MPH), preserving the calibration. Retune when
# the second runner's own labels exist.
WIND_BINS_MPH = [
    (5.44, "low"),
    (7.30, "moderate"),
    (8.70, "high"),
]
WIND_BIN_TOP = "extreme"


def wind_bin(wind_speed_mph) -> str | None:
    if wind_speed_mph is None:
        return None
    for threshold, label in WIND_BINS_MPH:
        if wind_speed_mph < threshold:
            return label
    return WIND_BIN_TOP
