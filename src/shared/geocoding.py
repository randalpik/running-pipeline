"""City-state -> (lat, lon) lookup with on-demand Nominatim geocoding.

Cache lives at ``data/city_coords.csv`` and is created empty on first run.
Successful lookups are appended; failures are logged and skipped (next run
retries). Designed so that adding a new city anywhere in the log results
in one extra geocode the next time the pipeline runs, with no manual step.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Iterable

from src.shared.paths import DATA_DIR

CACHE_PATH = DATA_DIR / 'city_coords.csv'
# ``tz`` is the IANA zone name (e.g. 'America/Chicago'), resolved automatically
# from the geocoded lat/lon at cache time (see ``_timezone_for``). Consumers use
# it with ``zoneinfo`` for DST-correct local time — notably the Misc. Trends
# Time panel, which converts each run's absolute UTC moment into the canonical
# city's local clock + solar gradient, rather than trusting the watch's
# (occasionally stale) reported offset.
CACHE_HEADER = ['city_state', 'latitude', 'longitude', 'tz', 'geocoded_at']

# Reverse cache maps a rounded (lat, lon) -> city_state, so repeated runs
# starting from the same area (the common case for watch imports) cost one
# lookup, not one per activity.
REVERSE_CACHE_PATH = DATA_DIR / 'reverse_coords.csv'
REVERSE_CACHE_HEADER = ['lat_round', 'lon_round', 'city_state', 'geocoded_at']
# ~1.1 km dedup grid. Reverse lookups resolve to city granularity (zoom=10),
# so a metro's many run-starts collapse to a handful of cells — keeping the
# one-time backfill's Nominatim call count (1.1 s each) to minutes, not hours.
REVERSE_ROUND = 2

NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
NOMINATIM_REVERSE_URL = 'https://nominatim.openstreetmap.org/reverse'
# Open-Meteo echoes the resolved IANA timezone for a lat/lon when queried with
# timezone=auto — keyless, no account, so it fits the geocoding step without a
# new Python dependency (timezonefinder would pull a ~50 MB polygon dataset).
OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
USER_AGENT = 'max-running-pipeline/1.0'
RATE_LIMIT_SEC = 1.1   # Nominatim ToS: max 1 req/sec
TZ_RATE_LIMIT_SEC = 0.4  # Open-Meteo is generous; stay polite

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC',
}
CA_PROVINCES = {
    'AB','BC','MB','NB','NL','NS','NT','NU','ON','PE','QC','SK','YT',
}


def _build_query(city_state: str) -> str:
    """Append country code to disambiguate US/CA city-states."""
    if ',' not in city_state:
        return city_state
    city, _, region = city_state.rpartition(',')
    region = region.strip()
    city = city.strip()
    if region in US_STATES:
        return f'{city}, {region}, US'
    if region in CA_PROVINCES:
        return f'{city}, {region}, CA'
    return f'{city}, {region}'


def _geocode_one(city_state: str) -> tuple[float, float] | None:
    q = _build_query(city_state)
    url = (f'{NOMINATIM_URL}?'
           f'{urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})}')
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'[geocode] {city_state!r}: request failed ({e})')
        return None
    if not data:
        print(f'[geocode] {city_state!r}: no results for query {q!r}')
        return None
    return float(data[0]['lat']), float(data[0]['lon'])


def _timezone_for(lat: float, lon: float) -> str | None:
    """IANA timezone name for a lat/lon via Open-Meteo (timezone=auto), or None.

    Used at cache time so each city carries a DST-aware zone alongside its
    coordinates. Failures are non-fatal (the city simply has no tz until a later
    run retries)."""
    url = (f'{OPEN_METEO_URL}?'
           f'{urllib.parse.urlencode({"latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 1})}')
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'[geocode-tz] ({lat},{lon}): request failed ({e})')
        return None
    tz = data.get('timezone') if isinstance(data, dict) else None
    return tz or None


def _region_from_address(addr: dict) -> str | None:
    """Pull a geocode-correct 'City, Region' city_state from a Nominatim
    reverse address. This is the stored bin key, so it must round-trip through
    forward geocoding — it is NOT the display form.

    US/CA use the 2-letter subdivision code ('Chicago, IL'). Everywhere else
    keeps the full English country name ('Berlin, Germany', 'Osaka, Japan')
    rather than a 2-letter country code: a code like 'DE' is ambiguous (it is
    also Delaware), which mis-resolves on forward geocoding. The 2-letter
    display form is applied separately at render time (see
    src/shared/country_codes.country_abbrev). accept-language=en still
    Anglicizes the names here (大阪市 -> Osaka, Deutschland -> Germany).
    """
    city = (addr.get('city') or addr.get('town') or addr.get('village')
            or addr.get('municipality') or addr.get('hamlet')
            or addr.get('suburb') or addr.get('county'))
    if not city:
        return None
    country_code = (addr.get('country_code') or '').upper()
    if country_code in ('US', 'CA'):
        iso = addr.get('ISO3166-2-lvl4') or ''   # e.g. 'US-IL'
        region = iso.split('-')[-1] if '-' in iso else ''
        if region:
            return f'{city}, {region}'
        if addr.get('state'):
            return f'{city}, {addr["state"]}'
        return city
    # Foreign: keep the full (English) country name — geocode-safe.
    return f'{city}, {addr["country"]}' if addr.get('country') else city


def _reverse_geocode_one(lat: float, lon: float) -> str | None:
    url = (f'{NOMINATIM_REVERSE_URL}?'
           f'{urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 10, "accept-language": "en"})}')
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'[reverse] ({lat},{lon}): request failed ({e})')
        return None
    addr = data.get('address') if isinstance(data, dict) else None
    if not addr:
        print(f'[reverse] ({lat},{lon}): no address in result')
        return None
    return _region_from_address(addr)


def _read_reverse_cache(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[(row['lat_round'], row['lon_round'])] = row['city_state']
            except KeyError:
                continue
    return out


def reverse_geocode(points: Iterable[tuple[float, float]],
                    cache_path: Path = REVERSE_CACHE_PATH,
                    ) -> dict[tuple[float, float], str]:
    """Map start-of-activity ``(lat, lon)`` points to ``'City, ST'`` strings.

    Caches by a rounded (lat, lon) grid so repeated starts from the same area
    cost a single Nominatim reverse lookup. Returns ``{(lat, lon): city_state}``
    for every input we could resolve (unresolvable points are omitted).
    """
    cache = _read_reverse_cache(cache_path)
    out: dict[tuple[float, float], str] = {}
    added = []
    seen_keys: set[tuple[str, str]] = set()
    for lat, lon in points:
        key = (f'{round(lat, REVERSE_ROUND):.{REVERSE_ROUND}f}',
               f'{round(lon, REVERSE_ROUND):.{REVERSE_ROUND}f}')
        if key not in cache and key not in seen_keys:
            cs = _reverse_geocode_one(lat, lon)
            time.sleep(RATE_LIMIT_SEC)
            seen_keys.add(key)
            if cs:
                cache[key] = cs
                added.append([key[0], key[1], cs, date.today().isoformat()])
        if key in cache:
            out[(lat, lon)] = cache[key]
    if added:
        path = cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(REVERSE_CACHE_HEADER)
            w.writerows(added)
        print(f'[reverse] appended {len(added)} new entries to {path.name}')
    return out


def _read_cache(path: Path) -> dict[str, dict]:
    """``{city_state: {'lat','lon','tz','geocoded_at'}}``. Tolerates a legacy
    cache file with no ``tz`` column (tz comes back None, to be backfilled)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(CACHE_HEADER)
        return {}
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[row['city_state']] = {
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude']),
                    'tz': (row.get('tz') or None) or None,
                    'geocoded_at': row.get('geocoded_at') or '',
                }
            except (KeyError, ValueError):
                continue
    return out


def _write_cache(path: Path, cache: dict[str, dict]) -> None:
    """Rewrite the whole cache (sorted by city_state) with the current header.
    Also migrates a legacy headerless-tz file to the new schema."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CACHE_HEADER)
        for cs in sorted(cache):
            v = cache[cs]
            w.writerow([cs, v['lat'], v['lon'], v.get('tz') or '',
                        v.get('geocoded_at') or ''])


def ensure_coords(city_states: Iterable[str],
                  cache_path: Path = CACHE_PATH,
                  overrides: dict[str, tuple[float, float]] | None = None,
                  ) -> dict[str, tuple[float, float, str | None]]:
    """Return ``{city_state: (lat, lon, tz)}`` for everything we have coords for.

    Geocodes any inputs missing from the cache (lat/lon via Nominatim, then the
    IANA ``tz`` via Open-Meteo) and backfills ``tz`` for any cached city that
    lacks one (e.g. a legacy cache predating the column). Rewrites the cache
    sorted if anything changed. Failures are skipped (the city won't be in the
    returned dict, or keeps a None tz to retry next run).

    ``overrides`` (snapshot ``coordinates`` section) wins over the cached coords
    in the returned dict but is NOT persisted — keeping ``city_coords.csv``
    regenerable from source. The existing cached ``tz`` is kept for an override
    (coordinate corrections almost never cross a timezone).
    """
    cache = _read_cache(cache_path)
    missing = sorted(set(city_states) - set(cache.keys()))
    if missing:
        print(f'[geocode] cache miss for {len(missing)} city-state(s); '
              f'geocoding via Nominatim')
    changed = False
    for cs in missing:
        result = _geocode_one(cs)
        time.sleep(RATE_LIMIT_SEC)
        if result is None:
            continue
        lat, lon = result
        tz = _timezone_for(lat, lon)
        time.sleep(TZ_RATE_LIMIT_SEC)
        cache[cs] = {'lat': lat, 'lon': lon, 'tz': tz,
                     'geocoded_at': date.today().isoformat()}
        changed = True

    # Backfill tz for any city missing one (legacy cache / earlier tz failure).
    backfill = [cs for cs, v in cache.items() if not v.get('tz')]
    if backfill:
        print(f'[geocode] resolving timezone for {len(backfill)} city-state(s) '
              f'via Open-Meteo')
        for cs in backfill:
            v = cache[cs]
            tz = _timezone_for(v['lat'], v['lon'])
            time.sleep(TZ_RATE_LIMIT_SEC)
            if tz:
                v['tz'] = tz
                changed = True

    if changed:
        _write_cache(cache_path, cache)
        print(f'[geocode] wrote {cache_path.name} ({len(cache)} city-states)')

    out = {cs: (v['lat'], v['lon'], v.get('tz')) for cs, v in cache.items()}
    if overrides:
        applied = 0
        for cs, (lat, lon) in overrides.items():
            prev = out.get(cs)
            tz = prev[2] if prev else None        # keep cached tz for overrides
            if prev is None or (prev[0], prev[1]) != (lat, lon):
                applied += 1
            out[cs] = (lat, lon, tz)
        if applied:
            print(f'[geocode] applied {applied} coordinate override(s) from snapshot')
    return out
