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
CACHE_HEADER = ['city_state', 'latitude', 'longitude', 'geocoded_at']

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
USER_AGENT = 'max-running-pipeline/1.0'
RATE_LIMIT_SEC = 1.1   # Nominatim ToS: max 1 req/sec

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


def _read_cache(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(CACHE_HEADER)
        return {}
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[row['city_state']] = (float(row['latitude']),
                                          float(row['longitude']))
            except (KeyError, ValueError):
                continue
    return out


def _append_cache(path: Path, city_state: str, lat: float, lon: float) -> None:
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([city_state, lat, lon, date.today().isoformat()])


def _resort_cache(path: Path) -> None:
    """Rewrite cache sorted by city_state for clean diffs."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sorted(reader, key=lambda r: r[0])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ensure_coords(city_states: Iterable[str],
                  cache_path: Path = CACHE_PATH,
                  overrides: dict[str, tuple[float, float]] | None = None,
                  ) -> dict[str, tuple[float, float]]:
    """Return ``{city_state: (lat, lon)}`` for everything we have coords for.

    Geocodes any inputs missing from the cache, appends to disk as it goes,
    and rewrites sorted at the end if anything was added. Failures are
    silently skipped (the city won't be in the returned dict).

    ``overrides`` is applied last and wins over cached / freshly-fetched
    Nominatim results — used by the snapshot's ``coordinates`` section to
    correct city-states whose Nominatim lookup is wrong, without mutating
    ``city_coords.csv`` (which is regenerable from source).
    """
    cache = _read_cache(cache_path)
    missing = sorted(set(city_states) - set(cache.keys()))
    if missing:
        print(f'[geocode] cache miss for {len(missing)} city-state(s); '
              f'geocoding via Nominatim')
    added = 0
    for cs in missing:
        result = _geocode_one(cs)
        if result is not None:
            lat, lon = result
            cache[cs] = (lat, lon)
            _append_cache(cache_path, cs, lat, lon)
            added += 1
        time.sleep(RATE_LIMIT_SEC)
    if added:
        _resort_cache(cache_path)
        print(f'[geocode] appended {added} new entries to {cache_path.name}')
    if overrides:
        applied = 0
        for cs, (lat, lon) in overrides.items():
            if cs in cache and cache[cs] != (lat, lon):
                applied += 1
            cache[cs] = (lat, lon)
        if applied:
            print(f'[geocode] applied {applied} coordinate override(s) from snapshot')
    return cache
