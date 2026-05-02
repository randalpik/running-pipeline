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

NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
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
                  cache_path: Path = CACHE_PATH
                  ) -> dict[str, tuple[float, float]]:
    """Return ``{city_state: (lat, lon)}`` for everything we have coords for.

    Geocodes any inputs missing from the cache, appends to disk as it goes,
    and rewrites sorted at the end if anything was added. Failures are
    silently skipped (the city won't be in the returned dict).
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
    return cache
