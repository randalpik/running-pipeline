"""INVESTIGATION: backfill DEM into recovery rows of elevation_measured.csv
(reuses measure_run_elevation, cache-served where warm). Easily reverted by
clearing the dem_* columns on recovery rows. Run in background."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.coros import mappings as M
from src.coros import dem_elevation as DEM

DET = Path('data/profiles/coros/details')
EM = Path('data/elevation_measured.csv')


def main():
    acts = pd.read_csv('data/watch_activities.csv', dtype={'labelId': str, 'date': str})
    ids = {}
    for _, r in acts.iterrows():
        if int(r['sport_type']) in {M.SPORT_RUN, M.SPORT_TRAIL_RUN} and \
           str(r['labelId']) not in M.EXCLUDED_LABEL_IDS:
            ids.setdefault(r['date'], []).append(str(r['labelId']))
    em = pd.read_csv(EM)
    need = em[(em['run_type'] == 'recovery') & em['dem_gain_ft'].isna()]
    print(f'recovery rows needing DEM: {len(need)}', flush=True)
    cache = DEM._load_cache()
    n = 0
    for i, row in need.iterrows():
        d = row['date']
        recs = []
        for lid in ids.get(d, []):
            p = DET / f'{lid}.json'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get('rich') == 2:
                    recs.append(rec)
        if not recs:
            continue
        res = DEM.measure_run_elevation(recs, cache)
        if res is None:
            continue
        for k, v in res.items():
            em.at[i, k] = v
        n += 1
        if n % 50 == 0:
            DEM._save_cache(cache)
            em.to_csv(EM, index=False)
            print(f'  {n} done (cache {len(cache)})', flush=True)
    DEM._save_cache(cache)
    em.to_csv(EM, index=False)
    print(f'recovery DEM backfilled: {n} rows', flush=True)


if __name__ == '__main__':
    main()
