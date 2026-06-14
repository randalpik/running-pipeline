"""Drill into the temperature and time-of-day disagreements from
spike_weather_compare, to judge whether they're watch faults or hand-log faults.

    python -m scripts.spike_weather_outliers

Read-only. Uses the same rep selection / scaling as production.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta, timezone

import pandas as pd

from src.coros import mappings as M
from src.coros.build_current_log import Activity
from src.coros.solar import sun_events_utc, time_of_day
from src.shared.paths import REPO_ROOT

DETAILS_DIR = REPO_ROOT / "data" / "profiles" / "coros" / "details"
HAND = REPO_ROOT / "data" / "daily.csv"
TOD_ORDER = ["early", "morning", "afternoon", "late"]


def load_details():
    out = []
    for p in sorted(DETAILS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (ValueError, OSError):
            continue
    return out


def reps(details):
    by_day = defaultdict(list)
    for d in details:
        s = d.get("summary") or {}
        if s.get("sportType") is None:
            continue
        a = Activity(d)
        if a.sport_type not in M.RUN_SPORTS:
            continue
        by_day[a.local_date].append(a)
    out = {}
    for day, items in by_day.items():
        items.sort(key=lambda a: a.start_utc)
        out[day.isoformat()] = (items[0], len(items))
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    details = load_details()
    rep = reps(details)
    hand = pd.read_csv(HAND, dtype=str).set_index("date")

    # ===================== TEMPERATURE OUTLIERS =====================
    print("=" * 78)
    print("TEMPERATURE outliers  (|watch - hand| > 3C)")
    print("=" * 78)
    rows = []
    for d, (a, ndoubles) in rep.items():
        if d not in hand.index or a.is_indoor or a.temp_c is None:
            continue
        h = fnum(hand.loc[d].get("temp_c"))
        if h is None:
            continue
        err = round(a.temp_c - h, 1)
        rows.append({
            "date": d, "hand": h, "watch": round(a.temp_c, 1), "err": err,
            "ae": abs(err), "loc": (hand.loc[d].get("city_state") or
                                    hand.loc[d].get("location") or "")[:22],
            "doubles": ndoubles, "month": int(d[5:7]),
        })
    t = pd.DataFrame(rows)
    out = t[t["ae"] > 3].sort_values("ae", ascending=False)
    print(f"total compared {len(t)}, outliers >3C: {len(out)} "
          f"({len(out)/len(t)*100:.1f}%)")
    print(f"of outliers: watch-cooler {int((out['err']<0).sum())}, "
          f"watch-warmer {int((out['err']>0).sum())}; "
          f"mean signed err {out['err'].mean():+.1f}C")
    print(f"outliers on double-run days: {int((out['doubles']>1).sum())} "
          f"(rep run may not be the run Max logged)")
    print("  by month:", dict(sorted(Counter(out["month"]).items())))
    print("  top locations:", dict(Counter(out["loc"]).most_common(8)))
    print("\n  all outliers (date: hand -> watch = err  @loc  [doubles]):")
    for _, r in out.iterrows():
        dbl = f"  x{r['doubles']}" if r["doubles"] > 1 else ""
        print(f"    {r['date']}: {r['hand']:6.1f} -> {r['watch']:6.1f} "
              f"= {r['err']:+6.1f}   {r['loc']}{dbl}")

    # ===================== TIME-OF-DAY MISSES =====================
    print("\n" + "=" * 78)
    print("TIME-OF-DAY misses  (midpoint rule != hand)")
    print("=" * 78)
    miss = []
    for d, (a, _) in rep.items():
        if d not in hand.index:
            continue
        htod = (hand.loc[d].get("time_of_day") or "").strip()
        if not htod:
            continue
        lat = None if a.is_indoor else a.lat
        lon = None if a.is_indoor else a.lon
        mid = a.start_utc + timedelta(seconds=a.total_s / 2.0)
        wmid = time_of_day(mid, lat, lon, a.tz_min)
        if wmid == htod:
            continue
        tz = timezone(timedelta(minutes=a.tz_min))
        ls, lm = a.start_utc.astimezone(tz), mid.astimezone(tz)
        sr = ss = None
        if lat is not None:
            sru, ssu = sun_events_utc(lm.date(), lat, lon)
            sr = sru.astimezone(tz).strftime("%H:%M") if sru else None
            ss = ssu.astimezone(tz).strftime("%H:%M") if ssu else None
        # distance (min) from midpoint to nearest relevant clock boundary
        miss.append({
            "date": d, "hand": htod, "watch": wmid,
            "start": ls.strftime("%H:%M"), "mid": lm.strftime("%H:%M"),
            "dur_min": round(a.total_s / 60), "sunrise": sr, "sunset": ss,
            "indoor": a.is_indoor,
        })
    md = pd.DataFrame(miss)
    print(f"total misses: {len(md)}")
    print(f"  indoor (clock-fallback) misses: {int(md['indoor'].sum())}")
    print("  miss directions (hand -> watch):",
          dict(Counter(zip(md["hand"], md["watch"])).most_common()))
    print("\n  every miss (date hand->watch | start mid dur | sunrise/sunset):")
    for _, r in md.sort_values("date").iterrows():
        ind = " INDOOR" if r["indoor"] else ""
        sun = (f"  sr {r['sunrise']} ss {r['sunset']}"
               if r["sunrise"] else "  (no gps)")
        print(f"    {r['date']}  {r['hand']:>9s} -> {r['watch']:<9s} | "
              f"start {r['start']} mid {r['mid']} dur {r['dur_min']:>3d}m{sun}{ind}")


if __name__ == "__main__":
    main()
