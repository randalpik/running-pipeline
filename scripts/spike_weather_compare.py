"""Compare Coros watch-derived weather fields against Max's hand log.

Read-only spike (Phase 1 of the watch-weather accuracy effort). Produces the
per-field metrics needed to decide whether the watch can replace hand-logged
temperature / weather / wind / time-of-day, and whether humidity (watch-only)
is worth enriching daily rows with.

Watch bins come straight from build_current_log() so they equal exactly what
the coros profile ships — no re-implementation. The rep run (first run of the
day) drives every field, matching production. wind m/s, humidity, and the
midpoint time-of-day variant aren't in the current_log frame, so they're read
from a parallel rep map built from the same Activity scaling.

    python -m scripts.spike_weather_compare

Writes output/weather_compare_diff.csv (per-day hand vs watch) for spot checks.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta

import pandas as pd

from src.coros import mappings as M
from src.coros.build_current_log import Activity, build_current_log
from src.coros.solar import time_of_day
from src.shared.paths import REPO_ROOT

DETAILS_DIR = REPO_ROOT / "data" / "profiles" / "coros" / "details"
HAND_DAILY = REPO_ROOT / "data" / "daily.csv"
OUT_CSV = REPO_ROOT / "output" / "weather_compare_diff.csv"

WIND_ORDER = ["low", "moderate", "high", "extreme"]
WIND_CODE = {b: i for i, b in enumerate(WIND_ORDER)}
TOD_ORDER = ["early", "morning", "afternoon", "late"]
# hand spelling -> watch bin spelling, for the "normalized" weather match
WEATHER_NORMALIZE = {"foggy": "fog"}


def _sep(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_details():
    out = []
    for p in sorted(DETAILS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (ValueError, OSError):
            continue
    return out


def build_rep_map(details):
    """date(iso) -> rep Activity + (wind_ms, humidity_pct), mirroring
    build_current_log's rep selection (RUN_SPORTS grouped by local date, first
    by start). Carries the fields the current_log frame doesn't expose."""
    by_day = defaultdict(list)          # date -> [(Activity, humidity_raw)]
    for d in details:
        s = d.get("summary") or {}
        if s.get("sportType") is None:
            continue
        a = Activity(d)
        if a.sport_type not in M.RUN_SPORTS:
            continue
        hum = (d.get("weather") or {}).get("humidity")
        by_day[a.local_date].append((a, hum))
    reps = {}
    for day, items in by_day.items():
        items.sort(key=lambda t: t[0].start_utc)
        a, hum = items[0]
        reps[day.isoformat()] = {
            "act": a,
            "wind_ms": a.wind_ms,
            "humidity_pct": None if hum is None else float(hum) / M.WEATHER_DIV,
        }
    return reps


def midpoint_tod(a: Activity):
    mid = a.start_utc + timedelta(seconds=a.total_s / 2.0)
    lat = None if a.is_indoor else a.lat
    lon = None if a.is_indoor else a.lon
    return time_of_day(mid, lat, lon, a.tz_min)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def confusion(pairs, order_rows=None, order_cols=None):
    """Print a hand(row) x watch(col) confusion matrix from (hand, watch) pairs."""
    rows = order_rows or sorted({h for h, _ in pairs})
    cols = order_cols or sorted({w for _, w in pairs})
    cnt = Counter(pairs)
    w = max(8, *(len(c) for c in cols)) + 1
    head = "hand\\watch".ljust(12) + "".join(c.rjust(w) for c in cols) + "   total"
    print(head)
    for r in rows:
        rtot = sum(cnt[(r, c)] for c in cols)
        line = r.ljust(12) + "".join(str(cnt[(r, c)]).rjust(w) for c in cols)
        print(line + str(rtot).rjust(8))


def main():
    details = load_details()
    df_watch, meta = build_current_log(details, geocode=False)
    reps = build_rep_map(details)
    print(f"[load] {len(details)} details -> {meta['run_activities']} run "
          f"activities over {meta['days']} watch-days")
    print(f"[load] weatherTypes seen: {meta['weather_types_seen']}")

    hand = pd.read_csv(HAND_DAILY, dtype=str).set_index("date")
    watch = df_watch.set_index("date")
    common = [d for d in watch.index if d in hand.index]
    print(f"[join] watch-days {len(watch)}, hand-days {len(hand)}, "
          f"overlap {len(common)}")

    # assemble one joined record per overlap day
    recs = []
    for d in common:
        wr, hr = watch.loc[d], hand.loc[d]
        rep = reps.get(d)
        recs.append({
            "date": d,
            "h_temp": fnum(hr.get("temp_c")),
            "w_temp": fnum(wr.get("temp_c")),
            "h_weather": (hr.get("weather") or "") or None,
            "w_weather": wr.get("weather"),
            "h_wind": (hr.get("wind") or "") or None,
            "w_wind_bin": wr.get("wind"),
            "w_wind_ms": rep["wind_ms"] if rep else None,
            "h_tod": (hr.get("time_of_day") or "") or None,
            "w_tod_start": wr.get("time_of_day"),
            "w_tod_mid": midpoint_tod(rep["act"]) if rep else None,
            "w_humidity": rep["humidity_pct"] if rep else None,
            "indoor": rep["act"].is_indoor if rep else None,
        })
    df = pd.DataFrame(recs)
    indoor_n = int(df["indoor"].fillna(False).sum())

    # ---- TEMPERATURE ----
    _sep("TEMPERATURE  (watch temp_c - hand temp_c)")
    t = df[(df["h_temp"].notna()) & (df["w_temp"].notna()) & (~df["indoor"])].copy()
    t["err"] = t["w_temp"] - t["h_temp"]
    t["ae"] = t["err"].abs()
    n = len(t)
    print(f"N = {n}  (excluded {indoor_n} indoor-rep days)")
    print(f"  mean error (bias) : {t['err'].mean():+.2f} C")
    print(f"  mean abs error    : {t['ae'].mean():.2f} C")
    print(f"  median abs error  : {t['ae'].median():.2f} C")
    print(f"  RMSE              : {(t['err']**2).mean()**0.5:.2f} C")
    print(f"  p90 |err|         : {t['ae'].quantile(0.90):.2f} C")
    print(f"  within 1C / 2C / 3C: "
          f"{(t['ae']<=1).mean()*100:.0f}% / {(t['ae']<=2).mean()*100:.0f}% / "
          f"{(t['ae']<=3).mean()*100:.0f}%")
    worst = t.reindex(t["ae"].sort_values(ascending=False).index).head(10)
    print("  worst 10 outliers (date: hand -> watch  = err):")
    for _, r in worst.iterrows():
        print(f"    {r['date']}: {r['h_temp']:5.1f} -> {r['w_temp']:5.1f}  "
              f"= {r['err']:+.1f}")

    # ---- WIND ----
    _sep("WIND  (watch m/s vs hand qualitative bin)")
    wv = df[(df["h_wind"].notna()) & (df["w_wind_ms"].notna()) & (~df["indoor"])].copy()
    print(f"N = {len(wv)}  (hand wind only exists from the 2025 schema)")
    print("  watch m/s distribution per hand bin (is it monotone?):")
    print("    hand-bin      n   mean   min    max")
    for b in WIND_ORDER:
        sub = wv[wv["h_wind"] == b]["w_wind_ms"]
        if len(sub):
            print(f"    {b:10s} {len(sub):4d}  {sub.mean():5.2f}  "
                  f"{sub.min():5.2f}  {sub.max():5.2f}")
    wv["w_bin"] = wv["w_wind_ms"].map(M.wind_bin)
    exact = (wv["w_bin"] == wv["h_wind"]).mean() * 100
    wv["h_code"] = wv["h_wind"].map(WIND_CODE)
    wv["w_code"] = wv["w_bin"].map(WIND_CODE)
    within1 = ((wv["h_code"] - wv["w_code"]).abs() <= 1).mean() * 100
    rho = wv["h_code"].corr(wv["w_wind_ms"], method="spearman")
    print(f"  hand bin vs wind_bin(watch m/s): exact {exact:.0f}%, "
          f"within-1-bin {within1:.0f}%")
    print(f"  Spearman(hand bin rank, watch m/s): {rho:.2f}  "
          f"(prior 21-day calibration: 0.62)")
    print("  confusion (hand bin x current wind_bin(watch)):")
    confusion(list(zip(wv["h_wind"], wv["w_bin"])), WIND_ORDER, WIND_ORDER)

    # ---- TIME OF DAY ----
    _sep("TIME OF DAY  (start-based vs midpoint 'majority' rule)")
    tod = df[(df["h_tod"].notna()) & (df["w_tod_start"].notna())].copy()
    n = len(tod)
    start_m = (tod["w_tod_start"] == tod["h_tod"]).mean() * 100
    mid_m = (tod["w_tod_mid"] == tod["h_tod"]).mean() * 100
    print(f"N = {n}")
    print(f"  exact match vs hand -- START-based  : {start_m:.1f}%")
    print(f"  exact match vs hand -- MIDPOINT rule: {mid_m:.1f}%  "
          f"(delta {mid_m - start_m:+.1f} pts)")
    changed = tod[tod["w_tod_start"] != tod["w_tod_mid"]]
    print(f"  midpoint differs from start on {len(changed)} days")
    print("  confusion MIDPOINT (hand x watch):")
    confusion(list(zip(tod["h_tod"], tod["w_tod_mid"])), TOD_ORDER, TOD_ORDER)

    # ---- WEATHER ----
    _sep("WEATHER  (watch bin vs hand bin)")
    we = df[(df["h_weather"].notna()) & (df["w_weather"].notna())
            & (df["w_weather"] != "indoors")].copy()
    n = len(we)
    raw_m = (we["w_weather"] == we["h_weather"]).mean() * 100
    we["h_norm"] = we["h_weather"].map(lambda x: WEATHER_NORMALIZE.get(x, x))
    norm_m = (we["w_weather"] == we["h_norm"]).mean() * 100
    print(f"N = {n}")
    print(f"  exact match           : {raw_m:.1f}%")
    print(f"  normalized (foggy=fog): {norm_m:.1f}%")
    showery = we[we["w_weather"].isin(["showers", "drizzle"])]
    if len(showery):
        print(f"  watch said showers/drizzle on {len(showery)} days; "
              f"hand labels there: {dict(Counter(showery['h_weather']))}")
    rows = sorted(set(we["h_weather"]))
    cols = sorted(set(we["w_weather"]))
    print("  confusion (hand x watch):")
    confusion(list(zip(we["h_weather"], we["w_weather"])), rows, cols)

    # ---- HUMIDITY ----
    _sep("HUMIDITY  (watch-only; never hand-logged)")
    hu = df[df["w_humidity"].notna()]["w_humidity"]
    print(f"N with humidity = {len(hu)}  (coverage "
          f"{len(hu)/len(df)*100:.0f}% of overlap days)")
    print(f"  range {hu.min():.0f}-{hu.max():.0f}%, mean {hu.mean():.0f}%, "
          f"median {hu.median():.0f}%")
    hcorr = df[df["w_humidity"].notna() & df["w_temp"].notna()]
    if len(hcorr):
        print(f"  corr(humidity, watch temp): "
              f"{hcorr['w_humidity'].corr(hcorr['w_temp']):+.2f}")
    # plausibility: mean humidity by watch weather bin
    hb = df[df["w_humidity"].notna() & df["w_weather"].notna()]
    by_w = hb.groupby("w_weather")["w_humidity"].agg(["count", "mean"])
    print("  mean humidity by watch weather bin (rain should be wetter):")
    for w, row in by_w.sort_values("mean").iterrows():
        print(f"    {w:12s} n={int(row['count']):4d}  {row['mean']:.0f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[out] per-day diff -> {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
