"""Lake analysis used by the AI service. Reads the files made by
fetch_lake_outlines.py and lake_water_timeseries.py (nothing is invented).

Honest limits (also returned to callers):
  * Only lakes that exist in the OpenStreetMap outline file can be analysed.
  * "Normal for the season" = median of the same calendar month (+-1) in OTHER
    years. With ~5-10 years of data this is a rough guide, not a statistic.
  * Rainfall is not used, so a dry year looks like lake loss.
  * Scenes from 2015-2016 (your own files, 17-20 m) and from the archive
    (10 m) can differ slightly; the source is returned with every number.
"""

import csv
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAKES_PATH = ROOT / "data" / "lakes" / "bengaluru_lakes.geojson"
SERIES_PATH = ROOT / "reports" / "lakes" / "lake_water_timeseries.csv"
PRED_DIR = ROOT / "reports" / "predictions_6band"

BELOW_FRACTION = 0.15      # this far under normal counts as "below normal"
FULL_SCORE_DEFICIT = 0.50  # 50% under normal -> anomaly score 1.0
MIN_OTHER_YEARS = 2
MIN_NORMAL_HA = 5.0        # below this a lake is dry/tiny: no judgement
MAX_AGE_DAYS = 400         # older newest-measurement = stale
SEASON_MONTHS = (2, 3, 4, 5)   # pre-monsoon level used for change points
MIN_CP_YEARS = 5
MIN_CP_SHIFT = 0.25
MIN_CP_SSE_GAIN = 0.5
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"], start=1)}


# ---------------------------------------------------------------- loading
_cache = {}
EXCLUDED_PATH = SERIES_PATH.parent / "excluded_dates.txt"
MIN_NORMAL_HA = 5.0        # a normal under this is too small to judge


def _excluded():
    """Image dates left out because the whole image looked wrong (made by
    lake_seasonality.py --save-exclusions after you checked them)."""
    if not EXCLUDED_PATH.exists():
        return frozenset()
    with open(EXCLUDED_PATH, encoding="utf-8") as handle:
        return frozenset(line.strip()[:10] for line in handle
                         if line.strip() and not line.startswith("#"))


def _stamp():
    out = SERIES_PATH.stat().st_mtime if SERIES_PATH.exists() else None
    ex = EXCLUDED_PATH.stat().st_mtime if EXCLUDED_PATH.exists() else None
    return (out, ex)


def _load_lakes():
    if "lakes" not in _cache:
        lakes = []
        if LAKES_PATH.exists():
            from shapely.geometry import shape
            with open(LAKES_PATH, encoding="utf-8") as handle:
                for feature in json.load(handle)["features"]:
                    lakes.append({**feature["properties"],
                                  "geom": shape(feature["geometry"])})
        _cache["lakes"] = lakes
    return _cache["lakes"]


def _load_series():
    """lake_id -> list of measured rows sorted by date."""
    key = _stamp()
    if _cache.get("series_key") != key or "series" not in _cache:
        series = {}
        excluded = _excluded()
        if SERIES_PATH.exists():
            with open(SERIES_PATH, encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["status"] != "ok" or row["date"][:10] in excluded:
                        continue
                    try:
                        row["_date"] = datetime.strptime(
                            row["date"][:10], "%Y-%m-%d").date()
                        row["_water"] = float(row["water_ha"])
                        row["_veg"] = float(row.get("vegetation_ha") or 0)
                    except (ValueError, KeyError):
                        continue
                    series.setdefault(row["lake_id"], []).append(row)
        for rows in series.values():
            rows.sort(key=lambda r: r["_date"])
        _cache["series"], _cache["series_key"] = series, key
    return _cache["series"]


def status():
    return {
        "lakes_file": LAKES_PATH.exists(),
        "series_file": SERIES_PATH.exists(),
        "lakes": len(_load_lakes()),
        "lakes_with_measurements": len(_load_series()),
    }


def parse_date(text):
    if not text:
        return None
    if isinstance(text, date):
        return text
    try:
        return datetime.strptime(str(text)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# --------------------------------------------------------------- matching
def match_lake(bbox=None, lake_id=None):
    """The OSM lake that overlaps the bbox most (bbox = minlon,minlat,maxlon,maxlat)."""
    lakes = _load_lakes()
    if lake_id:
        for lake in lakes:
            if lake["id"] == lake_id:
                return lake
    if not bbox or len(bbox) != 4:
        return None
    from shapely.geometry import box
    window = box(*bbox)
    best, best_area = None, 0.0
    for lake in lakes:
        if not lake["geom"].intersects(window):
            continue
        overlap = lake["geom"].intersection(window).area
        if overlap > best_area:
            best, best_area = lake, overlap
    return best


def lake_rows(lake_id):
    return _load_series().get(lake_id, [])


def pick_measurement(lake_id, when=None, max_gap_days=60):
    """Measured row nearest to `when` (latest if None). None if too far."""
    rows = lake_rows(lake_id)
    if not rows:
        return None
    if when is None:
        return rows[-1]
    best = min(rows, key=lambda r: abs((r["_date"] - when).days))
    if abs((best["_date"] - when).days) > max_gap_days:
        return None
    return best


# ---------------------------------------------------------- land cover
def landcover_percentages(bbox, when=None, max_gap_days=45):
    """Four-class shares inside the bbox from the saved model predictions."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    if not PRED_DIR.exists() or not bbox:
        return None
    scenes = []
    for path in PRED_DIR.glob("*_landcover.tif"):
        name = path.name[:-len("_landcover.tif")]
        m = re.fullmatch(r"(\d{1,2})([a-z]+)(\d{2,4})", name.lower())
        if not m or m.group(2)[:3] not in MONTHS:
            continue
        year = int(m.group(3))
        year += 2000 if year < 100 else 0
        scenes.append((date(year, MONTHS[m.group(2)[:3]], int(m.group(1))),
                       name, path))
    if not scenes:
        return None
    scenes.sort()
    if when is None:
        chosen = scenes[-1]
    else:
        chosen = min(scenes, key=lambda s: abs((s[0] - when).days))
        if abs((chosen[0] - when).days) > max_gap_days:
            return None
    with rasterio.open(chosen[2]) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window)
    data = data[data < 4]
    if data.size == 0:
        return None
    share = np.bincount(data.ravel(), minlength=4)[:4] / data.size * 100
    return {
        "scene": chosen[1], "scene_date": chosen[0].isoformat(),
        "pixels": int(data.size),
        "class_percentages": {
            "water": round(float(share[0]), 1),
            "vegetation": round(float(share[1]), 1),
            "bare_land": round(float(share[2]), 1),
            "built_up": round(float(share[3]), 1),
        },
    }


# ---------------------------------------------------------------- anomaly
def seasonal_normal(lake_id, when, key="_water"):
    """Median of the same month (+-1) in other years (open water by default)."""
    rows = lake_rows(lake_id)
    values, years = [], set()
    for r in rows:
        if r["_date"].year == when.year:
            continue
        gap = abs(r["_date"].month - when.month)
        if min(gap, 12 - gap) <= 1:
            values.append(r[key])
            years.add(r["_date"].year)
    if len(years) < MIN_OTHER_YEARS:
        return None, len(years)
    return float(np.median(values)), len(years)


def anomaly(lake_id, when=None):
    row = pick_measurement(lake_id, when, max_gap_days=10_000)
    if row is None:
        return {"anomaly_score": None, "change_type": "no_measurements",
                "note": "No usable (cloud-free) measurement for this lake."}
    age_days = (date.today() - row["_date"]).days
    if when is None and age_days > MAX_AGE_DAYS:
        return {"anomaly_score": None, "change_type": "stale_data",
                "date": row["_date"].isoformat(), "age_days": age_days,
                "note": "The newest measurement of this lake is more than "
                        f"{MAX_AGE_DAYS} days old, so no current anomaly is "
                        "reported."}
    expected, n_years = seasonal_normal(lake_id, row["_date"])
    out = {
        "date": row["_date"].isoformat(), "source": row["source"],
        "observed_water_ha": round(row["_water"], 2),
        "expected_water_ha": None if expected is None else round(expected, 2),
        "years_in_normal": n_years,
        "expected_ndwi": None, "observed_ndwi": None,
    }
    if expected is not None and expected < MIN_NORMAL_HA:
        out.update(anomaly_score=None, change_type="normal_too_small",
                   note="This lake normally holds under 5 ha of open water "
                        "in this season, so a shortfall is not judged.")
        return out
    if expected is not None and 0 <= expected < MIN_NORMAL_HA:
        out.update(anomaly_score=None, change_type="normal_too_small",
                   note=f"This lake's normal for the season is under "
                        f"{MIN_NORMAL_HA:.0f} ha (often dry or tiny), so a "
                        "percentage change would be noise. Not judged.")
        return out
    if expected is None or expected <= 0:
        out.update(anomaly_score=None, change_type="insufficient_history",
                   note=f"Need the same season in at least {MIN_OTHER_YEARS} "
                        "other years; have " + str(n_years) + ".")
        return out
    deficit = (expected - row["_water"]) / expected
    out["deficit_fraction"] = round(float(deficit), 3)
    out["anomaly_score"] = round(float(np.clip(
        deficit / FULL_SCORE_DEFICIT, 0, 1)), 3)
    if deficit < -BELOW_FRACTION:
        out["change_type"] = "above_seasonal_normal"
    elif deficit < BELOW_FRACTION:
        out["change_type"] = "normal_for_season"
    else:
        earlier = [r for r in lake_rows(lake_id)
                   if r["_date"] < row["_date"]][-2:]
        low = 0
        for r in earlier:
            exp, _ = seasonal_normal(lake_id, r["_date"])
            if exp and (exp - r["_water"]) / exp >= BELOW_FRACTION:
                low += 1
        out["change_type"] = ("persistent_degradation"
                              if len(earlier) == 2 and low == 2
                              else "below_seasonal_normal")
    # open water lost AND green growth gained: weeds / plants over the lake
    exp_veg, _ = seasonal_normal(lake_id, row["_date"], "_veg")
    outline = float(row.get("outline_ha") or 0)
    out["vegetation_ha"] = round(row["_veg"], 2)
    out["expected_vegetation_ha"] = (None if exp_veg is None
                                     else round(exp_veg, 2))
    if (exp_veg is not None and outline > 0 and deficit >= BELOW_FRACTION
            and row["_veg"] - exp_veg >= 0.10 * outline):
        out["interpretation"] = (
            "open_water_lost_and_vegetation_gained: green growth inside the "
            "outline rose while open water fell (floating weeds, marsh, or "
            "plants growing on an exposed lake bed - the image alone cannot "
            "tell which).")
    out["note"] = ("Compared with the same season in other years. Rainfall "
                   "is not considered, so a dry year also shows as below "
                   "normal.")
    return out


# ------------------------------------------------------------ change point
def yearly_levels(lake_id):
    """Pre-monsoon (Feb-May) median open water per year."""
    by_year = {}
    for r in lake_rows(lake_id):
        if r["_date"].month in SEASON_MONTHS:
            by_year.setdefault(r["_date"].year, []).append(r["_water"])
    return {y: float(np.median(v)) for y, v in sorted(by_year.items())}


def changepoint(lake_id):
    levels = yearly_levels(lake_id)
    years = list(levels)
    base = {"change_point_year": None, "before_mean": None,
            "after_mean": None, "yearly_levels_ha": {
                str(y): round(v, 2) for y, v in levels.items()}}
    if len(years) < MIN_CP_YEARS:
        base["narrative"] = (
            f"Only {len(years)} pre-monsoon years measured; at least "
            f"{MIN_CP_YEARS} are needed to look for a change point.")
        return base
    values = np.array([levels[y] for y in years])
    total = float(((values - values.mean()) ** 2).sum())
    best = None
    for k in range(2, len(years) - 1):
        left, right = values[:k], values[k:]
        sse = float(((left - left.mean()) ** 2).sum()
                    + ((right - right.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k)
    sse, k = best
    before, after = float(values[:k].mean()), float(values[k:].mean())
    shift = (after - before) / before if before > 0 else 0.0
    gain = 1 - sse / total if total > 0 else 0.0
    if abs(shift) >= MIN_CP_SHIFT and gain >= MIN_CP_SSE_GAIN:
        direction = "dropped" if shift < 0 else "rose"
        base.update(
            change_point_year=int(years[k]),
            before_mean=round(before, 2), after_mean=round(after, 2),
            narrative=(f"Pre-monsoon open water {direction} from about "
                       f"{before:.0f} ha to {after:.0f} ha ({shift:+.0%}) "
                       f"from {years[k]}. This compares one season across "
                       "years but ignores rainfall."))
    else:
        base["narrative"] = ("No clear step change in pre-monsoon open "
                             "water across the measured years.")
    return base


# ------------------------------------------------------- priority / cause
def priority(severity=0, persistence=50, ecological_importance=50,
             population_impact=50):
    def clip(x):
        return float(np.clip(float(x or 0), 0, 100))
    comps = {"severity": clip(severity), "persistence": clip(persistence),
             "ecological_importance": clip(ecological_importance),
             "population_impact": clip(population_impact)}
    weights = {"severity": 0.40, "persistence": 0.25,
               "ecological_importance": 0.20, "population_impact": 0.15}
    score = sum(comps[k] * weights[k] for k in comps)
    return {"priority_score": round(score, 1), "components": comps,
            "weights": weights,
            "note": "Weighted sum; weights are a project choice, not "
                    "learned. Inputs left at 50 are placeholders."}


def cause(segmentation=None, anomaly_result=None, lake_summary=None):
    hyp = []
    anomaly_result = anomaly_result or {}
    segmentation = segmentation or {}
    change = anomaly_result.get("change_type")
    low = change in ("below_seasonal_normal", "persistent_degradation")
    if low:
        hyp.append({"cause": "Below-normal rainfall or inflow",
                    "confidence": "unchecked",
                    "evidence": "Open water is below this lake's normal for "
                                "the season. No rainfall data is used."})
    veg_share = (lake_summary or {}).get("vegetation_share_of_outline")
    if veg_share is not None and veg_share > 0.4:
        hyp.append({"cause": "Weed or marsh vegetation covering the lake",
                    "confidence": "medium",
                    "evidence": f"{veg_share:.0%} of the outline is green "
                                "growth rather than open water."})
    built = segmentation.get("built_up")
    if low and built is not None and built > 40:
        hyp.append({"cause": "Encroachment or urban runoff around the lake",
                    "confidence": "low",
                    "evidence": f"{built:.0f}% of the area is built-up."})
    if not hyp:
        hyp.append({"cause": "No unusual pattern detected",
                    "confidence": "n/a", "evidence": change or "no data"})
    return {"hypotheses": hyp,
            "note": "Hypotheses from simple rules, not confirmed causes."}


def lake_summary(lake, row):
    if lake is None or row is None:
        return None
    outline = float(row.get("outline_ha") or lake["area_ha"] or 0)
    veg = float(row.get("vegetation_ha") or 0)
    return {
        "lake_id": lake["id"], "name": lake["name"],
        "outline_ha": round(outline, 1),
        "open_water_ha": round(row["_water"], 2),
        "vegetation_ha": round(veg, 2),
        "vegetation_share_of_outline": (veg / outline) if outline else None,
        "date": row["_date"].isoformat(), "source": row["source"],
    }


# ================================================================ extras for
# the dashboard: series, seasonality, summary, outlines, ranking
MIN_LAKE_HA = 5.0           # lakes under this are not judged
PERSISTENCE_LOOKBACK = 4    # newest N usable dates used for "persistence"


def _all_rows():
    """lake_id -> every row (ok, small, cloudy) sorted by date."""
    key = _stamp()
    if _cache.get("all_key") != key or "all" not in _cache:
        rows = {}
        excluded = _excluded()
        if SERIES_PATH.exists():
            with open(SERIES_PATH, encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["date"][:10] in excluded:
                        continue
                    try:
                        row["_date"] = datetime.strptime(
                            row["date"][:10], "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    rows.setdefault(row["lake_id"], []).append(row)
        for items in rows.values():
            items.sort(key=lambda r: r["_date"])
        _cache["all"], _cache["all_key"] = rows, key
        for name in ("summary", "geojson", "rank"):
            _cache.pop(name, None)
    return _cache["all"]


def _lake_by_id(lake_id):
    for lake in _load_lakes():
        if lake["id"] == lake_id:
            return lake
    return None


def series(lake_id, start=None, end=None):
    lake = _lake_by_id(lake_id)
    points = []
    for r in _all_rows().get(lake_id, []):
        if start and r["_date"] < start:
            continue
        if end and r["_date"] > end:
            continue
        ok = r["status"] in ("ok", "small")
        normal = None
        if r["status"] == "ok":
            normal, _ = seasonal_normal(lake_id, r["_date"])
        points.append({
            "date": r["_date"].isoformat(),
            "open_water_ha": round(float(r["water_ha"]), 2) if ok else None,
            "vegetation_ha": round(float(r["vegetation_ha"] or 0), 2) if ok else None,
            "source": r["source"],
            "pixel_m": float(r["pixel_m"]) if r.get("pixel_m") else None,
            "status": "cloudy" if r["status"] == "cloudy" else r["status"],
            "normal_ha": None if normal is None else round(normal, 2),
        })
    return {"lake_id": lake_id,
            "name": lake["name"] if lake else None,
            "outline_ha": lake["area_ha"] if lake else None,
            "points": points}


def seasonality(lake_id):
    rows = lake_rows(lake_id)
    by_month = {}
    for r in rows:
        by_month.setdefault(r["_date"].month, {}).setdefault(
            r["_date"].year, []).append(r["_water"])
    normal = []
    for month in range(1, 13):
        years = by_month.get(month, {})
        if len(years) >= MIN_OTHER_YEARS:
            per_year = [float(np.mean(v)) for v in years.values()]
            normal.append({"month": month,
                           "normal_ha": round(float(np.median(per_year)), 2),
                           "low_ha": round(float(np.min(per_year)), 2),
                           "high_ha": round(float(np.max(per_year)), 2),
                           "years": len(years)})
    pre, post = {}, {}
    for r in rows:
        d = r["_date"]
        if d.month in (3, 4, 5):
            pre.setdefault(d.year, []).append(r["_water"])
        if d.month >= 10:
            post.setdefault(d.year, []).append(r["_water"])
        elif d.month == 1:
            post.setdefault(d.year - 1, []).append(r["_water"])
    yearly = [{"year": y,
               "pre_monsoon_min_ha": round(min(pre[y]), 2) if y in pre else None,
               "pre_monsoon_median_ha": round(float(np.median(pre[y])), 2) if y in pre else None,
               "post_monsoon_max_ha": round(max(post[y]), 2) if y in post else None}
              for y in sorted(set(pre) | set(post))]
    return {"lake_id": lake_id, "monthly_normal": normal, "yearly": yearly,
            "change_point": changepoint(lake_id),
            "note": "Normal = median over years of the month's mean; "
                    "rainfall is not used."}


def lake_status(lake):
    """Status + key numbers of one lake from its newest measurement."""
    if lake["area_ha"] < MIN_LAKE_HA:
        return {"status": "not_measured", "open_water_ha": None,
                "date": None, "deficit_fraction": None}
    if not lake_rows(lake["id"]):
        return {"status": "no_measurements", "open_water_ha": None,
                "date": None, "deficit_fraction": None}
    a = anomaly(lake["id"], None)
    return {"status": a.get("change_type"),
            "open_water_ha": a.get("observed_water_ha"),
            "date": a.get("date"),
            "deficit_fraction": a.get("deficit_fraction"),
            "vegetation_ha": a.get("vegetation_ha"),
            "interpretation": a.get("interpretation")}


def summary():
    _all_rows()
    if "summary" in _cache:
        return _cache["summary"]
    counts, replaced, newest, measured = {}, 0, None, 0
    for lake in _load_lakes():
        s = lake_status(lake)
        counts[s["status"]] = counts.get(s["status"], 0) + 1
        if lake["area_ha"] >= MIN_LAKE_HA and lake_rows(lake["id"]):
            measured += 1
        if s.get("interpretation"):
            replaced += 1
        if s["date"] and (newest is None or s["date"] > newest):
            newest = s["date"]
    below = (counts.get("below_seasonal_normal", 0)
             + counts.get("persistent_degradation", 0))
    out = {"lakes_measured": measured, "lakes_below_normal": below,
           "lakes_open_water_replaced_by_plants": replaced,
           "status_counts": counts, "newest_image": newest,
           "data_processed": datetime.fromtimestamp(
               SERIES_PATH.stat().st_mtime).date().isoformat()
           if SERIES_PATH.exists() else None}
    _cache["summary"] = out
    return out


def geojson():
    _all_rows()
    if "geojson" in _cache:
        return _cache["geojson"]
    from shapely.geometry import mapping
    features = []
    for lake in _load_lakes():
        s = lake_status(lake)
        c = lake["geom"].representative_point()
        features.append({
            "type": "Feature",
            "properties": {"lake_id": lake["id"], "name": lake["name"],
                           "outline_ha": lake["area_ha"],
                           "lat": round(c.y, 6), "lon": round(c.x, 6),
                           "bbox": [round(v, 5) for v in lake["geom"].bounds],
                           **s},
            "geometry": mapping(lake["geom"].simplify(
                0.00005, preserve_topology=True)),
        })
    out = {"type": "FeatureCollection", "features": features}
    _cache["geojson"] = out
    return out


def _persistence(lake_id):
    ok = [r for r in lake_rows(lake_id)][-PERSISTENCE_LOOKBACK:]
    low = n = 0
    for r in ok:
        exp, _ = seasonal_normal(lake_id, r["_date"])
        if exp and exp > 0:
            n += 1
            if (exp - r["_water"]) / exp >= BELOW_FRACTION:
                low += 1
    return (100.0 * low / n) if n else None


def rank():
    """Priority of every judged lake, computed from the data (no stored rows).
    ecological_importance is a SIZE PROXY (outline / 300 ha, capped at 100);
    population_impact is a placeholder (50) until real data exists."""
    _all_rows()
    if "rank" in _cache:
        return _cache["rank"]
    items = []
    for lake in _load_lakes():
        s = lake_status(lake)
        if s.get("deficit_fraction") is None:
            continue
        anom = anomaly(lake["id"], None)
        severity = (anom.get("anomaly_score") or 0) * 100
        persistence = _persistence(lake["id"])
        eco = min(100.0, 100.0 * lake["area_ha"] / 300.0)
        p = priority(severity, persistence if persistence is not None else 50,
                     eco, 50)
        c = lake["geom"].representative_point()
        items.append({
            "lake_id": lake["id"], "name": lake["name"],
            "lat": round(c.y, 6), "lon": round(c.x, 6),
            "bbox": [round(v, 5) for v in lake["geom"].bounds],
            "outline_ha": lake["area_ha"], "status": s["status"],
            "open_water_ha": s["open_water_ha"], "date": s["date"],
            "priority_score": p["priority_score"], "severity": round(severity, 1),
            "persistence": None if persistence is None else round(persistence, 1),
            "ecological_importance": round(eco, 1), "population_impact": 50.0,
        })
    items.sort(key=lambda i: -i["priority_score"])
    for position, item in enumerate(items, start=1):
        item["rank"] = position
    out = {"items": items,
           "note": "Ecological importance is a size proxy and population "
                   "impact a placeholder (50); weights are a project choice."}
    _cache["rank"] = out
    return out


def nearest_lake(lat, lon, max_km=5.0):
    """Lake whose outline contains the point, else the nearest centre."""
    from shapely.geometry import Point
    point = Point(lon, lat)
    best, best_d = None, None
    for lake in _load_lakes():
        if lake["geom"].contains(point):
            return lake, 0.0
        d = lake["geom"].distance(point) * 111.32   # degrees -> km (rough)
        if best_d is None or d < best_d:
            best, best_d = lake, d
    if best is not None and best_d <= max_km:
        return best, round(best_d, 2)
    return None, None
