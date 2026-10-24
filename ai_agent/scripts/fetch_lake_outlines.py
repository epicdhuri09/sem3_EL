"""Download the outline of every lake, tank and pond in Bengaluru.

Source: OpenStreetMap (via the public Overpass API). Free, no account.
Licence: ODbL - credit "(c) OpenStreetMap contributors" in your report.

Writes to ai_agent/data/lakes/:
    bengaluru_lakes.geojson   outlines (longitude/latitude)
    bengaluru_lakes.csv       id, name, area_ha, centre
    lakes_preview.png         a map to check by eye
    overpass_raw.json         the raw download (re-use it with --from-file)

Important limits:
  * OpenStreetMap is volunteer-made. It is NOT complete, and an outline is a
    fixed shape (roughly the lake at full level), not the water on any date.
    The next step measures the water INSIDE each outline, date by date.
  * Unnamed ponds are included; small ones are hard to measure from space.

Run from the project root:
    python ai_agent/scripts/fetch_lake_outlines.py
Options:
    --min-ha 0.5      smallest water body to keep
    --from-file F     use a saved Overpass JSON instead of downloading
    --bbox S W N E    search box (default covers greater Bengaluru)
Needs:  pip install shapely
"""

import argparse
import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.warp import transform_geom
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "lakes"
UTM = "EPSG:32643"          # metres; Bengaluru is in UTM zone 43N
DEFAULT_BBOX = (12.80, 77.40, 13.20, 77.85)   # south, west, north, east
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# OSM "water=*" values that are NOT lakes/tanks
EXCLUDE_WATER = {
    "river", "canal", "stream", "drain", "ditch", "wastewater", "pool",
    "swimming_pool", "fountain", "reflecting_pool", "basin", "moat", "lock",
    "stream_pool", "fish_pass", "rapids",
}
KEY_NAMES = ["Bellandur", "Varthur", "Hebbal", "Ulsoor"]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def build_query(bbox):
    s, w, n, e = bbox
    area = f"({s},{w},{n},{e})"
    return (
        "[out:json][timeout:180];\n(\n"
        f'  way["natural"="water"]{area};\n'
        f'  relation["natural"="water"]{area};\n'
        f'  way["landuse"="reservoir"]{area};\n'
        f'  relation["landuse"="reservoir"]{area};\n'
        ");\nout geom;"
    )


def fetch_overpass(query):
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_error = None
    for url in ENDPOINTS:
        try:
            print(f"Asking {url} ... (can take a minute)")
            request = urllib.request.Request(
                url, data=body,
                headers={"User-Agent": "sem3_EL-student-project/1.0"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            print(f"  failed: {error}")
    raise SystemExit(
        f"Could not download from OpenStreetMap ({last_error}).\n"
        "Check your internet/VPN and try again. Or open "
        "https://overpass-turbo.eu, run the same query, export the raw "
        "JSON, and pass it with --from-file."
    )


# --------------------------------------------------------------------------
# Turning OSM elements into polygons
# --------------------------------------------------------------------------

def to_xy(geometry):
    return [(point["lon"], point["lat"]) for point in geometry]


def element_geometry(element):
    """Polygon (lon/lat) for an OSM way or relation, or None."""
    if element["type"] == "way":
        points = to_xy(element.get("geometry", []))
        if len(points) >= 4 and points[0] == points[-1]:
            polygon = Polygon(points)
            return polygon if polygon.is_valid else polygon.buffer(0)
        return None                      # open way = a stream/shoreline

    if element["type"] == "relation":
        outer, inner = [], []
        for member in element.get("members", []):
            if member.get("type") != "way" or not member.get("geometry"):
                continue
            line = LineString(to_xy(member["geometry"]))
            (inner if member.get("role") == "inner" else outer).append(line)
        if not outer:
            return None
        outer_polys = list(polygonize(unary_union(outer)))
        if not outer_polys:
            return None
        geometry = unary_union(outer_polys)
        if inner:
            islands = list(polygonize(unary_union(inner)))
            if islands:
                geometry = geometry.difference(unary_union(islands))
        return geometry if geometry.is_valid else geometry.buffer(0)
    return None


def wanted(tags):
    if tags.get("water") in EXCLUDE_WATER:
        return False
    if tags.get("leisure") == "swimming_pool":
        return False
    return tags.get("natural") == "water" or tags.get("landuse") == "reservoir"


def to_utm(geometry):
    return shape(transform_geom("EPSG:4326", UTM, mapping(geometry)))


def covered_fraction(geometry, kept):
    """How much of `geometry` lies inside shapes we already kept (0-1)."""
    if geometry.area == 0:
        return 1.0
    inside = 0.0
    for other in kept:
        ox0, oy0, ox1, oy1 = other["utm"].bounds
        gx0, gy0, gx1, gy1 = geometry.bounds
        if ox1 < gx0 or gx1 < ox0 or oy1 < gy0 or gy1 < oy0:
            continue
        inside += geometry.intersection(other["utm"]).area
        if inside >= geometry.area:
            return 1.0
    return inside / geometry.area


def build_records(data, min_ha):
    records, seen = [], 0
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        if not wanted(tags):
            continue
        geometry = element_geometry(element)
        if geometry is None or geometry.is_empty:
            continue
        seen += 1
        utm = to_utm(geometry)
        area_ha = utm.area / 10_000.0
        if area_ha < min_ha:
            continue
        name = tags.get("name:en") or tags.get("name") or ""
        records.append({
            "osm": f"{element['type']}/{element['id']}",
            "name": name, "ll": geometry, "utm": utm, "area_ha": area_ha,
        })

    # remove duplicates / shapes mostly covered by a bigger one
    records.sort(key=lambda r: r["area_ha"], reverse=True)
    kept = []
    for record in records:
        if covered_fraction(record["utm"], kept) < 0.8:
            kept.append(record)
    print(f"Water shapes found: {seen}; kept {len(kept)} "
          f"(>= {min_ha} ha, duplicates removed)")
    return kept


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def write_outputs(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features, rows = [], []
    for index, record in enumerate(records, start=1):
        lake_id = f"lake_{index:04d}"
        centre = record["ll"].representative_point()
        props = {
            "id": lake_id, "name": record["name"],
            "area_ha": round(record["area_ha"], 2), "osm": record["osm"],
            "lat": round(centre.y, 6), "lon": round(centre.x, 6),
        }
        features.append({"type": "Feature", "properties": props,
                         "geometry": mapping(record["ll"])})
        rows.append(props)

    geojson = OUT_DIR / "bengaluru_lakes.geojson"
    with open(geojson, "w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features},
                  handle, ensure_ascii=False)
    table = OUT_DIR / "bengaluru_lakes.csv"
    with open(table, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 9))
    for record in records:
        geoms = (record["ll"].geoms if record["ll"].geom_type == "MultiPolygon"
                 else [record["ll"]])
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="#2a78d6", lw=0)
    for row in rows:
        if any(key.lower() in row["name"].lower() for key in KEY_NAMES):
            ax.annotate(row["name"], (row["lon"], row["lat"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_aspect(1 / 0.975)     # roughly equal scale at Bengaluru's latitude
    ax.set_title(f"Bengaluru water bodies from OpenStreetMap "
                 f"({len(records)} shapes)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    preview = OUT_DIR / "lakes_preview.png"
    fig.savefig(preview, dpi=120)
    plt.close(fig)
    return geojson, table, preview, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ha", type=float, default=0.5)
    parser.add_argument("--from-file", default=None)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"),
                        default=DEFAULT_BBOX)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = fetch_overpass(build_query(tuple(args.bbox)))
        with open(OUT_DIR / "overpass_raw.json", "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    records = build_records(data, args.min_ha)
    if not records:
        raise SystemExit("No water bodies found. Check the search box.")
    geojson, table, preview, rows = write_outputs(records)

    total = sum(r["area_ha"] for r in records)
    named = [r for r in rows if r["name"]]
    print(f"\nTotal outline area: {total:,.0f} ha | named: {len(named)} | "
          f"unnamed: {len(rows) - len(named)}")
    bands = [("under 1 ha", 0, 1), ("1-5 ha", 1, 5), ("5-20 ha", 5, 20),
             ("over 20 ha", 20, 1e9)]
    for label, lo, hi in bands:
        count = sum(1 for r in rows if lo <= r["area_ha"] < hi)
        print(f"  {label:<11}{count:>5}")

    print("\nLargest 15:")
    for row in rows[:15]:
        print(f"  {row['id']}  {row['area_ha']:>8.1f} ha  "
              f"{row['name'] or '(unnamed)'}")

    print("\nKey lakes found?")
    for key in KEY_NAMES:
        hits = [r for r in rows if key.lower() in r["name"].lower()]
        if hits:
            print(f"  {key:<10} yes  " + ", ".join(
                f"{h['name']} ({h['area_ha']:.0f} ha)" for h in hits[:3]))
        else:
            print(f"  {key:<10} NOT FOUND (try the name in Kannada, or add "
                  "it by hand)")

    print(f"\nSaved:\n  {geojson}\n  {table}\n  {preview}")
    print("Open lakes_preview.png and check the shapes look like Bengaluru's "
          "lakes.\nCredit: (c) OpenStreetMap contributors (ODbL).")


if __name__ == "__main__":
    main()
