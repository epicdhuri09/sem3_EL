import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lakewater import (choose_items, load_archive_scene, load_lakes,
                       measure_lake, search_archive)

LEVELS = (0.02, 0.03, 0.04)

def dark_stats(debug, px_ha):
    inside, clear = debug["inside"], debug["clear"]
    use = inside & clear
    g, n, s = debug["green"], debug["nir"], debug["swir"]
    out = {"pixels": int(use.sum())}
    out["water_now_ha"] = round(float((debug["water"] & inside).sum() * px_ha), 2)
    if not use.any():
        return out
    for name, band in (("green", g), ("nir", n), ("swir", s)):
        v = band[use]
        out[f"{name}_p5"] = round(float(np.percentile(v, 5)), 4)
        out[f"{name}_p50"] = round(float(np.percentile(v, 50)), 4)
    for level in LEVELS:
        dark = use & (n < level) & (s < level) & (g < level + 0.02)
        out[f"dark<{level}_ha"] = round(float(dark.sum() * px_ha), 2)
        out[f"dark<{level}_not_counted_ha"] = round(float((dark & ~debug["water"]).sum() * px_ha), 2)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--dates", required=True)
    args = parser.parse_args()
    only = [o.strip() for o in args.only.split(",") if o.strip()]
    lakes = load_lakes(min_ha=5.0, only=only)
    if not lakes:
        raise SystemExit("No lake matched.")
    boxes = [lake["geom_ll"].bounds for lake in lakes]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    items = search_archive(bbox, args.year, args.year)
    chosen, _ = choose_items(items, bbox, 1, 70.0)
    for day in [d.strip() for d in args.dates.split(",") if d.strip()]:
        match = [i for i in chosen if i["properties"]["datetime"][:10] == day]
        if not match:
            print(f"\n{day}: no chosen scene on that date"); continue
        scene = load_archive_scene(match[0], lakes)
        if scene is None:
            print(f"\n{day}: scene could not be read"); continue
        print(f"\n=== {day}  {scene.item_id}  cloud {scene.cloud_pct}%  {scene.extra.get('offset_note')}")
        px_ha = scene.pixel_m ** 2 / 10_000.0
        for lake in lakes:
            row, debug = measure_lake(scene, lake, keep_debug=True)
            if row is None or debug is None:
                print(f"{lake['name']}: {row['status'] if row else 'outside'}"); continue
            print(f"\n{lake['name']} ({lake['area_ha']:.0f} ha outline)")
            for key, value in dark_stats(debug, px_ha).items():
                print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
