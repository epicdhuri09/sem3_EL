"""Add Bengaluru lakes from the OpenStreetMap outline file to the database.

Run from inside backend/ (venv active), AFTER fetch_lake_outlines.py:
    python seed_from_osm.py              # the 30 largest NAMED lakes
    python seed_from_osm.py --top 100    # more
    python seed_from_osm.py --all        # every lake of 5 ha and over (use this)
    python seed_from_osm.py --names Bellandur,Varthur,Hebbal,Ulsoor
Lakes already in the database (same name) are skipped.
"""
import argparse
import json
import re
from pathlib import Path

from app import models
from app.database import SessionLocal, engine

LAKES = (Path(__file__).resolve().parents[1] / "ai_agent" / "data" / "lakes"
         / "bengaluru_lakes.geojson")
PAD = 0.002   # degrees (~200 m) around the outline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--names", default="")
    parser.add_argument("--all", action="store_true",
                        help="every lake of 5 ha and over, named or not "
                             "(what the map and ranking need)")
    args = parser.parse_args()
    if not LAKES.exists():
        raise SystemExit(f"{LAKES} not found. Run "
                         "ai_agent/scripts/fetch_lake_outlines.py first.")
    from shapely.geometry import shape
    with open(LAKES, encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    wanted = [n.strip().lower() for n in args.names.split(",") if n.strip()]
    chosen = []
    for feature in features:
        props = feature["properties"]
        name = props["name"]
        if args.all:
            if props["area_ha"] < 5:
                continue
            name = name or f"Unnamed lake {props['id'][-4:]}"
            props = {**props, "name": name}
        elif not name:
            continue
        if wanted and not any(w in name.lower() for w in wanted):
            continue
        chosen.append((props, shape(feature["geometry"])))
    chosen.sort(key=lambda item: -item[0]["area_ha"])
    if not wanted and not args.all:
        chosen = chosen[:args.top]

    models.Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    added = 0
    for props, geom in chosen:
        slug0 = re.sub(r"[^a-z0-9]+", "", props["name"].lower())[:12]
        new_id = f"wb_{slug0}_{props['id'][-4:]}"
        if db.query(models.Waterbody).filter(
                (models.Waterbody.name == props["name"])
                | (models.Waterbody.id == new_id)).first():
            print(f"skip   {props['name']} (already there)")
            continue
        minx, miny, maxx, maxy = geom.bounds
        bbox = [round(minx - PAD, 5), round(miny - PAD, 5),
                round(maxx + PAD, 5), round(maxy + PAD, 5)]
        slug = re.sub(r"[^a-z0-9]+", "", props["name"].lower())[:12]
        db.add(models.Waterbody(
            id=f"wb_{slug}_{props['id'][-4:]}", name=props["name"],
            bbox_geojson=json.dumps(bbox), ward=None, region="Bengaluru"))
        added += 1
        print(f"added  {props['name']}  {props['area_ha']:.0f} ha  {bbox}")
    db.commit()
    db.close()
    print(f"\nAdded {added} lakes.")


if __name__ == "__main__":
    main()
