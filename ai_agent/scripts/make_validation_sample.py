"""Build a random validation sample of lake pictures, and score your ratings.

Step 1 - make the sample (needs internet; saves pictures + a rating sheet):
    python ai_agent/scripts/make_validation_sample.py --n 28 --seed 1
Step 2 - fill the `rating` column of validation_ratings.csv:
    ok / under / over / bad_image
Step 3 - score it:
    python ai_agent/scripts/make_validation_sample.py --score
"""

import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "reports" / "lakes" / "lake_water_timeseries.csv"
OUT_DIR = ROOT / "reports" / "validation"
SHEET = OUT_DIR / "validation_ratings.csv"
STRATA = [("almost none (<1% of outline)", 0.0, 1.0),
          ("a little (1-10%)", 1.0, 10.0),
          ("some (10-40%)", 10.0, 40.0),
          ("a lot (40%+)", 40.0, 1000.0)]
RATINGS = ("ok", "under", "over", "bad_image")


def pick(rows, n, seed, per_lake=2):
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for row in rows:
        pct = float(row["water_pct_of_outline"] or 0)
        for name, lo, hi in STRATA:
            if lo <= pct < hi:
                by_stratum[name].append(row)
                break
    chosen, used = [], Counter()
    share = max(1, n // len(STRATA))
    for name, _, _ in STRATA:
        pool = by_stratum[name][:]
        rng.shuffle(pool)
        took = 0
        for row in pool:
            if took >= share:
                break
            if used[row["lake_id"]] >= per_lake:
                continue
            row = dict(row, stratum=name)
            chosen.append(row)
            used[row["lake_id"]] += 1
            took += 1
    rng.shuffle(chosen)
    return chosen


def make(n, seed):
    import lakewater as lw
    if not SERIES.exists():
        raise SystemExit(f"{SERIES} not found.")
    with open(SERIES, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["source"] == "archive" and r["status"] == "ok"
                and float(r["outline_ha"] or 0) >= 5.0
                and r["water_pct_of_outline"] != ""]
    sample = pick(rows, n, seed)
    print(f"Sampled {len(sample)} lake-dates from {len(rows)} usable rows.")
    lakes = {l["id"]: l for l in lw.load_lakes(min_ha=5.0)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_item = defaultdict(list)
    for row in sample:
        if row["lake_id"] in lakes:
            by_item[(row["date"][:4], row["item_id"])].append(row)

    sheet, number = [], 0
    for year in sorted({y for y, _ in by_item}):
        wanted = [(i, rs) for (y, i), rs in by_item.items() if y == year]
        group = [lakes[r["lake_id"]] for _, rs in wanted for r in rs]
        boxes = [g["geom_ll"].bounds for g in group]
        bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))
        try:
            items = {i["id"]: i for i in lw.search_archive(bbox, int(year), int(year))}
        except Exception as error:
            print(f"{year}: archive search failed ({error})")
            continue
        for item_id, rs in wanted:
            item = items.get(item_id)
            if item is None:
                print(f"{item_id}: not found in the archive search, skipped")
                continue
            scene = None
            for attempt in range(3):
                try:
                    scene = lw.load_archive_scene(item, [lakes[r["lake_id"]] for r in rs])
                    break
                except Exception as error:
                    print(f"  {item_id}: read failed ({error}); retry {attempt + 1}/3")
                    time.sleep(4)
            if scene is None:
                continue
            for r in rs:
                new_row, debug = lw.measure_lake(scene, lakes[r["lake_id"]],
                                                 keep_debug=True)
                if new_row is None or debug is None:
                    continue
                number += 1
                name = f"{number:02d}_{r['lake'].replace(' ', '_')[:24]}_{r['date']}.png"
                lw.save_chip(OUT_DIR / name, new_row, debug)
                sheet.append({"file": name, "lake": r["lake"], "date": r["date"],
                              "stratum": r["stratum"],
                              "water_ha_stored": r["water_ha"],
                              "water_ha_now": new_row["water_ha"],
                              "rating": "", "notes": ""})
                print(f"  {name}")
    sheet.sort(key=lambda s: s["file"])
    with open(SHEET, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    print(f"\n{len(sheet)} pictures in {OUT_DIR}\nFill the `rating` column in {SHEET}")


def score():
    if not SHEET.exists():
        raise SystemExit(f"{SHEET} not found.")
    with open(SHEET, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rated = [r for r in rows if r["rating"].strip().lower() in RATINGS]
    print(f"Rated {len(rated)} of {len(rows)} pictures.")
    count = Counter(r["rating"].strip().lower() for r in rated)
    judged = [r for r in rated if r["rating"].strip().lower() != "bad_image"]
    good = sum(1 for r in judged if r["rating"].strip().lower() == "ok")
    print(f"  ok {count['ok']}, under {count['under']}, over {count['over']}, "
          f"bad_image {count['bad_image']}")
    if judged:
        print(f"\nCorrect: {good} of {len(judged)} judged pictures "
              f"({100 * good / len(judged):.0f}%)")
    print("\nBy amount of open water:")
    for name, _, _ in STRATA:
        part = [r for r in judged if r["stratum"] == name]
        if part:
            ok = sum(1 for r in part if r["rating"].strip().lower() == "ok")
            print(f"  {name:32s} {ok} of {len(part)} correct")
    wrong = [r for r in judged if r["rating"].strip().lower() != "ok"]
    if wrong:
        print("\nPictures to look at first:")
        for r in wrong:
            print(f"  {r['file']}  {r['rating']}  {r['notes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=28)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    score() if args.score else make(args.n, args.seed)
