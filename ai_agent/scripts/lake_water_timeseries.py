"""Open-water area of every Bengaluru lake, date by date.

For each lake outline (from fetch_lake_outlines.py) and each satellite image
it measures the open-water area inside the outline and writes one CSV row.

Two image sources:
  archive  free public Sentinel-2 archive (AWS Earth Search), 10 m pixels,
           with an official cloud mask. Covers Bengaluru from 2017.
  local    the scenes already in ai_agent/data/extracted (2015-2016 and some
           later dates, about 17-20 m pixels, no cloud mask).
With --source both (default) you get the local scenes AND the archive. Where
dates overlap you can compare the two.

Outputs in ai_agent/reports/lakes/:
    lake_water_timeseries.csv   one row per lake per date (appended; safe to
                                stop and run again - finished images are
                                skipped)
    lake_series.png             water area over time for the main lakes
    qa/*.png                    pictures to check the measurement by eye

Run from the project root. FIRST try a small test (a few minutes):
    python ai_agent/scripts/lake_water_timeseries.py --test
then everything (the archive download is several GB; it can take an hour or
more, and you can stop it with Ctrl+C and continue later):
    python ai_agent/scripts/lake_water_timeseries.py
Useful options: --only Bellandur,Varthur   --per-month 2   --qa 10
                --source local|archive|both   --fresh (ignore old results)
                --out-dir DIR (separate check run, main results untouched)
                --reprocess-from DATE (measure archive images again)
"""

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from lakewater import (
    LAKES_PATH, ROOT, ROW_FIELDS, choose_items, load_archive_scene,
    load_lakes, load_local_scene, measure_lake, save_chip, search_archive,
    lake_geom, union_bounds,
)
from landcover6 import DATA_DIR, find_band, scene_date_key

OUT_DIR = ROOT / "reports" / "lakes"
QA_DIR = OUT_DIR / "qa"
CSV_PATH = OUT_DIR / "lake_water_timeseries.csv"
KEY_NAMES = ["Bellandur", "Varthur", "Hebbal", "Ulsoor"]
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e"

GDAL_OPTIONS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="TRUE", GDAL_CACHEMAX=256,
)


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------

def finished_keys():
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done.add((row["source"], row["item_id"]))
    return done


def append_rows(rows):
    new = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS,
                                extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Measuring a scene
# --------------------------------------------------------------------------

def measure_scene(scene, lakes, qa):
    rows, saved = [], 0
    for rank, lake in enumerate(lakes):
        want = qa["on"] and rank < 3 and saved < 2
        row, debug = measure_lake(scene, lake, keep_debug=want)
        if row is None:
            continue
        rows.append(row)
        if want and debug is not None and qa["left"] > 0:
            QA_DIR.mkdir(parents=True, exist_ok=True)
            name = f"{row['lake_id']}_{row['date']}_{row['source']}.png"
            save_chip(QA_DIR / name, row, debug)
            saved += 1
            qa["left"] -= 1
    return rows


def summarise_scene(scene, rows):
    ok = [r for r in rows if r["status"] in ("ok", "small")]
    cloudy = sum(1 for r in rows if r["status"] == "cloudy")
    total = sum(float(r["water_ha"]) for r in ok)
    note = scene.extra.get("offset_note", "baseline")
    note = "" if note == "baseline" else f" | {note}"
    print(f"  {scene.date} [{scene.source}] {scene.item_id}: "
          f"{len(rows)} lakes in view, {len(ok)} measured, {cloudy} cloudy, "
          f"water {total:,.0f} ha{note}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def read_results():
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report(lakes):
    results = read_results()
    by_lake = {}
    for row in results:
        by_lake.setdefault(row["lake_id"], []).append(row)

    def usable(rows):
        return [r for r in rows if r["status"] in ("ok", "small")
                and r["water_ha"] != ""]

    key_ids = [l["id"] for l in lakes
               if any(k.lower() in l["name"].lower() for k in KEY_NAMES)]
    biggest = [l["id"] for l in sorted(lakes, key=lambda l: -l["area_ha"])[:10]]
    shown = list(dict.fromkeys(key_ids + biggest))[:14]
    names = {l["id"]: l for l in lakes}

    print(f"\n{'=' * 96}\nPER-LAKE SUMMARY (open water inside the outline)\n{'=' * 96}")
    print(f"{'lake':<26}{'outline ha':>11}{'dates':>7}{'cloudy':>8}"
          f"{'median ha':>11}{'min':>8}{'max':>8}  note")
    for lake_id in shown:
        rows = by_lake.get(lake_id, [])
        good = usable(rows)
        if not good:
            print(f"{names[lake_id]['name'][:25]:<26}"
                  f"{names[lake_id]['area_ha']:>11.0f}{'0':>7}  no usable dates")
            continue
        water = np.array([float(r["water_ha"]) for r in good])
        pct = np.median([float(r["water_pct_of_outline"]) for r in good])
        ring = np.median([float(r["ring_water_ha"]) for r in good])
        cloudy = sum(1 for r in rows if r["status"] == "cloudy")
        notes = []
        if pct < 20:
            notes.append("mostly not open water: dry, weed-covered, or the "
                         "outline is off")
        if np.median(water) > 0 and ring > 0.25 * np.median(water):
            notes.append("water outside the outline: check the outline")
        print(f"{names[lake_id]['name'][:25]:<26}"
              f"{names[lake_id]['area_ha']:>11.0f}{len(good):>7}{cloudy:>8}"
              f"{np.median(water):>11.1f}{water.min():>8.1f}{water.max():>8.1f}"
              f"  {'; '.join(notes)}")

    plot_ids = (key_ids or biggest)[:6]
    if not plot_ids:
        return
    cols = 2 if len(plot_ids) > 1 else 1
    rows_n = int(np.ceil(len(plot_ids) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3.4 * rows_n),
                             squeeze=False, sharex=True)
    for ax, lake_id in zip(axes.ravel(), plot_ids):
        good = usable(by_lake.get(lake_id, []))
        for source, color, marker, label in (
                ("archive", BLUE, "o", "archive, 10 m"),
                ("local", ORANGE, "s", "your scenes, 17-20 m")):
            pts = sorted((r["date"], float(r["water_ha"])) for r in good
                         if r["source"] == source)
            if pts:
                ax.plot([dt.date.fromisoformat(d) for d, _ in pts],
                        [v for _, v in pts], color=color, marker=marker,
                        ms=4, lw=1.4, label=label)
        scl = sorted((r["date"], float(r["scl_water_ha"])) for r in good
                     if r["scl_water_ha"] != "")
        if scl:
            ax.plot([dt.date.fromisoformat(d) for d, _ in scl],
                    [v for _, v in scl], color=GREY, lw=1, ls="--",
                    label="provider's own water class")
        ax.axhline(names[lake_id]["area_ha"], color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{names[lake_id]['name']} (outline "
                     f"{names[lake_id]['area_ha']:.0f} ha, dotted)",
                     loc="left", fontsize=10, color=INK)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e3e2de", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("open water (ha)", color=INK2, fontsize=9)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[len(plot_ids):]:
        ax.axis("off")
    fig.autofmt_xdate()
    fig.tight_layout()
    chart = OUT_DIR / "lake_series.png"
    fig.savefig(chart, dpi=120)
    plt.close(fig)
    print(f"\nChart: {chart}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_local(lakes, done, qa, limit):
    scene_dirs = sorted((p for p in DATA_DIR.iterdir() if p.is_dir()),
                        key=lambda p: scene_date_key(p.name))
    print(f"\nLocal scenes: {len(scene_dirs)}")
    for scene_dir in scene_dirs:
        if ("local", scene_dir.name) in done:
            print(f"  {scene_dir.name}: already done, skipping")
            continue
        y, m, d = scene_date_key(scene_dir.name)
        scene = load_local_scene(scene_dir, find_band,
                                 f"{y:04d}-{m:02d}-{d:02d}")
        if scene is None:
            print(f"  {scene_dir.name}: missing green/NIR/SWIR band, skipped")
            continue
        rows = measure_scene(scene, lakes, qa)
        if rows:
            append_rows(rows)
        summarise_scene(scene, rows)


def run_archive(lakes, done, qa, args):
    ll_bounds = [l["geom_ll"].bounds for l in lakes]
    bbox = (min(b[0] for b in ll_bounds), min(b[1] for b in ll_bounds),
            max(b[2] for b in ll_bounds), max(b[3] for b in ll_bounds))
    last_year = args.end_year or dt.date.today().year
    print(f"\nSearching the archive {args.start_year}-{last_year} "
          f"for lakes in {tuple(round(v, 3) for v in bbox)} ...")
    try:
        items = search_archive(bbox, args.start_year, last_year)
    except Exception as error:                     # network problems
        print(f"Archive search failed: {error}\nCheck your internet/VPN. "
              "Local scenes were still processed.")
        return
    chosen, tile = choose_items(items, bbox, args.per_month, args.max_cloud)
    print(f"Found {len(items)} images; tile {tile}; using {len(chosen)} "
          f"(best {args.per_month} per month, cloud under {args.max_cloud:.0f}%)")
    if args.max_items and len(chosen) > args.max_items:
        step = len(chosen) / args.max_items
        chosen = [chosen[int(i * step)] for i in range(args.max_items)]
        print(f"Test mode: sampling {len(chosen)} images across the years")

    for item in chosen:
        if ("archive", item["id"]) in done:
            continue
        scene = None
        for attempt in range(3):
            try:
                scene = load_archive_scene(item, lakes)
                break
            except Exception as error:
                print(f"  {item['id']}: read failed ({error}); "
                      f"retry {attempt + 1}/3")
                time.sleep(4)
        if scene is None:
            continue
        rows = measure_scene(scene, lakes, qa)
        if rows:
            append_rows(rows)
        summarise_scene(scene, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "archive", "both"],
                        default="both")
    parser.add_argument("--lakes", default=str(LAKES_PATH))
    parser.add_argument("--min-ha", type=float, default=1.0)
    parser.add_argument("--only", default="", help="comma-separated name parts")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=0)
    parser.add_argument("--per-month", type=int, default=1)
    parser.add_argument("--max-cloud", type=float, default=70.0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--qa", type=int, default=0,
                        help="save this many quality-control pictures")
    parser.add_argument("--test", action="store_true",
                        help="key lakes only, few images, with pictures")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--reprocess-from", default="",
                        help="YYYY-MM-DD: forget archive results from this "
                             "date on (a backup copy is kept) and measure "
                             "them again")
    parser.add_argument("--out-dir", default="",
                        help="write results/pictures here instead (keeps the "
                             "main results untouched)")
    args = parser.parse_args()
    global OUT_DIR, QA_DIR, CSV_PATH
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        QA_DIR = OUT_DIR / "qa"
        CSV_PATH = OUT_DIR / "lake_water_timeseries.csv"

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.test:
        only = only or KEY_NAMES
        args.max_items = args.max_items or 6
        args.qa = args.qa or 8
    lakes = load_lakes(args.lakes, args.min_ha, only or None)
    if not lakes:
        raise SystemExit("No lakes matched. Check --only / --min-ha, and that "
                         "fetch_lake_outlines.py found them.")
    lakes.sort(key=lambda l: -l["area_ha"])
    print(f"Lakes to measure: {len(lakes)} "
          f"(total outline {sum(l['area_ha'] for l in lakes):,.0f} ha)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and CSV_PATH.exists():
        CSV_PATH.unlink()
    if args.reprocess_from and CSV_PATH.exists():
        import shutil
        backup = CSV_PATH.with_name("lake_water_timeseries.backup_before_redo.csv")
        if not backup.exists():
            shutil.copy(CSV_PATH, backup)
            print(f"Backup of the old results: {backup}")
        kept, dropped = [], 0
        with open(CSV_PATH, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["source"] == "archive" and row["date"] >= args.reprocess_from:
                    dropped += 1
                else:
                    kept.append(row)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
        print(f"Forgot {dropped:,} archive rows from {args.reprocess_from} on.")
    done = finished_keys()
    qa = {"on": args.qa > 0, "left": args.qa}

    with rasterio.Env(**GDAL_OPTIONS):
        if args.source in ("local", "both"):
            run_local(lakes, done, qa, args)
        if args.source in ("archive", "both"):
            run_archive(lakes, done, qa, args)

    if not CSV_PATH.exists():
        raise SystemExit("Nothing was measured.")
    report(lakes)
    print(f"\nResults: {CSV_PATH}")
    if args.qa:
        print(f"Check the pictures in {QA_DIR} - do the yellow outline and "
              "the blue water match the lake you can see?")


if __name__ == "__main__":
    main()
