"""Is the model's low water "precision" real, or is the reference missing water?

The land-cover model scores about 37% water precision against ESA WorldCover.
That counts every place the model says "water" and WorldCover says "not water"
as a mistake. But WorldCover only maps permanent water and misses small or
seasonal ponds, so some of those "mistakes" may be real water.

This script finds those disagreement spots on the held-out strip (never trained
on), saves a small picture of each one, and a rating sheet. You look at each
picture and say whether there is water. Then it estimates the TRUE precision.

  1) python ai_agent/scripts/audit_water_false_positives.py --n 30
  2) open ai_agent/reports/audit/ and fill `rating` in audit_ratings.csv:
        water      there is water here (WorldCover missed it)
        not_water  no water (the model was wrong)
        unsure     cannot tell
  3) python ai_agent/scripts/audit_water_false_positives.py --score
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "audit"
SHEET = OUT_DIR / "audit_ratings.csv"
HALF = 24                      # picture is 48 x 48 pixels (about 480 m)
MIN_PIXELS = 4                 # smaller disagreement spots are ignored
WC_NAMES = {0: "water", 1: "vegetation", 2: "bare land", 3: "built-up"}


def find_spots(pred, labels, valid, start, rng, per_scene):
    """Connected groups of 'model says water, WorldCover says not water'."""
    from scipy import ndimage
    bad = np.zeros(pred.shape, bool)
    bad[:, start:] = ((pred[:, start:] == 0) & (labels[:, start:] != 0)
                      & (labels[:, start:] < 4) & valid[:, start:])
    groups, count = ndimage.label(bad)
    true_pos = int(((pred[:, start:] == 0) & (labels[:, start:] == 0)
                    & valid[:, start:]).sum())
    if count == 0:
        return [], 0, true_pos
    sizes = np.bincount(groups.ravel())[1:]
    keep = [i + 1 for i, size in enumerate(sizes) if size >= MIN_PIXELS]
    chosen = rng.sample(keep, min(per_scene, len(keep)))
    spots = []
    for gid in chosen:
        ys, xs = np.where(groups == gid)
        wc = np.bincount(labels[ys, xs][labels[ys, xs] < 4], minlength=4)
        spots.append({"y": int(ys.mean()), "x": int(xs.mean()),
                      "pixels": int(len(ys)),
                      "wc_class": WC_NAMES[int(wc.argmax())]})
    return spots, int(bad.sum()), true_pos


def picture(features, spot, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, height, width = features.shape
    y0 = int(np.clip(spot["y"] - HALF, 0, height - 2 * HALF))
    x0 = int(np.clip(spot["x"] - HALF, 0, width - 2 * HALF))
    chip = features[:, y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    false_colour = np.clip(np.dstack([chip[3], chip[2], chip[1]]), 0, 1)
    true_colour = np.clip(np.dstack([chip[2], chip[1], chip[0]]) * 1.5, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    for ax, img, name in zip(axes, (false_colour, true_colour),
                             ("false colour (water dark, plants red)",
                              "natural colour")):
        ax.imshow(img)
        ax.plot(spot["x"] - x0, spot["y"] - y0, "+", color="yellow", ms=14)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def make(n, seed):
    import landcover6 as lc
    import rasterio
    rng = random.Random(seed)
    model = lc.load_model()
    names = sorted(p.name[len("reference_"):-len("_worldcover.tif")]
                   for p in lc.WC_DIR.glob("reference_*_worldcover.tif"))
    if not names:
        raise SystemExit("No WorldCover reference files found.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet, bad_total, tp_total = [], 0, 0
    per_scene = max(1, n // len(names))
    for name in names:
        features, valid, _, reason = lc.load_scene(lc.DATA_DIR / name)
        if features is None:
            print(f"SKIP {name}: {reason}")
            continue
        with rasterio.open(lc.WC_DIR / f"reference_{name}_worldcover.tif") as src:
            labels = src.read(1).copy()
        labels[~valid] = lc.IGNORE
        start = int(labels.shape[1] * 0.70)
        pred = lc.predict_scene(model, features, valid)
        spots, bad, tp = find_spots(pred, labels, valid, start, rng, per_scene)
        bad_total += bad
        tp_total += tp
        print(f"{name}: {bad} pixels where the model says water and "
              f"WorldCover does not; {tp} where both say water; "
              f"{len(spots)} spots sampled")
        for spot in spots:
            number = len(sheet) + 1
            file = f"{number:02d}_{name}.png"
            picture(features, spot, OUT_DIR / file,
                    f"{name}: model says water, WorldCover says "
                    f"{spot['wc_class']} ({spot['pixels']} px)")
            sheet.append({"file": file, "scene": name,
                          "pixels": spot["pixels"],
                          "worldcover_says": spot["wc_class"],
                          "rating": "", "notes": ""})
    if not sheet:
        raise SystemExit("No disagreement spots found.")
    with open(SHEET, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    with open(OUT_DIR / "audit_totals.txt", "w") as handle:
        handle.write(f"{tp_total} {bad_total}\n")
    print(f"\n{len(sheet)} pictures in {OUT_DIR}\nFill `rating` in {SHEET}")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def score():
    with open(SHEET, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rated = [r for r in rows if r["rating"].strip().lower() in
             ("water", "not_water")]
    unsure = sum(1 for r in rows if r["rating"].strip().lower() == "unsure")
    print(f"Rated {len(rated)} of {len(rows)} spots ({unsure} unsure, left out).")
    if not rated:
        return
    water = sum(1 for r in rated if r["rating"].strip().lower() == "water")
    lo, hi = wilson(water, len(rated))
    share = water / len(rated)
    print(f"\nOf the spots WorldCover calls 'not water', {water} of "
          f"{len(rated)} ({share * 100:.0f}%) really show water "
          f"(95% interval {lo * 100:.0f}-{hi * 100:.0f}%).")
    totals = (OUT_DIR / "audit_totals.txt").read_text().split()
    tp, bad = int(totals[0]), int(totals[1])
    print(f"\nHeld-out strip: {tp} pixels where model and WorldCover agree "
          f"on water; {bad} pixels where only the model says water.")
    for label, s in (("low", lo), ("best estimate", share), ("high", hi)):
        precision = (tp + bad * s) / (tp + bad) if tp + bad else float("nan")
        print(f"  water precision, {label:13s}: {precision * 100:.0f}%")
    print("\nThis is an estimate from a small sample: report it with the "
          "interval, and say it came from looking at the pictures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    score() if args.score else make(args.n, args.seed)
