"""Lake and land-cover timeline on ONE COMMON AREA.

Why this exists: the scenes cover different ground (different image sizes and
pixel sizes), so raw hectare totals are not comparable between dates. This
script
  1. finds the area that EVERY usable scene covers and measures that area only,
  2. cuts false water: it tests several ways of combining the model's water
     class with the NDWI water index against ESA WorldCover (on the held-out
     strips) and uses whichever scores best,
  3. flags odd scenes (clouds, jumps) instead of silently plotting them,
  4. writes a per-date table (CSV) and a chart (PNG).

Needs: predict_6band.py already run (it reuses the saved class maps; if one is
missing it predicts that scene itself), and evaluate_vs_worldcover.py run once
(for the WorldCover reference used to pick the water rule).

Run from the project root:
    python ai_agent/scripts/lake_timeline.py
Optional:  --drop 3mar26,14oct26   (leave scenes out of the common area)
"""

import argparse
import csv
import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from landcover6 import (
    BANDS, DATA_DIR, ROOT, WC_DIR, find_band, load_model, load_scene,
    predict_scene, scene_date_key,
)

PRED_DIR = ROOT / "reports" / "predictions_6band"
OUT_DIR = ROOT / "reports" / "timeline"
PIXEL_M = 20.0                 # common grid pixel size (metres)
PIXEL_HA = PIXEL_M * PIXEL_M / 10_000.0
HELDOUT_FRACTION = 0.30        # same strip train_worldcover_6band.py held out
BRIGHT_REFLECTANCE = 0.25      # blue-band reflectance above this = cloud-like
Z_LIMIT = 3.5                  # robust z-score beyond which a scene is "odd"

# (use model's water class?, NDWI threshold or None)
RULES = [
    (True, None), (False, 0.0), (False, 0.1), (False, 0.2),
    (True, -0.1), (True, 0.0), (True, 0.1), (True, 0.2),
]
DEFAULT_RULE = (True, 0.0)

# Colours: validated reference palette (blue, orange, aqua) + neutral greys.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
YELLOW = "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"


def rule_name(rule):
    use_model, thr = rule
    parts = []
    if use_model:
        parts.append("model water")
    if thr is not None:
        parts.append(f"NDWI > {thr:g}")
    return " AND ".join(parts)


def water_mask(labels, ndwi, rule):
    use_model, thr = rule
    mask = np.ones(labels.shape, dtype=bool)
    if use_model:
        mask &= labels == 0
    if thr is not None:
        mask &= ndwi > thr
    return mask


# --------------------------------------------------------------------------
# Loading one scene
# --------------------------------------------------------------------------

def analyse_scene(name, holder):
    """Load one scene; return a dict of arrays, or None if unusable."""
    scene_dir = DATA_DIR / name
    features, valid, profile, reason = load_scene(scene_dir)
    if features is None:
        print(f"SKIP {name}: {reason}")
        return None

    labels = None
    saved = PRED_DIR / f"{name}_landcover.tif"
    if saved.exists():
        with rasterio.open(saved) as src:
            candidate = src.read(1)
        if candidate.shape == valid.shape:
            labels = candidate
    if labels is None:
        if holder.get("model") is None:
            holder["model"] = load_model()
        print(f"  {name}: no saved class map, predicting now...")
        labels = predict_scene(holder["model"], features, valid)

    with rasterio.open(find_band(scene_dir, "B02")) as src:
        blue = src.read(1, masked=True).astype(np.float32).filled(0.0)
    values = blue[valid]
    scale = 10000.0 if values.size and np.percentile(values, 99) > 5 else 1.0

    return {
        "labels": labels,
        "valid": valid,
        "ndvi": features[6] * 2 - 1,
        "ndwi": features[7] * 2 - 1,
        "blue": blue / scale,
        "transform": profile["transform"],
        "crs": profile["crs"],
    }


def footprint(name):
    with rasterio.open(find_band(DATA_DIR / name, "B03")) as src:
        return src.bounds, src.crs, src.transform.a


# --------------------------------------------------------------------------
# Choosing the water rule against WorldCover
# --------------------------------------------------------------------------

def choose_water_rule(cache):
    totals = {rule: np.zeros(3) for rule in RULES}   # tp, fp, fn
    used = []
    for name, scene in cache.items():
        ref_path = WC_DIR / f"reference_{name}_worldcover.tif"
        if not ref_path.exists():
            continue
        with rasterio.open(ref_path) as src:
            ref = src.read(1)
        if ref.shape != scene["valid"].shape:
            continue
        start = int(ref.shape[1] * (1 - HELDOUT_FRACTION))
        cols = np.s_[:, start:]
        ok = (ref[cols] < 4) & scene["valid"][cols]
        truth = ref[cols] == 0
        for rule in RULES:
            pred = water_mask(scene["labels"][cols], scene["ndwi"][cols], rule)
            totals[rule] += [(pred & truth & ok).sum(),
                             (pred & ~truth & ok).sum(),
                             (~pred & truth & ok).sum()]
        used.append(name)

    if not used:
        print("No WorldCover reference found, so the water rule cannot be "
              f"tested. Using the default: {rule_name(DEFAULT_RULE)}.")
        return DEFAULT_RULE, []

    print(f"\nWater rule test against WorldCover (held-out strips of "
          f"{', '.join(used)}):")
    print(f"{'rule':<34}{'precision':>10}{'recall':>9}{'IoU':>8}")
    rows, best, best_iou = [], DEFAULT_RULE, -1.0
    for rule in RULES:
        tp, fp, fn = totals[rule]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        rows.append((rule_name(rule), precision, recall, iou))
        print(f"{rule_name(rule):<34}{precision * 100:>9.1f}%"
              f"{recall * 100:>8.1f}%{iou * 100:>7.1f}%")
        if iou > best_iou:
            best, best_iou = rule, iou
    print(f"-> Using: {rule_name(best)}  (best IoU {best_iou * 100:.1f}%)")
    print("   Chosen on only these scenes, so treat the score as slightly "
          "optimistic.")
    return best, rows


# --------------------------------------------------------------------------
# Common grid
# --------------------------------------------------------------------------

def make_common_grid(names):
    bounds, crs_list = {}, set()
    for name in names:
        box, crs, _ = footprint(name)
        bounds[name] = box
        crs_list.add(str(crs))
    if len(crs_list) != 1:
        raise SystemExit(f"Scenes use different coordinate systems: {crs_list}")

    left = max(b.left for b in bounds.values())
    bottom = max(b.bottom for b in bounds.values())
    right = min(b.right for b in bounds.values())
    top = min(b.top for b in bounds.values())
    if right - left < 5000 or top - bottom < 5000:
        raise SystemExit(
            "The scenes share almost no common area. Try --drop with the "
            "scene(s) that sit furthest away (see the footprints above)."
        )

    width = int((right - left) // PIXEL_M)
    height = int((top - bottom) // PIXEL_M)
    grid = {
        "w": width, "h": height,
        "transform": from_origin(left, top, PIXEL_M, PIXEL_M),
        "crs": rasterio.open(find_band(DATA_DIR / names[0], "B03")).crs,
    }

    print(f"\nCommon area: {(right - left) / 1000:.1f} km x "
          f"{(top - bottom) / 1000:.1f} km = "
          f"{width * height * PIXEL_HA:,.0f} ha")
    print(f"{'scene':<12}{'footprint (km)':>16}{'common area is':>16}")
    for name in names:
        box = bounds[name]
        w, h = box.right - box.left, box.top - box.bottom
        share = (right - left) * (top - bottom) / (w * h)
        print(f"{name:<12}{w / 1000:>8.1f} x {h / 1000:<5.1f}{share * 100:>14.0f}%")
    return grid


def on_grid(array, scene, grid):
    """Area-average an array onto the common grid (so sums keep their area)."""
    out = np.zeros((grid["h"], grid["w"]), dtype=np.float32)
    reproject(
        array.astype(np.float32), out,
        src_transform=scene["transform"], src_crs=scene["crs"],
        dst_transform=grid["transform"], dst_crs=grid["crs"],
        resampling=Resampling.average,
    )
    return out


def measure(scene, grid, rule):
    valid = scene["valid"]
    labels = scene["labels"]
    water = water_mask(labels, scene["ndwi"], rule) & valid
    layers = {
        "valid": valid,
        "water": water,
        "water_model": (labels == 0) & valid,
        "vegetation": labels == 1,
        "bare_land": labels == 2,
        "built_up": labels == 3,
        "bright": (scene["blue"] > BRIGHT_REFLECTANCE) & valid,
    }
    grids = {key: on_grid(mask, scene, grid) for key, mask in layers.items()}
    ndvi_sum = on_grid(np.where(valid, scene["ndvi"], 0.0), scene, grid).sum()
    blue_sum = on_grid(np.where(valid, scene["blue"], 0.0), scene, grid).sum()

    valid_px = max(grids["valid"].sum(), 1e-9)
    row = {
        "valid_ha": grids["valid"].sum() * PIXEL_HA,
        "coverage_pct": 100 * grids["valid"].sum() / (grid["w"] * grid["h"]),
        "mean_ndvi": ndvi_sum / valid_px,
        "mean_blue": blue_sum / valid_px,
        "bright_pct": 100 * grids["bright"].sum() / valid_px,
    }
    for key in ("water", "water_model", "vegetation", "bare_land", "built_up"):
        row[f"{key}_ha"] = grids[key].sum() * PIXEL_HA
        row[f"{key}_pct"] = 100 * grids[key].sum() / valid_px
    return row


# --------------------------------------------------------------------------
# Flags, outputs
# --------------------------------------------------------------------------

def robust_z(values):
    v = np.asarray(values, dtype=float)
    median = np.median(v)
    mad = max(np.median(np.abs(v - median)), 0.01 * abs(median), 1e-6)
    return 0.6745 * (v - median) / mad


def add_flags(rows):
    checks = {"vegetation_pct": "vegetation share", "built_up_pct":
              "built-up share", "mean_ndvi": "average greenness (NDVI)",
              "mean_blue": "average blue brightness"}
    zs = {key: robust_z([r[key] for r in rows]) for key in checks}
    for i, row in enumerate(rows):
        reasons = []
        if row["bright_pct"] > 5:
            reasons.append(f"{row['bright_pct']:.0f}% of pixels are very "
                           "bright (possible cloud)")
        if row["coverage_pct"] < 90:
            reasons.append(f"only {row['coverage_pct']:.0f}% of the common "
                           "area has valid data")
        for key, label in checks.items():
            if abs(zs[key][i]) > Z_LIMIT:
                reasons.append(f"{label} unusual (z={zs[key][i]:+.1f})")
        row["flag"] = "; ".join(reasons)


def save_chart(rows, rule):
    dates = [dt.date.fromisoformat(r["date"]) for r in rows]
    flagged = [bool(r["flag"]) for r in rows]
    april = [r["month"] == 4 for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    water = [r["water_ha"] for r in rows]
    model_only = [r["water_model_ha"] for r in rows]
    ax1.plot(dates, model_only, color=GREY, lw=1, marker="o", ms=4,
             label="model water, before cleaning")
    ax1.plot(dates, water, color=BLUE, lw=2, marker="o", ms=7,
             label=f"water area, cleaned ({rule_name(rule)})")
    a_dates = [d for d, a in zip(dates, april) if a]
    a_vals = [w for w, a in zip(water, april) if a]
    if len(a_dates) > 1:
        ax1.plot(a_dates, a_vals, color=ORANGE, lw=2.5, ls="--",
                 label="April scenes only (same season)")
    for d, w, f in zip(dates, water, flagged):
        if f:
            ax1.plot([d], [w], marker="o", ms=11, mfc="none", mec=INK, mew=1.5)
    ax1.plot([], [], marker="o", ms=11, mfc="none", mec=INK, mew=1.5, ls="",
             label="odd scene, check before trusting")
    ax1.set_ylabel("Water area in common area (ha)", color=INK2)
    ax1.set_title("Water area per date (same ground in every scene)",
                  loc="left", color=INK)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=9, loc="best")

    for key, color, label in (("vegetation_pct", AQUA, "vegetation"),
                              ("built_up_pct", ORANGE, "built-up"),
                              ("bare_land_pct", YELLOW, "bare land")):
        values = [r[key] for r in rows]
        ax2.plot(dates, values, color=color, lw=2, marker="o", ms=5)
        ax2.annotate(label, (dates[-1], values[-1]), xytext=(8, 0),
                     textcoords="offset points", va="center", color=INK2,
                     fontsize=9)
    ax2.set_ylabel("Share of valid area (%)", color=INK2)
    ax2.set_title("Land cover shares (model, not cleaned)", loc="left",
                  color=INK)
    ax2.set_ylim(bottom=0)
    ax2.margins(x=0.08)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e3e2de", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = OUT_DIR / "lake_timeline.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", default="",
                        help="comma-separated scene names to leave out")
    args = parser.parse_args()
    dropped = {s.strip() for s in args.drop.split(",") if s.strip()}

    names = []
    for scene_dir in sorted((p for p in DATA_DIR.iterdir() if p.is_dir()),
                            key=lambda p: scene_date_key(p.name)):
        if scene_dir.name in dropped:
            print(f"DROP {scene_dir.name}: left out by --drop")
        elif any(find_band(scene_dir, b) is None for b in BANDS):
            print(f"SKIP {scene_dir.name}: missing a band")
        else:
            names.append(scene_dir.name)
    if len(names) < 2:
        raise SystemExit("Need at least two usable scenes.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = make_common_grid(names)

    holder, cache = {}, {}
    for ref_path in sorted(WC_DIR.glob("reference_*_worldcover.tif")):
        name = ref_path.name[len("reference_"):-len("_worldcover.tif")]
        if name in names:
            scene = analyse_scene(name, holder)
            if scene is not None:
                cache[name] = scene
    rule, rule_rows = choose_water_rule(cache)

    rows = []
    print("\nMeasuring each date on the common area...")
    for name in names:
        scene = cache.pop(name, None) or analyse_scene(name, holder)
        if scene is None:
            continue
        year, month, day = scene_date_key(name)
        row = {"scene": name, "date": f"{year:04d}-{month:02d}-{day:02d}",
               "month": month}
        row.update(measure(scene, grid, rule))
        rows.append(row)
        del scene
        print(f"  {name}: water {row['water_ha']:.0f} ha "
              f"(model alone {row['water_model_ha']:.0f} ha)")

    add_flags(rows)

    columns = ["scene", "date", "valid_ha", "coverage_pct", "water_ha",
               "water_pct", "water_model_ha", "vegetation_ha", "vegetation_pct",
               "bare_land_ha", "bare_land_pct", "built_up_ha", "built_up_pct",
               "mean_ndvi", "mean_blue", "bright_pct", "flag"]
    out_csv = OUT_DIR / "lake_timeline.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in row.items()})

    if rule_rows:
        with open(OUT_DIR / "water_rule_test.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rule", "precision", "recall", "iou"])
            for line in rule_rows:
                writer.writerow([line[0]] + [round(x, 4) for x in line[1:]])

    chart = save_chart(rows, rule)

    print(f"\n{'=' * 78}\nTIMELINE (common area, water rule: "
          f"{rule_name(rule)})\n{'=' * 78}")
    print(f"{'scene':<12}{'date':<12}{'water ha':>9}{'veg %':>7}"
          f"{'built %':>9}{'NDVI':>7}{'bright %':>9}  note")
    for row in rows:
        mark = "CHECK" if row["flag"] else ""
        print(f"{row['scene']:<12}{row['date']:<12}{row['water_ha']:>9.0f}"
              f"{row['vegetation_pct']:>7.1f}{row['built_up_pct']:>9.1f}"
              f"{row['mean_ndvi']:>7.2f}{row['bright_pct']:>9.1f}  {mark}")

    flagged = [r for r in rows if r["flag"]]
    if flagged:
        print("\nScenes to check before trusting (odd is not the same as wrong):")
        for row in flagged:
            print(f"  {row['scene']}: {row['flag']}")

    aprils = [r for r in rows if r["month"] == 4]
    if len(aprils) > 1:
        print("\nApril scenes only (same season, the fairest comparison):")
        for row in aprils:
            print(f"  {row['date']}: {row['water_ha']:.0f} ha of water")

    print(f"\nSaved: {out_csv}\nSaved: {chart}")
    print("Reminder: clouds are only flagged roughly, and water numbers carry "
          "the model's error.")


if __name__ == "__main__":
    main()
