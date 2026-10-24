"""Score the land-cover model against ESA WorldCover (an independent map).

ESA WorldCover is a free 10 m land-cover map made from Sentinel satellites
(CC-BY 4.0). It exists for 2020 (v100) and 2021 (v200), so we can only use it
for scenes from those years:  8april20 -> 2020,  3april21 -> 2021.

What this script does, per scene:
  1. Finds which WorldCover 3x3-degree tile(s) cover the scene and downloads
     them (once; they are cached in ai_agent/data/worldcover/).
  2. Re-projects WorldCover onto the scene's own pixel grid and converts its
     11 classes into our 4 (water / vegetation / bare land / built-up).
  3. Scores BOTH the trained model AND the original rule-based masks against
     that independent reference, and saves a side-by-side picture.

Notes you should keep in mind:
  * WorldCover is itself a computer-made map. It is a strong independent
    reference, not perfect ground truth.
  * It is a yearly product; the scene is one day in April. Cropland that is
    bare in April is called "cropland" (-> vegetation) by WorldCover, so some
    disagreement on vegetation vs bare land is expected. Edit WC_TO_CLASS
    below if you want to treat cropland differently.
  * These two scenes were part of the model's training set (trained on the
    rule-based labels, never on WorldCover), so the comparison is fair for
    "does it match an independent map", but not a pure unseen-image test.

Run from the project root:
    python ai_agent/scripts/evaluate_vs_worldcover.py
"""

import csv
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds

from evaluate_model import CHECKPOINT, CLASS_NAMES, fmt, metrics_from
from train_model import IMAGE_DIR, MASK_DIR, NUM_CLASSES, SmallUNet

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"
WC_DIR = ROOT / "data" / "worldcover"
REPORT_DIR = ROOT / "reports" / "evaluation"

WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# scene folder -> (WorldCover version, year)
SCENES = {
    "8april20": ("v100", "2020"),
    "3april21": ("v200", "2021"),
}

IGNORE = 255
# WorldCover class code -> our class (0 water, 1 vegetation, 2 bare, 3 built)
WC_TO_CLASS = {
    10: 1,   # tree cover
    20: 1,   # shrubland
    30: 1,   # grassland
    40: 1,   # cropland   (may look bare in April - see notes above)
    50: 3,   # built-up
    60: 2,   # bare / sparse vegetation
    70: IGNORE,  # snow and ice
    80: 0,   # permanent water bodies
    90: 1,   # herbaceous wetland
    95: 1,   # mangroves
    100: IGNORE,  # moss and lichen
}
LUT = np.full(256, IGNORE, dtype=np.uint8)
for _code, _cls in WC_TO_CLASS.items():
    LUT[_code] = _cls

PLOT_COLORS = ["#0064ff", "#14b43c", "#dcb450", "#e64141", "#e8e8e8"]


# --------------------------------------------------------------------------
# Finding and downloading WorldCover tiles
# --------------------------------------------------------------------------

def find_band(scene_dir, band):
    matches = [
        p for p in scene_dir.rglob("*.tiff")
        if band in p.name.upper() and "(RAW)" in p.name.upper()
    ]
    return matches[0] if matches else None


def tile_ids_for_bounds(lon_min, lat_min, lon_max, lat_max):
    """WorldCover tiles are 3x3 degrees, named by their lower-left corner."""
    eps = 1e-9
    ids = []
    for lat0 in range(math.floor(lat_min / 3) * 3,
                      math.floor((lat_max - eps) / 3) * 3 + 1, 3):
        for lon0 in range(math.floor(lon_min / 3) * 3,
                          math.floor((lon_max - eps) / 3) * 3 + 1, 3):
            ns = "N" if lat0 >= 0 else "S"
            ew = "E" if lon0 >= 0 else "W"
            ids.append(f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}")
    return ids


def parse_listing(xml_text):
    """Pull object keys out of an S3 ListObjectsV2 XML response."""
    return re.findall(r"<Key>(.*?)</Key>", xml_text)


def lookup_key(version, year, tile):
    """Ask the public bucket which file exists for this tile (no guessing)."""
    prefix = f"{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}"
    url = f"{WC_BASE}/?list-type=2&prefix={prefix}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            keys = parse_listing(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise SystemExit(
            f"Could not reach the WorldCover bucket ({error}).\n"
            "Check your internet connection / VPN and run again."
        )
    keys = [k for k in keys if k.endswith("_Map.tif")]
    return keys[0] if keys else None


def download(key, destination):
    url = f"{WC_BASE}/{key}"
    temp = destination.with_suffix(".part")
    print(f"  Downloading {destination.name} (large file, please wait)...")
    with urllib.request.urlopen(url, timeout=60) as response, \
            open(temp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 25 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                pct = f" ({done * 100 // total}%)" if total else ""
                print(f"    {done // (1024 * 1024)} MB{pct}")
                next_report += 25 * 1024 * 1024
    temp.rename(destination)


def get_tile_file(version, year, tile):
    """Return a local path for the tile, downloading it only if needed."""
    WC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    local = WC_DIR / name
    if local.exists() and local.stat().st_size > 0:
        return local
    key = lookup_key(version, year, tile)
    if key is None:
        print(f"  No WorldCover file found for tile {tile} ({year}); skipping")
        return None
    download(key, local)
    return local


# --------------------------------------------------------------------------
# Building the reference mask on the scene grid
# --------------------------------------------------------------------------

def build_reference(scene, version, year):
    """Return (reference_in_our_4_classes, scene_profile) on the scene grid."""
    band_path = find_band(DATA_DIR / scene, "B03")
    if band_path is None:
        raise SystemExit(f"No B03 band found for scene {scene}")

    with rasterio.open(band_path) as src:
        shape, transform, crs = src.shape, src.transform, src.crs
        profile = src.profile.copy()
        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            crs, "EPSG:4326", *src.bounds
        )

    tiles = tile_ids_for_bounds(lon_min, lat_min, lon_max, lat_max)
    print(f"  Scene covers lon {lon_min:.3f}..{lon_max:.3f}, "
          f"lat {lat_min:.3f}..{lat_max:.3f} -> WorldCover tile(s): "
          f"{', '.join(tiles)}")

    wc = np.zeros(shape, dtype=np.uint8)
    used = 0
    for tile in tiles:
        local = get_tile_file(version, year, tile)
        if local is None:
            continue
        piece = np.zeros(shape, dtype=np.uint8)
        with rasterio.open(local) as tile_src:
            reproject(
                source=rasterio.band(tile_src, 1),
                destination=piece,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
        wc = np.where(wc == 0, piece, wc)
        used += 1

    if used == 0:
        raise SystemExit(f"No WorldCover data could be loaded for {scene}")

    reference = LUT[wc]
    reference[wc == 0] = IGNORE
    return reference, profile


def save_reference_tif(scene, reference, profile):
    out = WC_DIR / f"reference_{scene}_worldcover.tif"
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=IGNORE,
                   compress="deflate")
    profile.pop("photometric", None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(reference, 1)
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def tile_position(path):
    scene, row, col = path.stem.rsplit("_", 2)
    return scene, int(row), int(col)


def add_to_matrix(matrix, truth, pred):
    keep = truth < NUM_CLASSES
    idx = truth[keep].astype(np.int64) * NUM_CLASSES + pred[keep]
    matrix += np.bincount(
        idx, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def print_block(title, matrix):
    precision, recall, iou, accuracy, miou = metrics_from(matrix)
    share = matrix.sum(axis=1) / max(matrix.sum(), 1)
    print(f"\n{title}")
    print(f"{'class':<12}{'in WorldCover':>14}{'precision':>11}"
          f"{'recall':>9}{'IoU':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{fmt(share[i]):>14}{fmt(precision[i]):>11}"
              f"{fmt(recall[i]):>9}{fmt(iou[i]):>8}")
    print(f"Overall accuracy: {fmt(accuracy).strip()}   "
          f"Mean IoU: {fmt(miou).strip()}")


def save_figure(scene, reference, model_canvas, rules_canvas, year):
    cmap = ListedColormap(PLOT_COLORS)
    step = 4

    def prep(array):
        out = array[::step, ::step].copy()
        out[out > 3] = 4
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (f"WorldCover {year} (independent reference)", prep(reference)),
        ("Trained model", prep(model_canvas)),
        ("Original rule-based labels", prep(rules_canvas)),
    ]
    for ax, (title, data) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"{scene}   blue=water  green=vegetation  tan=bare land  "
        f"red=built-up  grey=no data / no tile", fontsize=11
    )
    out = REPORT_DIR / f"worldcover_comparison_{scene}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def evaluate_scene(scene, version, year, model):
    scene_dir = DATA_DIR / scene
    tiles = sorted(IMAGE_DIR.glob(f"{scene}_*.png"))
    if not scene_dir.is_dir() or not tiles:
        print(f"\nSKIP {scene}: scene folder or training tiles not found")
        return None

    print(f"\n{'=' * 64}\nScene {scene}  (WorldCover {year} {version})\n"
          f"{'=' * 64}")
    reference, profile = build_reference(scene, version, year)
    ref_tif = save_reference_tif(scene, reference, profile)
    print(f"  Reference mask saved: {ref_tif}")

    model_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rules_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    model_canvas = np.full(reference.shape, IGNORE, dtype=np.uint8)
    rules_canvas = np.full(reference.shape, IGNORE, dtype=np.uint8)
    size = 256

    for start in range(0, len(tiles), 8):
        batch = tiles[start:start + 8]
        images = np.stack([
            np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            for p in batch
        ])
        x = torch.from_numpy(images).permute(0, 3, 1, 2)
        with torch.inference_mode():
            predictions = model(x).argmax(dim=1).numpy().astype(np.uint8)

        for path, prediction in zip(batch, predictions):
            _, row, col = tile_position(path)
            truth = reference[row:row + size, col:col + size]
            rules = np.asarray(
                Image.open(MASK_DIR / f"{path.stem}.png"), dtype=np.uint8
            )
            if truth.shape != prediction.shape:
                continue
            add_to_matrix(model_matrix, truth, prediction)
            add_to_matrix(rules_matrix, truth, rules)
            model_canvas[row:row + size, col:col + size] = prediction
            rules_canvas[row:row + size, col:col + size] = rules

    print_block(f"{scene}: TRAINED MODEL vs WorldCover", model_matrix)
    print_block(f"{scene}: ORIGINAL RULE-BASED LABELS vs WorldCover",
                rules_matrix)

    water_truth = int(model_matrix[0].sum())
    print(f"\nWater check: WorldCover marks {water_truth:,} pixels as water "
          f"in the tiled area.\n  Model found {int(model_matrix[0, 0]):,}, "
          f"rules found {int(rules_matrix[0, 0]):,}.")

    figure = save_figure(scene, reference, model_canvas, rules_canvas, year)
    print(f"Picture saved: {figure}")
    return model_matrix, rules_matrix


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"No trained model at {CHECKPOINT}. "
                         "Run train_model.py first.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = SmallUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_model = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_rules = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    done = 0

    for scene, (version, year) in SCENES.items():
        result = evaluate_scene(scene, version, year, model)
        if result is None:
            continue
        done += 1
        for source, matrix in zip(("model", "rules"), result):
            precision, recall, iou, accuracy, miou = metrics_from(matrix)
            rows.append({
                "scene": scene, "source": source,
                "accuracy": round(float(accuracy), 4),
                "mean_iou": round(float(miou), 4),
                **{f"{n}_recall": round(float(recall[i]), 4)
                   for i, n in enumerate(CLASS_NAMES)},
                **{f"{n}_precision": round(float(precision[i]), 4)
                   for i, n in enumerate(CLASS_NAMES)},
            })
        total_model += result[0]
        total_rules += result[1]

    if done == 0:
        raise SystemExit("No scenes were evaluated.")

    print(f"\n{'=' * 64}\nALL SCENES COMBINED vs WorldCover\n{'=' * 64}")
    print_block("TRAINED MODEL", total_model)
    print_block("ORIGINAL RULE-BASED LABELS", total_rules)

    out_csv = REPORT_DIR / "worldcover_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")
    print("WorldCover is a strong independent reference but is itself a "
          "computer-made map.")


if __name__ == "__main__":
    main()
