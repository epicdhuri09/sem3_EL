"""Run the all-bands model on every scene and save GEOREFERENCED results.

For each scene it writes (to ai_agent/reports/predictions_6band/):
    <scene>_landcover.tif   class map with real map coordinates (GeoTIFF)
    <scene>_landcover.png   coloured picture for quick viewing
and one table, area_summary.csv, with the area of each class in hectares.

Scenes missing a band (26oct18 has no B04) are skipped with a message.

Run from the project root:
    python ai_agent/scripts/predict_6band.py
"""

import csv

import numpy as np
import rasterio
from PIL import Image

from landcover6 import (
    CLASS_NAMES, DATA_DIR, IGNORE, MODEL_PATH, ROOT, load_model, load_scene,
    predict_scene, scene_date_key,
)

OUT_DIR = ROOT / "reports" / "predictions_6band"
COLORS = np.array([
    [0, 100, 255],    # water
    [20, 180, 60],    # vegetation
    [220, 180, 80],   # bare land
    [230, 65, 65],    # built-up
    [235, 235, 235],  # no data
], dtype=np.uint8)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"No model at {MODEL_PATH}. "
                         "Run train_worldcover_6band.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()

    scene_dirs = sorted((p for p in DATA_DIR.iterdir() if p.is_dir()),
                        key=lambda p: scene_date_key(p.name))
    rows = []
    for scene_dir in scene_dirs:
        features, valid, profile, reason = load_scene(scene_dir)
        if features is None:
            print(f"SKIP {scene_dir.name}: {reason}")
            continue

        labels = predict_scene(model, features, valid)

        profile.update(driver="GTiff", count=1, dtype="uint8",
                       nodata=IGNORE, compress="deflate")
        profile.pop("photometric", None)
        with rasterio.open(OUT_DIR / f"{scene_dir.name}_landcover.tif", "w",
                           **profile) as dst:
            dst.write(labels, 1)

        picture = COLORS[np.where(labels == IGNORE, 4, labels)]
        Image.fromarray(picture).save(
            OUT_DIR / f"{scene_dir.name}_landcover.png")

        transform = profile["transform"]
        pixel_ha = abs(transform.a * transform.e) / 10_000.0
        counts = [int((labels == c).sum()) for c in range(4)]
        valid_px = max(sum(counts), 1)
        row = {"scene": scene_dir.name,
               "pixel_size_m2": round(abs(transform.a * transform.e), 1)}
        for name, count in zip(CLASS_NAMES, counts):
            row[f"{name}_ha"] = round(count * pixel_ha, 1)
            row[f"{name}_pct"] = round(100 * count / valid_px, 2)
        rows.append(row)
        print(f"{scene_dir.name}: water {row['water_ha']:.1f} ha "
              f"({row['water_pct']:.2f}%), vegetation "
              f"{row['vegetation_pct']:.1f}%, built-up "
              f"{row['built_up_pct']:.1f}%")

    if not rows:
        raise SystemExit("No scenes could be processed.")

    out_csv = OUT_DIR / "area_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved maps and table to: {OUT_DIR}")
    print("Caution: clouds are not screened, and different dates cover "
          "slightly different areas,\nso treat water-area changes between "
          "dates as rough until checked.")


if __name__ == "__main__":
    main()
