
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "extracted"
TILES = ROOT / "data" / "training" / "images"
MASKS = ROOT / "data" / "training" / "masks"
MASKS.mkdir(parents=True, exist_ok=True)

def band_path(scene, band):
    return next((p for p in scene.rglob("*.tiff")
                 if band in p.name.upper()
                 and "(RAW)" in p.name.upper()), None)

def normdiff(a, b):
    return (a - b) / (a + b + 1e-6)

cache = {}
made = 0

for tile_path in sorted(TILES.glob("*.png")):
    parts = tile_path.stem.rsplit("_", 2)
    scene_name, y, x = parts[0], int(parts[1]), int(parts[2])
    scene = DATA / scene_name

    if scene_name not in cache:
        names = ["B02", "B03", "B04", "B08", "B11"]
        paths = [band_path(scene, b) for b in names]
        if any(p is None for p in paths):
            cache[scene_name] = None
            continue

        arrays = []
        for p in paths:
            with rasterio.open(p) as src:
                arrays.append(src.read(1).astype("float32"))
        cache[scene_name] = arrays

    arrays = cache[scene_name]
    if arrays is None:
        continue

    blue, green, red, nir, swir = [
        a[y:y+256, x:x+256] for a in arrays
    ]
    if any(a.shape != (256, 256) for a in
           [blue, green, red, nir, swir]):
        continue

    ndwi = normdiff(green, nir)
    ndvi = normdiff(nir, red)
    ndbi = normdiff(swir, nir)

    # Provisional heuristic labels; not verified ground truth.
    mask = np.full((256, 256), 2, dtype=np.uint8)  # bare land
    mask[ndvi > 0.35] = 1                           # vegetation
    mask[ndbi > 0.10] = 3                           # built-up candidate
    mask[ndwi > 0.15] = 0                           # water candidate

    Image.fromarray(mask).save(MASKS / tile_path.name)
    made += 1

print(f"Provisional masks created: {made}")
print(f"Folder: {MASKS}")
print("Labels: 0 water, 1 vegetation, 2 bare land, 3 built-up")
print("WARNING: heuristic labels require visual validation.")
