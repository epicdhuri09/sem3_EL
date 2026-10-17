
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"
OUT_DIR = ROOT / "data" / "training" / "images"

TILE_SIZE = 256
STRIDE = 256
MIN_VALID_FRACTION = 0.90


def find_band(scene, band):
    matches = [
        p for p in scene.rglob("*.tiff")
        if band in p.name.upper()
        and "(RAW)" in p.name.upper()
    ]
    return matches[0] if matches else None


def normalize_band(array, valid):
    values = array[valid]
    if values.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low, high = np.percentile(values, [2, 98])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array.astype(np.float32) - low) / (high - low)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


OUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0

for scene in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    paths = [find_band(scene, band) for band in ("B04", "B03", "B02")]

    if any(path is None for path in paths):
        print(f"SKIP {scene.name}: missing RGB band")
        continue

    with rasterio.open(paths[0]) as red_src, \
         rasterio.open(paths[1]) as green_src, \
         rasterio.open(paths[2]) as blue_src:

        sources = [red_src, green_src, blue_src]

        if not all(
            src.shape == red_src.shape
            and src.crs == red_src.crs
            and src.transform == red_src.transform
            for src in sources
        ):
            print(f"SKIP {scene.name}: RGB bands are not aligned")
            continue

        bands = [src.read(1, masked=True).astype(np.float32)
                 for src in sources]

        arrays = [band.filled(np.nan) for band in bands]
        valid = np.logical_and.reduce([
            ~np.ma.getmaskarray(band) & np.isfinite(array)
            for band, array in zip(bands, arrays)
        ])

        rgb = np.stack([
            normalize_band(array, valid)
            for array in arrays
        ], axis=-1)

        height, width = valid.shape
        scene_count = 0

        for y in range(0, height - TILE_SIZE + 1, STRIDE):
            for x in range(0, width - TILE_SIZE + 1, STRIDE):
                tile_valid = valid[
                    y:y + TILE_SIZE, x:x + TILE_SIZE
                ]

                if tile_valid.mean() < MIN_VALID_FRACTION:
                    continue

                tile = rgb[
                    y:y + TILE_SIZE, x:x + TILE_SIZE
                ]

                filename = f"{scene.name}_{y}_{x}.png"
                Image.fromarray(tile).save(OUT_DIR / filename)
                scene_count += 1

        total += scene_count
        print(f"{scene.name}: saved {scene_count} tiles")

print(f"\nTotal tiles saved: {total}")
print(f"Output folder: {OUT_DIR}")
