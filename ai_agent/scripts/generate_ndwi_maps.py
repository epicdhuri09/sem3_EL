
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"
OUT_DIR = ROOT / "reports" / "ndwi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_band(scene, band):
    matches = [
        p for p in scene.rglob("*.tiff")
        if band in p.name.upper()
        and "(RAW)" in p.name.upper()
    ]
    return matches[0] if matches else None


for scene in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    green_path = find_band(scene, "B03")
    nir_path = find_band(scene, "B08")

    if green_path is None or nir_path is None:
        print(f"SKIP {scene.name}: missing B03 or B08")
        continue

    with rasterio.open(green_path) as green_src, \
         rasterio.open(nir_path) as nir_src:

        if (
            green_src.shape != nir_src.shape
            or green_src.crs != nir_src.crs
            or green_src.transform != nir_src.transform
        ):
            print(f"SKIP {scene.name}: bands are not aligned")
            continue

        green = green_src.read(1, masked=True).astype("float32")
        nir = nir_src.read(1, masked=True).astype("float32")

        g = green.filled(np.nan)
        n = nir.filled(np.nan)
        denominator = g + n

        valid = (
            ~np.ma.getmaskarray(green)
            & ~np.ma.getmaskarray(nir)
            & np.isfinite(g)
            & np.isfinite(n)
            & (np.abs(denominator) > 1e-10)
        )

        ndwi = np.full(g.shape, np.nan, dtype="float32")
        np.divide(g - n, denominator, out=ndwi, where=valid)

        profile = green_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
        )

        output_tif = OUT_DIR / f"{scene.name}_ndwi.tif"
        saved = np.where(np.isfinite(ndwi), ndwi, -9999.0)

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(saved, 1)

        output_png = OUT_DIR / f"{scene.name}_ndwi.png"

        plt.figure(figsize=(10, 6))
        plt.imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
        plt.colorbar(label="NDWI")
        plt.title(f"NDWI — {scene.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        plt.close()

        print(f"Saved {output_tif.name} and {output_png.name}")

print(f"\nOutputs saved to: {OUT_DIR}")
