
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def find_band(scene_dir, band):
    matches = list(scene_dir.rglob(f"*{band}*(Raw).tiff"))
    if not matches:
        matches = [
            p for p in scene_dir.rglob("*.tiff")
            if band in p.name.upper()
            and "TRUE_COLOR" not in p.name.upper()
        ]
    return matches[0] if matches else None


def main():
    scenes = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    if not scenes:
        print("No scene folders found:", DATA_DIR)
        return

    for scene in scenes:
        print(f"\n{'=' * 60}\nSCENE: {scene.name}")
        datasets = {}

        for band in BANDS:
            path = find_band(scene, band)
            if path is None:
                print(f"{band}: MISSING")
                continue

            try:
                with rasterio.open(path) as src:
                    datasets[band] = {
                        "path": path,
                        "shape": (src.height, src.width),
                        "crs": str(src.crs),
                        "transform": src.transform,
                        "bounds": src.bounds,
                        "resolution": src.res,
                    }
                    print(
                        f"{band}: {src.width}x{src.height}, "
                        f"CRS={src.crs}, resolution={src.res}"
                    )
            except Exception as exc:
                print(f"{band}: ERROR - {exc}")

        if "B03" not in datasets or "B08" not in datasets:
            print("NDWI skipped: B03 or B08 is missing.")
            continue

        green = datasets["B03"]
        nir = datasets["B08"]

        aligned = (
            green["shape"] == nir["shape"]
            and green["crs"] == nir["crs"]
            and green["transform"] == nir["transform"]
        )

        print("B03/B08 grid aligned:", aligned)

        if not aligned:
            print("NDWI skipped until grids are aligned.")
            continue

        try:
            with rasterio.open(green["path"]) as gsrc, \
                 rasterio.open(nir["path"]) as nsrc:

                g = gsrc.read(1, masked=True).astype("float32")
                n = nsrc.read(1, masked=True).astype("float32")

                valid = (
                    ~np.ma.getmaskarray(g)
                    & ~np.ma.getmaskarray(n)
                    & np.isfinite(g.filled(np.nan))
                    & np.isfinite(n.filled(np.nan))
                    & ((g.filled(0) + n.filled(0)) != 0)
                )

                denominator = g.filled(0) + n.filled(0)
                ndwi = np.full(g.shape, np.nan, dtype="float32")
                np.divide(
                    g.filled(0) - n.filled(0),
                    denominator,
                    out=ndwi,
                    where=valid,
                )

                if valid.any():
                    values = ndwi[valid]
                    print("Valid NDWI pixels:", int(valid.sum()))
                    print("NDWI min/max:", float(values.min()),
                          float(values.max()))
                    print("NDWI mean:", float(values.mean()))
                    print("Potential water pixels (NDWI > 0):",
                          int((values > 0).sum()))
                else:
                    print("No valid pixels for NDWI.")

        except Exception as exc:
            print("NDWI calculation failed:", exc)


if __name__ == "__main__":
    main()
