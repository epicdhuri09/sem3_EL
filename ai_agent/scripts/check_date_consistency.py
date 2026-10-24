"""Why do land-cover shares jump between dates? A diagnostic, not a model.

For every scene, on the same common area, it prints
  * raw band brightness (median reflectance of each band),
  * the 2nd/98th percentile "stretch" the model input uses for B04 and B08
    (landcover6.load_scene rescales every band PER SCENE with these),
  * the share of ground that is green by a simple NDVI rule (no model), and
  * the share the trained model calls vegetation.

How to read it:
  * If the bands themselves differ a lot between dates, the images are not
    radiometrically alike (season, haze, processing) and the model sees
    different inputs for the same ground.
  * If the model's vegetation share does NOT follow the simple NDVI share,
    the model is unstable across dates, not the land.

Run from the project root (after predict_6band.py):
    python ai_agent/scripts/check_date_consistency.py
"""

import csv

import numpy as np
import rasterio

from landcover6 import BANDS, DATA_DIR, find_band, scene_date_key
from lake_timeline import OUT_DIR, PRED_DIR, make_common_grid, on_grid

NDVI_GREEN = 0.30          # simple "green" threshold for the model-free share
BAND_DIFF_LIMIT = 0.30     # flag a band whose median differs >30% from typical


def read_raw(name):
    """Return raw bands as reflectance (dict), valid mask, scene dict."""
    arrays, grid_info = {}, None
    for band in BANDS:
        with rasterio.open(find_band(DATA_DIR / name, band)) as src:
            masked = src.read(1, masked=True).astype(np.float32)
            arrays[band] = masked.filled(0.0)
            this = (src.transform, src.crs, src.shape)
            grid_info = grid_info or this
            if this != grid_info:
                raise SystemExit(f"{name}: bands are not aligned")
            if band == BANDS[0]:
                valid = ~np.ma.getmaskarray(masked)
            else:
                valid &= ~np.ma.getmaskarray(masked)
    valid &= np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    # pixels that are exactly zero in every band are padding, not data
    all_zero = np.logical_and.reduce([a == 0 for a in arrays.values()])
    zero_pct = 100.0 * (valid & all_zero).sum() / max(valid.sum(), 1)
    valid &= ~all_zero
    sample = arrays["B02"][valid]
    scale = 10000.0 if sample.size and np.percentile(sample, 99) > 5 else 1.0
    bands = {b: a / scale for b, a in arrays.items()}
    scene = {"transform": grid_info[0], "crs": grid_info[1]}
    return bands, valid, scene, scale, zero_pct


def main():
    names = []
    for scene_dir in sorted((p for p in DATA_DIR.iterdir() if p.is_dir()),
                            key=lambda p: scene_date_key(p.name)):
        if all(find_band(scene_dir, b) is not None for b in BANDS):
            names.append(scene_dir.name)
        else:
            print(f"SKIP {scene_dir.name}: missing a band")
    if len(names) < 2:
        raise SystemExit("Need at least two usable scenes.")

    grid = make_common_grid(names)
    rows = []
    for name in names:
        bands, valid, scene, scale, zero_pct = read_raw(name)
        valid_c = on_grid(valid, scene, grid)
        ok = valid_c > 0.5
        mean = {b: on_grid(np.where(valid, bands[b], 0.0), scene, grid)
                / np.maximum(valid_c, 1e-6) for b in BANDS}
        ndvi = (mean["B08"] - mean["B04"]) / (mean["B08"] + mean["B04"] + 1e-6)

        row = {"scene": name, "scale": scale, "zero_pad_pct": zero_pct}
        for band in BANDS:
            row[f"{band}_median"] = float(np.median(mean[band][ok]))
        # the stretch the model input uses (whole scene, like load_scene)
        for band in ("B04", "B08"):
            lo, hi = np.percentile(bands[band][valid], [2, 98])
            row[f"{band}_p2"], row[f"{band}_p98"] = float(lo), float(hi)
        row["ndvi_median"] = float(np.median(ndvi[ok]))
        row["green_share_pct"] = 100 * float((ndvi[ok] > NDVI_GREEN).mean())

        saved = PRED_DIR / f"{name}_landcover.tif"
        if saved.exists():
            with rasterio.open(saved) as src:
                labels = src.read(1)
            veg = on_grid((labels == 1) & valid, scene, grid)
            row["model_veg_pct"] = 100 * float(veg[ok].mean())
        else:
            row["model_veg_pct"] = float("nan")
        rows.append(row)
        print(f"  measured {name}")

    # ---- table -----------------------------------------------------------
    print(f"\n{'=' * 92}\nRAW BAND BRIGHTNESS ON THE COMMON AREA "
          f"(median reflectance)\n{'=' * 92}")
    print(f"{'scene':<12}" + "".join(f"{b:>8}" for b in BANDS)
          + f"{'B04 p2-p98':>14}{'B08 p2-p98':>14}{'zero-pad %':>12}")
    for r in rows:
        print(f"{r['scene']:<12}"
              + "".join(f"{r[f'{b}_median']:>8.3f}" for b in BANDS)
              + f"{r['B04_p2']:>7.3f}-{r['B04_p98']:<6.3f}"
              + f"{r['B08_p2']:>7.3f}-{r['B08_p98']:<6.3f}"
              + f"{r['zero_pad_pct']:>11.1f}%")

    print(f"\n{'=' * 92}\nGREEN SHARE: SIMPLE NDVI RULE vs MODEL\n{'=' * 92}")
    print(f"{'scene':<12}{'median NDVI':>13}{f'NDVI>{NDVI_GREEN:g} %':>13}"
          f"{'model veg %':>13}{'model / NDVI':>14}")
    ratios = []
    for r in rows:
        ratio = r["model_veg_pct"] / max(r["green_share_pct"], 1e-6)
        ratios.append(ratio)
        r["model_to_ndvi_ratio"] = ratio
        print(f"{r['scene']:<12}{r['ndvi_median']:>13.2f}"
              f"{r['green_share_pct']:>13.1f}{r['model_veg_pct']:>13.1f}"
              f"{ratio:>14.2f}")

    # ---- verdict ---------------------------------------------------------
    print(f"\n{'=' * 92}\nWHAT THIS SUGGESTS\n{'=' * 92}")
    typical = {b: np.median([r[f"{b}_median"] for r in rows]) for b in BANDS}
    off = []
    for r in rows:
        bad = [b for b in BANDS
               if abs(r[f"{b}_median"] - typical[b]) / typical[b]
               > BAND_DIFF_LIMIT]
        if bad:
            off.append(f"{r['scene']} ({', '.join(bad)})")
    if off:
        print("Scenes whose band brightness differs >30% from the typical "
              "scene:\n  " + "; ".join(off))
    else:
        print("No scene has a band median more than 30% away from the "
              "typical scene.")

    finite = [x for x in ratios if np.isfinite(x)]
    if finite:
        spread = (max(finite) - min(finite)) / np.median(finite)
        print(f"\nModel-vegetation / NDVI-green ratio runs from "
              f"{min(finite):.2f} to {max(finite):.2f} across dates.")
        if spread > 0.25:
            print("That ratio should be roughly constant. It is not, so the "
                  "model's land-cover shares are NOT stable across dates "
                  "even where the greenness of the ground is similar. "
                  "Do not read land-cover change from them yet.")
        else:
            print("That ratio is fairly constant, so the model follows "
                  "greenness consistently and the shifts are probably real "
                  "or seasonal.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "date_consistency.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
