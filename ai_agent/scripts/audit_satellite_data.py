
from pathlib import Path
import csv
import re
import sys

import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "satellite_audit.csv"

EXPECTED_BANDS = {"B02", "B03", "B04", "B08", "B11", "B12"}


def identify_bands(path):
    """Identify Sentinel-2 band codes present in a filename."""
    name = path.name.upper()
    return sorted(
        band for band in EXPECTED_BANDS
        if re.search(rf"(?<![A-Z0-9]){band}(?![A-Z0-9])", name)
    )


def inspect_tiff(path):
    result = {
        "scene_folder": path.parent.name,
        "filename": path.name,
        "relative_path": str(path.relative_to(DATA_DIR)),
        "bands_in_filename": ",".join(identify_bands(path)),
        "width": "",
        "height": "",
        "band_count": "",
        "crs": "",
        "resolution_x": "",
        "resolution_y": "",
        "dtype": "",
        "nodata": "",
        "bounds": "",
        "status": "OK",
        "error": "",
    }

    try:
        with rasterio.open(path) as src:
            result.update({
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "crs": str(src.crs) if src.crs else "MISSING",
                "resolution_x": abs(src.transform.a),
                "resolution_y": abs(src.transform.e),
                "dtype": ",".join(src.dtypes),
                "nodata": src.nodata,
                "bounds": str(tuple(round(v, 6) for v in src.bounds)),
            })

            if src.crs is None:
                result["status"] = "CHECK: missing CRS"
            elif src.width == 0 or src.height == 0:
                result["status"] = "CHECK: empty raster"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Extracted data folder not found: {DATA_DIR}")
        sys.exit(1)

    files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    )

    if not files:
        print(f"No TIFF files found under: {DATA_DIR}")
        sys.exit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [inspect_tiff(path) for path in files]

    columns = list(rows[0].keys())
    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTIFF files found: {len(files)}")
    print(f"Readable files: {sum(r['status'] != 'ERROR' for r in rows)}")
    print(f"Files needing attention: {sum(r['status'] != 'OK' for r in rows)}")
    print(f"Audit report: {REPORT_FILE}\n")

    print("SUMMARY BY SCENE FOLDER")
    scene_names = sorted({r["scene_folder"] for r in rows})

    for scene in scene_names:
        scene_rows = [r for r in rows if r["scene_folder"] == scene]
        readable = [r for r in scene_rows if r["status"] != "ERROR"]
        crs_values = sorted({r["crs"] for r in readable})
        sizes = sorted({
            f"{r['width']}x{r['height']}"
            for r in readable
            if r["width"] != "" and r["height"] != ""
        })
        bands = sorted({
            band
            for r in scene_rows
            for band in r["bands_in_filename"].split(",")
            if band
        })

        print(f"\n{scene}")
        print(f"  TIFF files: {len(scene_rows)}")
        print(f"  CRS values: {', '.join(crs_values) or 'none'}")
        print(f"  Raster sizes: {', '.join(sizes) or 'unknown'}")
        print(f"  Band codes found in filenames: {', '.join(bands) or 'not detected'}")

    print("\nAudit complete. Open the CSV in Excel to inspect individual files.")


if __name__ == "__main__":
    main()
