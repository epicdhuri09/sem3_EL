"""Core code: measure open water INSIDE a lake outline on one satellite image.

Used by lake_water_timeseries.py. Not run directly.

Method for one lake on one date
  1. Cut out a window around the outline (outline + 150 m of surroundings).
  2. Mark unusable pixels: clouds, cirrus, no-data (and cloud shadow when the
     scene is cloudy). If under 80% of the outline is usable -> "cloudy", no
     number is reported.
  3. Water index MNDWI = (green - SWIR) / (green + SWIR). The threshold is
     chosen for THIS lake on THIS date with Otsu's method on the window (so it
     adapts to turbid or algae-coloured water). If the window is not clearly
     two-groups, a fixed 0.0 is used instead.
  4. Water = MNDWI above the threshold AND dark in near-infrared (this removes
     shadows and wet-looking roofs). Counted only inside the outline (+30 m).
  5. Also reported: water just OUTSIDE the outline (a sign the outline is
     wrong or the lake has spilled), green growth inside the outline (floating
     weeds, islands, trees), and the image provider's own water class.

The thresholds below are sensible starting values, NOT yet tuned on Bengaluru
lakes. Check the quality-control pictures and tune using hand-drawn outlines.
"""

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
LAKES_PATH = ROOT / "data" / "lakes" / "bengaluru_lakes.geojson"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

BUFFER_M = 30.0          # outline is widened by this when counting water
RING_M = 150.0           # surroundings used for the threshold / spill check
NIR_MAX = 0.20           # water must be darker than this in near-infrared
SHADOW_REACH_M = 1500.0   # cloud shadow must lie within this of a cloud
BODY_REACH_M = 100.0     # a water body touching the outline is counted
                         # whole, out to this far from the outline
BODY_MIN_TOUCH_PX = 2    # ... if at least this many pixels touch it
DARK_REFL = 0.03         # "black water": NIR and SWIR both below this ...
DARK_GREEN = 0.05        # ... and green below this (archive scenes only)
T_MIN, T_MAX, T_DEFAULT = -0.05, 0.30, 0.0
SEPARABILITY_MIN = 0.70  # Otsu quality needed to trust the adaptive threshold
MIN_VALID_FRAC = 0.80    # outline must be at least this clear
VEG_NDVI = 0.35          # green growth inside the outline
SMALL_PIXELS = 100       # fewer pixels than this = "small", least reliable
BRIGHT_BLUE = 0.25       # local scenes only: very bright blue = cloud-like

ROW_FIELDS = [
    "lake_id", "lake", "date", "source", "item_id", "status", "water_ha",
    "outline_ha", "water_pct_of_outline", "vegetation_ha", "ring_water_ha",
    "scl_water_ha", "valid_frac", "threshold", "threshold_method",
    "n_inside_px", "pixel_m", "scene_cloud_pct",
]


# --------------------------------------------------------------------------
# Lakes
# --------------------------------------------------------------------------

def load_lakes(path=LAKES_PATH, min_ha=1.0, only=None):
    if not Path(path).exists():
        raise SystemExit(f"No lake outlines at {path}.\n"
                         "Run fetch_lake_outlines.py first.")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    lakes = []
    for feature in data["features"]:
        props = feature["properties"]
        if props["area_ha"] < min_ha:
            continue
        if only and not any(o.lower() in props["name"].lower() for o in only):
            continue
        lakes.append({
            "id": props["id"], "name": props["name"] or props["id"],
            "area_ha": props["area_ha"],
            "geom_ll": shape(feature["geometry"]), "_cache": {},
        })
    return lakes


def lake_geom(lake, crs):
    """The lake outline in the image's coordinate system (cached)."""
    key = str(crs)
    if key not in lake["_cache"]:
        lake["_cache"][key] = shape(
            transform_geom("EPSG:4326", crs, mapping(lake["geom_ll"])))
    return lake["_cache"][key]


def union_bounds(lakes, crs, margin=200.0):
    boxes = [lake_geom(lake, crs).bounds for lake in lakes]
    return (min(b[0] for b in boxes) - margin, min(b[1] for b in boxes) - margin,
            max(b[2] for b in boxes) + margin, max(b[3] for b in boxes) + margin)


# --------------------------------------------------------------------------
# Scene container and loaders
# --------------------------------------------------------------------------

@dataclass
class SceneData:
    date: str
    source: str
    item_id: str
    green: np.ndarray
    nir: np.ndarray
    swir: np.ndarray
    clear: np.ndarray
    transform: object
    crs: object
    pixel_m: float
    red: np.ndarray = None
    scl_water: np.ndarray = None
    cloud_pct: float = float("nan")
    extra: dict = field(default_factory=dict)


def to_reflectance(dn, offset):
    out = (dn.astype(np.float32) - offset) / 10000.0
    out[dn == 0] = np.nan
    return out


def processing_offset(props):
    """Sentinel-2 files from baseline 04.00 (2022) on carry a +1000 offset."""
    try:
        baseline = float(props.get("s2:processing_baseline", "0"))
    except (TypeError, ValueError):
        baseline = 0.0
    return 1000.0 if baseline >= 4.0 else 0.0


def decide_offset(props, swir_dn, usable):
    """Which DN offset do THIS image's numbers carry? -> (offset, note).

    Images from processing baseline 04.00 on should carry +1000, but some
    archive copies already have it removed. Subtracting 1000 a second time
    pushes dark water below zero (black lake, a broken water index). A real
    +1000 offset means almost no pixel is below ~1000, so if the darkest 1%
    of the shortwave-infrared band sits far below 1000 the offset is not
    there, whatever the metadata says."""
    stated = processing_offset(props)
    if props.get("earthsearch:boa_offset_applied") is True:
        stated = 0.0
    sample = swir_dn[usable & (swir_dn > 0)]
    if stated >= 1000.0 and sample.size >= 500:
        p1 = float(np.percentile(sample, 1))
        if p1 < 600.0:
            return 0.0, (f"offset 0 (data check: darkest 1% of SWIR is "
                         f"{p1:.0f}, metadata said 1000)")
    return stated, "baseline"


def load_archive_scene(item, lakes):
    """Read just the lake window of one archive image (10 m bands)."""
    props, assets = item["properties"], item["assets"]
    cloud = props.get("eo:cloud_cover", float("nan"))

    with rasterio.open(assets["green"]["href"]) as src:
        crs = src.crs
        wanted = union_bounds(lakes, crs)
        try:
            win = windows.from_bounds(*wanted, transform=src.transform)
            win = win.round_offsets(op="floor").round_lengths(op="ceil")
            win = win.intersection(windows.Window(0, 0, src.width, src.height))
        except windows.WindowError:
            return None
        shape_ = (int(win.height), int(win.width))
        transform = src.window_transform(win)
        exact = windows.bounds(win, src.transform)
        green_dn = src.read(1, window=win)
        pixel_m = abs(src.transform.a)

    def aligned(key, resampling):
        with rasterio.open(assets[key]["href"]) as other:
            w = windows.from_bounds(*exact, transform=other.transform)
            return other.read(1, window=w, out_shape=shape_,
                              resampling=resampling)

    nir_dn = aligned("nir", Resampling.bilinear)
    swir_dn = aligned("swir16", Resampling.bilinear)
    red_dn = aligned("red", Resampling.bilinear) if "red" in assets else None
    scl = aligned("scl", Resampling.nearest)

    # 0 nodata, 1 saturated, 8/9 cloud, 10 cirrus. 3 = cloud shadow, but
    # Sen2Cor often mislabels DARK WATER as shadow, so count class 3 as bad
    # only when the scene really has clouds.
    bad = np.isin(scl, [0, 1, 8, 9, 10])
    if np.isfinite(cloud) and cloud > 5:
        # A real cloud shadow lies within about a kilometre of its cloud.
        # Class-3 pixels farther from any cloud are dark water or dark
        # ground wrongly labelled as shadow, so they are kept.
        shadow = scl == 3
        cloud_px = np.isin(scl, [8, 9, 10])
        if shadow.any() and cloud_px.any():
            from scipy import ndimage
            reach_px = SHADOW_REACH_M / abs(transform.a)
            near_cloud = ndimage.distance_transform_edt(~cloud_px) <= reach_px
            bad |= shadow & near_cloud
    offset, offset_note = decide_offset(props, swir_dn, scl != 0)
    green = to_reflectance(green_dn, offset)
    nir = to_reflectance(nir_dn, offset)
    swir = to_reflectance(swir_dn, offset)
    clear = (~bad & np.isfinite(green) & np.isfinite(nir) & np.isfinite(swir))
    return SceneData(
        date=props["datetime"][:10], source="archive", item_id=item["id"],
        green=green, nir=nir, swir=swir, clear=clear, transform=transform,
        crs=crs, pixel_m=pixel_m,
        red=to_reflectance(red_dn, offset) if red_dn is not None else None,
        scl_water=(scl == 6), cloud_pct=cloud,
        extra={"offset": offset, "offset_note": offset_note},
    )


def load_local_scene(scene_dir, find_band, date_text):
    """Your downloaded scenes (about 17-20 m pixels, no cloud mask)."""
    names = {"green": "B03", "nir": "B08", "swir": "B11", "red": "B04",
             "blue": "B02"}
    arrays, grid = {}, None
    for key, band in names.items():
        path = find_band(scene_dir, band)
        if path is None:
            arrays[key] = None
            continue
        with rasterio.open(path) as src:
            this = (src.transform, src.crs, src.shape)
            grid = grid or this
            if this != grid:
                return None
            arrays[key] = src.read(1, masked=True).astype(np.float32)
    if any(arrays[k] is None for k in ("green", "nir", "swir")):
        return None

    raw = {k: (v.filled(0.0) if v is not None else None)
           for k, v in arrays.items()}
    masked_bad = np.ma.getmaskarray(arrays["green"])
    sample = raw["green"][~masked_bad]
    scale = 10000.0 if sample.size and np.percentile(sample, 99) > 5 else 1.0
    allzero = np.logical_and.reduce(
        [v == 0 for v in raw.values() if v is not None])
    refl = {k: (v / scale if v is not None else None) for k, v in raw.items()}

    clear = ~masked_bad & ~allzero
    if refl["blue"] is not None:
        clear &= refl["blue"] < BRIGHT_BLUE
    transform, crs, _ = grid
    return SceneData(
        date=date_text, source="local", item_id=Path(scene_dir).name,
        green=refl["green"], nir=refl["nir"], swir=refl["swir"], clear=clear,
        transform=transform, crs=crs,
        pixel_m=float(np.sqrt(abs(transform.a * transform.e))),
        red=refl["red"],
    )


# --------------------------------------------------------------------------
# STAC search (archive)
# --------------------------------------------------------------------------

def _post(url, body):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "sem3_EL-student-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _next_body(data, body):
    for link in data.get("links", []):
        if link.get("rel") == "next" and link.get("body"):
            nxt = dict(link["body"])
            return {**body, **nxt} if link.get("merge") else nxt
    return None


def search_archive(bbox, first_year, last_year, collection="sentinel-2-l2a",
                   post=_post):
    """All matching images, year by year. `post` is replaceable for tests."""
    items = {}
    for year in range(first_year, last_year + 1):
        body = {"collections": [collection], "bbox": list(bbox),
                "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
                "limit": 200}
        pages = 0
        while body and pages < 30:
            data = post(STAC_URL, body)
            for feature in data.get("features", []):
                items[feature["id"]] = feature
            body = _next_body(data, body)
            pages += 1
    return list(items.values())


def choose_items(items, lakes_bbox, per_month=1, max_cloud=70.0):
    """Keep the tile that covers the lakes, then the clearest per month."""
    lx0, ly0, lx1, ly1 = lakes_bbox

    def covers(item):
        bx0, by0, bx1, by1 = item["bbox"]
        return bx0 <= lx0 and by0 <= ly0 and bx1 >= lx1 and by1 >= ly1

    full = [i for i in items if covers(i)]
    pool = full if full else items
    counts = {}
    for item in pool:
        code = item["properties"].get("grid:code", "")
        counts[code] = counts.get(code, 0) + 1
    best_tile = max(counts, key=counts.get) if counts else None
    pool = [i for i in pool
            if i["properties"].get("grid:code", "") == best_tile
            and i["properties"].get("eo:cloud_cover", 100.0) < max_cloud]

    months = {}
    for item in pool:
        months.setdefault(item["properties"]["datetime"][:7], []).append(item)
    chosen = []
    for month in sorted(months):
        ranked = sorted(months[month],
                        key=lambda i: i["properties"].get("eo:cloud_cover", 100))
        chosen.extend(ranked[:per_month])
    chosen.sort(key=lambda i: i["properties"]["datetime"])
    return chosen, best_tile


# --------------------------------------------------------------------------
# Measuring one lake
# --------------------------------------------------------------------------

def otsu(values, bins=128, lo=-1.0, hi=1.0):
    """Otsu threshold and how clearly two groups are separated (0-1)."""
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=bins,
                               range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centres)
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * w0 - mu) ** 2 / (w0 * w1)
    between = np.nan_to_num(between)
    k = int(np.argmax(between))
    total = float(np.sum(p * (centres - mu_total) ** 2))
    separability = float(between[k] / total) if total > 0 else 0.0
    return float(edges[k + 1]), separability


def measure_lake(scene, lake, keep_debug=False):
    """Return (row, debug). Row is None if the lake is outside the image."""
    geom = lake_geom(lake, scene.crs)
    height, width = scene.green.shape
    try:
        win = windows.from_bounds(*geom.buffer(RING_M).bounds,
                                  transform=scene.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = win.intersection(windows.Window(0, 0, width, height))
    except windows.WindowError:
        return None, None
    if win.width < 2 or win.height < 2:
        return None, None

    rows, cols = win.toslices()
    sub_t = windows.transform(win, scene.transform)
    out_shape = (int(win.height), int(win.width))
    inside = features.rasterize([(geom.buffer(BUFFER_M), 1)], out_shape=out_shape,
                                transform=sub_t, fill=0, dtype="uint8",
                                all_touched=True).astype(bool)
    ring = features.rasterize(
        [(geom.buffer(RING_M).difference(geom.buffer(BUFFER_M)), 1)],
        out_shape=out_shape, transform=sub_t, fill=0, dtype="uint8",
    ).astype(bool)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return None, None

    px_ha = scene.pixel_m ** 2 / 10_000.0
    clear = scene.clear[rows, cols]
    valid_frac = float((inside & clear).sum() / n_inside)
    row = {
        "lake_id": lake["id"], "lake": lake["name"], "date": scene.date,
        "source": scene.source, "item_id": scene.item_id,
        "outline_ha": round(lake["area_ha"], 2),
        "valid_frac": round(valid_frac, 3), "n_inside_px": n_inside,
        "pixel_m": round(scene.pixel_m, 2),
        "scene_cloud_pct": (round(scene.cloud_pct, 1)
                            if np.isfinite(scene.cloud_pct) else ""),
        "water_ha": "", "water_pct_of_outline": "", "vegetation_ha": "",
        "ring_water_ha": "", "scl_water_ha": "", "threshold": "",
        "threshold_method": "",
    }
    if valid_frac < MIN_VALID_FRAC:
        row["status"] = "cloudy"
        return row, None

    g, n, s = (scene.green[rows, cols], scene.nir[rows, cols],
               scene.swir[rows, cols])
    with np.errstate(divide="ignore", invalid="ignore"):
        mndwi = (g - s) / (g + s)
    mndwi = np.where(np.isfinite(mndwi), mndwi, -1.0)

    sample = mndwi[(inside | ring) & clear]
    method = "default"
    threshold = T_DEFAULT
    if sample.size >= 30:
        t, separability = otsu(sample)
        if separability >= SEPARABILITY_MIN:
            threshold, method = float(np.clip(t, T_MIN, T_MAX)), "otsu"

    water = (mndwi > threshold) & (n < NIR_MAX) & clear
    # Very dark in every band = clear deep water the index misses (archive only).
    if scene.source == "archive":
        dark = (n < DARK_REFL) & (s < DARK_REFL) & (g < DARK_GREEN) & clear
        if (dark & ~water & inside).any():
            method += "+dark"
        water = water | dark
    # OSM outlines are sometimes offset from the real lake, so water just
    # outside the outline is counted too when it is part of the same body
    # that touches the outline (connected pixels, within BODY_REACH_M).
    from scipy import ndimage
    reach = features.rasterize([(geom.buffer(BODY_REACH_M), 1)],
                               out_shape=out_shape, transform=sub_t, fill=0,
                               dtype="uint8", all_touched=True).astype(bool)
    groups, count = ndimage.label(water & reach, structure=np.ones((3, 3)))
    counted = water & inside
    if count:
        touch = np.bincount(groups[inside & (groups > 0)],
                            minlength=count + 1)
        keep = touch >= BODY_MIN_TOUCH_PX
        keep[0] = False
        body = keep[groups]
        if (body & ~inside).any():
            method += "+body"
        counted = counted | body
    water_ha = float(counted.sum() * px_ha)
    row.update({
        "water_ha": round(water_ha, 2),
        "water_pct_of_outline": round(100 * water_ha / (n_inside * px_ha), 1),
        "ring_water_ha": round(float((water & ring & ~counted).sum() * px_ha), 2),
        "threshold": round(threshold, 3), "threshold_method": method,
        "status": "small" if n_inside < SMALL_PIXELS else "ok",
    })
    if scene.red is not None:
        r = scene.red[rows, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (n - r) / (n + r)
        green_growth = inside & clear & ~water & (ndvi > VEG_NDVI)
        row["vegetation_ha"] = round(float(green_growth.sum() * px_ha), 2)
    if scene.scl_water is not None:
        row["scl_water_ha"] = round(
            float((scene.scl_water[rows, cols] & inside).sum() * px_ha), 2)

    debug = None
    if keep_debug:
        debug = {"green": g, "nir": n, "swir": s, "red": (scene.red[rows, cols]
                 if scene.red is not None else g), "mndwi": mndwi,
                 "water": counted, "inside": inside, "clear": clear,
                 "ring": ring}
    return row, debug


def save_chip(path, row, debug):
    """Quality-control picture: false colour + water mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.dstack([np.clip(debug["nir"] / 0.4, 0, 1),
                     np.clip(debug["red"] / 0.25, 0, 1),
                     np.clip(debug["green"] / 0.25, 0, 1)])
    rgb = np.nan_to_num(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("false colour (water dark, plants red)", fontsize=9)
    axes[1].imshow(debug["mndwi"], cmap="gray", vmin=-0.8, vmax=0.8)
    overlay = np.zeros(debug["water"].shape + (4,))
    overlay[debug["water"]] = (0.16, 0.47, 0.84, 0.55)
    axes[1].imshow(overlay)
    axes[1].set_title(f"water {row['water_ha']} ha  threshold "
                      f"{row['threshold']} ({row['threshold_method']})",
                      fontsize=9)
    for ax in axes:
        ax.contour(debug["inside"], [0.5], colors="yellow", linewidths=1)
        ax.axis("off")
    fig.suptitle(f"{row['lake']} - {row['date']} ({row['source']}); "
                 f"yellow = outline", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
