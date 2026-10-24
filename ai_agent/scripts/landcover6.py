"""Shared helpers for the all-bands (6 Sentinel bands + 3 indices) model.

Used by train_worldcover_6band.py and predict_6band.py. Not run directly.

Model input (9 channels, all scaled to 0-1):
    B02 blue, B03 green, B04 red, B08 near-infrared, B11 and B12 short-wave
    infrared (each scaled by that scene's 2nd-98th percentile), plus
    NDVI (greenness), NDWI (water) and NDBI (built-up) computed from the raw
    band values.
"""

import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "extracted"
WC_DIR = ROOT / "data" / "worldcover"
MODEL_PATH = ROOT / "models" / "landcover_6band_unet.pth"

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CLASS_NAMES = ["water", "vegetation", "bare_land", "built_up"]
NUM_CLASSES = 4
IN_CHANNELS = 9
IGNORE = 255
EPS = 1e-6


# --------------------------------------------------------------------------
# Reading a scene
# --------------------------------------------------------------------------

def find_band(scene_dir, band):
    matches = [
        p for p in Path(scene_dir).rglob("*.tiff")
        if band in p.name.upper() and "(RAW)" in p.name.upper()
    ]
    return matches[0] if matches else None


SCALING_VERSION = "fixed-v2"
# Reflectance that maps to 1.0 for each band (values above are clipped).
FIXED_RANGE = {"B02": 0.25, "B03": 0.30, "B04": 0.35,
               "B08": 0.50, "B11": 0.50, "B12": 0.40}


def fixed_scale(array, high):
    return np.clip(array / high, 0, 1).astype(np.float32)


def percentile_scale(array, valid):
    values = array[valid]
    if values.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    scaled = (array - low) / (high - low)
    return np.clip(scaled, 0, 1).astype(np.float32)


def index_scaled(numerator, denominator):
    """Normalised-difference style index, mapped from [-1, 1] to [0, 1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = numerator / (denominator + EPS)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return ((np.clip(value, -1, 1) + 1) / 2).astype(np.float32)


def load_scene(scene_dir):
    """Return (features, valid, profile, None) or (None, None, None, reason).

    features: float32 array [9, H, W];  valid: bool array [H, W].
    """
    paths = {band: find_band(scene_dir, band) for band in BANDS}
    missing = [band for band, path in paths.items() if path is None]
    if missing:
        return None, None, None, f"missing band(s): {', '.join(missing)}"

    arrays, grid, profile = {}, None, None
    for band, path in paths.items():
        with rasterio.open(path) as src:
            this_grid = (src.shape, src.crs, src.transform)
            if grid is None:
                grid, profile = this_grid, src.profile.copy()
            elif this_grid != grid:
                return None, None, None, "bands are not aligned"
            arrays[band] = src.read(1, masked=True).astype(np.float32)

    valid = np.logical_and.reduce([
        ~np.ma.getmaskarray(arrays[b]) & np.isfinite(arrays[b].filled(np.nan))
        for b in BANDS
    ])
    raw = {b: arrays[b].filled(0.0) for b in BANDS}

    # Pixels that are exactly zero in every band are image padding, not data.
    valid &= np.logical_or.reduce([raw[b] != 0 for b in BANDS])

    # Reflectance scale (0-1). Some exports store it x10000.
    sample = raw["B02"][valid]
    scale = 10000.0 if sample.size and np.percentile(sample, 99) > 5 else 1.0
    raw = {b: values / scale for b, values in raw.items()}

    # FIXED scaling: the same surface gets the same input value on every
    # date. (The old per-scene percentile stretch hid real differences.)
    channels = [fixed_scale(raw[b], FIXED_RANGE[b]) for b in BANDS]
    channels.append(index_scaled(raw["B08"] - raw["B04"],
                                 raw["B08"] + raw["B04"]))   # NDVI
    channels.append(index_scaled(raw["B03"] - raw["B08"],
                                 raw["B03"] + raw["B08"]))   # NDWI
    channels.append(index_scaled(raw["B11"] - raw["B08"],
                                 raw["B11"] + raw["B08"]))   # NDBI

    features = np.stack(channels).astype(np.float32)
    features[:, ~valid] = 0.0
    return features, valid, profile, None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, classes=NUM_CLASSES, base=24):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, base)
        self.enc2 = block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.out = nn.Conv2d(base, classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("scaling") != SCALING_VERSION:
        raise SystemExit(
            "This model was trained with the old per-scene stretch and "
            "cannot be used with the current inputs. Run "
            "train_worldcover_6band.py again."
        )
    model = UNet(in_channels=checkpoint["in_channels"],
                 base=checkpoint["base"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict_scene(model, features, valid, core=448, margin=32):
    """Predict a whole scene in overlapping windows; returns uint8 [H, W].

    Pixels without valid satellite data are set to 255 (no data).
    """
    _, height, width = features.shape
    pad_h = -height % core
    pad_w = -width % core
    padded = np.pad(
        features,
        ((0, 0), (margin, margin + pad_h), (margin, margin + pad_w)),
        mode="reflect",
    )
    result = np.full((height + pad_h, width + pad_w), IGNORE, dtype=np.uint8)

    for y in range(0, height, core):
        for x in range(0, width, core):
            window = padded[:, y:y + core + 2 * margin,
                            x:x + core + 2 * margin]
            logits = model(torch.from_numpy(window).unsqueeze(0))
            labels = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
            result[y:y + core, x:x + core] = labels[margin:-margin,
                                                    margin:-margin]

    result = result[:height, :width]
    result[~valid] = IGNORE
    return result


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------

def confusion(truth, pred):
    """4x4 matrix: rows = reference class, columns = predicted class."""
    keep = (truth < NUM_CLASSES) & (pred < NUM_CLASSES)
    idx = truth[keep].astype(np.int64) * NUM_CLASSES + pred[keep]
    return np.bincount(
        idx, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from(matrix):
    tp = np.diag(matrix).astype(float)
    truth_total = matrix.sum(axis=1).astype(float)
    pred_total = matrix.sum(axis=0).astype(float)

    def safe(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)

    precision = safe(tp, pred_total)
    recall = safe(tp, truth_total)
    iou = safe(tp, truth_total + pred_total - tp)
    accuracy = tp.sum() / matrix.sum() if matrix.sum() else float("nan")
    return precision, recall, iou, accuracy, np.nanmean(iou)


def fmt(value):
    return "  n/a " if np.isnan(value) else f"{value * 100:5.1f}%"


def print_metrics(title, matrix):
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
    print("Confusion matrix (rows = WorldCover, columns = model):")
    print(f"{'':<12}" + "".join(f"{n:>12}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}" + "".join(f"{v:>12,}" for v in matrix[i]))


MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def scene_date_key(name):
    """Sort key (year, month, day) for names like 22oct15 or 4april2018."""
    match = re.fullmatch(r"(\d{1,2})([a-z]+)(\d{2,4})", name.lower())
    if not match:
        return (9999, 0, 0)
    day, month, year = match.groups()
    year = int(year)
    year = year + 2000 if year < 100 else year
    return (year, MONTHS.get(month[:3], 0), int(day))
