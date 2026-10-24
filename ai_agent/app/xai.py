"""Explain what the land-cover model relies on, for one lake area.

Three methods, all on the same crop of the same scene:
  * OCCLUSION (fast, default): each of the 9 inputs is replaced by its typical
    value and we count how many predicted pixels change.
  * GRAD-CAM (default): a heatmap of where in the area the model's "water"
    evidence comes from, taken from an inner layer of the U-Net (gradient of
    the summed water score inside the area with respect to that layer).
  * SHAP (slow, optional): GradientSHAP on a small centre crop, giving each of
    the 9 inputs a signed contribution to the area's water probability.

Limits: these explain the MODEL, not the ground. Inputs are correlated (NDWI
is built from green and NIR), so credit can be shared or hidden. Grad-CAM is
coarse (the layer is half the image resolution). SHAP is run on a ~96 px crop
only and its numbers vary a little between runs. Needs PyTorch, the trained
model and the scene files on this machine (SHAP also needs `pip install shap`).
"""

import base64
import io
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CHANNEL_NAMES = ["B02 blue", "B03 green", "B04 red", "B08 near-infrared",
                 "B11 shortwave-infrared 1", "B12 shortwave-infrared 2",
                 "NDVI (greenness)", "NDWI (water index)",
                 "NDBI (built-up index)"]
MAX_WINDOW = 320
PAD = 32
MAX_GAP_DAYS = 45
CAM_LAYER = "dec2"          # decoder block at half resolution
SHAP_CROP = 96
SHAP_SAMPLES = 40
SHAP_BACKGROUND = 6


class XaiUnavailable(Exception):
    pass


def _lc():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    try:
        import landcover6
    except ModuleNotFoundError as error:
        missing = getattr(error, "name", "") or str(error)
        if missing == "landcover6":
            raise XaiUnavailable("ai_agent/scripts/landcover6.py not found.")
        raise XaiUnavailable(f"Missing package: {missing}. "
                             "Install it in the AI service environment.")
    return landcover6


@lru_cache(maxsize=1)
def _model():
    lc = _lc()
    if not lc.MODEL_PATH.exists():
        raise XaiUnavailable("Trained model file not found "
                             f"({lc.MODEL_PATH}). Run "
                             "train_worldcover_6band.py.")
    return lc.load_model()


@lru_cache(maxsize=1)
def _scene(name):
    lc = _lc()
    features, valid, profile, reason = lc.load_scene(lc.DATA_DIR / name)
    if features is None:
        raise XaiUnavailable(f"Scene {name}: {reason}")
    return features, valid, profile


def _scene_choices():
    lc = _lc()
    choices = []
    if not lc.DATA_DIR.exists():
        return choices
    for folder in lc.DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
        y, m, d = lc.scene_date_key(folder.name)
        if y == 9999 or m == 0:
            continue
        if all(lc.find_band(folder, b) is not None for b in lc.BANDS):
            choices.append((date(y, m, d), folder.name))
    return sorted(choices)


def prepare(bbox, when=None):
    """Crop the scene around the area; everything the methods share."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    lc = _lc()
    choices = _scene_choices()
    if not choices:
        raise XaiUnavailable("No scene folders with all six bands found.")
    chosen = choices[-1] if when is None else min(
        choices, key=lambda c: abs((c[0] - when).days))
    if when is not None and abs((chosen[0] - when).days) > MAX_GAP_DAYS:
        raise XaiUnavailable(
            f"No scene within {MAX_GAP_DAYS} days of {when} "
            f"(nearest is {chosen[0]}).")
    features, valid, profile = _scene(chosen[1])
    model = _model()

    _, height, width = features.shape
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", profile["crs"], *bbox)
    win = from_bounds(left, bottom, right, top, profile["transform"])
    col0, row0 = int(win.col_off), int(win.row_off)
    col1, row1 = int(win.col_off + win.width), int(win.row_off + win.height)
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, width), min(row1, height)
    if col1 - col0 < 4 or row1 - row0 < 4:
        raise XaiUnavailable("This area is outside the scene.")
    c0, r0 = max(col0 - PAD, 0), max(row0 - PAD, 0)
    c1, r1 = min(col1 + PAD, width), min(row1 + PAD, height)
    if c1 - c0 > MAX_WINDOW:
        mid = (col0 + col1) // 2
        c0, c1 = max(mid - MAX_WINDOW // 2, 0), min(mid + MAX_WINDOW // 2, width)
    if r1 - r0 > MAX_WINDOW:
        mid = (row0 + row1) // 2
        r0, r1 = max(mid - MAX_WINDOW // 2, 0), min(mid + MAX_WINDOW // 2, height)

    feat = features[:, r0:r1, c0:c1]
    ok = valid[r0:r1, c0:c1]
    inside = np.zeros(ok.shape, dtype=bool)
    inside[max(row0 - r0, 0):row1 - r0, max(col0 - c0, 0):col1 - c0] = True
    inside &= ok
    count = int(inside.sum())
    if count == 0:
        raise XaiUnavailable("No valid pixels in this area for that scene.")

    # lon/lat bounds of the crop, for drawing it on a map
    tr = profile["transform"]
    west, north = tr * (c0, r0)
    east, south = tr * (c1, r1)
    w84, s84, e84, n84 = transform_bounds(
        profile["crs"], "EPSG:4326", west, south, east, north)
    return {"lc": lc, "model": model, "scene": chosen, "feat": feat,
            "ok": ok, "inside": inside, "count": count,
            "image_bounds": [round(w84, 6), round(s84, 6),
                             round(e84, 6), round(n84, 6)],
            "all_features": features, "all_valid": valid,
            "origin": (r0, c0)}


# ------------------------------------------------------------------ occlusion
def occlusion(prep):
    lc, model = prep["lc"], prep["model"]
    feat, ok, inside, count = (prep["feat"], prep["ok"], prep["inside"],
                               prep["count"])
    base = lc.predict_scene(model, feat, ok)

    def shares(labels):
        return np.bincount(labels[inside], minlength=256)[:4] / count * 100

    base_share = shares(base)
    rows = []
    for index, name in enumerate(CHANNEL_NAMES):
        changed = feat.copy()
        changed[index][ok] = float(np.median(feat[index][ok]))
        pred = lc.predict_scene(model, changed, ok)
        new_share = shares(pred)
        rows.append({
            "input": name,
            "pixels_changed_pct": round(float((pred[inside] != base[inside]).mean() * 100), 1),
            "share_change_pts": {
                cls: round(float(new_share[k] - base_share[k]), 1)
                for k, cls in enumerate(lc.CLASS_NAMES)},
        })
    rows.sort(key=lambda r: -r["pixels_changed_pct"])
    top = rows[:3]
    text = (f"For {prep['scene'][1]}, the model leans most on "
            + ", ".join(f"{r['input']} ({r['pixels_changed_pct']}% of pixels "
                        "change if it is removed)" for r in top)
            + ". Occlusion explains the model, not the ground; correlated "
              "inputs can share or hide importance.")
    return {
        "method": "occlusion: each input replaced by its median in the window",
        "scene": prep["scene"][1], "scene_date": prep["scene"][0].isoformat(),
        "pixels_in_area": count,
        "predicted_shares_pct": {cls: round(float(base_share[k]), 1)
                                 for k, cls in enumerate(lc.CLASS_NAMES)},
        "inputs": rows, "explanation": text,
    }


# ------------------------------------------------------------------- Grad-CAM
def _false_colour(feat):
    """NIR, red, green as R, G, B (plants red, water dark), 0-1."""
    rgb = np.stack([feat[3], feat[2], feat[1]], axis=-1)
    return np.clip(rgb * 1.6, 0, 1)


def gradcam(prep, class_name="water"):
    import torch
    import torch.nn.functional as F
    lc, model = prep["lc"], prep["model"]
    if class_name not in lc.CLASS_NAMES:
        raise XaiUnavailable(f"Unknown class '{class_name}'.")
    k = lc.CLASS_NAMES.index(class_name)
    feat, ok, inside = prep["feat"], prep["ok"], prep["inside"]
    height, width = ok.shape
    ph, pw = -height % 4, -width % 4
    x = np.pad(feat, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    mask = np.pad(inside, ((0, ph), (0, pw)), mode="constant")
    layer = getattr(model, CAM_LAYER)

    store = {}

    def forward_hook(_, __, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            tensor = torch.from_numpy(x).unsqueeze(0).float()
            logits = model(tensor)
            score = logits[0, k][torch.from_numpy(mask)].sum()
            model.zero_grad()
            score.backward()
    finally:
        handle.remove()
    act, grad = store["act"][0].detach(), store["grad"][0].detach()
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(0, keepdim=True)).unsqueeze(0)
    cam = F.interpolate(cam, size=x.shape[1:], mode="bilinear",
                        align_corners=False)[0, 0].numpy()
    cam = cam[:height, :width]
    cam[~ok] = 0
    peak = float(cam.max())
    cam = cam / peak if peak > 0 else cam

    inside_mean = float(cam[inside].mean())
    outside = ok & ~inside
    outside_mean = float(cam[outside].mean()) if outside.any() else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 4.4 * height / width))
    ax.imshow(_false_colour(feat))
    ax.imshow(np.ma.masked_less(cam, 0.05), cmap="magma", alpha=0.65,
              vmin=0, vmax=1)
    ys, xs = np.where(inside)
    ax.plot([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()],
            [ys.min(), ys.min(), ys.max(), ys.max(), ys.min()],
            color="#ffe14d", lw=1.2)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=70)
    plt.close(fig)

    return {
        "method": f"Grad-CAM on layer {CAM_LAYER}, class {class_name}",
        "class": class_name, "scene": prep["scene"][1],
        "scene_date": prep["scene"][0].isoformat(),
        "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "image_bounds": prep["image_bounds"],
        "attention_inside_area": round(inside_mean, 3),
        "attention_outside_area": None if outside_mean is None else round(outside_mean, 3),
        "explanation": (
            f"Bright areas are where the model's {class_name} evidence comes "
            "from (false colour underneath: plants red, water dark; yellow "
            "box = the lake area). Grad-CAM is coarse because it reads an "
            "inner layer at half resolution."),
    }


# ----------------------------------------------------------------------- SHAP
def shap_bands(prep, class_name="water"):
    try:
        import shap
        import torch
    except ModuleNotFoundError as error:
        raise XaiUnavailable(f"SHAP needs `pip install shap` ({error.name}).")
    lc, model = prep["lc"], prep["model"]
    k = lc.CLASS_NAMES.index(class_name)
    feat, ok, inside = prep["feat"], prep["ok"], prep["inside"]

    # centre crop around the area, multiple of 4
    ys, xs = np.where(inside)
    cy, cx = int(ys.mean()), int(xs.mean())
    size = min(SHAP_CROP, (min(feat.shape[1], feat.shape[2]) // 4) * 4)
    y0 = int(np.clip(cy - size // 2, 0, feat.shape[1] - size))
    x0 = int(np.clip(cx - size // 2, 0, feat.shape[2] - size))
    crop = feat[:, y0:y0 + size, x0:x0 + size]
    area = inside[y0:y0 + size, x0:x0 + size]
    if not area.any():
        raise XaiUnavailable("The area is outside the SHAP crop.")
    mask = torch.from_numpy(area)

    class WaterProbability(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            prob = torch.softmax(self.net(x), dim=1)[:, k]
            return prob[:, mask].mean(dim=1, keepdim=True)

    rng = np.random.default_rng(0)
    all_feat, all_valid = prep["all_features"], prep["all_valid"]
    background = [np.stack([np.median(crop[c][ok[y0:y0 + size, x0:x0 + size]])
                            * np.ones((size, size), np.float32)
                            for c in range(crop.shape[0])])]
    tries = 0
    while len(background) < SHAP_BACKGROUND and tries < 200:
        tries += 1
        by = int(rng.integers(0, all_feat.shape[1] - size))
        bx = int(rng.integers(0, all_feat.shape[2] - size))
        if all_valid[by:by + size, bx:bx + size].mean() > 0.9:
            background.append(all_feat[:, by:by + size, bx:bx + size])
    background = torch.from_numpy(np.stack(background).astype(np.float32))

    wrapped = WaterProbability(model).eval()
    explainer = shap.GradientExplainer(wrapped, background)
    values = explainer.shap_values(torch.from_numpy(crop[None].astype(np.float32)),
                                   nsamples=SHAP_SAMPLES)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 5 and values.shape[-1] == 1:      # (batch, C, H, W, out)
        values = values[..., 0]
    values = values[0]                                  # drop batch -> (C, H, W)
    rows = []
    for index, name in enumerate(CHANNEL_NAMES):
        v = values[index][area]
        rows.append({"input": name,
                     "mean_abs_contribution": round(float(np.abs(v).mean()), 5),
                     "signed_mean_contribution": round(float(v.mean()), 5)})
    total = sum(r["mean_abs_contribution"] for r in rows) or 1.0
    for r in rows:
        r["share_of_total_pct"] = round(r["mean_abs_contribution"] / total * 100, 1)
    rows.sort(key=lambda r: -r["mean_abs_contribution"])
    with torch.no_grad():
        base_prob = float(wrapped(torch.from_numpy(crop[None])).item())
    return {
        "method": f"GradientSHAP on a {size}x{size} px crop, class {class_name}",
        "scene": prep["scene"][1], "scene_date": prep["scene"][0].isoformat(),
        "predicted_probability_in_area": round(base_prob, 3),
        "inputs": rows,
        "explanation": (
            "Positive = pushes the area's water probability up, negative = "
            "pushes it down, compared with typical scene pixels. Sampled "
            "crop only; values vary slightly between runs."),
    }


def explain(bbox, when=None, methods=("occlusion", "gradcam")):
    """Run the chosen methods; one failure never hides the others."""
    prep = prepare(bbox, when)
    out = {"scene": prep["scene"][1],
           "scene_date": prep["scene"][0].isoformat()}
    runners = {"occlusion": occlusion, "gradcam": gradcam, "shap": shap_bands}
    for name in methods:
        if name not in runners:
            continue
        try:
            out[name] = runners[name](prep)
        except XaiUnavailable as error:
            out[name] = {"unavailable": str(error)}
    return out
